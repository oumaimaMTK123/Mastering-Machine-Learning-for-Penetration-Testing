"""Frozen VideoMAE extraction, resumable linear probe, and held-out evaluation."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from data import decode, read_manifest
from model import embed, load_encoder


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def initialize(cfg):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    if cfg['device'].startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.reset_peak_memory_stats()


def peak(cfg):
    if cfg['device'].startswith('cuda'):
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return None


def extract(cfg, limit=None):
    rows = read_manifest(cfg['manifest'])
    if limit:
        rows = rows[:limit]
    model, processor, loading = load_encoder(cfg)
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    features, shuffled, metadata = [], [], []
    begin = time.perf_counter()
    generator = np.random.default_rng(cfg['seed'])
    for r in rows:
        clips, meta = decode(r['path'], cfg['frames'], cfg['stride'], cfg['clips'])
        normal = [embed(model, processor, v, cfg['device']) for v in clips]
        perturbed = [embed(model, processor, v[generator.permutation(len(v))], cfg['device']) for v in clips]
        features.append(torch.stack(normal).mean(0))
        shuffled.append(torch.stack(perturbed).mean(0))
        metadata.append({**r, **meta})
        print(json.dumps({'video': r['path'], 'feature_dim': len(features[-1])}), flush=True)
    torch.save({'features': torch.stack(features), 'shuffled': torch.stack(shuffled),
                'rows': rows, 'config': cfg,
                'resolved_revision': getattr(model.config, '_commit_hash', None)}, out/'features.pt')
    save_json(out/'extraction.json', {'seconds': time.perf_counter()-begin,
        'peak_gpu_bytes': peak(cfg), 'videos': metadata, 'loading_info': loading,
        'processor': processor.to_dict(), 'model_config': model.config.to_dict()})


def read_features(cfg):
    path = Path(cfg['output'])/'features.pt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = torch.load(path, map_location='cpu', weights_only=True)
    rows = data['rows']
    classes = sorted({r['label'] for r in rows if r['split']=='train'})
    y = torch.tensor([classes.index(r['label']) for r in rows])
    masks = {s: torch.tensor([r['split']==s for r in rows]) for s in ('train','val','test')}
    if not masks['train'].any() or not masks['val'].any():
        raise ValueError('Both train and val features are required; pilot subset is insufficient')
    return data, y, masks, classes, digest


@torch.inference_mode()
def scores(head, x, y, device):
    predicted = head(x.to(device)).argmax(1).cpu()
    return {'accuracy': accuracy_score(y, predicted),
            'macro_f1': f1_score(y, predicted, average='macro', zero_division=0,
                                labels=list(range(head.out_features))),
            'videos': len(y)}


def probe(cfg, resume=False):
    data, y, masks, classes, digest = read_features(cfg)
    x = data['features'].float()
    mean = x[masks['train']].mean(0)
    std = x[masks['train']].std(0, unbiased=False).clamp_min(1e-6)
    x = (x-mean)/std
    head = torch.nn.Linear(x.shape[1], len(classes)).to(cfg['device'])
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    out = Path(cfg['output']); start, best, history, elapsed = 0, -1., [], 0.
    if resume:
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        if state['feature_sha256'] != digest or state['classes'] != classes:
            raise ValueError('Features/classes changed; cannot resume')
        for k in ('lr','weight_decay','seed'):
            if state['config'][k] != cfg[k]:
                raise ValueError(f'Resume configuration changed: {k}')
        head.load_state_dict(state['head']); opt.load_state_dict(state['optimizer'])
        for state_values in opt.state.values():
            for key, value in state_values.items():
                if isinstance(value, torch.Tensor):
                    state_values[key] = value.to(cfg['device'])
        start, best, history, elapsed = state['epoch']+1, state['best'], state['history'], state['elapsed_seconds']
        torch.set_rng_state(state['rng'])
    begin = time.perf_counter()
    train_x, train_y = x[masks['train']].to(cfg['device']), y[masks['train']].to(cfg['device'])
    for epoch in range(start, cfg['epochs']):
        head.train(); opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(train_x), train_y)
        loss.backward(); opt.step(); head.eval()
        val = scores(head, x[masks['val']], y[masks['val']], cfg['device'])
        record = {'epoch': epoch, 'train_loss': loss.item(), 'val': val}
        history.append(record); print(json.dumps(record), flush=True)
        improved = val['accuracy'] > best
        best = max(best, val['accuracy'])
        state = {'head': head.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch,
                 'best': best, 'history': history, 'mean': mean, 'std': std, 'classes': classes,
                 'feature_sha256': digest, 'config': cfg, 'rng': torch.get_rng_state(),
                 'elapsed_seconds': elapsed+time.perf_counter()-begin}
        torch.save(state, out/'last.pt')
        if improved:
            torch.save(state, out/'best.pt')
    save_json(out/'probe.json', {'history': history, 'seconds': elapsed+time.perf_counter()-begin,
                              'peak_gpu_bytes': peak(cfg)})


def evaluate(cfg):
    data, y, masks, classes, digest = read_features(cfg)
    state = torch.load(Path(cfg['output'])/'best.pt', map_location='cpu', weights_only=True)
    if state['feature_sha256'] != digest or state['classes'] != classes:
        raise ValueError('Feature checkpoint mismatch')
    mask = masks['test']
    if not mask.any():
        raise ValueError('No test videos; evaluation cannot run')
    head = torch.nn.Linear(data['features'].shape[1],len(classes)).to(cfg['device'])
    head.load_state_dict(state['head']); head.eval()
    begin = time.perf_counter()
    results = {}
    for name, key in [('normal','features'),('shuffled','shuffled')]:
        x = (data[key][mask]-state['mean'])/state['std']
        results[name] = scores(head,x,y[mask],cfg['device'])
    results['accuracy_drop_shuffled'] = results['normal']['accuracy']-results['shuffled']['accuracy']
    results['seconds'] = time.perf_counter()-begin
    results['peak_gpu_bytes'] = peak(cfg)
    save_json(Path(cfg['output'])/'test.json',results)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['inspect','extract','probe','evaluate'])
    p.add_argument('--config',default='config.json')
    p.add_argument('--limit',type=int)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text()); initialize(cfg)
    if args.command=='inspect':
        rows=read_manifest(cfg['manifest'])
        report=[{**r, **decode(r['path'],cfg['frames'],cfg['stride'],1)[1]} for r in rows]
        save_json(Path(cfg['output'])/'dataset.json',report)
        print(json.dumps({'videos':len(rows),'splits':{s:sum(r['split']==s for r in rows) for s in ('train','val','test')}}))
    elif args.command=='extract': extract(cfg,args.limit)
    elif args.command=='probe': probe(cfg,args.resume)
    else: evaluate(cfg)
