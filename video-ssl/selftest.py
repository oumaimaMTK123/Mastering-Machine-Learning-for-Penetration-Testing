"""Offline engineering pilot. Synthetic videos and RANDOM tiny weights, no benchmark."""
import csv
import json
from pathlib import Path
from fractions import Fraction
import av
import numpy as np
import torch
from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel
from data import read_manifest
from pipeline import initialize, extract, probe, evaluate


def run():
    root=Path('runs/selftest'); root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(17)
    model=VideoMAEModel(VideoMAEConfig(image_size=32,patch_size=16,num_frames=4,
        hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4))
    checkpoint=root/'random-tiny'
    model.save_pretrained(checkpoint)
    VideoMAEImageProcessor(size={'shortest_edge':32},
        crop_size={'height':32,'width':32}).save_pretrained(checkpoint)
    rows=[]
    for split in ['train','val','test']:
        for label in [0,1]:
            name=f'{split}-{label}.mp4'
            with av.open(str(root/name),'w') as out:
                stream=out.add_stream('mpeg4',rate=Fraction(8,1))
                stream.width=48; stream.height=40; stream.pix_fmt='yuv420p'
                for i in range(12):
                    rgb=np.zeros((40,48,3),dtype=np.uint8)
                    rgb[:,:,label]=80
                    rgb[:,(i*3)%40:(i*3)%40+8,:]=255
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                        out.mux(packet)
                for packet in stream.encode(): out.mux(packet)
            rows.append({'path':name,'label':str(label),'split':split,'source_id':name})
    manifest=root/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg={'checkpoint':str(checkpoint),'revision':'main','manifest':str(manifest),
         'output':str(root/'features'),'frames':4,'stride':2,'clips':2,'seed':17,
         'device':'cpu','threads':2,'epochs':2,'lr':0.01,'weight_decay':0.01}
    initialize(cfg); extract(cfg); probe(cfg)
    before=torch.load(root/'features/last.pt',weights_only=True)['head']['weight'].clone()
    cfg['epochs']=3; probe(cfg,resume=True)
    after=torch.load(root/'features/last.pt',weights_only=True)['head']['weight']
    assert not torch.equal(before,after), 'Probe weights did not update on resume'
    evaluate(cfg)
    # Prove partition guard actually rejects a repeated source across splits.
    rows[-1]['source_id']=rows[0]['source_id']
    bad=root/'leaking.csv'
    with bad.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    try: read_manifest(bad)
    except ValueError as e: assert 'leakage' in str(e)
    else: raise AssertionError('Source leakage was accepted')
    results={'status':'passed','videos':6,'feature_shape':[6,32],
             'checkpoint':'random tiny LOCAL engineering fixture, not pretrained',
             'checks':['encode/decode','processor and shape','local checkpoint loading',
                       'forward','normal/shuffled embedding saving','probe updates',
                       'checkpoint resume','held-out metrics','source leakage rejection'],
             'cuda_available':torch.cuda.is_available()}
    (root/'validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': run()
