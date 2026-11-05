"""Encoder loading fails explicitly; random weights are never a fallback."""
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel


def load_encoder(cfg):
    processor = VideoMAEImageProcessor.from_pretrained(
        cfg['checkpoint'], revision=cfg['revision'])
    model, info = VideoMAEModel.from_pretrained(
        cfg['checkpoint'], revision=cfg['revision'], output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete encoder checkpoint: {info}')
    if model.config.num_frames != cfg['frames']:
        raise ValueError('Clip length does not match checkpoint')
    model.eval().requires_grad_(False).to(cfg['device'])
    return model, processor, info


@torch.inference_mode()
def embed(model, processor, video, device):
    pixels = processor(list(video), return_tensors='pt')['pixel_values'].to(device)
    tokens = model(pixel_values=pixels).last_hidden_state
    # A true frozen-feature protocol: mean of all space-time tokens, no attention probe.
    return tokens.mean(dim=1).squeeze(0).cpu()
