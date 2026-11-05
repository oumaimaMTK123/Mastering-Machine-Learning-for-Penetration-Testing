"""Strict source-grouped manifests and bounded-memory, timestamp-based decoding."""
import csv
from pathlib import Path
import av
import numpy as np


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'path', 'label', 'split', 'source_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty manifest')
    groups, files, labels = {}, set(), {}
    for r in rows:
        p = (path.parent / r['path']).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if p in files:
            raise ValueError(f'Duplicate video: {p}')
        files.add(p)
        r['path'] = str(p)
        if r['split'] not in {'train', 'val', 'test'} or not r['source_id']:
            raise ValueError('Invalid split or missing source_id')
        group = r['source_id']
        if group in groups and groups[group] != r['split']:
            raise ValueError(f'Source leakage: {group}')
        if group in labels and labels[group] != r['label']:
            raise ValueError(f'Conflicting source label: {group}')
        groups[group], labels[group] = r['split'], r['label']
    train_labels = {r['label'] for r in rows if r['split'] == 'train'}
    if len(train_labels) < 2:
        raise ValueError('Need at least two training classes')
    if any(r['label'] not in train_labels for r in rows):
        raise ValueError('Evaluation class absent from training')
    return rows


def decode(path, frames=16, stride=4, clips=3):
    """Two passes: measure timestamps, then retain only selected RGB frames.

    Sampling at stride/average_fps seconds approximates the 16x4 K400 recipe.
    VFR is handled by selecting the closest presentation timestamp. Short videos
    repeat their boundary frames. Each returned array is T,H,W,C uint8.
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 0)
        if fps <= 0:
            raise ValueError('Video has no usable frame rate')
        times = []
        for f in c.decode(s):
            if f.time is None:
                raise ValueError('Video has no presentation timestamps')
            times.append(float(f.time))
    if not times or np.any(np.diff(times) < 0):
        raise ValueError('Empty video or nonmonotonic timestamps')
    times = np.array(times)
    span = (frames - 1) * stride / fps
    starts = np.linspace(times[0], max(times[0], times[-1] - span), clips)
    if clips == 1:
        starts[0] = times[0] + max(0, times[-1] - times[0] - span) / 2
    targets = starts[:, None] + np.arange(frames)[None, :] * stride / fps
    right = np.searchsorted(times, targets).clip(0, len(times)-1)
    left = (right-1).clip(0, len(times)-1)
    indices = np.where(abs(times[left]-targets) <= abs(times[right]-targets), left, right)
    wanted, selected = set(indices.ravel().tolist()), {}
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in wanted:
                selected[i] = f.to_ndarray(format='rgb24')
            if len(selected) == len(wanted):
                break
    videos = [np.stack([selected[int(i)] for i in row]) for row in indices]
    return videos, {'fps': fps, 'decoded_frames': len(times),
                    'duration_seconds': float(times[-1]-times[0]+1/fps),
                    'clip_endpoint_span_seconds': span,
                    'indices': indices.tolist()}
