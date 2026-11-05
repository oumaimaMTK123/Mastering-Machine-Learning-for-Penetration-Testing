# Video representation learning: executable frozen baseline

## Decision

| Item | Recommendation | Reason |
|---|---|---|
| Objective family | Masked pixel reconstruction, VideoMAE | Compact official Transformers implementation; no teacher/predictor infrastructure needed for the initial frozen experiment |
| Encoder | VideoMAE ViT-B/16, 12 blocks, width 768, 12 heads, tubelet 2 | 16 RGB frames, 224 pixels; full architecture forward tested on this CPU |
| Candidate checkpoint | `MCG-NJU/videomae-base` (not `*-finetuned-kinetics`) | Intended Kinetics-400 SSL base checkpoint; remote config/card/weights remain **unverified because Hugging Face is blocked** |
| Dataset | UCF101 official split 1, user-supplied videos and split lists | No real videos provided; action recognition is the initial task |
| Training strategy | A: frozen pretrained encoder + supervised **linear** probe | CPU only, no target videos; no evidence justifies SSL adaptation |
| SSL adaptation B | Not performed or recommended yet | First establish baseline and domain shift; requires target training videos and a measured GPU pilot |
| Random-initialization SSL C | Not recommended | No dataset or accelerator; tiny random selftest is an engineering fixture only |
| Supervised encoder fine-tuning D | Separate future experiment, not implemented/run | Would answer a different question and require labels and compute |

### V-JEPA comparison and verified sources

The [official V-JEPA repository](https://github.com/facebookresearch/vjepa2) lists V-JEPA 2.1
ViT-B/16, 80M parameters at 384 resolution, checkpoint
`https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`,
and hub entry `vjepa2_1_vit_base_384`. This is **distilled** from ViT-G.
The release uses dense prediction of context and masked latent tokens and deep
supervision. The repository identifies the paper as
“V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning”, but its
paper URL currently contains `TODO`; no independently verified paper or exact
2.1 teacher data inventory is claimed here. Do not assume absence of data overlap,
labels or text supervision solely from the name. Code is mostly MIT, with separately
licensed portions described in its README. Check checkpoint-specific terms before use.
Its larger spatial token grid and teacher distillation recipe make it a poorer
first CPU experiment despite similar parameter count. No comparative accuracy or
GPU-memory claim has been measured.

[VideoMAE paper, NeurIPS 2022](https://arxiv.org/abs/2203.12602),
[official implementation](https://github.com/MCG-NJU/VideoMAE),
[model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md),
[official K400 800-epoch recipe](https://github.com/MCG-NJU/VideoMAE/blob/main/scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/pretrain.sh),
[checkpoint page](https://huggingface.co/MCG-NJU/videomae-base),
[Transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/videomae).
Fetched source evidence is retained under `research/`. VideoMAE upstream code uses
CC BY-NC 4.0; the checkpoint card/license still needs verification after network access.
The implementation here uses Transformers, rather than copied upstream training code.

VideoMAE masks about 90% of spatial tubes (same spatial mask across time), encodes
visible tubelets and uses a lightweight decoder to reconstruct masked normalized
RGB patches with MSE. Targets are actual video pixels, not an EMA teacher;
gradients flow through decoder and visible-token encoder. Pixel-specific targets
make constant collapsed predictions inadequate. No negatives or target encoder
are needed. Latent JEPA prediction avoids reconstructing texture and is attractive
for motion, but brings teacher/predictor and checkpoint compatibility requirements.
Contrastive learning adds view/negative-batch choices and can learn static cues;
it is not selected for this CPU baseline.

The chosen run **uses previously learned representations**. It does not newly
learn a video encoder through SSL. The only trained parameters are one affine
classification layer. No attentive head is substituted for a linear probe.
For future continued SSL, use `VideoMAEForPreTraining` only with a verified complete
encoder+decoder checkpoint (never silently initialize a missing decoder), training
partition videos only, tube masking, normalized masked-patch loss, then verify
finite loss and encoder gradients before scaling. That stage is deliberately not
presented as implemented or run here.

## Environment and protocol

Observed: Python 3.12.14, PyTorch 2.7.1+cpu, Transformers 4.57.1, PyAV 16.0.1;
CUDA unavailable and `nvidia-smi` absent; approximately 60 GiB free initially.
No GPU VRAM measurement is possible. No video dataset size, fps, duration,
domain-shift estimate or labels are known because no real data was supplied.
`inspect` measures fps/duration/frame counts on supplied files; variable-rate
videos use closest presentation timestamps, with bounded retained RGB frames.
Decoding makes two full passes per video; this is reliable but not optimized for
very long recordings. Pre-cut long recordings into externally defined source groups.

Protocol: original UCF101 train/test lists for split 1. Hold out 20% of original
training source groups per class (seed 17) for validation. Test is untouched until
hyperparameters are fixed. `prepare_ucf101.py` groups all `_gXX_cYY` siblings by
source group, and `data.py` rejects cross-partition source leakage and repeated
paths. This is one split with a reduced training partition, not the standard
three-split published benchmark. Repeat fixed protocols across three splits/seeds
before making research claims. Select probe hyperparameters on validation only.

Use 16 frames at a stride of four **native-fps** frame intervals, matching the
official K400 temporal recipe. First-to-last temporal span is 60/fps seconds
(2 seconds at 30fps); short clips repeat boundary frames. Three deterministic
equally spaced temporal windows per video, one spatial center crop each. Actual
resize, crop, rescaling, mean and std come from the checkpoint's own processor;
never replace them with guessed ImageNet constants. Spatial transforms are the
same deterministic operations for every frame. No random augmentation in frozen
extraction. Mean pool all space-time tokens, then average clips into one 768-D
video embedding; train-only feature standardization before the affine classifier.

External pretraining on Kinetics-400 may overlap UCF101 by source or near-duplicate
content; overlap is not audited here. Record that limitation, checkpoint card,
resolved revision (saved in features), processor and config for every run. `main`
is only a discovery revision: after authorized download, pin the returned full
commit SHA in config before a substantive experiment and regenerate embeddings.
Reject any card revealing label/text training or distillation if a pure SSL
comparison is required. Loading fails on missing/mismatched encoder weights;
unexpected decoder keys from a pretraining checkpoint are logged.

## Exact commands

```bash
cd /workspace/video-ssl
bash install.sh
source /workspace/.venvs/ml-pentesting/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export MPLCONFIGDIR=/workspace/.cache/matplotlib
python selftest.py
```

Acquire the UCF101 videos and official recognition splits from its
[dataset owner](https://www.crcv.ucf.edu/data/UCF101.php), respecting its terms.
No automatic dataset download or additional domain allowlist is assumed.
Keep the original class directories and filenames. Then:

```bash
python prepare_ucf101.py --videos /workspace/datasets/UCF-101 \
  --splits /workspace/datasets/ucfTrainTestlist --split 1 --output data/manifest.csv
python pipeline.py inspect --config config.json
# First real pretrained smoke test: one video, including three clips and shuffled controls.
# Use a separate output to preserve full features:
python - <<'PY'
import json
c=json.load(open('config.json')); c['output']='runs/pretrained-pilot'
json.dump(c,open('pilot.json','w'),indent=2)
PY
python pipeline.py extract --config pilot.json --limit 1
# Only proceed after inspecting pilot extraction.json and checkpoint provenance.
python pipeline.py extract --config config.json
python pipeline.py probe --config config.json
# Increase epochs in config before resuming; feature fingerprint must match.
python pipeline.py probe --config config.json --resume
# Final test operation, after all model/hyperparameter decisions:
python pipeline.py evaluate --config config.json
```

For a custom dataset, CSV columns are `path,label,split,source_id`. Relative paths
are relative to the CSV. Each row is one video; source IDs must identify shared
original recordings, not just individual cuts. At least two training classes
and train/val partitions are required. Use `test` only for final evaluation.
Labels need not be numeric. All three partitions must use the same vocabulary.

`features.pt` saves normal and frame-shuffled embeddings with metadata;
`extraction.json` logs wall time, peak allocated CUDA bytes (null on CPU),
decoding metadata, processor, model config and loading warnings. `probe.json`
records per-epoch training loss and validation metrics; `last.pt` holds optimizer,
RNG, standardization, classes, configuration and feature SHA256; `best.pt` is
selected on validation accuracy. `test.json` reports accuracy, macro-F1 across
all train classes, and normal-minus-shuffled accuracy. Shuffle order is seeded;
the same trained probe is applied without retraining. Shuffling can introduce
unnatural transitions and distribution shift; an accuracy drop is a diagnostic,
not proof of causal temporal reasoning. Time is measured, not a projected full-run
budget. GPU memory is allocated-tensor peak, not total device usage.

## Actual validation and blockers

`runs/selftest/validation.json`: six encoded/decoded synthetic videos; local tiny
random checkpoint loading; [6,32] normal and shuffled embeddings; linear head
updates; save/resume; evaluation; deliberate source leakage rejected. Selftest
metrics are fixtures and have no scientific meaning. `runs/full-shape.json`
records a full ViT-B forward on random weights with [1,16,3,224,224] input and
[1,1568,768] output. This does **not** validate pretrained weight loading.

No real pretrained download/forward, UCF101 pilot, scientific accuracy, adaptation,
fine-tuning, GPU timing or VRAM measurement has run. Hugging Face HTTPS requests
returned policy HTTP 403; required host additions are saved in the cloud draft.
Review/save network settings and publish the environment when appropriate;
draft saving alone does not grant runtime access. Then verify checkpoint config,
card, license, exact revision and the first pretrained forward before calling
the scientific baseline ready. No secret/API key is presently required.

Original book checkout is untouched. Its Chapter02 decision-tree/logistic scripts
and Chapter10 feature selectors can use the same venv; legacy Python2/Theano/TF1
examples are not modernized by this environment setup.
