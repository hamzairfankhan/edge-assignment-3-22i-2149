# Edge Computing Assignment 3

**Hamza Khan | 22i-2149 | Section A**

Repository for the two assignment questions: static post-training quantization,
quantization-aware training, iterative magnitude pruning, and physical channel pruning.

## Setup

Python 3.12. Install the packages in `requirements.txt`:

```sh
python -m pip install -r requirements.txt
```

The measured environment is Windows on an Intel CPU, four PyTorch threads,
with the `onednn` quantized backend. Set `EDGE_THREADS` to change the thread count.
Quantized inference requires a PyTorch build supporting `onednn`; the scripts fail
explicitly if it is unavailable. This project uses PyTorch eager quantization and
TorchScript for directly reloadable INT8 checkpoints. PyTorch emits deprecation
notices for these APIs, but the submitted runtime and artifacts were tested.

CIFAR-10 downloads automatically into `data/`. To reuse an existing download, set
the `CIFAR_ROOT` environment variable to the folder containing `cifar-10-batches-py`.
The dataset is not committed.

## Reproduce

Run from this repository:

```sh
python run_all.py
```

Individual stages, in dependency order:

```sh
python train_baseline.py --epochs 15
python ptq.py
python qat.py --epochs 3
python prune_unstructured.py
python prune_structured.py
python benchmark.py
python export_onnx.py
python verify.py
```

The optimization methods independently start from the same trained FP32 checkpoint.
No method starts from another method's optimized model. `calibrate.py` supplies the
PTQ calibration routine. Both pruning scripts include fine-tuning.

## Inference without retraining

All five `.pt` files are executable TorchScript models and need no architecture code
for loading. Float `.pth` files also contain state dictionaries and architecture metadata
for training and reconstruction.

```sh
python inference.py --model fp32
python inference.py --model ptq_int8
python inference.py --model qat_int8
python inference.py --model unstructured
python inference.py --model structured
python inference.py --model structured --image photo.png
python onnx_inference.py --model qat_int8
```

Image inputs are resized to 32 x 32 RGB, divided by 255, then normalized by
mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`.
Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Model and experimental protocol

`common.py` implements a CIFAR-adapted **ResNet-18 with reduced stage widths
16/32/64/128**. It has a 3 x 3 stride-1 stem, no initial max pool, eight residual
blocks with two convolutions each, global average pooling, and a 10-class head.
It is an 18-layer residual network, not the canonical 64/128/256/512-width
ImageNet model. This choice makes the full training and edge deployment experiment
practical on CPU. All methods use the same architecture and data split initially.

Seed 2149 defines the 45,000 training / 5,000 validation split; all 10,000 official
test images are reserved for final evaluation. Training uses random horizontal
flips, batch 256, SGD with momentum 0.9, weight decay 0.0005, and cosine learning
rate decay. Baseline: 15 epochs, initial LR 0.08. QAT: 3 epochs, LR 0.002.
PTQ calibration: first 1,024 images of the fixed training split.

Unstructured pruning removes 20% of remaining convolution/linear weights at each
of three global L1 iterations (48.8% total), with one recovery epoch at LR 0.005
per round. Masks remain active throughout recovery and are removed permanently
before export. Structured pruning retains the strongest 50% of each residual
block's internal conv1 channels, slices BN1 and conv2 inputs consistently, and
fine-tunes for 3 epochs at LR 0.01. Residual output widths remain unchanged.

## Results and deliverables

- `models/`: five executable `.pt` models, float training `.pth` checkpoints,
  and all five `.onnx` deployment files.
- `results/benchmark.csv`: test accuracy, size, sparsity, MACs and CPU timing.
- `results/predictions.csv`: 50,000 per-model test predictions with ground truth,
  correctness and the associated model's measured median/p95 inference performance.
  These repeated performance values are model-level measurements, not individual
  image timings.
- `results/latency_samples.csv`: all 1,000 measured batch-1 latency samples.
- `results/onnx_validation.csv`: full-test ONNX validation and logit parity checks.
- Training histories, pruning rounds, retained channel indices and environment metadata
  are saved as JSON. `results/verification.json` records artifact checks and SHA-256 hashes.
- The PDF report explains the method, important code, results and conclusions
  for each question. Screenshots show actual recorded outputs displayed in Colab;
  the numerical experiments ran on the documented Windows CPU.

Latency protocol: 30 warm-up runs, then 200 batch-1 forward passes using
`perf_counter_ns`, with median and p95 reported. Model loading, preprocessing and
disk access are excluded. Test throughput is measured separately at batch 256.
Dense MAC counts include convolutions and linear layers, not BN, ReLU or pooling.
Zero-valued weights still count as dense work; no sparse kernel is claimed.
File size is serialized deployment size, not peak RAM. No energy or MCU measurement
is claimed. ONNX exports preserve INT8 quantize/dequantize operations and run with
ONNX Runtime's CPU provider.

## Submission format

The instructor's November 13-14 Classroom announcements supersede the older ZIP
instruction in the PDF: submit **this GitHub repository link and the PDF report**.
No compressed archive is submitted.

## Measured results

| Model | Test accuracy | Executable MiB | Batch-1 median ms |
|---|---:|---:|---:|
| fp32 | 84.27% | 2.762 | 0.693 |
| ptq_int8 | 84.24% | 0.738 | 4.883 |
| qat_int8 | 84.52% | 0.738 | 4.882 |
| unstructured | 84.13% | 2.770 | 0.679 |
| structured | 82.74% | 1.466 | 0.540 |

INT8 is about 3.75 times smaller here, but slower for batch-1 inference. QAT has the highest measured test accuracy; structured pruning has the lowest median latency and physically fewer parameters. At batch 256, QAT completed the test pass in 0.627 seconds versus 2.451 seconds for FP32. These are CPU measurements, not MCU or energy claims.
