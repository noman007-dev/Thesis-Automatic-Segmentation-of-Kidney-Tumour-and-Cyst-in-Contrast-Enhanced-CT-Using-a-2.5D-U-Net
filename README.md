# Kidney, Tumour & Cyst Segmentation in CT using a 2.5D U-Net with Uncertainty Quantification (KiTS23)

Automatic segmentation of **kidney, tumour, and cyst** in contrast-enhanced CT using a lightweight **2.5D U-Net**, with a controlled **2D vs 2.5D ablation** and **Monte Carlo Dropout** uncertainty estimation, trained and evaluated on the [KiTS23](https://github.com/neheller/kits23) dataset.

This repository accompanies the BSc thesis *"Automatic Segmentation of Kidney, Tumour, and Cyst in Contrast-Enhanced CT Using a 2.5D U-Net with Uncertainty Quantification on KiTS23"*, American International University-Bangladesh (AIUB), Faculty of Science and Technology.

---

## Overview

Full 3D networks (e.g. nnU-Net ensembles) perform strongly on KiTS23 but are expensive to train and deploy. Plain 2D networks are cheap but ignore context between slices. This project studies the middle ground:

- **2.5D input**: three neighbouring axial slices are stacked as a 3-channel input to a standard 2D U-Net.
- **Depth ablation**: the same model is retrained with a single slice (depth = 1) to isolate the effect of inter-slice context.
- **Uncertainty**: MC Dropout (p = 0.3, 20 stochastic passes) with calibration and error-detection analysis.
- **Volume-stratified analysis**: cyst Dice is analysed against cyst volume instead of being averaged away.

Everything runs end to end in a single Colab notebook on one GPU (about 11 GPU-hours for the whole study).

## Pipeline

```
KiTS23 download
  -> HU window [-100, 300] -> per-volume z-score -> resize to 192x192 -> .npz cache
  -> patient-level split (342 / 73 / 74, stratified on tumour presence)
  -> foreground-biased slice sampling (13 fg + 3 bg slices per patient per epoch)
  -> 3-slice stack [3, 192, 192] -> 2D U-Net -> softmax (4 classes)
  -> loss = 0.5 * weighted CE + 0.5 * soft Dice
  -> test evaluation + depth ablation + MC Dropout + volume-stratified analysis
```

## Model and training

| Item | Setting |
|---|---|
| Architecture | 2D U-Net, 4 encoder / 4 decoder stages, filters 32-64-128-256, 512 bottleneck |
| Parameters | 7,763,140 (2.5D) / 7,762,564 (2D, depth = 1) |
| Input / output | 3 x 192 x 192 / 4 x 192 x 192 (background, kidney, tumour, cyst) |
| Loss | 0.5 CE (class weights 0.1 / 1 / 3 / 5) + 0.5 soft Dice |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine annealing |
| Training | 40 epochs, batch size 32, gradient clip 1.0, seed 42 |
| Augmentation | Horizontal flip (p = 0.5), Gaussian noise (sigma = 0.05) |
| Checkpoint selection | Best mean of validation kidney and tumour Dice |
| Hardware | 1 x NVIDIA A100-SXM4-80GB (Google Colab), about 3.2 h training |

## Results (74 held-out test patients)

| Class | Global voxel Dice | Per-patient mean Dice | 95% CI (bootstrap) |
|---|---|---|---|
| Kidney | 0.879 | 0.879 | [0.858, 0.897] |
| Tumour | 0.672 | 0.433 | [0.359, 0.501] |
| Cyst | 0.434 | 0.219 | [0.150, 0.297] |

### 2D vs 2.5D ablation (per-patient mean Dice)

| Class | 2D (depth 1) | 2.5D (depth 3) | Delta |
|---|---|---|---|
| Kidney | 0.877 | 0.879 | +0.003 |
| Tumour | 0.386 | 0.433 | +0.046 |
| Cyst | 0.162 | 0.219 | +0.057 |

### Key findings

- Adding neighbouring slices improves **tumour** segmentation and leaves **kidney** essentially unchanged.
- Cyst Dice rises with cyst size (Pearson r = 0.61, n = 42), so small-lesion resolution and class imbalance explain much of the failure.
- MC Dropout **predictive entropy** ranks errors reasonably well (AUC = 0.861; variance of tumour probability: 0.749), but thresholding uncertainty finds very few errors (RIU < 0.05).
- Global voxel Dice is higher than per-patient Dice for small classes because large lesions dominate voxel counts. Both are reported.

> This is a single small model, cheaper than the KiTS23 winning ensemble but also less accurate. It is a research and teaching baseline, **not a clinical tool**.

## Repository structure

```
.
├── kits23_segmentation_FINAL.ipynb   # Full Colab notebook (data -> figures)
├── README.md
├── requirements.txt                  # Python package versions
├── LICENSE
├── results/                          # CSVs and figures reported in the thesis
├── models/                           # Trained model checkpoints
└── splits/                           # Deterministic patient-level split files
```

## Getting started

### Requirements

- Google Colab with an A100 GPU (the notebook mounts Google Drive for caching)
- Python 3.13, PyTorch 2.11.0 (CUDA 12.8)
- `nibabel`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `huggingface_hub`, `tqdm`

### Reproduce the results

1. Clone this repository and open `kits23_segmentation_FINAL.ipynb` in Colab.
2. Select a GPU runtime and run the cells in order. The notebook will:
   - check the environment and install dependencies,
   - mount Google Drive for the preprocessing cache,
   - download KiTS23 from its HuggingFace mirror,
   - preprocess and cache all 489 cases (resumable),
   - build the 342 / 73 / 74 split, train, evaluate, run the ablation and MC Dropout analysis.
3. Figures and CSVs are written to `results/`.

Seeds are fixed (`SEED = 42`) and split IDs are saved in `splits/`, so every run uses the same patient-level split.

### Data

KiTS23 is not redistributed here. Get it from the official [KiTS23 repository](https://github.com/neheller/kits23) or the HuggingFace mirror used by the notebook. Please follow the dataset's own license and terms.

## Limitations

- Class weights are hand-tuned, not cross-validated.
- Checkpoint selection ignores cyst Dice (cyst peaked earlier, at epoch 12, than the selected epoch 39).
- Validation and test global metrics use a slice stride of 12, so they are strided estimates.
- The effective MC Dropout rate (dropout injected after every ReLU) was not measured; p = 0.3 is the configured value.
- Versions of some auxiliary libraries are not recorded, and the notebook uses Colab-specific Drive mounting.

## Authors

- Abdullah Al Noman
- Md. Irfan Hossain
- Abir Ahosan Ratul
- Md. Fahim Siddiki

**Supervisor:** Supta Richard Philip, Lecturer, Department of Computer Science, AIUB

## Citation

```bibtex
@thesis{kits23_25d_unet_uncertainty,
  title  = {Automatic Segmentation of Kidney, Tumour, and Cyst in Contrast-Enhanced CT
            Using a 2.5D U-Net with Uncertainty Quantification on KiTS23},
  author = {Noman, Abdullah Al and Hossain, Md. Irfan and Ratul, Abir Ahosan and Siddiki, Md. Fahim},
  school = {American International University-Bangladesh},
  year   = {2026},
  type   = {BSc thesis}
}
```

Please also cite the KiTS23 challenge:

> N. Heller et al., "The 2023 Kidney and Kidney Tumor Segmentation Challenge," 2023. https://kits-challenge.org/kits23/

## License

See [LICENSE](LICENSE).

## Acknowledgements

Thanks to our supervisor Supta Richard Philip for his guidance, and to the KiTS23 organisers for releasing the dataset.
