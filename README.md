# Progetto_EIS

## Overview

Progetto_EIS is a research-grade toolkit for multimodal retrieval, segmentation, and report generation for brain tumor MRI studies. It integrates state-of-the-art deep learning models to segment brain tumors, retrieve similar cases, and automatically generate professional medical reports. The project is designed for clinical research, educational, and benchmarking purposes on datasets such as BraTS2023 and MultiCaRe.

The repository is organized in two main retrieval paradigms:
- **MMRAG**: Multimodal Retrieval-Augmented Generation based on deep fusion of imaging and text.
- **VRAG**: Visual Retrieval-Augmented Generation using CLIP-based methods.

It also includes a segmentation pipeline utilizing MONAI's SegResNet, and interactive chat/report tools via Gradio.

## Directory Structure

- `ASNR-MICCAI-BraTS2023-Challenge-TrainingData/` — Contains BraTS2023 MRI images.
- `medical_datasets/` — Contains the MultiCaRe dataset for multimodal brain tumor analysis.
- `models/monai_brats_mri_segmentation/` — MONAI segmentation model weights and documentation.
- `evaluation_results/` — Stores evaluation outputs.
- `MMRAG/` — Scripts for multimodal retrieval, fusion, and report generation.
- `VRAG/` — CLIP-based retrieval tools and reporting.
- `vector_db/` — Vector databases for retrieval.
- `pesi_fusion/` — Weights for the multimodal fusion model.

## Main Components

### Segmentation

- **segmentator.py**: Implements the `Segmentator` class to preprocess MRIs, run segmentation, generate heatmaps, plot slices, save masks, and compute tumor volumes and spatial localization (hemisphere/lobe).
- Utilizes MONAI's SegResNet and torchcam's GradCAM++ for interpretability.

### Retrieval & Report Generation

- **MMRAG/MMRAG_rad_genome.py**, **MMRAG/MMRAG_multicare.py**: `MRAG` class for multimodal search using BiomedCLIP, vector DBs, and fusion modules.
- **VRAG/VRAG_rad_genoma.py**, **VRAG/VRAG_multicare.py**: `VRAG` class for CLIP-based retrieval and report generation.
- Both paradigms allow querying by image or text, returning similar cases and associated professional reports.

### Interactive Chat

- **MMRAG/main_chat.py**: Gradio interface for uploading new cases and interacting with the system via diagnostic chat, including image segmentation display and report generation.

## Setup & Installation

1. **Requirements**  
   - Python 3.10 or higher
   - NVIDIA GPU with CUDA support
   - Install required packages:
     ```
     pip install -r requirements.txt
     ```

2. **Datasets**  
   - Place BraTS2023 images in `ASNR-MICCAI-BraTS2023-Challenge-TrainingData`.
   - Place MultiCaRe dataset in `medical_datasets`.

3. **Model Weights**  
   - Download MONAI SegResNet weights and place in `models/monai_brats_mri_segmentation`.

4. **Run Segmentation and Report**  
   Example usage from `MMRAG/main.py`:
   ```python
   from pipeline import run
   import matplotlib.pyplot as plt

   case_name = 'BraTS-GLI-00000-000'
   user_prompt = 'Generate a professional medical report'
   image, report = run(case_name, user_prompt)
   print(report)
   plt.imshow(image)
   plt.show()
   ```

## Usage

- **Interactive Chat**  
  Launch the Gradio app to interact with the system, upload cases and get segmentation & report:
  ```
  python MMRAG/main_chat.py
  ```

- **Batch Processing**  
  See scripts in `MMRAG/` and `VRAG/` for batch evaluation and retrieval (`test_pipeline_MMRAG.py`, `test_pipeline_VRAG.py`, etc.).

- **Model Training**  
  For training MONAI models, see documentation in `models/monai_brats_mri_segmentation/docs/README.md`.

## Main Classes & Modules

- `Segmentator`: Preprocesses and segments MRI scans, generates visualizations, computes tumor metrics.
- `MRAG` (MMRAG): Multimodal retrieval and fusion, vector DB building, report generation.
- `VRAG`: CLIP-based image/text retrieval and report generation.
- `Trainer`: Training utilities for fusion modules (see `MMRAG/training_*` scripts).

## Citation & License

Please refer to `models/monai_brats_mri_segmentation/docs/README.md` and `models/monai_brats_mri_segmentation/LICENSE` for details about model performance, reproducibility, and licensing.

## References

- MONAI: https://monai.io/
- CLIP: https://openai.com/blog/clip/

---

For further details, consult the documentation in each module and the [MONAI segmentation README](models/monai_brats_mri_segmentation/docs/README.md).
