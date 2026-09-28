# Computer Vision — Experiments and Applications

> **Area:** Computer Vision
> **Task:** Image classification (multiclass and multi-label), detection and face recognition
> **Primary metric:** Accuracy (CIFAR-10) / F1-macro (multi-label)
> **Status:** Completed
> **Datasets:** CIFAR-10 (50,000 train / 10,000 test, 10 classes); own pet dataset (44 images, 2 multi-label classes); local face dataset; COCO/custom for YOLO

## 1. Abstract

This folder gathers four computer vision notebooks: a comparison of three paradigms (HOG+SVM, ResNet18 and ViT) on CIFAR-10 — won by **ViT with 0.9805 accuracy** —, a multi-label study of pet classification with four approaches (ResNet18, VGG16, CLIP zero-shot and EfficientNet) — won by **ResNet18 with F1-macro 1.000** —, a face recognition app with LBPH/CNN/transfer (YuNet) modes and a detection notebook with YOLO (OpenCV DNN). It is concluded that pretrained transformers and supervised fine-tuning are the highest-accuracy paths, while manual features (HOG) fail on low-resolution images.

## 2. Context and Objectives

The group investigates the range of computer vision techniques available for problems at scale:
1. **CIFAR-10 (cv-methods-comparison.ipynb):** quantify the jump from manual representations (HOG) to deep networks (ResNet18 residual CNN) and to vision transformers (ViT pretrained on ImageNet-21k), besides the computational cost of each.
2. **Multi-label pets (animal-classifier.ipynb):** compare 4 flows (PyTorch/Keras/CLIP) for the problem of activating multiple labels on the same image — two cats (Dime and Frida) that appear together in some photos.
3. **Face recognition and YOLO:** provide functional applications (app embedded in the notebook and YOLO detection via OpenCV DNN) without depending on external scripts.

Research questions: *does the architectural jump matter more in vision than in text? Does supervised fine-tuning beat frozen backbones and zero-shot in low-data scenarios?*

## 3. Theoretical Background (brief)

- **HOG (Histogram of Oriented Gradients):** classical representation based on local gradients per cell (cell/block); effective for pedestrian detection at medium resolution, but with low generalization capacity for varied classes.
- **Residual CNNs (ResNet18):** blocks with residual connections allow deep training without gradient degradation; fine-tuning on ImageNet weights transfers generic texture/shape features.
- **Vision Transformer (ViT):** splits the image into linearized patches and applies global self-attention; pretraining on giant corpora (ImageNet-21k, 14M images/21k classes) yields qualitative advantages over CNNs pretrained on ImageNet-1k.
- **Multi-label learning:** BCEWithLogitsLoss + sigmoid per class; Exact Match, Hamming Loss, F1-micro/macro, precision/recall metrics.
- **CLIP zero-shot:** aligns text-image (ViT-B/32); classification via class prototypes (mean embedding) and cosine similarity with a threshold.
- **EfficientNet-B0:** compound scaling (depth × width × resolution).
- **Face recognition:** LBPH (LBP histograms + distance), a CNN trained from scratch and transfer learning with MobileNetV2 on faces detected by YuNet.
- **YOLO (OpenCV DNN):** one-stage detection (YOLOv3-tiny COCO) for object classification in uploaded images.

## 4. Methodology

### 4.1 Data

**CIFAR-10 experiment (cv-methods-comparison.ipynb):**
- CIFAR-10: 50,000 training images, 10,000 test, 10 classes, 32×32, color (RGB).
- HOG+SVM used a subsample of **10k train / 2k test** due to computational limits; ResNet18 and ViT used **50k train / 10k test**.
- Hardware: NVIDIA RTX 4070 Laptop GPU (8GB), Python 3.8, PyTorch 2.4.

**Multi-label experiment (animal-classifier.ipynb):**
- 44 labeled images (22 per class: Dime and Frida); multi-label because the two cats appear together in some photos.
- Split: 60% training (30), 15% validation (7), 25% test (7), stratified by dominant class.
- Caution: a very small dataset prevents robust generalization (perfect values should be interpreted with reservations).

**Face recog & YOLO:** local dataset `dataset/<name>/` (collection by upload); YOLO uses COCO (YOLOv3-tiny) or a custom model (car, motorbike, threewheel, van, bus, truck), downloaded automatically via `.env`.

### 4.2 Preprocessing

- CIFAR-10: resizing to 224×224 with ImageNet normalization (ResNet18 and ViT); for ViT, the model's own normalization; HOG based on 9 orientations, cell 8×8, block 3×3, generating **2,916 features**.
- Pets: Data Augmentation (horizontal flip, ±15° rotation, color jitter, affine) applied to the ResNet18 and EfficientNet flows; ImageNet normalization.
- Faces: crop of detected faces, saved under `dataset/<name>/`.

### 4.3 Methods compared

| Notebook | Flows/Architectures | Strategy |
|---|---|---|
| cv-methods-comparison.ipynb | HOG+SVM; ResNet18 (fine-tune 5 epochs, Adam lr=1e-4, batch 128); ViT `google/vit-base-patch16-224-in21k` (fine-tune 2 epochs, Adam lr=2e-5, batch 32) | Paradigmatic: manual → CNN → Transformer |
| animal-classifier.ipynb | ResNet18+Aug (fine layer4+FC, adam lr=1e-4, 10 epochs); VGG16 (frozen + head 128/Dropout0.2 + sigmoid, 6 epochs); CLIP zero-shot (prototypes, threshold 0.75); EfficientNet-B0+Aug (blocks 4-5+FC, 10 epochs) | Supervised vs zero-shot |
| face_recognition_app.ipynb | LBPH (baseline), CNN (from scratch), transfer_yunet (MobileNetV2 + YuNet detection) | Face recognition |
| yolo_notebook.ipynb | YOLOv3-tiny COCO (OpenCV DNN) | One-stage detection |

Env settings for the face app: `FACE_DETECTOR=yunet\|haar`; `FACE_TL_EPOCHS`, `FACE_TL_BATCH`; `FACE_CNN_EPOCHS`, `FACE_CNN_BATCH`; `YUNET_SCORE_THRESHOLD`, `YUNET_NMS_THRESHOLD`, `YUNET_TOP_K`.

### 4.4 Evaluation

- CIFAR-10: **accuracy** on test and training time; per-class analysis (F1) for each method.
- Multi-label: **Exact Match, Hamming Loss, F1-micro, F1-macro, micro precision, micro recall** on the test set (7 images).
- Face recog: prediction by upload with visualization of the result.
- Deterministic seeds and the same split for all multi-label flows.

### 4.5 Reproduction

- Notebooks with embedded outputs; open in Jupyter (Jupyter Notebook / VS Code) and run cell by cell.
- `cv-methods-comparison.ipynb` requires an NVIDIA GPU (RTX 4070) and PyTorch 2.4.
- Structural validation scripts: `python scripts/validate_notebooks.py`.
- Output bundle (atypical for this model group): `experiments/artifacts/<experiment>_<timestamp>_<sha>/`.

## 5. Results

### 5.1 CIFAR-10 — Paradigm comparison (cv-methods-comparison.ipynb)

| Method | Accuracy | Time | Paradigm | Data |
|--------|----------|-------|-----------|-------|
| **ViT** | **0.9805** | ~17 min (1 epoch) | Pretrained visual transformer (ImageNet-21k) | 50k train |
| **ResNet18** | **0.9362** | 12.5 min (5 epochs) | Pretrained residual CNN (ImageNet) | 50k train |
| HOG+SVM | 0.3970 | 27 min | Manual features + SVM | 10k train |

**Per-class analysis (F1):** HOG+SVM best on `automobile` (0.54 F1, straight edges) and worst on `cat` (0.25 F1, non-rigid shape). ResNet18: best on `ship` (0.99 precision), `bird` (0.97), `horse` (0.97); worst `cat` (0.84 precision, 0.87 F1). ViT dominates every class by a margin.

ResNet18 training behaviour: fast saturation (epoch 1 = 0.9323, oscillates around ~0.94). ViT reached 0.9805 in **1 epoch**.

**Fairness note (derived from the numbers above, without a new run):** HOG used 5× less data (10k vs 50k) and still cost more (27 min vs 12.5 min ResNet / ~17 min ViT). Cost per 1k samples: HOG ~2.7 min (+SVM O(n²·d) at d=2,916), ResNet ~0.25 min, ViT ~0.34 min. That is, even normalized per sample HOG loses on acc (0.3970) and on cost — the qualitative conclusion (avoid HOG on CIFAR) holds, but a head-to-head comparison requires HOG at 50k or everyone at 10k.

### 5.1b HOG at 50k/10k — fair comparison (`run_hog_full.py`)

Same recipe (gray → 64×64 → HOG 9/8×8/3×3 → StandardScaler → LinearSVC C=1),
parallelized extraction (joblib), HF `cifar10` data (the local tarball
`data/cifar-10-python.tar.gz` is truncated — EOFError verified in gzip):

| Method | Accuracy | Time | Data |
|--------|----------|-------|-------|
| **ViT** | **0.9805** | ~17 min | 50k train |
| **ResNet18** | **0.9362** | 12.5 min | 50k train |
| HOG+SVM (full) | 0.5381 | ~33 min (1958 s) | 50k train / 10k test |
| HOG+SVM (subsample, original) | 0.3970 | 27 min | 10k / 2k |

5× more data lifted HOG from 0.3970 → **0.5381 (+14 pp)**, and the per-class
pattern held (`automobile` best, `cat` worst) — but the gap to ResNet
(−39.8 pp) remains abyssal: manual features do not scale, and they still cost
more time than fine-tuning. Artifacts: `experiments/artifacts/hog_cifar10_20260908_121832/metrics.json`.

### 5.2 Multi-label pets — 4 approaches (animal-classifier.ipynb)

| Metric | ResNet18 + Aug | VGG16 | CLIP zero-shot | EfficientNet + Aug |
|---------|:--------------:|:-----:|:--------------:|:------------------:|
| **Exact Match** | **1.000** | 0.429 | 0.000 | 0.714 |
| **Hamming Loss** | 0.000 | 0.286 | 0.500 | 0.143 |
| **F1-micro** | **1.000** | 0.714 | 0.667 | 0.833 |
| **F1-macro** | **1.000** | 0.714 | 0.664 | 0.829 |
| Micro precision | 1.000 | 0.714 | 0.500 | 1.000 |
| Micro recall | 1.000 | 0.714 | 1.000 | 0.714 |

**Details per flow:**
- **ResNet18**: perfect performance (F1-macro 1.000), a result of the low generalization complexity (7 test images); selective fine-tuning (layer4+FC) is sufficient and BCEWithLogitsLoss is adequate for multi-label.
- **VGG16**: winner at 0.714; better for Frida (F1=0.86) than for Dime (0.57); the frozen backbone limits domain adaptation (−28.6 pp vs ResNet18).
- **CLIP**: F1-macro 0.664, recall 1.0, precision 0.5, exact match 0.0; the 0.75 threshold is too permissive (false positives); the prototypes capture the classes, but threshold calibration is critical.
- **EfficientNet+Aug**: 2nd place (0.829), +11.5 pp over VGG16, conservative profile (precision 1.0, recall 0.714) — it omits 28.6% of the positive predictions; compound scaling needs more fine-tuning to calibrate the sigmoid.

### 5.3 Face Recognition App (face_recognition_app.ipynb)

- Flow embedded in the notebook: face collection by upload, training (LBPH/CNN/YuNet) and prediction by upload with visualization.
- Objective evaluation: `eval_detection.py::classification_metrics(y_true, y_pred)` (accuracy, F1-macro/micro, confusion matrix) on a labeled split; `face_verification_metrics(distances, same_person)` sweeps the distance threshold and returns the best acc + curve. Tests in `tests/test_eval_detection.py`.
- Modes: `lbph` (OpenCV baseline), `cnn` (small CNN), `transfer_yunet` (MobileNetV2 + YuNet).

### 5.4 YOLO (yolo_notebook.ipynb)

- Classification/detection by upload using YOLOv3-tiny COCO via OpenCV DNN; the custom classes are shown when a trained model is downloaded.
- Objective evaluation: `eval_detection.py::detection_map(pred_boxes, pred_scores, true_boxes, iou_thr=0.5)` (single-class mAP + AP per image). Annotate a validation subset with boxes and run the harness — without relying on visual inspection.

## 6. Discussion

- **Manual features do not scale**: HOG+SVM reaches 0.3970 on CIFAR-10; the gradient representation is sufficient for rigid shapes (automobile) but not for the variability of cats — and the cost (27 min) does not even compensate.
- **Transformers ≈ the new standard**: ViT beats ResNet18 by 4.4 pp with just 1 epoch. In the 16th experiment (DistilBERT vs TF-IDF+SVC in NLP) the architectural jump was smaller (0.9 pp), suggesting that in vision pretraining on 21k classes gives a larger qualitative advantage over mid-sized data (50k).
- **Supervised fine-tuning dominates multi-label**, but the 44-image dataset prevents strong conclusions; ResNet18's perfect results should be read with caution (beneficial overfitting).
- **Zero-shot is an option for zero data**, yet threshold calibration is the decisive factor: at 0.75 CLIP gained recall and lost precision (exact match 0.000).
- **Limitations**: room for hardware budgets (GPU mandatory in the CIFAR-10 comparison); face app and YOLO now have a metrics protocol in `eval_detection.py` (requires an annotated subset for mAP).

## 7. Conclusions and Recommendations

- For **medium-sized image classification**: use ViT for maximum accuracy (~0.985+ with 3 epochs, ~50 min) or ResNet18 for fast prototyping (0.9362, 12.5 min); avoid HOG (not recommended).
- For **multi-label with few data**: prefer selective supervised fine-tuning (ResNet18), which reduced overfitting with data augmentation; if there are no labels, CLIP zero-shot requires careful threshold calibration (0.75 is too permissive).
- For **production applications**: the face app notebook runs on CPU (LBPH runs on CPU; `transfer_yunet` is faster with a GPU, but also works on CPU) and YOLO enables object detection via OpenCV DNN without training.

## 8. References and Files

- Notebooks (relative to this folder):
  - `./cv-methods-comparison.ipynb` — CIFAR-10 comparison (HOG+SVM / ResNet18 / ViT)
  - `./animal-classifier.ipynb` — multi-label pets (ResNet18, VGG16, CLIP, EfficientNet)
  - `./face_recognition_app.ipynb` — face recognition app (LBPH / CNN / transfer_yunet)
  - `./yolo_notebook.ipynb` — YOLO detection via OpenCV DNN
  - `./kd-cifar10-comparison.ipynb` — knowledge distillation on CIFAR-10
  - `./yolo_inference.py`, `./eval_detection.py` — inference and detection
    evaluation used by the YOLO notebook
  - `./tests/test_yolo_inference.py`, `./tests/test_eval_detection.py` — the
    unit tests for both (run with `pytest experiments/computer_vision/tests`)
- References: Deng et al. (2009) CIFAR-10; He et al. (2016) *Deep Residual Learning*; Dosovitskiy et al. (2021) *An Image is Worth 16x16 Words*; Radford et al. (2021) *Learning Transferable Visual Models From Natural Language Supervision* (CLIP); Tan & Le (2019) *EfficientNet: Rethinking Model Scaling*; Sedhain et al. (2015) — see also the documents in `docs/academic-readme-template.md`.