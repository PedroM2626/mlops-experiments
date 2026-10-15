# Boxing Stance Recognition from Pose Landmarks (MediaPipe + Gradient Boosting)

> **Area:** Computer Vision
> **Task:** 4-class classification of boxing stances from body pose landmarks (real-time webcam application)
> **Primary metric:** Accuracy on a 80-sample holdout test split (F1 per class reported)
> **Status:** Completed (migrated from the standalone repository `PedroM2626/boxing-pose-detector` on 2026-10-15)
> **Datasets:** Two self-collected sessions of 400 pose samples each (100 per class, 19 features), captured 2026-05-12 from a webcam by the author

## 1. Abstract

This experiment asks whether a boxer's **stance** — a defensive/attitudinal configuration of the whole body — can be recognised from a cheap geometric representation of the skeleton instead of from pixels. MediaPipe Holistic supplies 33 pose landmarks per frame, 19 hand-designed quantities (widths, angles, extensions and symmetry terms) are derived from them, and three gradient-boosting/random-forest classifiers are compared on 800 self-labelled samples. All four stances are separated almost perfectly: the best recorded run reaches **0.9875 accuracy (79/80)** and the worst **0.95 (76/80)**. The conclusion is deliberately narrow: pose geometry is a sufficient representation for *this* stance vocabulary under *this* collector, while the margin between the three models (one test sample) does not support any claim about which algorithm is better.

## 2. Context and Objectives

Boxing coaching software usually reacts to punches, not to the guard a fighter holds between them. The four classes here come from a well-known visual vocabulary of boxing stances in manga/anime (Mashiba's "Hitman", Ippo's "Peek-a-Boo", Glasses Joe's "No Guard" and Miyata's "Cross Guard"), which made the target postures easy to specify and to demonstrate in front of a camera.

Research questions:

1. Can stance be read off joint geometry alone, without a CNN on the image?
2. How much of the residual error is class-specific (which stances get confused)?
3. Does a boosted tree beat a random forest on 19 engineered features and 320 training samples?

The working hypothesis is recorded in the code itself: `pose_classifier.py` adds the inter-hand distance with the comment *"Specifically for boxing: distance between hands (helps differentiate Peek-a-boo and others)"* — that is, hand separation was expected to be the discriminative signal. No feature-importance run was ever logged, so the hypothesis was never tested as such; § 5.4 checks it indirectly through the error pattern.

## 3. Theoretical Background (brief)

- **MediaPipe Holistic Landmarker** (`.task` bundle): a single graph that emits pose (33 landmarks), face mesh and hands; here only `pose_landmarks` are consumed, as normalised `(x, y, z)` triples.
- **Pose features as a representation:** distances, joint angles and axis offsets are invariant to image scale only insofar as MediaPipe's normalisation is; the angle terms are computed with the dot-product/arccos rule in `PoseClassifier._calculate_angle`.
- **Tree ensembles on tabular data:** RandomForest (bagging, 100 trees, depth 10) versus boosted trees (XGBoost and LightGBM, 150 rounds, depth 6, lr 0.1, L2 1.0, L1 0.1 for XGBoost). With ~10² samples per class this is the regime where boosting's extra capacity is most likely to overfit, which is why the comparison is run under one fixed split rather than tuned.
- **Metrics:** accuracy plus per-class precision/recall/F1 from `classification_report`, and the confusion matrix, because with 80 test samples a single cell is 1.25 pp.

## 4. Methodology

### 4.1 Data

| Collection | File (in `./pose_data/`) | Samples | Features | Per class |
|---|---|---|---|---|
| A (18:46) | `pose_features_20260512_184633.npy` | 400 | 19 | 100 / 100 / 100 / 100 |
| B (18:57) | `pose_features_20260512_185753.npy` | 400 | 19 | 100 / 100 / 100 / 100 |

- Classes: `0 Hitman`, `1 Peek-a-Boo`, `2 No Guard`, `3 Cross Guard`.
- Capture: `data_collector.py` opens the webcam, holds the MediaPipe detector, and while auto-collect is on (`a`) stores the feature vector of every 5th frame (`collection_delay = 5`) under the class selected with keys `0`–`3`; 100 samples per class per session.
- `pose_features_latest.npy` is byte-identical to collection B (verified with `np.array_equal`), which is what `main.py` trains from.
- One person, one room, one camera. The labels are the stance the collector was *attempting* at capture time.

### 4.2 Preprocessing / feature engineering

No cleaning, no scaling (tree models), no outlier removal. `PoseClassifier.extract_features` maps the `(33, 3)` landmark array to 19 numbers: shoulder width, hip width, torso height, left/right elbow angle, left/right hand height relative to shoulder, left/right knee angle, torso orientation, left/right shoulder-to-wrist extension, left/right leg angle (recomputed from the knee triple), body symmetry (left-vs-right mean landmark offset), left/right hand horizontal offset from the shoulder midline, nose-vs-shoulder-midline vertical alignment, and inter-hand distance.

### 4.3 Methods compared

| Model | Configuration (`pose_classifier.py`) | Role |
|---|---|---|
| RandomForest | `n_estimators=100, max_depth=10, random_state=42` | default in `train_model`, and one arm of the comparison |
| XGBoost | `n_estimators=150, max_depth=6, learning_rate=0.1, reg_lambda=1.0, reg_alpha=0.1, random_state=42` | comparison arm |
| LightGBM | `n_estimators=150, max_depth=6, learning_rate=0.1, reg_lambda=1.0, num_leaves=31, random_state=42` | comparison arm |

### 4.4 Evaluation

- Single holdout: `train_test_split(test_size=0.2, random_state=42)`, **not stratified** → test class counts 26/18/18/18 (n=80).
- Metrics: accuracy and per-class precision/recall/F1; confusion matrix plotted per model.
- Tracking: each fit is an MLflow run (`boxing-pose-classification` experiment) logging parameters, accuracy, the classification report, the confusion matrix and, for the best model, the sklearn artefact.
- Hardware: CPU laptop, no GPU used; MediaPipe runs the landmarker in `IMAGE` mode per frame.

### 4.5 Reproduction

```bash
# one-time: the MediaPipe bundle is NOT in git (13.7 MB)
mkdir -p models && curl -L -o models/holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

pip install -r requirements.txt

python data_collector.py     # collect/extend ./pose_data (webcam)
python main.py               # live detection; 't' retrains from pose_data/*_latest.npy
```

Headless reproduction of the recorded numbers from the committed data (no webcam needed):

```python
import numpy as np, sys
sys.path.insert(0, "experiments/computer_vision/boxing_pose_detector")
from pose_classifier import PoseClassifier
b = "experiments/computer_vision/boxing_pose_detector/pose_data/"
X = np.load(b + "pose_features_latest.npy"); y = np.load(b + "pose_labels_latest.npy")
PoseClassifier(model_path="/tmp/scratch.pkl").compare_models(X, y)
```

This was executed on 2026-10-15 with the monorepo environment (Python 3.11.9, numpy 1.26.4, scikit-learn 1.4.2, xgboost 3.2.0, lightgbm 4.6.0) and reproduced **all five recorded accuracies exactly** (§ 5). Output bundle: `../../artifacts/boxing_pose_detector_20260512_192700_419b421/`.

## 5. Results

### 5.1 RandomForest, one run per collection (recorded 2026-05-12)

| Collection | Accuracy | Correct / n | Hitman P/R | Peek-a-Boo P/R | No Guard P/R | Cross Guard P/R |
|---|---|---|---|---|---|---|
| A (18:46) | **0.95** | 76 / 80 | 1.00 / 0.92 | 0.90 / 1.00 | 0.90 / 1.00 | 1.00 / 0.89 |
| B (18:57) | **0.975** | 78 / 80 | 0.96 / 0.96 | 1.00 / 0.94 | 0.95 / 1.00 | 1.00 / 1.00 |

Evidence: `rf_run_7535870b_*` and `rf_run_6e292fe0_*` (classification report + confusion matrix) in the artefact bundle.

### 5.2 Three-model comparison, collection B (recorded 19:27, and re-executed 2026-10-15)

| Model | Accuracy (recorded) | Correct / 80 | Accuracy (reproduced 2026-10-15) | Confusions observed |
|---|---|---|---|---|
| **XGBoost** | **0.9875** | 79 | **0.9875** | 1: Hitman → Peek-a-Boo |
| RandomForest | 0.9750 | 78 | 0.9750 | 2: Hitman → No Guard, Peek-a-Boo → Hitman |
| LightGBM | 0.9750 | 78 | 0.9750 | 2: Hitman → Peek-a-Boo, Peek-a-Boo → Hitman |

The comparison runs committed only their confusion matrices to git (the MLflow `metrics/` files were not versioned), so their accuracies are read off the annotated matrices as diagonal/80 — which is consistent with the two classification reports that *were* committed.

### 5.3 Which model ships

`pose_classifier_model.pkl` (488,684 bytes) is **XGBoost**, not the RandomForest the class default suggests: it is byte-size-identical to the MLflow model `m-193baba3c9db46f39776055a188dfc88`, its pickle stream names `XGBClassifier`/`xgboost.sklearn`, and that model's logged requirements list `xgboost==2.1.4` while the two other logged models (both `RandomForestClassifier`) do not. This is the `best_model_XGBoost` written by `compare_models`.

### 5.4 Error structure

The 11 misclassifications across the five recorded runs, read off the committed matrices:

| Run | Errors | True → predicted |
|---|---|---|
| A / RandomForest (0.95) | 4 | Hitman → Peek-a-Boo ×2, Cross Guard → No Guard ×2 |
| B / RandomForest (0.975) | 2 | Hitman → No Guard, Peek-a-Boo → Hitman |
| B / RandomForest (comparison) | 2 | Hitman → No Guard, Peek-a-Boo → Hitman |
| B / XGBoost | 1 | Hitman → Peek-a-Boo |
| B / LightGBM | 2 | Hitman → Peek-a-Boo, Peek-a-Boo → Hitman |

By **true** class: Hitman 6, Peek-a-Boo 3, Cross Guard 2, **No Guard 0** — No Guard is the only stance never misclassified in any run. The Hitman ↔ Peek-a-Boo pair accounts for 4 of the 11 errors, which is the direction § 2 predicted, but collection A's two `Cross Guard → No Guard` errors are not explained by it.

## 6. Discussion

- **The representation is enough, the models are indistinguishable.** 76–79 correct out of 80 for every arm of the experiment. The spread between the best and worst model is **one test sample = 1.25 pp**, well inside what a different 80-sample split would produce; the "XGBoost is best" line printed by `compare_models` is a tie-break, not a finding.
- **The split is optimistic.** Frames were captured every 5th frame of continuous takes, so near-duplicate feature vectors sit on both sides of a frame-level split. The honest reading is *recognition of the collector's own takes*, not of a new boxer or a new camera. A subject-level or take-level split is the single most informative change to make here.
- **Only one collector**, self-labelled, so the four classes are one person's interpretation of four named stances; class definitions are never operationalised in the code (no angle thresholds), which means the label semantics live entirely in the collector's head.
- **Environment drift.** The runs were logged under Python 3.13.1 / scikit-learn 1.7.1 / numpy 2.4.4 / xgboost 2.1.4 / MLflow 3.8.1 (`mlflow_logged_requirements.txt`), while `requirements.txt` now pins scikit-learn 1.8.0 / numpy 2.3.5 / MLflow 3.12.0. Reproduction under yet another stack (sklearn 1.4.2 / xgboost 3.2.0) gave identical numbers, so these results are not version-sensitive — but that is an observation about this dataset, not a general guarantee.
- **Known defects carried over unchanged** (the migration was a move, not a refactor):
  1. `extract_features(None)` returns `np.zeros(20)` while every real sample has **19** features, and `tests/test_detector.py` asserts the 20. The mismatch is masked because `predict()` wraps inference in `except Exception` and returns `("No Model", 0.0)`, so the no-pose path silently degrades instead of failing loudly.
  2. `main.py` calls `show_reference_image(None, ...)` on line 93 but `show_reference_image("None", ...)` on line 61; both happen to fall through to the "No Pose Detected" card, so the inconsistency is cosmetic.
  3. The `t` key in `main.py` re-trains by calling `load_sample_data()`, which re-fits from `pose_data/*_latest.npy` — it does not collect anything new, so "Model retrained with real data!" can mean "retrained on the same file".
  4. `holistic_detector.py` returns `None` from `_get_model_buffer()` when the `.task` file is absent, which surfaces as an opaque MediaPipe `BaseOptions` error rather than a message about the missing download.
  5. `requirements.txt` carries `pygame==2.6.1`, which **no file in the project imports** (checked across all five `.py` files) — a leftover from an earlier input-handling idea. It is kept so the migrated manifest matches what the original repository shipped.
- **Environment fit.** Of the migrated dependencies, the monorepo environment already provides opencv, numpy, scikit-learn, xgboost, lightgbm, mlflow, matplotlib, seaborn and python-dotenv; only **MediaPipe** is missing (listed commented-out at the bottom of the root `requirements.txt`), which is why the live webcam path is untested here while the classifier was reproduced in full.
- **What is not in git.** `mlruns/`, `*.db` and `*.pkl` are ignored by this monorepo, so the MLflow store and the pickled model were left behind on purpose; the reports and matrices were copied into the artefact bundle so the conclusions remain checkable, and the model regenerates from `compare_models` in seconds.

## 7. Conclusions and Recommendations

- For **stance recognition from skeleton geometry**, 19 engineered quantities over MediaPipe pose landmarks are sufficient: no CNN, no training data beyond a few minutes in front of a webcam, and CPU-only inference.
- For **model choice at this sample size**, prefer RandomForest or LightGBM over XGBoost on principle (fewer knobs, same measured result): the recorded ranking is one sample wide and should not drive a decision.
- Before quoting any number from this experiment externally, **re-split by take or by subject**; the current 0.95–0.99 band is an upper bound on generalisation, not an estimate of it.
- If this is extended, the useful next steps are: multiple collectors, an operational definition per stance (angle thresholds) to validate the labels, a take-level grouped CV with confidence intervals, and feature importance to test the § 2 hypothesis about hand height and hand separation.
- Keep `pose_classifier_model.pkl` out of the inference path assumptions: the shipped pickle is an XGBoost model, and `PoseClassifier.__init__` would otherwise build a RandomForest — loading a model of a different class than the constructor's default is a silent behaviour change.

## 8. References and Files

Code (relative to this folder):

- `./holistic_detector.py` — MediaPipe Holistic wrapper: detect, draw, extract `(33, 3)` pose landmarks
- `./pose_classifier.py` — 19-feature extractor, `train_model`, `compare_models`, predict/confidence API, MLflow logging
- `./data_collector.py` — interactive webcam collection loop (keys `0`–`3`, `a`, `s`, `t`, `l`, `q`)
- `./main.py` — live detection app with the reference-stance window
- `./tests/test_detector.py` — 4 unit tests, using the monorepo's `sys.path.insert` convention. MediaPipe is not part of the root environment, so the module-level `importorskip` skips all four there (`pytest experiments/computer_vision/boxing_pose_detector/tests` → 1 skipped); `test_detector_initialization` also skips when `models/holistic_landmarker.task` is absent.
- `./requirements.txt`, `./Dockerfile` — environment (mediapipe, opencv, xgboost, lightgbm, mlflow)
- `./images/` — the four reference stance photographs used by the overlay window
- `./pose_data/` — the two 400-sample collections and their metadata JSON

Evidence bundle: `../../artifacts/boxing_pose_detector_20260512_192700_419b421/` — `metrics.json` (all recorded and reproduced values with provenance), the two RandomForest classification reports, the five confusion matrices, and the environment recorded by MLflow for the shipped model.

History: the project's 4 commits live in the local clone of the former `PedroM2626/boxing-pose-detector` repository; the standalone GitHub repository was deleted after this migration.

References: Gravity et al. (2009) *Realtime and Lightweight Pose Estimation with Automated Aggregation* (Holistic) — MediaPipe Holistic Landmarker documentation, https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker ; Chen & Guestrin (2016) *XGBoost*; Ke et al. (2017) *LightGBM*; Breiman (2001) *Random Forests*.
