# Deep Learning for Healthcare (CS 598, UIUC — MCS-DS)

Graduate coursework in deep learning applied to clinical and healthcare data. All notebooks render directly in GitHub — click any link in the table below to read without downloading or running Jupyter.

---

## Assignments

| Notebook | Topic | Key Techniques | View |
|---|---|---|---|
| HW2 | Neural Networks for Healthcare | PyTorch, feedforward NN, BCELoss, precision/recall/F1/AUC | [notebook](HW2_NN.ipynb) · [pdf](HW2_NN.pdf) |
| HW4 — pyhealth | EHR Data Preprocessing | pyhealth, patient-visit sequences, feature dictionaries, ICD/drug/procedure coding | [notebook](HW4_pyhealth_preprocess_1.ipynb) · [pdf](HW4_pyhealth_preprocess_1%20(1).pdf) |
| HW4 — RETAIN | Sequential EHR Event Prediction | RETAIN architecture, reverse-time attention (α/β), GRU encoders, sequence masking, BCELoss | [notebook](HW4_RETAIN.ipynb) · [pdf](HW4_RETAIN.pdf) |
| HW4 — MINA | ECG Arrhythmia Classification | Multi-level attention CNN (beat/rhythm/frequency), replication of MINA paper | [notebook](HW4_MINA_1.ipynb) · [pdf](HW4_MINA_1%20(1).pdf) |
| HW4 — Autoencoders | Unsupervised EHR Feature Learning | Vanilla, sparse, and denoising autoencoders in PyTorch, reconstruction loss comparison | [notebook](HW4_Autoencoder.ipynb) · [pdf](HW4_Autoencoder.pdf) |
| Final Project | Hospital Readmission Prediction | CONTENT model replication — topic modeling + neural readmission prediction architecture | [report](Replication%20of%20Paper%20Proposing%20the%20CONTENT%20Readmission%20Prediction%20Model.docx) |

---

## What each assignment demonstrates

### HW2 — Neural Networks
Feedforward network trained on a healthcare classification task in PyTorch. Covers architecture construction, training loop, loss function selection (BCELoss for binary outcomes), and evaluation across precision, recall, F1, and ROC-AUC — the standard evaluation suite for imbalanced clinical outcomes.

### HW4 — pyhealth Preprocessing
End-to-end EHR preprocessing using the `pyhealth` toolkit: loading raw patient records, constructing patient-visit sequence objects, building feature dictionaries from ICD codes, drug codes, and procedure codes, and packaging tensors for downstream model training. Foundation step for the RETAIN and autoencoder assignments.

### HW4 — RETAIN
Implementation of the RETAIN architecture (Choi et al., 2016) for sequential clinical event prediction over longitudinal patient visit histories. Key components:
- **Dual attention mechanisms:** alpha (visit-level importance) and beta (variable-level importance within a visit)
- **Two GRU encoders** running in reverse time order over the visit sequence
- **Sequence masking** for variable-length patient timelines
- BCELoss training loop with validation reporting

RETAIN's interpretable attention weights make it directly applicable to payer ML use cases where model explainability is required (prior auth, risk stratification, HEDIS measure prediction).

### HW4 — MINA (Multi-Level Attention for ECG)
Replication of the MINA architecture for ECG arrhythmia classification. Combines CNN feature extraction with multi-level attention operating at beat, rhythm, and frequency granularities. Demonstrates the pattern of stacking local feature extractors with hierarchical attention — applicable beyond ECG to any structured temporal clinical signal.

### HW4 — Autoencoders
Three autoencoder variants implemented in PyTorch for unsupervised feature learning over EHR-style input features:
- **Vanilla autoencoder:** standard encoder-decoder with reconstruction loss
- **Sparse autoencoder:** L1 penalty on activations to learn sparse representations
- **Denoising autoencoder:** trained to reconstruct clean input from corrupted input — more robust representations, applicable to noisy claims data

### Final Project — CONTENT Model Replication
Replication of *"Predicting Hospital Readmission via the CONTENT Model"* — a topic-modeling-augmented neural architecture that extracts latent clinical topics from patient notes/codes and feeds them alongside structured features into a readmission prediction network. Full written report included.

---

## Stack
Python · PyTorch · pyhealth · NumPy · scikit-learn · Matplotlib · Jupyter

## Course
CS 598 — Deep Learning for Healthcare · University of Illinois Urbana-Champaign · MCS-DS (School of Engineering)
