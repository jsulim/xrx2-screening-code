# RxRx2 reliability-aware immune-stimulant screening

Reproducible analysis code for the accompanying ICCE paper. The project studies broad functional-group screening from public RxRx2 HUVEC Cell Painting embeddings and a raw-image pretrained CNN extension.

## Scope

- 37 pharmacy-reviewed immune/vascular treatments grouped into four broad functional classes.
- Type-III interferon ligands IL-28A, IL-28B and IL-29 are excluded from the principal HUVEC analysis.
- Primary protocol: treatment-held-out evaluation. No treatment is shared across train, validation and test partitions.
- Raw images, archives and checkpoints are deliberately excluded from Git.

## Layout

- `src/`: embedding baselines and reliability analyses.
- `notebooks/`: GPU notebook for raw six-channel image CNN and channel ablation.
- `results/embedding_reliability/`: committed CSV outputs from the completed embedding experiments.
- `figures/`: committed analysis figures.

## Data

Obtain RxRx2 metadata, released deep-learning embeddings and images from the dataset provider. Set the paths described at the top of each script/notebook. Do not commit downloaded data or model checkpoints.

## Key completed results

- Treatment-held-out released-embedding benchmark: XGBoost Macro-F1 0.447; Logistic balanced accuracy 0.442.
- Well aggregation from one to six replicates: balanced accuracy 0.416 to 0.465; low-concentration subset 0.294 to 0.396.
- Same-plate within-condition cosine similarity 0.885 versus 0.862 across plates; unsupervised plate centering modestly improved Logistic BA from 0.443 to 0.450.
- ImageNet-pretrained 6-channel MobileNetV3-Small raw-image strict test: Macro-F1 0.402 and BA 0.400. This value is recorded in the paper text; the associated checkpoint remains external.

These results should be interpreted as reliability-aware screening evidence, not as a clinical diagnostic classifier.
