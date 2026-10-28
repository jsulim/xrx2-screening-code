"""Re-run the RxRx2 embedding baseline with the pharmacy-reviewed labels.

The main analysis excludes the type-III interferons IL-28A, IL-28B, and
IL-29 (IFN-lambda ligands).  Splits are treatment-held-out: no treatment is
shared between train, validation, and test.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_ROOT = PROJECT / "rxrx2-dl-embeddings.v2" / "rxrx2"
OUT = HERE / "outputs" / "stage1_final_4class_no_ifnlambda"

GROUPS = {
    "type_I_II_interferons": ["IFN-alpha-A", "IFNB-1", "IFNE", "IFN-omega", "IFN-gamma"],
    "il1_tnf_superfamily": ["IL-1-alpha", "IL-1-beta", "TNF-alpha", "TNF-beta", "IL-18", "IL-33", "IL-36-gamma", "TL-1A"],
    "chemokines_cxc_cc": ["ENA-78", "GCP-2", "GRO-alpha", "GRO-B", "GRO-gamma", "IL-8", "IP-10", "I-TAC", "CCL-1", "CCL14a", "CCL21", "CCL28", "CCL-4", "MCP-1", "MCP-2", "MCP-3"],
    "angiogenic_vascular_regulatory": ["Angiopoietin-1", "Angiopoietin-2", "Angiopoietin-4", "VEGF-165", "VEGFC", "VEGF-D", "EG-VEGF", "PLGF-4"],
}

# Retain the prior exploratory 4-class split where possible.  Removing
# IFN-lambda leaves 3/1/1 treatments in the interferon train/val/test sets.
SPLITS = {
    "type_I_II_interferons": {"train": ["IFN-omega", "IFN-gamma", "IFNE"], "val": ["IFNB-1"], "test": ["IFN-alpha-A"]},
    "il1_tnf_superfamily": {"train": ["IL-1-alpha", "TNF-alpha", "TL-1A", "IL-1-beta", "IL-18"], "val": ["IL-33"], "test": ["TNF-beta", "IL-36-gamma"]},
    "chemokines_cxc_cc": {"train": ["I-TAC", "MCP-3", "GRO-gamma", "GRO-alpha", "MCP-1", "IL-8", "CCL-1", "GCP-2", "CCL21", "CCL-4"], "val": ["CCL28", "CCL14a"], "test": ["IP-10", "MCP-2", "GRO-B", "ENA-78"]},
    "angiogenic_vascular_regulatory": {"train": ["PLGF-4", "Angiopoietin-2", "VEGF-D", "VEGFC", "Angiopoietin-1"], "val": ["EG-VEGF"], "test": ["VEGF-165", "Angiopoietin-4"]},
}


def split_frame():
    rows = []
    for label, by_split in SPLITS.items():
        for split, treatments in by_split.items():
            rows.extend((treatment, label, split) for treatment in treatments)
    return pd.DataFrame(rows, columns=["treatment", "label", "split"])


def evaluate(name, model, x_train, y_train, x_test, y_test, test_meta):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    test_meta = test_meta.copy()
    test_meta["prediction"] = prediction
    test_meta["correct"] = prediction == y_test
    test_meta.groupby(["treatment", "label"])["correct"].agg(["mean", "count"]).reset_index().to_csv(OUT / f"{name}_per_treatment.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, prediction, labels=list(GROUPS), normalize="true"), index=GROUPS, columns=GROUPS).to_csv(OUT / f"{name}_confusion_matrix.csv")
    return {"model": name, "macro_f1": float(f1_score(y_test, prediction, average="macro")), "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(next(DATA_ROOT.rglob("metadata.csv")))
    embeddings = pd.read_csv(next(DATA_ROOT.rglob("embeddings.csv")))
    splits = split_frame()
    label_map = dict(zip(splits.treatment, splits.label))
    selected = metadata[metadata.treatment.isin(label_map)].copy()
    selected["label"] = selected.treatment.map(label_map)
    data = selected.merge(splits[["treatment", "split"]], on="treatment", validate="many_to_one").merge(embeddings, on="site_id", validate="one_to_one")
    features = [column for column in data.columns if column.startswith("feature_")]
    train, test = data[data.split == "train"], data[data.split == "test"]
    x_train, y_train = train[features].to_numpy(np.float32), train.label.to_numpy()
    x_test, y_test = test[features].to_numpy(np.float32), test.label.to_numpy()
    test_meta = test[["site_id", "treatment", "label", "experiment", "plate"]]
    models = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]),
        "small_mlp": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.15, max_iter=300, random_state=SEED))]),
    }
    results = [evaluate(name, model, x_train, y_train, x_test, y_test, test_meta) for name, model in models.items()]
    pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_csv(OUT / "baseline_summary.csv", index=False)
    splits.to_csv(OUT / "treatment_splits.csv", index=False)
    with open(OUT / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"seed": SEED, "groups": GROUPS, "excluded_treatments": ["IL-28A", "IL-28B", "IL-29"], "n_sites": len(data)}, handle, indent=2)
    print(pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_string(index=False))
    print("Sites per class/split:\n", data.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
