import pandas as pd
from sklearn.metrics import f1_score
from constants import CONDITIONS   # all labels across sections


def load_section(path):
    return pd.read_csv(path)


def merge_sections(section_files):
    dfs = [load_section(p) for p in section_files]

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(
            merged,
            df,
            on="Report Impression",
            how="outer"
        )

    # remove duplicate columns if any
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def compute_final_f1(gt_csv, section_files):
    pred_df = merge_sections(section_files)
    gt_df = pd.read_csv(gt_csv)

    merged = pd.merge(
        gt_df,
        pred_df,
        on="Report Impression",
        suffixes=("_true", "_pred")
    )

    

    y_true = merged[[c + "_true" for c in CONDITIONS]].values
    y_pred = merged[[c + "_pred" for c in CONDITIONS]].values

    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")

    print("FINAL ECG F1")
    print("Macro:", macro)
    print("Micro:", micro)

section_files = [
    "output_rhythm/labeled_reports.csv",
    "output_form/labeled_reports.csv",
    "output_sub/labeled_reports.csv",
    "output_super/labeled_reports.csv",
]

compute_final_f1("ground_truth.csv", section_files)

