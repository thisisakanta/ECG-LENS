import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from models.bert_labeler import bert_labeler
from collections import OrderedDict
from datasets.unlabeled_dataset import UnlabeledDataset
from constants import *
from tqdm import tqdm
from sklearn.metrics import f1_score


# -------------------------
# Collate
# -------------------------
def collate_fn_no_labels(sample_list):
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch


# -------------------------
# Data loader
# -------------------------
def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, shuffle=False):
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_no_labels
    )
    return loader


# -------------------------
# Prediction
# -------------------------
def label(checkpoint_path, csv_path):
    ld = load_unlabeled_data(csv_path)

    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 0:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling:")
    with torch.no_grad():
        for data in tqdm(ld):
            batch = data['imp'].to(device)
            src_len = data['len']
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

    y_pred = [t.tolist() for t in y_pred]
    return y_pred


# -------------------------
# Save predictions
# -------------------------
def save_preds(y_pred, csv_path, out_path):
    y_pred = np.array(y_pred).T

    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    df = df[['Report Impression'] + CONDITIONS]

    df.replace(3, -1, inplace=True)
    df.replace(2, 0, inplace=True)

    out_file = os.path.join(out_path, 'labeled_reports.csv')
    df.to_csv(out_file, index=False)

    print("Saved predictions to:", out_file)
    return df


# -------------------------
# F1 Evaluation
# -------------------------
def compute_f1_scores(gt_csv, pred_df):
    gt_df = pd.read_csv(gt_csv)

    y_true = gt_df[CONDITIONS].values
    y_pred = pred_df[CONDITIONS].values

    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    weighted = f1_score(y_true, y_pred, average="weighted")

    print("\n===== F1 SCORES =====")
    print(f"Macro F1:    {macro:.4f}")
    print(f"Micro F1:    {micro:.4f}")
    print(f"Weighted F1: {weighted:.4f}")

    # per-label
    per_label = f1_score(y_true, y_pred, average=None)
    print("\nPer-label F1:")
    for cond, s in zip(CONDITIONS, per_label):
        print(f"{cond}: {s:.4f}")


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Label ECG reports and compute F1'
    )
    parser.add_argument('-d', '--data', required=True,
                        help='CSV with reports')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output folder')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Model checkpoint')
    parser.add_argument('-g', '--ground_truth', required=False,
                        help='Ground truth CSV for F1')

    args = parser.parse_args()

    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint
    gt_csv = args.ground_truth

    y_pred = label(checkpoint_path, csv_path)
    pred_df = save_preds(y_pred, csv_path, out_path)

    if gt_csv is not None:
        compute_f1_scores(gt_csv, pred_df)
