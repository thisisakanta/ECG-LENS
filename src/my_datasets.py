import torch
from torch.utils.data import Dataset
import h5py
import os
import ast

import numpy as np
import pandas as pd

import prompting


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder=None, split=None, transform=None, ecgs=None, captions=None, caplens=None, read_from_hdf5=True):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        if read_from_hdf5:
            self.split = split
            assert self.split in {'TRAIN', 'VAL', 'TEST'}

            # Open hdf5 file where images are stored
            self.h = h5py.File(os.path.join(data_folder, self.split + '_DATASET.hdf5'), 'r')
            self.secgs = self.h['secgs']
            self.captions = self.h['captions']
            self.caplens = self.h['caplens']
        else:
            self.secgs = ecgs
            self.captions = captions
            self.caplens = caplens

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform


    def __getitem__(self, i):
        secg = self.secgs[i]
        if not isinstance(secg, torch.Tensor):
            secg = torch.FloatTensor(secg)
        if self.transform is not None:
            secg = self.transform(secg)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([round(self.caplens[i])])

        return secg, caption, caplen

    def __len__(self):
        return self.dataset_size


def load_and_split_data(data_folder, train_csv, val_csv, test_csv):
    """reads an h5py file and split it to train/val/test according to csv files"""

class TransformerDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, use_scp_labels=False, merl_predictor=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        :param merl_predictor: optional MerlLabelPredictor. When provided, the
            clinical prompt is built from MERL-predicted diagnostic labels
            (predict-then-generate) instead of ground-truth ``scp_codes``.
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_DATASET.hdf5'), 'r')
        self.secgs = self.h['secgs'][:]
        self.captions = self.h['captions'][:]
        self.caplens = self.h['caplens'][:]

        self.use_scp_labels = use_scp_labels
        self.merl_predictor = merl_predictor
        self.scp_labels = None
        self.scp_class_names = []
        self.scp_label_texts = []
        if self.use_scp_labels:
            self.scp_labels, self.scp_class_names, self.scp_label_texts = self._build_scp_labels(data_folder, split)

        self.dataset_size = len(self.captions)

    @staticmethod
    def _aggregate_diagnostic_classes(scp_codes, diagnostic_df):
        labels = []
        if not isinstance(scp_codes, dict):
            return labels
        for code in scp_codes.keys():
            if code in diagnostic_df.index:
                diag_class = diagnostic_df.loc[code].diagnostic_class
                if isinstance(diag_class, str) and diag_class:
                    labels.append(diag_class)
        return sorted(list(set(labels)))

    @staticmethod
    def _aggregate_diagnostic_subclasses(scp_codes, diagnostic_df):
        labels = []
        if not isinstance(scp_codes, dict):
            return labels
        for code in scp_codes.keys():
            if code in diagnostic_df.index:
                diag_subclass = diagnostic_df.loc[code].diagnostic_subclass
                if isinstance(diag_subclass, str) and diag_subclass:
                    labels.append(diag_subclass)
        return sorted(list(set(labels)))

    # The prompt format and its clinical phrasing live in prompting.py so that
    # training and scripts/predict.py build the prompt identically.
    _class_name_to_full_form = staticmethod(prompting.class_name_to_full_form)
    _build_subclass_description_map = staticmethod(prompting.build_subclass_description_map)

    @staticmethod
    def _aggregate_full_descriptions(scp_codes, diagnostic_df):
        labels = []
        if not isinstance(scp_codes, dict):
            return labels
        for code in scp_codes.keys():
            if code in diagnostic_df.index:
                description = diagnostic_df.loc[code].description
                if isinstance(description, str) and description:
                    labels.append(description)
        return sorted(list(set(labels)))

    @staticmethod
    def _aggregate_codes_by_flag(scp_codes, agg_df, flag_column):
        labels = []
        if not isinstance(scp_codes, dict) or flag_column not in agg_df.columns:
            return labels

        for code in scp_codes.keys():
            if code not in agg_df.index:
                continue
            flag_value = agg_df.at[code, flag_column]
            flag_numeric = pd.to_numeric(flag_value, errors="coerce")
            if pd.notna(flag_numeric) and float(flag_numeric) == 1.0:
                labels.append(code)

        return sorted(list(set(labels)))

    def _build_scp_labels(self, data_folder, split):
        filtered_csv = os.path.join(data_folder, 'ptbxl_database_filtered.csv')
        fallback_csv = os.path.join(data_folder, 'ptbxl_database.csv')
        db_csv = filtered_csv if os.path.exists(filtered_csv) else fallback_csv

        Y = pd.read_csv(db_csv, index_col='ecg_id', engine='python', on_bad_lines='skip')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})

        scp_path = os.path.join(data_folder, 'scp_statements.csv')
        agg_df = pd.read_csv(scp_path, index_col=0)
        diagnostic_df = agg_df[agg_df.diagnostic == 1]

        if split == 'TRAIN':
            split_Y = Y[Y.strat_fold < 9]
        elif split == 'VAL':
            split_Y = Y[Y.strat_fold == 9]
        else:
            split_Y = Y[Y.strat_fold == 10]

        superclass_names = sorted(
            [c for c in diagnostic_df.diagnostic_class.dropna().unique().tolist() if isinstance(c, str)]
        )
        subclass_names = sorted(
            [c for c in diagnostic_df.diagnostic_subclass.dropna().unique().tolist() if isinstance(c, str)]
        )
        form_series = agg_df["form"] if "form" in agg_df.columns else pd.Series(index=agg_df.index, dtype=float)
        rhythm_series = agg_df["rhythm"] if "rhythm" in agg_df.columns else pd.Series(index=agg_df.index, dtype=float)
        form_mask = pd.to_numeric(form_series, errors="coerce") == 1.0
        rhythm_mask = pd.to_numeric(rhythm_series, errors="coerce") == 1.0
        form_codes = sorted(agg_df.index[form_mask].tolist())
        rhythm_codes = sorted(agg_df.index[rhythm_mask].tolist())
        subclass_description_map = self._build_subclass_description_map(diagnostic_df)
        form_description_map = {
            code: str(agg_df.at[code, "description"]).strip().lower()
            for code in form_codes
            if code in agg_df.index and isinstance(agg_df.at[code, "description"], str)
            and str(agg_df.at[code, "description"]).strip()
        }
        rhythm_description_map = {
            code: str(agg_df.at[code, "description"]).strip().lower()
            for code in rhythm_codes
            if code in agg_df.index and isinstance(agg_df.at[code, "description"], str)
            and str(agg_df.at[code, "description"]).strip()
        }
        class_names = (
            [f"superclass:{name}" for name in superclass_names]
            + [f"subclass:{name}" for name in subclass_names]
            + [f"form:{name}" for name in form_codes]
            + [f"rhythm:{name}" for name in rhythm_codes]
        )
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        # When a MERL predictor is provided, replace ground-truth labels with
        # MERL's zero-shot predictions from the ECG signal (predict-then-generate,
        # leakage-free). Predictions are aligned with self.secgs row order.
        merl_preds = None
        if self.merl_predictor is not None:
            families = {
                "super": superclass_names,
                "sub": subclass_names,
                "form": form_codes,
                "rhythm": rhythm_codes,
            }
            code_to_text = lambda c: (
                subclass_description_map.get(c)
                or form_description_map.get(c)
                or rhythm_description_map.get(c)
                or self._class_name_to_full_form(c, subclass_description_map)
            )
            merl_preds = self.merl_predictor.predict_active_codes(
                self.secgs, families, code_to_text=code_to_text
            )
            if len(merl_preds) != len(split_Y):
                raise ValueError(
                    f"MERL prediction count {len(merl_preds)} != split rows {len(split_Y)} "
                    f"for split={split}; check that secgs order matches strat_fold rows."
                )

        labels = np.zeros((len(split_Y), len(class_names)), dtype=np.float32)
        label_texts = []
        for row_idx, scp_codes in enumerate(split_Y.scp_codes.values):
            if merl_preds is not None:
                active_superclasses = merl_preds[row_idx]["super"]
                active_subclasses = merl_preds[row_idx]["sub"]
                active_form_codes = merl_preds[row_idx]["form"]
                active_rhythm_codes = merl_preds[row_idx]["rhythm"]
            else:
                active_superclasses = self._aggregate_diagnostic_classes(scp_codes, diagnostic_df)
                active_subclasses = self._aggregate_diagnostic_subclasses(scp_codes, diagnostic_df)
                active_form_codes = self._aggregate_codes_by_flag(scp_codes, agg_df, "form")
                active_rhythm_codes = self._aggregate_codes_by_flag(scp_codes, agg_df, "rhythm")

            for class_name in active_superclasses:
                labels[row_idx, class_to_idx[f"superclass:{class_name}"]] = 1.0
            for subclass_name in active_subclasses:
                labels[row_idx, class_to_idx[f"subclass:{subclass_name}"]] = 1.0
            for form_code in active_form_codes:
                labels[row_idx, class_to_idx[f"form:{form_code}"]] = 1.0
            for rhythm_code in active_rhythm_codes:
                labels[row_idx, class_to_idx[f"rhythm:{rhythm_code}"]] = 1.0

            label_texts.append(
                prompting.format_clinical_prompt(
                    active_superclasses,
                    active_subclasses,
                    active_form_codes,
                    active_rhythm_codes,
                    subclass_description_map=subclass_description_map,
                    form_description_map=form_description_map,
                    rhythm_description_map=rhythm_description_map,
                )
            )

        if labels.shape[0] != self.secgs.shape[0]:
            raise ValueError(
                f"SCP label count mismatch for split={split}: labels={labels.shape[0]} vs secgs={self.secgs.shape[0]}"
            )

        return labels, class_names, label_texts

    def print_sample(self, i=0):
        if self.scp_labels is None:
            print(f"split={self.split} idx={i} has no SCP labels enabled")
            return

        label_vector = self.scp_labels[i]
        active_indices = np.where(label_vector > 0)[0].tolist()
        active_classes = [self.scp_class_names[idx] for idx in active_indices]

        print(f"split={self.split} idx={i}")
        print(f"label_columns={self.scp_class_names}")
        print(f"label_vector={label_vector.tolist()}")
        print(f"active_classes={active_classes}")
        print(f"label_text={self.scp_label_texts[i]}")


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        secg = torch.FloatTensor(self.secgs[i])
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([round(self.caplens[i])])

        if self.scp_label_texts:
            x = (secg, caption[:-1], self.scp_label_texts[i])
        else:
            x = (secg, caption[:-1])
        y = (caption)

        # if self.split == 'TRAIN':
        return x, y, caplen
            # return combined, secg, caption, caplen
        # else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            # all_captions = torch.LongTensor(
            #     self.captions[i])
            # return secg, caption, caplen, all_captions
            # return x,y, all_captions

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    ds = TransformerDataset('data/ptbxl/', 'TRAIN', use_scp_labels=True)
    ds.print_sample(0)
    x = next(iter(ds))
    print(type(x))


class SECGDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_ASSESSMENT.hdf5'), 'r')
        self.secgs = self.h['secgs']
        # assessment per sECG
        self.assessments = self.h['assessments']


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # # Total number of datapoints
        self.dataset_size = len(self.secgs)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        secg = torch.FloatTensor(self.secgs[i])
        assessment = torch.FloatTensor(self.assessments[i])
        if self.transform is not None:
            secg = torch.FloatTensor([self.secgs[i]])
            secg = self.transform(secg)[0]

        return secg, assessment

    def __len__(self):
        return self.dataset_size
