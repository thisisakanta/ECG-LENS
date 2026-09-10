# ECG-LENS: Lead-Aware Clinical Context Enriched ECG Report Generation and Evaluation

ECG-LENS is an end-to-end ECG report generation framework that integrates lead-specific and global ECG representations with structured diagnostic context to generate clinically faithful reports. It combines lead-wise encoders that preserve localized waveform morphology with a global encoder that captures inter-lead dependencies, and fuses these signal representations with clinically enriched textual prompts that condition a GPT-2 decoder. We further introduce an ECG-specific report preprocessing strategy that removes repetitive and non-informative text, focusing the model on clinically meaningful findings. Finally, lexical metrics may not reflect clinical fidelity, as equivalent reports can differ in wording. We therefore devise F1-ECGBERT, a BERT-based evaluation procedure that measures diagnostic-label agreement between generated and reference reports. Experiments on PTB-XL and cross-domain evaluation on MIMIC-IV-ECG show that ECG-LENS consistently outperforms state-of-the-art methods, with absolute gains of 9.9%, 8.5%, and 10.9% in METEOR, ROUGE-L, and F1-ECGBERT over the strongest baselines.


Reference implementation of ECG-LENS, an end-to-end framework that turns a raw
12-lead ECG into a clinical report, together with **F1-ECGBERT**, an ECG-specific
metric that scores the diagnostic content of a generated report rather than its
wording.

ECG-LENS keeps lead identity intact instead of collapsing the recording into a
single vector: twelve **lead-specific 1-D ResNet-18 encoders** preserve localized
waveform morphology while a **global 1-D ResNet-18 encoder** reads all twelve
channels jointly to capture inter-lead relationships. A **frozen pretrained ECG
classifier** supplies zero-shot diagnostic labels, which are rendered into a
short **clinical prompt** that gives the decoder an explicit diagnostic plan. A
**GPT-2-style decoder trained from scratch** over a domain vocabulary then
generates the report, conditioned on both the ECG tokens and the prompt.

## Repository layout

```
ecg-lens/
├── src/
│   ├── models.py                     # lead-specific + global encoders (Encoder), lead embeddings
│   ├── ResNet1D.py                   # 1-D ResNet backbone (resnet18 / resnet34)
│   ├── GPT.py                        # GPT-2-style decoder, ECG tokens + prompt as prefix
│   ├── my_datasets.py                # hdf5 datasets, SCP label aggregation
│   ├── prompting.py                  # clinical prompt format (shared by training + inference)
│   ├── merl_predictor.py             # frozen zero-shot label predictor (external checkpoint)
│   ├── utils.py                      # tokenizer, report normalization, PTB-XL data preparation
│   ├── lexical.py                    # BLEU / ROUGE / METEOR (single source of truth)
│   ├── create_input_files.py         # CLI: raw PTB-XL -> hdf5 + encoder.pkl
│   ├── train_transformer_decoder.py  # main training / validation / test loop
│   ├── caption.py                    # qualitative figures and per-record inspection
│   └── vit1d.py, utils_builder.py    # transformer encoder + ECG-text model used for ablations
├── scripts/
│   ├── predict.py                    # inference: checkpoint + ECG -> generated report
│   ├── prepare_mimic.py              # MIMIC-IV WFDB + reports CSV -> test hdf5
│   ├── generate_reports.py           # trained checkpoint -> hypothesis / reference pickles
│   └── run_lexical.py                # pickles -> lexical metrics table
├── ecgbert/                          # F1-ECGBERT evaluator (see ecgbert/README.md)
├── configs/
│   ├── config_create_input_files.ini      # data preparation
│   ├── config_train_transformer_decoder.ini  # model + training
│   ├── ecgbert_eval/{super,sub,rhythm,form}.yaml
│   └── eval/mimic_crossdomain.yaml        # cross-domain settings for reference
├── requirements.txt
└── LICENSE
```

`data/` and `checkpoints/` are not tracked; the configs expect them at the paths
below.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The WordNet data is used by METEOR. `nltk >= 3.9` is required on Python 3.12;
earlier versions break BLEU on that interpreter. All commands below are run from
the repository root, which is where the scripts resolve `configs/` and `data/`
from.

## Data

**PTB-XL** — 21,799 ten-second 12-lead recordings from 18,869 patients. Download
from PhysioNet into `data/ptbxl/` so that `ptbxl_database.csv`,
`scp_statements.csv` and `records500/` are present. We use the 500 Hz signals
(`12 x 5000`) and the official `strat_fold` split: folds 1–8 train, fold 9
validation, fold 10 test.

**MIMIC-IV-ECG** — used for cross-domain evaluation only, never for training.
Place the raw WFDB records under `data/mimic/` alongside a CSV of reports.

```
data/
├── ptbxl/    # ptbxl_database.csv + scp_statements.csv + records500/ (and records100/)
└── mimic/    # raw MIMIC-IV-ECG WFDB records + a reports CSV
```

Everything else (`encoder.pkl` and the `*_DATASET.hdf5` files) is generated by
step 0 below. Both datasets are credentialed PhysioNet resources and are not
redistributed here.

### External checkpoints

Two components are pretrained elsewhere and loaded frozen. Neither is included:

- **Zero-shot ECG label predictor** — a pretrained ECG–text model used to obtain
  the diagnostic labels behind the clinical prompt. Place its repository at
  `src/MERL-ICML2024/` (so its `utils_builder` and zero-shot prompt file are
  importable) and set `merl_checkpoint` and `merl_ecg_model` in
  `configs/config_train_transformer_decoder.ini`. It is used for inference only
  and is excluded from backpropagation; no classification or contrastive loss is
  added to training.
- **F1-ECGBERT labelers** — a pretrained biomedical BERT plus one trained labeler
  per diagnostic segment. See [`ecgbert/README.md`](ecgbert/README.md).

## Quickstart

### Step 0 — build the PTB-XL inputs

```bash
python src/create_input_files.py        # reads configs/config_create_input_files.ini
```

Reads the raw records and reports, normalizes the report text, builds the
domain-specific word tokenizer from the **training** reports only, and writes:

```
data/ptbxl/{TRAIN,VAL,TEST}_DATASET.hdf5    # secgs (N,12,5000), captions, caplens
data/ptbxl/encoder.pkl                      # the tokenizer, reused everywhere downstream
```

### Step 1 — train on PTB-XL

```bash
python src/train_transformer_decoder.py   # reads configs/config_train_transformer_decoder.ini
```

Trains end-to-end, validates every epoch, selects the checkpoint by validation
METEOR, and finally evaluates that checkpoint on the fold-10 test split. Writes
`BEST_checkpoint_<model_name>_*.pth.tar`, a resumable
`checkpoint_<model_name>.pth.tar`, a training trace, per-epoch curves in
`training_and_test_results.csv`, and the tokenized hypothesis/reference pickles.
Training resumes automatically from the latest checkpoint if one is present.

### Step 2 — cross-domain evaluation on MIMIC-IV-ECG

No retraining: the PTB-XL checkpoint is run as-is on MIMIC-IV. The preparation
step must use the same sampling rate, signal length, caption length and
tokenizer as training.

```bash
python scripts/prepare_mimic.py --reports-csv data/mimic_reports.csv \
    --records-root data/mimic --encoder data/ptbxl/encoder.pkl \
    --out-dir data/mimic_prepared --sampling-rate 500 --length 5000 --max-len 64

python scripts/generate_reports.py --checkpoint checkpoints/ecglens_ptbxl.pth \
    --dataset-folder data/mimic_prepared --encoder data/ptbxl/encoder.pkl \
    --output-dir mimic_out --label-prompts mimic_out/merl_prompts.pkl

python scripts/run_lexical.py --hyp mimic_out/transformer_tokenized_hypotheses.pkl \
    --ref mimic_out/transformer_tokenized_references.pkl --tag MIMIC-IV \
    --out mimic_out/lexical_metrics.csv
```

`--label-prompts` is a pickled `List[str]` with one clinical prompt per record,
aligned to the hdf5 row order, produced by running `src/merl_predictor.py` on the
MIMIC signals. It is required when the checkpoint was trained with prompt
conditioning and omitted otherwise. `configs/eval/mimic_crossdomain.yaml` records
the settings to pass.

### Step 3 — clinical consistency (F1-ECGBERT)

```bash
python ecgbert/evaluate.py \
    --config configs/ecgbert_eval/super.yaml configs/ecgbert_eval/sub.yaml \
             configs/ecgbert_eval/rhythm.yaml configs/ecgbert_eval/form.yaml \
    --gt ground_truth_reports.csv --gen generated_reports.csv
```

Both CSVs need a `Report Impression` column and must be row-aligned. See
[`ecgbert/README.md`](ecgbert/README.md) for the metric, the four segments and how
to train the labelers.

## Inference

`scripts/predict.py` runs a trained checkpoint on new signals. It rebuilds the
model from the encoder/decoder configs stored inside the checkpoint, so it needs
only the checkpoint and the tokenizer — no config file and no dataset preparation.
Input is a WFDB record, a `.npy` array, or a prepared hdf5 folder:

```bash
# one WFDB record (PTB-XL or MIMIC-IV), prompt predicted from the signal
python scripts/predict.py \
    --checkpoint checkpoints/ecglens_ptbxl.pth.tar \
    --encoder data/ptbxl/encoder.pkl \
    --record data/ptbxl/records500/00000/00001_hr \
    --merl-checkpoint checkpoints/merl.pth \
    --scp-statements data/ptbxl/scp_statements.csv

# a whole prepared split, written as a CSV for F1-ECGBERT
python scripts/predict.py \
    --checkpoint checkpoints/ecglens_ptbxl.pth.tar \
    --encoder data/ptbxl/encoder.pkl \
    --hdf5 data/ptbxl --split TEST \
    --merl-checkpoint checkpoints/merl.pth \
    --scp-statements data/ptbxl/scp_statements.csv \
    --out generated_reports.csv
```

A `--record` is reordered to the standard lead order, resampled to
`--sampling-rate` and cropped or padded to `--length` (500 Hz and 5000 samples by
default, matching training). Decoding is greedy (`--top-k 1`) to match the
reported setting.

Because the reported checkpoint is trained with clinical-prompt conditioning, each
record also needs a prompt. `--merl-checkpoint` predicts the diagnostic labels from
the signal and builds the prompt through `src/prompting.py`, the same module
training uses, so the prompt format is identical. Alternatively pass `--prompt` to
supply one explicitly or `--prompts` for a pickled list. A checkpoint trained
without conditioning needs none of these.

The output CSV carries a `Report Impression` column, so it can be passed straight
to `ecgbert/evaluate.py --gen`. To label reports with the F1-ECGBERT evaluators
without computing F1, see `ecgbert/inference.py`.

## Configuration

Everything runtime-configurable lives in `configs/`; no source edits are needed
to reproduce the reported setup.

`config_create_input_files.ini` — `sampling_rate` (500 selects `records500/`),
`max_len` (caption length, 64), `min_word_freq` and `vocab_size` for the
tokenizer, `translate_comments` (prefer the refined `report_en` column when
present), and `split_method = strat_fold`.

`config_train_transformer_decoder.ini` — the settings used for the reported
results:

- **Encoder** — 1-D ResNet-18, convolution kernels `[9,7,7,5]`, stem kernel 9,
  `encoder_dim = 512`, and `ecg_tokens_per_lead = 1`, giving 12 lead tokens plus
  one global token with learned lead-identity embeddings.
- **Decoder** — GPT-2-style, `num_layers = 12`, `num_heads = 8`, `emb_dim = 512`,
  `dropout = 0.5`, trained from scratch over the domain vocabulary.
- **Optimization** — Adam with `encoder_lr = 4e-4` and `decoder_lr = 1e-4`,
  `batch_size = 32`, `grad_clip = 5`, learning rate scaled by 0.8 after 8 epochs
  without improvement, up to 200 epochs with early stopping patience 30. The
  reported checkpoint is selected on validation METEOR.
- **Clinical prompt** — `use_scp_label_conditioning` and `use_merl_labels` both
  enabled, with `merl_checkpoint` pointing at the frozen label predictor and
  `merl_threshold = 0.5`.

Set `merl_checkpoint` before training: it ships empty, and leaving it empty while
`use_merl_labels = true` fails at startup when the checkpoint is loaded.

## License

MIT — see [`LICENSE`](LICENSE).
