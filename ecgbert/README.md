# F1-ECGBERT

Reference implementation of **F1-ECGBERT**, the ECG-specific clinical evaluation
metric used in ECG-LENS. Lexical metrics (BLEU/ROUGE/METEOR) reward surface
overlap and cannot tell a valid paraphrase from a fluent report that drops,
alters, or invents a finding. F1-ECGBERT instead extracts diagnostic labels from
the reference and the generated report with the _same_ frozen labeler and reports
the F1 agreement between the two label sets.

## How it works

```
reference report   ---+
                      +--> frozen report-to-label BERT --> binary label vectors --> F1
generated report   ---+           (one per segment)
```

The metric is computed independently for four diagnostic segments, each with its
own class list and its own trained checkpoint:

| Segment  | Classes | Content                                           |
| -------- | ------- | ------------------------------------------------- |
| `super`  | 5       | diagnostic superclasses (NORM, MI, STTC, CD, HYP) |
| `sub`    | 23      | diagnostic subclasses                             |
| `rhythm` | 12      | rhythm statements                                 |
| `form`   | 19      | form / morphology statements                      |

Each labeler is a BERT encoder (`models/bert_labeler.py`) whose `[CLS]`
representation feeds one 4-way linear head per condition
(blank / positive / negative / uncertain). At evaluation time only the
_positive_ class counts as a present finding; every other head output is mapped
to absent. The labelers are trained separately and stay frozen while reports are
scored, so the metric is identical across all methods being compared.

## Layout

```
ecgbert/
├── evaluate.py              # main entry point: config-driven F1 for 1–4 segments
├── run_bert.py              # trains a report-to-label BERT labeler for one segment
├── models/bert_labeler.py   # BERT + one 4-way head per condition
├── datasets/
│   ├── impressions_dataset.py   # labeled reports (training)
│   └── unlabeled_dataset.py     # report text only (evaluation)
├── bert_tokenizer.py        # report text to BERT ids (truncated at 512)
├── utils.py                 # attention masks, per-class F1 / kappa, dev evaluation
├── constants.py             # training hyper-parameters + the 23-class `sub` list
├── label.py                 # dump per-report predicted labels to CSV
├── bert_f1.py               # label a CSV and score it against a labeled CSV
├── final_f1.py              # merge per-segment label CSVs into one overall F1
└── inference.py             # label a single report typed on stdin
```

`evaluate.py` is the entry point that produces the paper's F1-ECGBERT numbers;
it takes the class list and checkpoint from a YAML config, so a segment is
selected by pointing `--config` at a different file — no source edits. The
remaining scripts (`label.py`, `bert_f1.py`, `final_f1.py`, `inference.py`) are
single-segment helpers that read their class list from `constants.CONDITIONS`
(the 23-class `sub` list) and are convenience tools rather than part of the
reported metric.

## Requirements

Segment configs live in `configs/ecgbert_eval/` at the repository root and point
to two external artifacts:

- `pretrain_path` — a pretrained biomedical BERT (e.g. BlueBERT or BioBERT),
  by default `checkpoints/bert_pretrained`.
- `checkpoint` — the trained labeler for that segment, by default
  `checkpoints/ecgbert_{super,sub,rhythm,form}.pth`.

Edit those two fields to your local paths. `num_classes` must equal the number of
entries under `conditions`; `evaluate.py` validates this on load.

## Evaluation

Run from the repository root. Both CSVs need a report-text column (default
`Report Impression`) and must be **row-aligned** — the same records in the same
order, so that row _i_ of `--gen` is the model output for row _i_ of `--gt`.

```bash
# one segment
python ecgbert/evaluate.py --config configs/ecgbert_eval/super.yaml \
    --gt ground_truth_reports.csv --gen generated_reports.csv

# all four segments, each with its own class list and checkpoint
python ecgbert/evaluate.py \
    --config configs/ecgbert_eval/super.yaml configs/ecgbert_eval/sub.yaml \
             configs/ecgbert_eval/rhythm.yaml configs/ecgbert_eval/form.yaml \
    --gt ground_truth_reports.csv --gen generated_reports.csv
```

For each segment this prints macro-F1, micro-F1 and per-class F1; with more than
one config it also prints a summary table and the mean across segments. The
per-segment F1 is the F1-ECGBERT score reported for that diagnostic category.

## Labeling reports without computing F1

`inference.py` runs the labelers over reports and prints the conditions each one
asserts. It takes the same segment configs as `evaluate.py`, loads each checkpoint
once, and needs no reference reports — useful for inspecting a single generated
report or for producing a label table.

```bash
# one report, all four segments
python ecgbert/inference.py \
    --config configs/ecgbert_eval/super.yaml configs/ecgbert_eval/sub.yaml \
             configs/ecgbert_eval/rhythm.yaml configs/ecgbert_eval/form.yaml \
    --report "sinus rhythm, inferior myocardial infarction, st depression"

# a CSV of reports -> one label CSV
python ecgbert/inference.py --config configs/ecgbert_eval/sub.yaml \
    --csv generated_reports.csv --out predicted_labels.csv
```

`label.py` is the older single-segment equivalent, kept because it writes the
`labeled_reports.csv` layout that `bert_f1.py` and `final_f1.py` expect:

```bash
python ecgbert/label.py --data reports.csv --output_dir out/ \
    --checkpoint checkpoints/ecgbert_sub.pth
```

## Training a labeler

Needed only to rebuild the evaluators from scratch. Reports are tokenized once to
a JSON id list, then the labeler is trained with cross-entropy over the 4-way
heads and checkpointed on the best mean Cohen's kappa on the dev split.

```bash
# 1. tokenize the report text of each split
python ecgbert/bert_tokenizer.py --data train.csv --output_path train_imps.json
python ecgbert/bert_tokenizer.py --data dev.csv   --output_path dev_imps.json

# 2. train (label columns in the CSVs must match the segment's class list)
python ecgbert/run_bert.py \
    --train_csv train.csv --train_imp_list train_imps.json \
    --dev_csv   dev.csv   --dev_imp_list   dev_imps.json \
    --output_dir checkpoints/
```
