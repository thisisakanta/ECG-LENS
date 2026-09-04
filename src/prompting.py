"""
Clinical prompt construction: the single definition of the prompt format.

The decoder is conditioned on a clinical prompt built from predicted diagnostic
labels across four categories (superclass / subclass / form / rhythm). Training
(`my_datasets.TransformerDataset`) and inference (`scripts/predict.py`) both build
that prompt through this module, so the prompt a checkpoint sees at inference time
is byte-identical to the one it was trained on.

Contents:
  * `class_name_to_full_form` -- SCP code or class name -> clinical phrase
  * `build_subclass_description_map` -- subclass -> description from scp_statements
  * `load_label_vocabulary` -- read scp_statements.csv into the four category
    class lists plus their description maps
  * `format_clinical_prompt` -- active labels -> the prompt string
"""

# Canonical clinical phrasing for PTB-XL superclass / subclass codes.
CLASS_NAME_MAP = {
    "NORM": "normal ecg",
    "MI": "myocardial infarction",
    "HYP": "ventricular hypertrophy",
    "CD": "conduction disturbance",
    "STTC": "st t changes",
    "AMI": "anterior myocardial infarction",
    "CLBBB": "complete left bundle branch block",
    "IRBBB": "incomplete right bundle branch block",
    "CRBBB": "complete right bundle branch block",
    "ILBBB": "incomplete left bundle branch block",
    "IMI": "inferior myocardial infarction",
    "ISCA": "ischemic st t changes in anterior or anterolateral leads",
    "ISCI": "ischemic st t changes in inferior or inferolateral leads",
    "ISC_": "non specific ischemic st t changes",
    "LAFB/LPFB": "left anterior or left posterior fascicular block",
    "LAO/LAE": "left atrial overload or enlargement",
    "LMI": "lateral myocardial infarction",
    "LVH": "left ventricular hypertrophy",
    "NST_": "non specific st changes",
    "PMI": "posterior myocardial infarction",
    "RAO/RAE": "right atrial overload or enlargement",
    "WPW": "wolff parkinson white syndrome",
    "1AVB": "first degree atrioventricular block",
    "_AVB": "atrioventricular block",
    "IVCD": "non specific intraventricular conduction disturbance",
    "RVH": "right ventricular hypertrophy",
    "SEHYP": "septal hypertrophy",
}

# Order of the four diagnostic categories in the prompt.
CATEGORIES = ("super", "sub", "form", "rhythm")


def class_name_to_full_form(class_name, subclass_description_map=None):
    """Map an SCP code / class name to the clinical phrase used in the prompt."""
    if class_name in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[class_name]
    if subclass_description_map is not None and class_name in subclass_description_map:
        return subclass_description_map[class_name]

    cleaned = class_name.replace("_", " ").replace("/", " or ")
    cleaned = cleaned.replace("AV", "atrioventricular")
    return cleaned.lower()


def build_subclass_description_map(diagnostic_df):
    """subclass name -> ' / '-joined descriptions from the diagnostic statements."""
    subclass_description_map = {}
    grouped = diagnostic_df.groupby("diagnostic_subclass")
    for subclass_name, group in grouped:
        if not isinstance(subclass_name, str) or not subclass_name:
            continue

        descriptions = sorted(
            {
                str(desc).strip().lower()
                for desc in group["description"].tolist()
                if isinstance(desc, str) and str(desc).strip()
            }
        )
        if descriptions:
            subclass_description_map[subclass_name] = " / ".join(descriptions)
        else:
            subclass_description_map[subclass_name] = class_name_to_full_form(subclass_name)

    return subclass_description_map


def _description_map_for_codes(agg_df, codes):
    return {
        code: str(agg_df.at[code, "description"]).strip().lower()
        for code in codes
        if code in agg_df.index and isinstance(agg_df.at[code, "description"], str)
        and str(agg_df.at[code, "description"]).strip()
    }


def load_label_vocabulary(scp_statements_path):
    """Read scp_statements.csv into the four category class lists + description maps.

    Returns a dict with keys:
        families   -- {"super": [...], "sub": [...], "form": [...], "rhythm": [...]}
        maps       -- {"sub": {...}, "form": {...}, "rhythm": {...}} description maps
    The class lists match the ones the training dataset derives, so MERL scores the
    same candidate conditions at inference time as during training.
    """
    import pandas as pd

    agg_df = pd.read_csv(scp_statements_path, index_col=0)
    diagnostic_df = agg_df[agg_df.diagnostic == 1]

    superclass_names = sorted(
        [c for c in diagnostic_df.diagnostic_class.dropna().unique().tolist() if isinstance(c, str)]
    )
    subclass_names = sorted(
        [c for c in diagnostic_df.diagnostic_subclass.dropna().unique().tolist() if isinstance(c, str)]
    )

    form_series = agg_df["form"] if "form" in agg_df.columns else pd.Series(index=agg_df.index, dtype=float)
    rhythm_series = agg_df["rhythm"] if "rhythm" in agg_df.columns else pd.Series(index=agg_df.index, dtype=float)
    form_codes = sorted(agg_df.index[pd.to_numeric(form_series, errors="coerce") == 1.0].tolist())
    rhythm_codes = sorted(agg_df.index[pd.to_numeric(rhythm_series, errors="coerce") == 1.0].tolist())

    return {
        "families": {
            "super": superclass_names,
            "sub": subclass_names,
            "form": form_codes,
            "rhythm": rhythm_codes,
        },
        "maps": {
            "sub": build_subclass_description_map(diagnostic_df),
            "form": _description_map_for_codes(agg_df, form_codes),
            "rhythm": _description_map_for_codes(agg_df, rhythm_codes),
        },
    }


def code_to_text(code, vocab):
    """Clinical phrase for a code, checked against every category description map.

    Used to build MERL's zero-shot text prompts for the candidate conditions.
    """
    maps = vocab["maps"]
    return (
        maps["sub"].get(code)
        or maps["form"].get(code)
        or maps["rhythm"].get(code)
        or class_name_to_full_form(code, maps["sub"])
    )


def format_clinical_prompt(active_super, active_sub, active_form, active_rhythm,
                           subclass_description_map=None,
                           form_description_map=None,
                           rhythm_description_map=None):
    """Render active labels per category into the decoder's clinical prompt.

    Empty categories are rendered as "none" so the prompt keeps a fixed structure
    and a constant token budget. Superclass phrases come from CLASS_NAME_MAP only,
    matching how the training targets were built.
    """
    def join(codes, description_map):
        if not codes:
            return "none"
        return ", ".join(class_name_to_full_form(code, description_map) for code in codes)

    superclass_text = join(active_super, None)
    subclass_text = join(active_sub, subclass_description_map)
    form_text = join(active_form, form_description_map)
    rhythm_text = join(active_rhythm, rhythm_description_map)

    return (
        f"diagnosis superclass: {superclass_text}; diagnosis subclass: {subclass_text}; "
        f"ecg form: {form_text}; ecg rhythm: {rhythm_text}"
    )
