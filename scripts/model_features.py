"""Engineered features for the stage-05 models.

This lives in a module rather than in the notebook on purpose. The engineered block is
wrapped in a ``FunctionTransformer`` so it travels *inside* the fitted pipeline; if the
function were defined in the notebook, joblib would pickle it by reference to ``__main__``
and the saved .pkl would fail to load in a fresh process. Importing it from here keeps the
round-trip test (load a .pkl, score raw CSV rows) honest.

Two properties matter, and both are deliberate:

1. **Every feature is row-wise.** Each one is computed from that member's own fields only,
   with no statistic pooled across members (no means, no quantiles, no group aggregates).
   A transform of that shape cannot leak information from the test set into the training
   set, whether it is fitted before or after the split.
2. **Every input is renewal-legal.** Nothing here reads the barred fields (premium, total
   claims paid, average claim amount, loss ratio, risk score, is-high-risk) or the
   same-period fields (claims count, visits, procedure counts, had-major-procedure). The
   notebook re-runs the leakage guard after the block is added to confirm this.
"""

import numpy as np
import pandas as pd

# Conditions weighted by how expensive they typically are to run, so that "three chronic
# conditions" is not treated as one number regardless of which three.
CONDITION_WEIGHTS = {
    "Cancer History": 3.0,
    "Kidney Disease": 3.0,
    "Cardiovascular Disease": 2.5,
    "COPD": 2.0,
    "Liver Disease": 2.0,
    "Diabetes": 1.5,
    "Hypertension": 1.0,
    "Asthma": 1.0,
    "Arthritis": 1.0,
    "Mental Health Condition": 1.0,
}

ENGINEERED_NAMES = [
    "eng_metabolic_syndrome_score",
    "eng_bp_stage",
    "eng_undiagnosed_dyslipidemia",
    "eng_comorbidity_load",
    "eng_meds_per_condition",
    "eng_days_per_admission",
    "eng_chronic_x_age",
    "eng_bmi_x_smoker",
    "eng_deductible_to_income",
    "eng_oop_exposure",
    "eng_dependents_per_head",
]


def _bp_stage(sbp, dbp):
    """ACC/AHA blood-pressure staging as one ordinal column."""
    stage = np.zeros(len(sbp))
    stage = np.where((sbp >= 120) | (dbp >= 80), 1, stage)   # elevated
    stage = np.where((sbp >= 130) | (dbp >= 80), 2, stage)   # stage 1
    stage = np.where((sbp >= 140) | (dbp >= 90), 3, stage)   # stage 2
    return stage


def add_engineered(X):
    """Append the engineered columns to a feature frame, leaving the originals in place."""
    X = X.copy()

    def col(name, default=0.0):
        return X[name] if name in X.columns else pd.Series(default, index=X.index)

    bmi = col("BMI")
    sbp = col("Systolic Blood Pressure")
    dbp = col("Diastolic Blood Pressure")
    ldl = col("LDL Cholesterol")
    hba1c = col("HbA1c Level")
    age = col("Age")
    chronic = col("Chronic Conditions Count")
    meds = col("Medication Count")
    hosp = col("Hospitalizations in Last 3 Years")
    days = col("Days Hospitalized in Last 3 Years")
    income = col("Income", 1.0)
    deductible = col("Deductible")
    copay = col("Copay")
    household = col("Household Size", 1.0)
    dependents = col("Dependents")

    # Clinical composites: several borderline readings that no single flag would catch.
    X["eng_metabolic_syndrome_score"] = (
        (bmi >= 30).astype(int) + (sbp >= 130).astype(int) + (dbp >= 85).astype(int)
        + (hba1c >= 5.7).astype(int) + (ldl >= 160).astype(int)
    )
    X["eng_bp_stage"] = _bp_stage(sbp.to_numpy(), dbp.to_numpy())

    # The same idea as undiagnosed_htn_flag from stage 03, applied to lipids: a bad reading
    # with no corresponding diagnosis on file.
    X["eng_undiagnosed_dyslipidemia"] = (
        (ldl >= 160) & (col("Cardiovascular Disease") == 0)
    ).astype(int)

    load = pd.Series(0.0, index=X.index)
    for cond, w in CONDITION_WEIGHTS.items():
        load = load + w * col(cond)
    X["eng_comorbidity_load"] = load

    # Burden and intensity: the same condition count can mean very different treatment.
    X["eng_meds_per_condition"] = meds / (chronic + 1)
    X["eng_days_per_admission"] = np.where(hosp > 0, days / hosp.replace(0, np.nan), 0.0)
    X["eng_chronic_x_age"] = chronic * age
    if "Smoker Status" in X.columns:
        # The field records Never / Former / Current, so the interaction keys off current smoking.
        X["eng_bmi_x_smoker"] = bmi * X["Smoker Status"].astype(str).str.lower().eq("current")
    else:
        X["eng_bmi_x_smoker"] = 0.0

    # Affordability and plan design. Income, deductible and copay are inputs to the premium
    # formula, not outputs of the cost target, so they are legal here; premium itself is not.
    X["eng_deductible_to_income"] = deductible / income.replace(0, np.nan)
    X["eng_oop_exposure"] = deductible + 12 * copay
    X["eng_dependents_per_head"] = dependents / household.replace(0, np.nan)

    return X.replace([np.inf, -np.inf], np.nan)
