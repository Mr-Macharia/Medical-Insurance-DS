"""Section 1 of notebook.ipynb — dictionary.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
# Britam Health Insurance — Portfolio Analysis

One extract: 100,000 members, one year each, 54 fields. This notebook runs top to bottom —
document the schema, inspect the raw values, clean, explore, chart, model.

| Section | Question |
|:---|:---|
| 1. The fields I have been given | What does each column mean, and who uses it? |
| 2. Looking at the raw extract | What condition is this data in? |
| 3. Cleaning | Which records cannot describe a real person? |
| 4. Exploring the book | What drives cost, and what can this data not answer? |
| 5. The dashboard picture | The headline charts |
| 6. Predicting cost and risk | What will a member cost, and who becomes expensive? |

Two constraints established below and enforced throughout. **`Annual Premium` is not an
underwritten price** (4.1), so no pricing-adequacy or profitability claim is available from this
book. And **the analysis frame is not the modelling frame** — section 6 builds a separate one.
"""),

md("""
---

# 1. The Fields I Have Been Given

Before I open anything I want to write down what each column is supposed to mean, who in the
business would use it, and which ones are going to cause trouble later. This section documents
and profiles. It changes nothing.
"""),

md("""
## Setting Up

Two packages to start with — `pathlib` to find the file and `pandas` to read it. Everything
else arrives at the section that first needs it.
"""),

code("""
# Importing what I need to open the file
from pathlib import Path

import pandas as pd
"""),

code("""
# Widening the display, since 54 columns will not fit at the default settings
pd.set_option('display.max_columns', 70)
pd.set_option('display.max_rows', 70)
pd.set_option('display.width', 1000)
"""),

md("""
I am resolving the project root by looking for the data folder rather than assuming a working
directory, so this runs whether the Jupyter server was started at the root or somewhere below
it.
"""),

code("""
# Finding the project root by looking for the data folder, rather than assuming where I started
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'data' / 'medical_insurance.csv').exists())

RAW_PATH = ROOT / 'data' / 'medical_insurance.csv'
PROCESSED_DIR = ROOT / 'data' / 'processed'
FIGS = ROOT / 'reports' / 'figures'
MODELS = ROOT / 'updated_models'    # kept separate from models/, which holds earlier runs

for d in (PROCESSED_DIR, FIGS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

ROOT
"""),

code("""
# Reading the raw extract
df = pd.read_csv(RAW_PATH)
df.shape
"""),

md("""
100,000 members and 54 fields.
"""),

md("""
## Who Uses What

The business has several audiences for this data and they want different things from it. I am
using these codes throughout, so that every conclusion below can be addressed to somebody in
particular rather than left hanging.

| Code | Stakeholder | What they want from this data |
|:---|:---|:---|
| UW | Underwriting and risk selection | Assess member risk, review eligibility |
| ACT | Actuarial and pricing | Estimate claims, frequency, severity, adequacy |
| CLM | Claims operations | Process and control claims |
| CM | Care management | Decide which members to enrol in a programme |
| FIN | Finance | Revenue, cost, profitability |
| RE | Reinsurance and capital | Severe and catastrophic exposure |
"""),

md("""
## The Column Dictionary

### Identity and demographics

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `person_id` | Unique member identifier | Joins records; never a risk feature | UW, ACT, CLM |
| `age` | Member age in years | Age-related morbidity and care planning | UW, ACT |
| `sex` | Recorded sex category | Population risk; review for fairness | ACT, UW |
| `region` | Broad geographic area | Regional cost, access and market analysis | ACT |
| `urban_rural` | Settlement classification | Healthcare access and utilisation | ACT |
| `income` | Annual income estimate | Affordability and segmentation | ACT, FIN |
| `education` | Highest education level | Health literacy and segmentation | ACT, UW |
| `marital_status` | Marital status | Household and family cover | ACT |
| `employment_status` | Employment category | Segments individual and employment-linked pools | UW, ACT |
| `household_size` | People in household | Family exposure and product design | ACT |
| `dependents` | Number of dependents | Family pricing and coverage | ACT, FIN |

### Lifestyle and clinical measurements

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `bmi` | Body mass index | Obesity-related risk, wellness targeting | UW, ACT |
| `smoker` | Never, former or current | Respiratory, cardiovascular and cancer risk | UW, ACT |
| `alcohol_freq` | Reported drinking frequency | Lifestyle risk; **missing means unknown** | UW |
| `systolic_bp` | Upper blood-pressure reading | Hypertension and cardiovascular risk | UW |
| `diastolic_bp` | Lower blood-pressure reading | Blood-pressure control | UW |
| `ldl` | LDL cholesterol | Cardiovascular risk | UW, ACT |
| `hba1c` | Average blood glucose over 2-3 months | Diabetes risk and care management | UW, ACT |
| `medication_count` | Active medications | Chronic burden and polypharmacy | UW, ACT |

### Utilisation and chronic conditions

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `visits_last_year` | Outpatient visits in the prior 12 months | Care demand | CLM, ACT |
| `hospitalizations_last_3yrs` | Admissions in the prior three years | Prior severe events | CLM, ACT |
| `days_hospitalized_last_3yrs` | Inpatient days over three years | Length of stay and severity | CLM, ACT, RE |
| `chronic_count` | Number of chronic conditions | Comorbidity segmentation | UW, ACT |
| `hypertension`, `diabetes`, `asthma`, `copd`, `cardiovascular_disease`, `cancer_history`, `kidney_disease`, `liver_disease`, `arthritis`, `mental_health` | Ten condition flags, 0 or 1 | Morbidity mix and cost risk | UW, ACT, CM, RE |

### Procedure mix

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `proc_imaging_count` | X-ray, CT, MRI | Diagnostic intensity | CLM, ACT |
| `proc_surgery_count` | Surgeries | Severity and reinsurance exposure | CLM, RE |
| `proc_physio_count` | Physiotherapy sessions | Rehabilitation utilisation | CLM |
| `proc_consult_count` | Specialist consultations | Referral patterns | CLM, ACT |
| `proc_lab_count` | Laboratory tests | Diagnostic and monitoring intensity | CLM, ACT |

### Product

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `plan_type` | HMO, PPO, EPO or POS | Network and utilisation analysis | ACT |
| `network_tier` | Bronze through Platinum | Benefit richness | ACT, FIN |
| `deductible` | Paid by the member before cover contributes | Cost sharing and plan design | ACT |
| `copay` | Fixed fee per visit | Utilisation steering | ACT |
| `policy_term_years` | Policy tenure | Retention and maturity | ACT, FIN |
| `policy_changes_last_2yrs` | Changes in the prior two years | Engagement and lapse risk | UW |
| `provider_quality` | Provider rating, roughly 1.5 to 5.0 | Network contracting | ACT |
| `risk_score` | Composite member risk, 0 to 1 | Underwriting triage; **check for leakage** | UW, ACT |

### Financial and target fields

| Column | Meaning | Insurance use | Users |
|:---|:---|:---|:---|
| `annual_medical_cost` | Total medical spending for the year | The cost target | ACT, FIN, RE |
| `annual_premium` | Annual premium charged | Revenue; **check how it was set** | ACT, FIN |
| `monthly_premium` | Monthly premium | Billing; likely derived from annual | FIN |
| `claims_count` | Claims submitted | Frequency | ACT, CLM |
| `avg_claim_amount` | Average paid per claim | Severity | ACT, CLM, RE |
| `total_claims_paid` | Total paid on claims | Loss ratio; **likely derived from cost** | ACT, FIN |
| `is_high_risk` | Existing high-risk flag | Triage; candidate target | UW |
| `had_major_procedure` | Major procedure flag | Severe-event monitoring | CLM, RE |
"""),

md("""
## Profiling Every Column

Before I read a single value I want the shape of each field — what type it holds, how much of it
is missing, and how many distinct values it takes. A field with two distinct values is a flag; a
field with 100,000 is an identifier; a field with nothing missing is one less thing to worry
about.
"""),

code("""
# Profiling every column: type, completeness, and how many distinct values it holds
profile = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing': df.isna().sum(),
    'missing %': (df.isna().mean() * 100).round(2),
    'distinct': df.nunique(dropna=False),
})
profile
"""),

md("""
Only one field has anything missing, and the identifier is unique on every row. Both of those
are worth confirming rather than assuming, so I look at them properly in the next section.
"""),

md("""
## Checks I Can Run Without Changing Anything

Several fields in the dictionary look as though they might be calculated from other fields. If
they are, that matters enormously later — a column computed from the answer is not a predictor,
it is the answer wearing a different name.

I am testing those suspicions now, while nothing has been touched, so the results describe the
source file rather than anything I did to it.
"""),

code("""
# Testing the identities the dictionary hints at, before anything is modified
chronic_cols = ['hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease',
                'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis', 'mental_health']

pd.Series({
    'duplicate person_id': int(df['person_id'].duplicated().sum()),
    'chronic_count != sum of the ten flags': int((df['chronic_count'] != df[chronic_cols].sum(axis=1)).sum()),
    'worst gap: annual_premium vs 12 x monthly': float((df['annual_premium'] - 12 * df['monthly_premium']).abs().max()),
    'worst gap: total_claims_paid vs count x average': float((df['total_claims_paid'] - df['claims_count'] * df['avg_claim_amount']).abs().max()),
    'members recorded at age 0': int((df['age'] == 0).sum()),
    'alcohol_freq missing': int(df['alcohol_freq'].isna().sum()),
}, name='value')
"""),

md("""
Four results, and each one sets up work further down.

`person_id` is unique and `chronic_count` reconciles exactly against its ten flags, so the file
is internally consistent where it claims to be.

**`monthly_premium` is `annual_premium / 12` to the cent, and `total_claims_paid` is
`claims_count × avg_claim_amount` to the cent.** Neither field carries anything the other two do
not. That is the first sign of a pattern I chase down properly in section 4.

**165 members are recorded at age 0**, and **30,083 have no alcohol frequency recorded** — 30% of
the book in a single field. Both need investigating before I decide anything, which is section 2.
"""),

md("""
## What I Will and Will Not Model

Recorded now, before seeing which fields score well.

| Objective | Target | Fields needing a leakage review |
|:---|:---|:---|
| What will a member cost? | `annual_medical_cost` | Premium fields, claims-paid fields, `risk_score` |
| Who becomes expensive? | Top decile of `annual_medical_cost` | The same, plus anything recorded in the same year |
| What should we charge? | No model | Premium looks computed from cost — tested in 4.1 |

`person_id` is never a predictor and a target is never an input. Feature selection also depends on
**when** the prediction is made: a field present in the file is not necessarily available when
somebody needs the answer.
"""),

]
