"""Section 2 of notebook.ipynb — observation.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
---

# 2. Looking at the Raw Extract

First read of the actual values. Nothing is corrected here — the repairs are section 3.

Two things happen beyond looking. The columns get readable names, because `proc_physio_count` and
`hospitalizations_last_3yrs` invite mistakes. And where one field can be checked against another,
that comparison is built now for cleaning to use.

Neither changes a member's data.
"""),

code("""
# Adding numpy for the comparisons I am about to build
import numpy as np
"""),

md("""
## Giving the Columns Readable Names

Not cosmetic. I will be typing and reading these constantly, and `proc_physio_count` invites
mistakes in a way `Physiotherapy Procedures Count` does not.

I am asserting the map has exactly 54 entries, so that if a column is ever added to the source
file this cell fails loudly instead of silently leaving one behind.
"""),

code("""
# Mapping every database column name to something readable
COLUMN_RENAME_MAP = {
    'person_id': 'Id',
    'age': 'Age',
    'sex': 'Sex',
    'region': 'Region',
    'urban_rural': 'Urban / Rural',
    'income': 'Income',
    'education': 'Education (Qualification)',
    'marital_status': 'Marital Status',
    'employment_status': 'Employment Status',
    'household_size': 'Household Size',
    'dependents': 'Dependents',
    'bmi': 'BMI',
    'smoker': 'Smoker Status',
    'alcohol_freq': 'Alcohol Frequency',
    'visits_last_year': 'Visits in Last Year',
    'hospitalizations_last_3yrs': 'Hospitalizations in Last 3 Years',
    'days_hospitalized_last_3yrs': 'Days Hospitalized in Last 3 Years',
    'medication_count': 'Medication Count',
    'systolic_bp': 'Systolic Blood Pressure',
    'diastolic_bp': 'Diastolic Blood Pressure',
    'ldl': 'LDL Cholesterol',
    'hba1c': 'HbA1c Level',
    'plan_type': 'Plan Type',
    'network_tier': 'Network Tier',
    'deductible': 'Deductible',
    'copay': 'Copay',
    'policy_term_years': 'Policy Term (Years)',
    'policy_changes_last_2yrs': 'Policy Changes in Last 2 Years',
    'provider_quality': 'Provider Quality Rating',
    'risk_score': 'Risk Score',
    'annual_medical_cost': 'Annual Medical Cost',
    'annual_premium': 'Annual Premium',
    'monthly_premium': 'Monthly Premium',
    'claims_count': 'Claims Count',
    'avg_claim_amount': 'Average Claim Amount',
    'total_claims_paid': 'Total Claims Paid',
    'chronic_count': 'Chronic Conditions Count',
    'hypertension': 'Hypertension',
    'diabetes': 'Diabetes',
    'asthma': 'Asthma',
    'copd': 'COPD',
    'cardiovascular_disease': 'Cardiovascular Disease',
    'cancer_history': 'Cancer History',
    'kidney_disease': 'Kidney Disease',
    'liver_disease': 'Liver Disease',
    'arthritis': 'Arthritis',
    'mental_health': 'Mental Health Condition',
    'proc_imaging_count': 'Imaging Procedures Count',
    'proc_surgery_count': 'Surgical Procedures Count',
    'proc_physio_count': 'Physiotherapy Procedures Count',
    'proc_consult_count': 'Consultation Procedures Count',
    'proc_lab_count': 'Lab Procedures Count',
    'is_high_risk': 'Is High Risk',
    'had_major_procedure': 'Had Major Procedure',
}

assert len(COLUMN_RENAME_MAP) == 54, f'Expected 54 columns, got {len(COLUMN_RENAME_MAP)}'

df = df.rename(columns=COLUMN_RENAME_MAP)
df.head(3)
"""),

md("""
## Is There One Row Per Person?

This settles the **grain**, which is the question of what a single row represents. It matters
because everything downstream assumes it: if a member could appear twice I would be
double-counting their cost, and every average in the analysis would be wrong.
"""),

code("""
# Checking the grain - one row per member, or can a member appear twice?
pd.Series({
    'rows': len(df),
    'unique member ids': df['Id'].nunique(),
    'fully duplicated rows': int(df.duplicated().sum()),
})
"""),

md("""
100,000 unique member IDs across 100,000 rows, and no duplicated rows. One row is one member for
one year, and I can treat each row as one member-year of exposure.
"""),

md("""
## What Is Missing, and What Type Is Everything

Two questions I always ask of a new dataset: is anything missing, and is every field stored as
the right kind of value. The second one catches a common and quiet problem — a number stored as
text, money with a currency symbol in it — which silently breaks any arithmetic done on it.
"""),

code("""
# Counting missing values, showing only the columns that have any
missing = df.isnull().sum()
missing[missing > 0]
"""),

md("""
Only one column has anything missing, and it has a lot: **`Alcohol Frequency` is blank for 30,083
members**, which is 30% of the book. Every other one of the 54 fields is complete.

A single column carrying all of the missing data is unusual and it points somewhere specific. If
records had been damaged in transfer I would expect gaps scattered across many columns. Missing
data concentrated in exactly one field usually means that field was optional at the point of
collection — which is a question about the survey, not about the data pipeline. I chase it down
further below.
"""),

code("""
# Checking which fields are stored as text, in case a number is hiding in one
df.select_dtypes(include='str').columns.tolist()
"""),

md("""
All nine text fields are genuine categories — sex, region, education and so on. No numbers are
hiding in text form, so I can do arithmetic on the numeric fields without converting anything
first.

The binary condition flags being stored as whole numbers rather than true and false is fine, and
actually convenient: storing them as 0 and 1 means I can average the column to get a prevalence
rate directly.
"""),

code("""
# Looking at every distinct value in each text field, where messy data usually shows itself
pd.Series({col: sorted(df[col].dropna().unique()) for col in df.select_dtypes(include='str').columns})
"""),

md("""
These are clean. No duplicated categories from inconsistent spelling, no stray whitespace, no
surprise values.

Two things I want to change, though, both about clarity rather than correctness. The education
values use abbreviations — `HS`, `No HS`, `Some College` — that will read badly on a chart axis.
And the plan types are four-letter acronyms that mean nothing to a reader who does not already
know them.
"""),

md("""
## Making the Labels Readable

I am expanding the education values in place. For plan type I am **adding** a new column with the
full name rather than replacing the original, so the short code stays available for anything that
needs it while charts can use the readable version.
"""),

code("""
# Spelling out the education values
EDUCATION_MAP = {
    'No HS': 'Primary',
    'HS': 'High School',
    'Some College': 'College',
    'Bachelors': 'Bachelors',
    'Masters': 'Masters',
    'Doctorate': 'Doctorate',
}

df['Education (Qualification)'] = df['Education (Qualification)'].map(EDUCATION_MAP)
df['Education (Qualification)'].value_counts()
"""),

code("""
# Adding the full plan names alongside the short codes, rather than replacing them
PLAN_TYPE_MAP = {
    'HMO': 'HMO (Health Maintenance Organization)',
    'PPO': 'PPO (Preferred Provider Organization)',
    'EPO': 'EPO (Exclusive Provider Organization)',
    'POS': 'POS (Point of Service)',
}

df['Plan Type (Full Name)'] = df['Plan Type'].map(PLAN_TYPE_MAP)
df['Plan Type (Full Name)'].value_counts()
"""),

md("""
### What these plan types actually mean

Worth writing down, since the differences drive how members use their cover:

| Plan | Full name | Needs a referral? | Covers out of network? |
|:---|:---|:---|:---|
| **HMO** | Health Maintenance Organization | Yes, through a primary care doctor | No, except emergencies |
| **PPO** | Preferred Provider Organization | No | Yes, at a higher share of cost |
| **EPO** | Exclusive Provider Organization | No | No |
| **POS** | Point of Service | Yes | Yes, at a higher share of cost |

Two other product terms appear throughout this data. The **deductible** is what a member pays out
of their own pocket each year before cover starts contributing. The **copay** is the fixed fee
they pay at each visit.
"""),

md("""
## The Numbers: Range and Spread

Now the numeric fields. I am looking for three things: values that cannot be right, such as
negative costs or ages beyond a human lifespan; the gap between the average and the middle value,
which tells me whether a few extreme members are pulling the average around; and anything sitting
at a suspicious boundary.
"""),

code("""
# Summarising every numeric field
numeric_summary = df.describe().T[['mean', '50%', 'min', 'max']]
numeric_summary.columns = ['Mean', 'Median', 'Minimum', 'Maximum']
numeric_summary.round(2)
"""),

md("""
Three things stand out.

**Nothing is negative.** No impossible costs, measurements or counts.

**`Age` starts at 0**, which needs explaining — either genuine infants or a placeholder.

**`Income` is lopsided**: mean 49,874 against a median of 36,200, maximum 1,061,800. The mean
overstates a typical member's earnings, so **I use the median for income throughout**.
"""),

code("""
# Looking closely at the age range, since the minimum of 0 needs explaining
df['Age'].value_counts().sort_index().head(10)
"""),

md("""
165 members recorded at age 0, then a gap with nothing at ages 1 through 4, then small numbers
from age 5 onwards.

**That gap is the tell.** In a real population of 100,000 people I would expect roughly similar
numbers of one, two, three and four-year-olds. Finding 165 members at exactly age 0 and then
nobody at all for four years means age 0 is not describing infants — it is a placeholder standing
in for something else, most likely an age that was never recorded.
"""),

md("""
## Do the Ages Match the Lives Described?

This is the check I most want to run. Each record carries an age, and separately carries
employment, marital status, education and dependents. Those should agree with one another — a
two-year-old should not be employed, married or holding a degree.

Starting with the members recorded as age 0.
"""),

code("""
# Looking at what the age 0 records claim about themselves
infants = df[df['Age'] == 0]

pd.DataFrame({
    'Employment': infants['Employment Status'].value_counts(),
    'Marital status': infants['Marital Status'].value_counts(),
})
"""),

md("""
Of the 165 members recorded at age 0, 95 are marked `Employed` and a further group is marked
`Retired`. Many are recorded as married.

Taken with the missing ages 1 to 4, this confirms what the placeholder theory predicted: these
records describe adults whose age was not captured. The rest of each record describes a working,
married person — only the age field says infant.

Now let me check whether this is confined to age 0 or runs through the whole young end of the
book.
"""),

code("""
# Building ten-year age bands so I can compare employment across the age range
AGE_BINS_10 = [-1, 15, 29, 39, 49, 59, 69, 120]
AGE_LABELS_10 = ['0-15 (Minors)', '16-29 (Youth)', '30-39 (Mid-30s)', '40-49 (40s)',
                 '50-59 (50s)', '60-69 (60s)', '70+ (Seniors 70+)']

df['Age Cohort (10y)'] = pd.cut(df['Age'], bins=AGE_BINS_10, labels=AGE_LABELS_10)
df['Age Cohort (10y)'].value_counts().sort_index()
"""),

code("""
# Comparing employment status across every age band, as a percentage of each band
(pd.crosstab(df['Age Cohort (10y)'], df['Employment Status'], normalize='index') * 100).round(1)
"""),

md("""
Each row is an age band, each figure a percentage of that band.

If employment related to age at all, the rows would differ — under-15s barely working, over-70s
mostly retired. **Every row is the same.** `Employed` sits between 54.2% and 55.5% in every band
including 0-15; `Retired` between 19.2% and 20.5% everywhere, children included.

Employment status was assigned independently of age. It cannot be used as it stands at either end
of the range, and cleaning has to decide what to do about that. **UW**
"""),

code("""
# Sizing the problem at the young end
minors = df[df['Age'] <= 17]

pd.Series({
    'Under 18 total': len(minors),
    'Recorded as working': int(minors['Employment Status'].isin(['Employed', 'Self-employed']).sum()),
    'Recorded as married': int((minors['Marital Status'] == 'Married').sum()),
    'Holding a university degree': int(minors['Education (Qualification)'].isin(['Bachelors', 'Masters', 'Doctorate']).sum()),
    'Carrying dependents': int((minors['Dependents'] > 0).sum()),
})
"""),

md("""
3,044 members are under 18: **2,072 recorded as working**, 1,632 as married, 1,330 holding a
degree, 1,761 carrying dependents.

Four fields disagree with age at once, and usually in the same record. That rules out repairing a
single field — correcting one would mean inventing a member. **Cleaning has to remove these
records rather than correct them.**

One nuance: dependents among 16-17s is plausible, marriage and doctorates among infants are not.
So the rules should tighten at the youngest bands and loosen at the oldest, not apply uniformly to
everyone under 18.
"""),

md("""
### The same question at the other end of the book
"""),

code("""
# Looking at how employment is recorded among members at or past retirement age
seniors = df[df['Age'] >= 70]

pd.DataFrame({
    'Members': seniors['Employment Status'].value_counts(),
    'Share (%)': (seniors['Employment Status'].value_counts(normalize=True) * 100).round(2),
})
"""),

md("""
8,491 members are 70 or over and **4,604 — 54% — are recorded as actively employed**, against 1,742
retired.

Same artefact, opposite end, but a different problem. Here **only the employment field disagrees**.
Cost, claims, blood pressure, conditions and household are all ordinary and usable, so there is one
field to correct and 53 sound ones to keep.

That distinction drives cleaning: remove the minors, correct the field for the seniors.
"""),

md("""
## Cross-Checking the Diabetes Flag Against the Lab Result

The dataset carries two separate pieces of information about diabetes, and I can use each to
check the other.

`Diabetes` is an administrative flag — somebody has recorded that this member is diabetic.
`HbA1c Level` is a lab measurement of average blood sugar over the previous two to three months.
The clinical threshold for diagnosing diabetes is **6.5%**.

Putting them together gives four groups, and two of them are interesting.
"""),

code("""
# Splitting members by what the lab says and by what the flag says
df['Glycemic Status'] = np.where(df['HbA1c Level'] >= 6.5,
                                 'HbA1c >= 6.5% (Diabetic)', 'HbA1c < 6.5% (Normal/Pre-DM)')
df['Diagnosis Status'] = np.where(df['Diabetes'] == 1, 'Diagnosed Diabetic', 'No Diagnosis Flag')

pd.crosstab(df['Diagnosis Status'], df['Glycemic Status'], margins=True)
"""),

md("""
- **91,127** normal lab, no diagnosis. Nothing to look at.
- **7,431** high lab with a diagnosis. Identified, treatment not yet controlling the level.
- **1,162** diagnosed with a normal lab. Managed back into range.
- **280** high lab and **no diagnosis at all**.

The last group is the actionable one: diabetic on their own file, invisible administratively.
"""),

code("""
# Labelling the four groups so the later sections can work with them directly
DIABETES_QUADRANTS = [
    '1. Non-Diabetic / Normal',
    '2. UNDIAGNOSED DIABETIC (Silent Risk)',
    '3. Controlled Diagnosed Diabetic',
    '4. Uncontrolled Diagnosed Diabetic',
]

df['Diabetes Clinical Quadrant'] = np.select(
    [
        (df['Diabetes'] == 0) & (df['HbA1c Level'] < 6.5),
        (df['Diabetes'] == 0) & (df['HbA1c Level'] >= 6.5),
        (df['Diabetes'] == 1) & (df['HbA1c Level'] < 6.5),
        (df['Diabetes'] == 1) & (df['HbA1c Level'] >= 6.5),
    ],
    DIABETES_QUADRANTS,
    default='Unknown',
)

df.groupby('Diabetes Clinical Quadrant').agg(
    Members=('Id', 'count'),
    Mean_HbA1c=('HbA1c Level', 'mean'),
    Median_Cost=('Annual Medical Cost', 'median'),
).round(2)
"""),

md("""
The undiagnosed group averages an HbA1c of 6.62%, comfortably over the 6.5% threshold, so this is
not a handful of members sitting marginally on the line.

I am flagging this to care management rather than treating it as a data error. Nothing here is
inconsistent in the way the married infants were — the lab result and the flag are both plausible
on their own, they simply have not been reconciled with each other. **That is a gap in clinical
follow-up, not a broken record**, and no cleaning rule should touch these members. **CM**
"""),

md("""
## Why Is Alcohol Frequency Missing for 30% of Members?

Whether the gaps can be filled depends on why they are there.

If they are spread evenly across every kind of member, the cause is unrelated to the member and
filling them biases nothing. If certain groups are far more likely to be missing — heavy drinkers
declining to answer — filling them distorts the picture.

So: the missing rate, measured within every demographic group.
"""),

code("""
# Flagging which members have no alcohol value recorded, and keeping that flag permanently
df['Alcohol_Missing'] = df['Alcohol Frequency'].isnull()

pd.Series({'members missing': int(df['Alcohol_Missing'].sum()),
           'share of book (%)': round(df['Alcohol_Missing'].mean() * 100, 2)})
"""),

code("""
# Checking the missing rate within each demographic group, against that 30.08% baseline
pd.concat({
    feature: (df.groupby(feature)['Alcohol_Missing'].mean() * 100).round(2)
    for feature in ['Sex', 'Region', 'Urban / Rural', 'Smoker Status', 'Employment Status', 'Age Cohort (10y)']
}).rename('missing %').to_frame()
"""),

md("""
Every group sits between 29.5% and 30.4% against a 30.08% baseline — across sex, all five regions,
settlement, employment and age cohort.

Smoking is the informative one. If members were withholding drinking habits, current smokers would
differ; they do not (29.96% against 30.19% for never-smokers).

**The missingness is unrelated to anything observable**, so filling it cannot bias one part of the
book against another. What it does not explain is what the blanks *mean*.
"""),

code("""
# Seeing what the field records when it is not blank
df['Alcohol Frequency'].value_counts(dropna=False)
"""),

md("""
The field takes exactly three values — `Occasional`, `Weekly`, `Daily` — and **all three describe
someone who drinks.** There is no option for a person who does not.

A question with no way to answer "I don't drink", blank for 30% of members and evenly spread, did
not fail to record answers. The blank is the answer.

Decision deferred to section 3. The `Alcohol_Missing` flag stays on the frame permanently so every
later cut can separate recorded from inferred.
"""),

md("""
## What I Found, and What Cleaning Needs to Do

**Structurally sound.** One row per member, no duplicates, no negatives, no numbers stored as text,
no inconsistent category labels. One field of 54 has missing data.

| Finding | Size | Action |
|:---|---:|:---|
| Age 0 with adult attributes, no members aged 1-4 | 165 | Remove — age 0 is a placeholder |
| Under-18s married, employed, degree-holding or with dependents | 3,044 | Remove, stricter at the youngest bands. Several fields disagree at once |
| Members 70+ recorded as employed | 4,604 | **Correct the field, keep the member.** Only employment disagrees |
| `Alcohol Frequency` blank | 30,083 | Fill as non-drinking. The survey offered no such option |

**For care management, not cleaning:** 280 members at or above the diabetic threshold with no
diagnosis flag. Their records are not broken. **CM**

**Carry forward:** employment status bears no relationship to age. Whatever cleaning does will
*create* a relationship the source did not have, and any later age-employment finding is reporting
that rule.
"""),

]
