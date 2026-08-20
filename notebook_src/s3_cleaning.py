"""Section 3 of notebook.ipynb — cleaning.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
---

# 3. Cleaning

Section 2 changed no values. This is where the records that cannot be right get resolved.

The standard, since deleting rows is not neutral: **a record goes only when the combination of
values in it could not describe a real person.** A seven-year-old married with a doctorate is a
broken record. A 25-year-old with high blood pressure is unusual and stays.

There is a modelling reason too. Leave two thousand teenagers holding doctorates in the training
set and the model learns that education tells you nothing about age.

Where one field can be fixed instead of deleting a record, it is — once, at Rule 5.
"""),

md("""
## Keeping an Audit Trail

Before I change anything I want a record of every decision, because in six months nobody will
remember why the row count is what it is — including me. Each rule writes a line to a log, and I
keep an untouched copy of the frame so I can check at the end whether the members I removed
differed from the ones I kept.
"""),

code("""
# Keeping an untouched copy, so the deleted population can be compared against the retained one
df_before_cleaning = df.copy()
deleted_ids = []
"""),

code("""
# Somewhere to record every cleaning decision, so the final row count traces back to a reason
cleaning_log = []


def log_cleaning_action(step, column, action, records, rationale, impact, deleted_rows=0):
    # deleted_rows is recorded separately from Records_Affected, because a rule can touch
    # thousands of members without removing any of them. Rule 5 is exactly that case, and
    # matching on the word "deletion" in the action text would miscount it.
    cleaning_log.append({'Step': step, 'Column': column, 'Action': action,
                         'Records_Affected': records, 'Rows_Deleted': deleted_rows,
                         'Rationale': rationale, 'Downstream_Impact': impact})
"""),

md("""
## Rule 1 — Infants Recorded With Adult Lives

Starting with the members recorded as age 0, because that is where the clearest contradictions
are. An infant cannot be married, cannot have stopped smoking, cannot hold a degree, cannot be
retired, and cannot have dependents of their own.
"""),

code("""
# Finding age 0 records that carry attributes an infant cannot have
is_broken_infant = (df['Age'] == 0) & (
    df['Marital Status'].isin(['Married', 'Divorced', 'Widowed'])
    | df['Smoker Status'].isin(['Current', 'Former'])
    | df['Education (Qualification)'].isin(['Primary', 'High School', 'College', 'Bachelors', 'Masters', 'Doctorate'])
    | df['Alcohol Frequency'].isin(['Occasional', 'Weekly', 'Daily'])
    | df['Employment Status'].isin(['Employed', 'Retired', 'Self-employed', 'Unemployed'])
    | (df['Dependents'] > 0)
)

pd.Series({'age 0 records in total': int((df['Age'] == 0).sum()),
           'of those, carrying an impossible attribute': int(is_broken_infant.sum())})
"""),

md("""
Both figures are 165. **Every age-0 record carries at least one impossible adult attribute** — not
most, all.

Scattered data-entry errors would hit some. Hitting all of them means age 0 is not recording
infants; it stands in for a missing or unparsed age. There is no correct value to substitute and
the rest of each record describes an adult, so all 165 go.
"""),

code("""
# Removing the 165 broken infant records
deleted_ids += df.loc[is_broken_infant, 'Id'].tolist()
df = df[~is_broken_infant].copy().reset_index(drop=True)

log_cleaning_action(
    1, 'Age', 'Row deletion (corrupted age 0 records)', int(is_broken_infant.sum()),
    'Every age 0 record carried at least one impossible adult attribute, so age 0 is a placeholder '
    'for an unrecorded age rather than a real infant',
    f'Working dataset: {len(df):,} rows, ages 1 to 100',
    deleted_rows=int(is_broken_infant.sum()))

len(df)
"""),

md("""
## Building the Age Bands

Before the remaining age rules I need the bands themselves, because the rules differ by life
stage — what is impossible for a seven-year-old is ordinary for a seventeen-year-old.

The band that matters most for later is the last one. **I am setting retirement at 70**, so
anyone aged 70 or above belongs to the retiree cohort. That threshold drives Rule 5 below, and it
is a business decision rather than something the data told me.

I am building the broader life-stage grouping in the same cell, since the exploratory section
uses it to compare four coarse stages rather than nine fine ones.
"""),

code("""
# Cutting age into life-stage bands, with retirement starting at 70
AGE_GROUP_BINS = [0, 7, 15, 17, 25, 35, 45, 55, 69, 120]
AGE_GROUP_LABELS = ['1-7: Preteen', '8-15: Teenager', '16-17: Preadult', '18-25: Youth',
                    '26-35: Mature Adults', '36-45: Prime Adults', '46-55: Middle-Aged Adults',
                    '56-69: Pre-Retirement Adults', '70+: Retirees']

LIFE_STAGE_BINS = [-1, 15, 29, 49, 64, 120]
LIFE_STAGE_LABELS = ['0-15 (Minors)', '16-29 (Youth)', '30-49 (Mid-Career)',
                     '50-64 (Pre-Retire)', '65+ (Seniors)']

df['Age Groups'] = pd.cut(df['Age'], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS)
df['Age_Life_Stage'] = pd.cut(df['Age'], bins=LIFE_STAGE_BINS, labels=LIFE_STAGE_LABELS)

log_cleaning_action(
    2, 'Age Groups / Age_Life_Stage', 'Life-stage engineering', len(df),
    'Segmented ages into nine actuarial cohorts and five broader life stages, with retirees '
    'starting at age 70',
    'Enables life-stage stratified analysis, and sets the threshold Rule 5 depends on')

pd.DataFrame({
    'Members': df['Age Groups'].value_counts().sort_index(),
    'Share (%)': (df['Age Groups'].value_counts(normalize=True).sort_index() * 100).round(2),
})
"""),

md("""
The three childhood bands are small — a few hundred to a couple of thousand members each — while
the working-age bands hold tens of thousands. That matters for the next few rules: I am about to
examine the smallest groups in the book, so a rule that removes most of a childhood band still
removes very few members overall.
"""),

md("""
## Rule 2 — Preteens Aged 1 to 7

For a one to seven-year-old I am treating these as impossible: having smoked, having been
married, drinking alcohol, holding a school or university qualification, being employed or
retired, or having dependents of their own.

One deliberate exception. I am **allowing self-employment** in this band, because a child earning
small amounts from chores or a stall is plausible enough that I would rather keep the record than
lose it.
"""),

code("""
# Finding preteens carrying attributes a young child cannot have
is_broken_preteen = df['Age'].between(1, 7) & (
    df['Smoker Status'].isin(['Current', 'Former'])
    | df['Marital Status'].isin(['Married', 'Divorced', 'Widowed'])
    | df['Alcohol Frequency'].isin(['Occasional', 'Weekly', 'Daily'])
    | df['Education (Qualification)'].isin(['High School', 'College', 'Bachelors', 'Masters', 'Doctorate'])
    | df['Employment Status'].isin(['Employed', 'Retired'])
    | (df['Dependents'] > 0)
)

pd.Series({'preteens in total': int(df['Age'].between(1, 7).sum()),
           'of those, carrying an impossible attribute': int(is_broken_preteen.sum())})
"""),

md("""
472 preteens, and **all 472 match at least one impossible condition**. The same pattern as the
infants — the whole band is affected, not a scattering of it.

I am removing all of them. It costs me the entire 1-to-7 band, which is worth stating plainly:
after this step the dataset contains no young children at all. For a health insurance book that
is a real limitation, and anyone asking about paediatric cover needs to know this data cannot
answer them.
"""),

code("""
# Removing the broken preteen records
deleted_ids += df.loc[is_broken_preteen, 'Id'].tolist()
df = df[~is_broken_preteen].copy().reset_index(drop=True)

log_cleaning_action(
    3, 'Age Groups (preteens)', 'Row deletion (corrupted preteen records)', int(is_broken_preteen.sum()),
    'Ages 1-7 carrying marriage, employment, retirement, degrees, dependents, smoking or drinking',
    f'Working dataset: {len(df):,} rows. No members under 8 remain, so paediatric questions '
    'cannot be answered from this book',
    deleted_rows=int(is_broken_preteen.sum()))

len(df)
"""),

md("""
## Rule 3 — Teenagers Aged 8 to 15

For this band I loosen the rules, because more becomes possible as children get older.

I still treat drinking, marriage, smoking, university-level qualifications and being employed or
retired as impossible. But I now **allow Primary and High School education**, since a
fifteen-year-old having finished primary school is entirely normal, and I **allow up to two
dependents** rather than none — a teenager could legitimately appear on a policy alongside
siblings.

The point of loosening the rules is that I want to keep every record I reasonably can. Each one
is real exposure.
"""),

code("""
# Finding teenagers with attributes that do not fit the age band
is_broken_teen = df['Age'].between(8, 15) & (
    df['Alcohol Frequency'].isin(['Occasional', 'Weekly', 'Daily'])
    | df['Marital Status'].isin(['Married', 'Divorced', 'Widowed'])
    | df['Smoker Status'].isin(['Current', 'Former'])
    | df['Education (Qualification)'].isin(['College', 'Bachelors', 'Masters', 'Doctorate'])
    | df['Employment Status'].isin(['Employed', 'Retired'])
    | (df['Dependents'] > 2)
)

pd.Series({'teenagers in total': int(df['Age'].between(8, 15).sum()),
           'of those, carrying an impossible attribute': int(is_broken_teen.sum())})
"""),

md("""
1,643 teenagers, of whom 1,632 match. Eleven survive.

Loosening the rules did keep a handful of records that stricter criteria would have removed,
which is what I wanted. But eleven members is far too few to say anything about, and I suppress
the band in every later chart rather than draw conclusions from it.
"""),

code("""
# Removing the teenage records that do not fit
deleted_ids += df.loc[is_broken_teen, 'Id'].tolist()
df = df[~is_broken_teen].copy().reset_index(drop=True)

log_cleaning_action(
    4, 'Age Groups (teenagers)', 'Row deletion (corrupted teenager records)', int(is_broken_teen.sum()),
    'Ages 8-15 carrying alcohol, smoking, marriage, tertiary degrees, adult employment or '
    'retirement, or more than two dependents. Primary and high school education allowed',
    f'Working dataset: {len(df):,} rows. Eleven teenagers retained, too few to report on',
    deleted_rows=int(is_broken_teen.sum()))

len(df)
"""),

md("""
## Rule 4 — Preadults Aged 16 to 17

Looser again, because sixteen and seventeen-year-olds can legitimately do most adult things.

Here I only remove records showing regular drinking, marriage or widowhood, being a **former**
smoker (current smoking is plausible at this age, having already quit is a stretch), holding a
university degree, being **retired**, or carrying more than three dependents. Employment is
allowed entirely — a seventeen-year-old with a job is ordinary.
"""),

code("""
# Finding preadults with attributes that do not fit the age band
is_broken_preadult = df['Age'].between(16, 17) & (
    df['Alcohol Frequency'].isin(['Occasional', 'Weekly', 'Daily'])
    | df['Marital Status'].isin(['Married', 'Divorced', 'Widowed'])
    | (df['Smoker Status'] == 'Former')
    | df['Education (Qualification)'].isin(['College', 'Bachelors', 'Masters', 'Doctorate'])
    | (df['Employment Status'] == 'Retired')
    | (df['Dependents'] > 3)
)

pd.Series({'preadults in total': int(df['Age'].between(16, 17).sum()),
           'of those, carrying an impossible attribute': int(is_broken_preadult.sum())})
"""),

code("""
# Removing the preadult records that do not fit
deleted_ids += df.loc[is_broken_preadult, 'Id'].tolist()
df = df[~is_broken_preadult].copy().reset_index(drop=True)

log_cleaning_action(
    5, 'Age Groups (preadults)', 'Row deletion (corrupted preadult records)', int(is_broken_preadult.sum()),
    'Ages 16-17 carrying alcohol, marriage, divorce or widowhood, former smoking, tertiary '
    'degrees, retirement, or more than three dependents. Employment allowed',
    f'Working dataset: {len(df):,} rows',
    deleted_rows=int(is_broken_preadult.sum()))

len(df)
"""),

md("""
764 preadults, 740 removed, 24 retained.

Taking the three childhood rules together: they removed 2,844 records and left 35 members under
18 in a book of 97,000. **This is effectively an adult-only dataset**, and I treat it as one from
here on.
"""),

md("""
## Rule 5 — Seniors Recorded as Employed

**4,604 of 8,491 seniors** are recorded as employed, 54%, against 55.4% among the under-30s.
Employment was assigned without reference to age.

### Why recode rather than delete

The childhood rules deleted because **several fields were wrong at once**, leaving no single repair
that made the record coherent.

Here **one field of 54 is wrong**. Deleting discards 53 good fields to fix one — and cuts the 70+
cohort to 3,887, **less than half**, leaving questions about older members answered by whichever
seniors the source happened not to mark as employed.

**Consequence:** `Employed` becomes 0% at 70+ by construction. That is this rule in the data, not a
finding about the portfolio.
"""),

code("""
# Recoding employment for seniors, instead of deleting the records
is_working_senior = (df['Age'] >= 70) & (df['Employment Status'] == 'Employed')
n_recoded = int(is_working_senior.sum())

df.loc[is_working_senior, 'Employment Status'] = 'Retired'

log_cleaning_action(
    6, 'Employment Status (70+)', 'Field recode, not deletion', n_recoded,
    'Recoded Employed to Retired for members aged 70 and over. Retirement age is 70 and the '
    'source assigned employment independently of age (54.2% employed at 70+ vs 55.4% under 30). '
    'Only the employment field is implausible; every other attribute is valid and needed for exposure',
    f'No rows lost. Full 70+ cohort of {int((df["Age"] >= 70).sum()):,} retained. Employed is now '
    '0% at 70+ by construction and must not be reported as a finding')

df[df['Age'] >= 70]['Employment Status'].value_counts()
"""),

md("""
No seniors are marked employed any more, and the cohort still holds all 8,491 members. The row
count has not moved, which is exactly what I wanted — this rule corrected a field, it did not
remove anyone. **UW / ACT**
"""),

md("""
## Rule 6 — Missing Alcohol Frequency

Section 2 established the two facts this decision rests on. The blanks are spread evenly across
every observable group, so filling them cannot bias one part of the book against another. And the
field only ever takes three values — `Occasional`, `Weekly`, `Daily` — none of which describes a
person who does not drink.

A survey that offers no way to say "I don't drink" and then shows 30% blanks is not showing me
30% missing data. It is showing me the non-drinkers, who had nothing they could tick.

So I am filling the blanks with `Non Alcoholic`.
"""),

code("""
# Filling the blanks with a category the survey never offered
n_missing_alcohol = int(df['Alcohol Frequency'].isnull().sum())
df['Alcohol Frequency'] = df['Alcohol Frequency'].fillna('Non Alcoholic')

log_cleaning_action(
    7, 'Alcohol Frequency', 'Domain-specific categorical imputation', n_missing_alcohol,
    'Filled blanks as Non Alcoholic. DISCLOSURE: this category does not exist in the recorded '
    'data, so 100% of Non Alcoholic members are imputed. Missingness is uniform across every '
    'observable segment, under which most of these members would in fact be Occasional drinkers. '
    'Alcohol_Missing flag retained so every downstream cut can separate recorded from imputed',
    'Alcohol Frequency is complete, but no underwriting or care-management decision may use it')

df['Alcohol Frequency'].value_counts()
"""),

md("""
**Disclosure — read before using any alcohol figure.** `Non Alcoholic` is synthetic. The source
recorded only Occasional, Weekly and Daily, so all 29,092 members in that category are imputed,
none observed.

The uniform missingness cuts both ways: it means filling is unbiased across groups, but it is
equally consistent with ordinary non-response, in which case most of these members drink.

An honest `Unknown` category was the alternative; it leaves 30% of the book uninterpretable on any
chart. The `Alcohol_Missing` flag is retained and 4.3 tests whether any finding depends on the
field. **No underwriting, pricing or care-management decision should use it. UW / ACT**
"""),

md("""
## Rule 7 — Young Members With Graduate Degrees

The last rule. A Masters or Doctorate takes a Bachelors first — three or four years of university
after school — and then another two to five on top. Somebody aged 18 to 21 has not had time.
"""),

code("""
# Finding members too young to have completed a graduate degree
is_early_graduate = df['Age'].between(18, 21) & df['Education (Qualification)'].isin(['Masters', 'Doctorate'])

df.loc[is_early_graduate, 'Education (Qualification)'].value_counts()
"""),

md("""
352 members aged 18 to 21 hold a Masters or Doctorate.

This is the rule I am least confident in. Unlike a married infant, an advanced 21-year-old with a
Masters is rare rather than impossible. They go anyway: 352 in a four-year band is a pattern, not a
scattering of prodigies, and a model trained on it learns that age and education are unrelated.

At 0.36% of the book the cost of being wrong is small either way.
"""),

code("""
# Removing the records that claim a degree the member has not had time to earn
deleted_ids += df.loc[is_early_graduate, 'Id'].tolist()
df = df[~is_early_graduate].copy().reset_index(drop=True)

log_cleaning_action(
    8, 'Education (Qualification)', 'Row deletion (implausible early graduates)', int(is_early_graduate.sum()),
    'Ages 18-21 holding Masters or Doctorate. Genuine early graduates exist, so this is '
    'implausible at scale rather than strictly impossible; at 352 in a four-year band it is '
    'indistinguishable from the independent-assignment artefact seen in the minor bands',
    f'Working dataset: {len(df):,} rows',
    deleted_rows=int(is_early_graduate.sum()))

len(df)
"""),

md("""
## Two Flags Worth Carrying Forward

Section 2 found undiagnosed diabetics by cross-referencing a lab value against a diagnosis flag,
and that column is already on the frame. The same pattern applies to blood pressure, and there is
a second oddity worth marking: members recorded as retired well before retirement age.

Neither of these removes or corrects anything. They are labels for later.
"""),

code("""
# Two labels the later sections can use directly
df['undiagnosed_htn_flag'] = ((df['Systolic Blood Pressure'] >= 140) & (df['Hypertension'] == 0)).astype(int)
df['early_retired_flag'] = ((df['Age'] < 55) & (df['Employment Status'] == 'Retired')).astype(int)

log_cleaning_action(
    9, 'undiagnosed_htn_flag / early_retired_flag', 'Feature engineering', 0,
    'Marked members with a systolic reading at or above 140 and no hypertension diagnosis, and '
    'members recorded as retired before age 55. Neither changes any existing value',
    'Available to the care-management analysis in section 4')

pd.Series({'undiagnosed hypertension': int(df['undiagnosed_htn_flag'].sum()),
           'retired before 55': int(df['early_retired_flag'].sum())})
"""),

md("""
## Final Checks

Three things before I go any further: that nothing is missing, that I have not created
duplicates, and that the row count reconciles against the log.
"""),

code("""
# Reconciling the row count against the log, so the arithmetic is provable
log = pd.DataFrame(cleaning_log)
deleted_total = int(log['Rows_Deleted'].sum())

pd.Series({
    'Started with': len(df_before_cleaning),
    'Rows deleted': deleted_total,
    'Rows recoded, not deleted': int(n_recoded),
    'Expected remaining': len(df_before_cleaning) - deleted_total,
    'Actually remaining': len(df),
    'Missing values anywhere': int(df.isnull().sum().sum()),
    'Duplicated rows': int(df.duplicated().sum()),
    'Unique member ids': df['Id'].nunique(),
})
"""),

md("""
Reconciles exactly: 100,000 in, 3,361 deleted, 96,639 out, nothing missing, no duplicates, grain
intact.

The log tracks rows deleted separately from records affected, and has to. Rule 5 touched 4,604
members without removing any, so counting every rule's affected records as deletions gives 7,965
and disagrees with the row count by exactly that recode.

Of the 3,361: 165 infants, 2,844 children and teenagers, 352 early graduates. **The 4,604 seniors
were recoded, not deleted.** Deleting them instead would leave 92,035 rows and a senior cohort at
less than half its real size.
"""),

code("""
# The full decision log
log
"""),

md("""
## What Was Lost — Checking the Deletions for Bias

Deleting rows is not neutral, so I want to know whether the 3,361 members I removed differ from
the 96,639 I kept on the outcomes that matter. The deletions were driven by attribute
*combinations*, so similarity has to be demonstrated rather than assumed.
"""),

code("""
# Comparing deleted against retained on the fields a model or an actuary cares about
was_deleted = df_before_cleaning['Id'].isin(deleted_ids)
deleted_rows = df_before_cleaning[was_deleted]
retained_rows = df_before_cleaning[~was_deleted]


def outcome_profile(frame):
    return {
        'Mean annual medical cost': frame['Annual Medical Cost'].mean(),
        'Median annual medical cost': frame['Annual Medical Cost'].median(),
        'Mean claims count': frame['Claims Count'].mean(),
        'Mean chronic conditions': frame['Chronic Conditions Count'].mean(),
        'Mean risk score': frame['Risk Score'].mean(),
        'High-risk rate (%)': frame['Is High Risk'].mean() * 100,
        'Smoker, current or former (%)': frame['Smoker Status'].isin(['Current', 'Former']).mean() * 100,
        'Female (%)': (frame['Sex'] == 'Female').mean() * 100,
        'Mean income': frame['Income'].mean(),
    }


bias_report = pd.DataFrame({
    f'Deleted (n={len(deleted_rows):,})': outcome_profile(deleted_rows),
    f'Retained (n={len(retained_rows):,})': outcome_profile(retained_rows),
}).round(2)
bias_report['Difference (%)'] = ((bias_report.iloc[:, 0] / bias_report.iloc[:, 1] - 1) * 100).round(1)
bias_report
"""),

md("""
The demographics that should not matter — sex, income, smoking — match within a few percent, so the
deletions hit the artefact rather than a real sub-population.

The outcome columns do not match. Deleted members are 23% cheaper, 15% less morbid and 66% lower
risk (a 2% high-risk rate against 38%).

That is **age structure, not bias**: the deletions removed children and young adults, `Risk Score`
is age-driven, and cost rises with age. Testing it requires comparing within one age band.
"""),

code("""
# The one age band containing both deleted and retained members, so the comparison is like for like
band = df_before_cleaning[df_before_cleaning['Age'].between(18, 21)]
band_deleted = band[band['Id'].isin(deleted_ids)]
band_retained = band[~band['Id'].isin(deleted_ids)]

pd.DataFrame({
    f'Deleted 18-21 (n={len(band_deleted):,})': outcome_profile(band_deleted),
    f'Retained 18-21 (n={len(band_retained):,})': outcome_profile(band_retained),
}).round(2)
"""),

md("""
Within the 18-21 band the two groups are close on every outcome, which confirms the diagnosis: the
headline gap is age composition, not a systematic removal of one kind of member.

**Consequence to carry forward:** the cleaned book is older, sicker and more expensive than the
raw extract. Any comparison between raw and cleaned figures has to adjust for age before it means
anything. **ACT**
"""),

md("""
## Writing Out the Cleaned Data

One file leaves this notebook: the cleaned dataset. The audit script in `scripts/verify_claims.py`
reads it, so it needs to exist on disk. Everything else — the log, the bias report, the charts
below — stays in the notebook where it can be read in context.
"""),

code("""
# The one dataset written to disk, because it is consumed outside this notebook
CLEAN_PATH = PROCESSED_DIR / 'medical_insurance_clean.csv'
df.to_csv(CLEAN_PATH, index=False)

df.shape
"""),

md("""
## What Came Out Of This

**96,639 members, 66 columns, no missing values, no duplicates.**

| Rule | Records | Action |
|:---|---:|:---|
| Infants with adult attributes | 165 | Deleted — every age-0 record was broken |
| Age bands | — | Nine life stages, retirement at 70 |
| Preteens aged 1-7 | 472 | Deleted — the whole band matched |
| Teenagers aged 8-15 | 1,632 | Deleted, 11 retained |
| Preadults aged 16-17 | 740 | Deleted, 24 retained |
| **Seniors marked employed** | **4,604** | **Recoded — no rows lost** |
| Missing alcohol frequency | 29,185 | Filled as Non Alcoholic |
| Under-22s with graduate degrees | 352 | Deleted |

**Three constraints on everything below:**

1. **An adult book.** 35 members under 18 survived; paediatric questions are unanswerable.
2. **`Employed` is 0% at 70+ by construction.** Any age-employment finding reports this rule.
3. **`Non Alcoholic` is 100% imputed**, inferred from absence rather than recorded.
"""),

]
