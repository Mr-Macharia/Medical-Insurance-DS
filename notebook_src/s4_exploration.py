"""Section 4 of notebook.ipynb — exploration.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
---

# 4. Exploring the Book

Who the members are, what moves their cost, which conditions matter, and whether the products fit
the people holding them.

Underneath that sits a prior question: **can this data support those decisions at all?** Analysis
fails more often because a relationship was never there than because a chart was wrong. So each
business conclusion below is preceded by a test that the relationship exists.
"""),

md("""
## Setting Up the Charts

Everything this section needs, in one place. `matplotlib` for the plots, `scipy` for the
contingency test behind Cramér's V, `itertools` to enumerate every field pair, and `scikit-learn`
for the leakage tests at the end.

Three `scikit-learn` modules get pulled in whole here rather than split across two sections — I
need `LinearRegression`, `train_test_split` and `r2_score` below, and section 6 needs the rest of
what those same three modules offer. Importing each module exactly once, at the first point of
use, beats importing from it twice.
"""),

code("""
# Adding what the exploratory work needs on top of pandas and numpy
import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             mean_absolute_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, learning_curve,
                                     train_test_split)

warnings.filterwarnings('ignore')
"""),

md("""
A house style, so every chart in the notebook reads the same way. The palette is colour-blind safe
and kept in a fixed order, so a given category is always the same colour wherever it appears.

The figure size is deliberately modest. I learnt that the hard way — my first pass used figures up
to 19 inches wide and every label became unreadable once the notebook scaled them down.
"""),

code("""
# A colour-blind safe palette, kept in a fixed order so a category keeps its colour across charts
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#4a3aa7', '#e34948'
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK, INK_2, GRID_C = '#0b0b0b', '#52514e', '#e5e4e0'

# A single-hue ramp for "how much" charts, and a blue-to-red ramp with a grey middle for
# "above or below average" charts
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#3987e5', '#9ec5f4', '#f0efec', '#f2a3a2', '#e34948', '#a32b2a'])
"""),

code("""
# Chart defaults, so nothing has to be restated per figure
mpl.rcParams.update({
    'figure.figsize': (10, 6),       # fits a notebook column at roughly 1:1
    'font.size': 12,                 # big enough to survive being scaled down
    'axes.titlesize': 14, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelcolor': INK_2, 'axes.edgecolor': GRID_C,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'grid.color': GRID_C, 'legend.frameon': False,
    'figure.dpi': 100, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})
"""),

code("""
# One short line per chart instead of three, since every figure goes into the written report
def save(name):
    plt.savefig(FIGS / f'{name}.png', bbox_inches='tight', pad_inches=0.3)
"""),

md("""
And the orderings, condition list and derived fields I reuse throughout. The derived fields are
analysis conveniences — a loss ratio, income quintiles, a procedure total — and section 6 drops
every one of them before it builds anything, for reasons I set out there.
"""),

code("""
# Fixed orderings, so categories appear in a sensible sequence rather than alphabetically
AGE_COHORT = ['16-29 (Youth)', '30-39 (Mid-30s)', '40-49 (40s)', '50-59 (50s)',
              '60-69 (60s)', '70+ (Seniors 70+)']
LIFE_STAGE = ['16-29 (Youth)', '30-49 (Mid-Career)', '50-64 (Pre-Retire)', '65+ (Seniors)']
TIER = ['Bronze', 'Silver', 'Gold', 'Platinum']
EMPLOY = ['Employed', 'Self-employed', 'Unemployed', 'Retired']
EDU = ['Primary', 'High School', 'College', 'Bachelors', 'Masters', 'Doctorate']
"""),

code("""
# The ten recorded conditions, plus shorter labels for when they need to fit on a chart axis
CONDITIONS = ['Hypertension', 'Diabetes', 'Asthma', 'COPD', 'Cardiovascular Disease',
              'Cancer History', 'Kidney Disease', 'Liver Disease', 'Arthritis',
              'Mental Health Condition']

COND_SHORT = {'Cardiovascular Disease': 'Cardiovascular', 'Cancer History': 'Cancer',
              'Mental Health Condition': 'Mental health', 'Kidney Disease': 'Kidney',
              'Liver Disease': 'Liver'}
"""),

code("""
# Derived fields I need repeatedly in this section only
df['Loss Ratio'] = df['Annual Medical Cost'] / df['Annual Premium']
df['Income Band'] = pd.qcut(df['Income'], 5, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4', 'Q5 highest'])
df['Total Procedures'] = df[[c for c in df.columns if c.endswith('Procedures Count')]].sum(axis=1)
df['Top Decile'] = (df['Annual Medical Cost'] > df['Annual Medical Cost'].quantile(0.9)).astype(int)

# Any group smaller than this gets suppressed rather than reported on
MIN_N = 100
"""),

md("""
---

## 4.1 Is the Premium Actually a Price?

Before I chart anything about pricing I want to check one thing. `Annual Premium` is the field a
business would naturally build a pricing story on, so if there is something wrong with it, every
conclusion after that point inherits the problem.

Starting with how strongly each field moves with medical cost.
"""),

code("""
# Checking what correlates with the thing I am trying to explain
numeric = df.select_dtypes(include=[np.number]).drop(columns=['Id', 'Top Decile'])
numeric.corr()['Annual Medical Cost'].drop('Annual Medical Cost').abs().sort_values(ascending=False).head(8)
"""),

md("""
`Annual Premium` correlates with cost at **0.965**. `Total Claims Paid` at 0.74. The strongest
clinical field, chronic condition count, reaches 0.30.

That ordering is backwards. Health should predict cost; premium should be a risk judgement made in
advance. A premium tracking cost at 0.965 is not a judgement — it is arithmetic performed after the
cost was known.

Strong claim, so it gets tested directly. If premium is derived from cost, the scatter will show
lines rather than a cloud.
"""),

code("""
# Taking a sample so the scatter is readable, then plotting premium against cost by tier
sample = df.sample(9000, random_state=7)

plt.figure(figsize=(10, 6))
for i, tier in enumerate(TIER):
    members = sample[sample['Network Tier'] == tier]
    plt.scatter(members['Annual Medical Cost'], members['Annual Premium'], s=6, alpha=0.45,
                color=CAT[i], label=tier, linewidths=0)

plt.xlim(0, 22000)
plt.ylim(0, 4200)
plt.xlabel('Annual medical cost')
plt.ylabel('Annual premium')
plt.title('Annual premium against annual medical cost, by network tier')
plt.legend(title='Network tier', markerscale=3)
plt.grid(alpha=0.5)
save('v2_01_premium_vs_cost')
plt.show()
"""),

md("""
##### Reading the scatter

Each point is one member: cost horizontally, premium vertically.

Risk-priced premiums would give a **cloud** — two members with the same eventual cost assessed
differently, sitting at different heights above the same point. Sloping upward, but broad.

Instead: **four straight lines**, one per tier, with almost no scatter. A straight line from near
the origin means premium is a **fixed percentage of cost**, and four lines mean that percentage
changes only with tier.

The absence of scatter is the finding. If age, conditions or claims history moved premium even
slightly, members would sit off their tier's line. Nothing about the member shifts them.
"""),

code("""
# Fitting a line per tier to recover the rate each one charges per unit of cost
RATE = {tier: round(float(np.polyfit(g['Annual Medical Cost'], g['Annual Premium'], 1)[0]), 3)
        for tier, g in df.groupby('Network Tier')}

RATE
"""),

md("""
The slopes are the share of cost charged as premium: Bronze 9.6%, Silver 12%, Gold 14.4%, Platinum
17.4%.

Exact to three decimals and stepping up in order — rates from a business process are rarely this
tidy. The lines start slightly above zero, so a fixed amount is charged before cost enters. The
deductible is the only other product field that varies, which gives a full formula to test.
"""),

code("""
# Testing whether a closed-form formula reproduces the premium exactly
predicted_premium = 200 + 0.01 * df['Deductible'] + df['Network Tier'].map(RATE) * df['Annual Medical Cost']
premium_error = (df['Annual Premium'] - predicted_premium).abs()

# Half a cent is just rounding, so anything within that counts as an exact match
pd.Series({'share reproduced to the cent': (premium_error <= 0.005).mean(),
           'largest error anywhere in the book': premium_error.max()})
"""),

md("""
##### The finding

**99.997%** of premiums reproduce to within half a cent — 96,636 of 96,639 — and the largest error
anywhere is **0.005**, exactly what rounding to the nearest cent produces.

```
Annual Premium = 200 + 0.01 × Deductible + tier_rate × Annual Medical Cost
tier_rate:  Bronze 0.096 · Silver 0.120 · Gold 0.144 · Platinum 0.174
```

Not an approximation of how premiums were set — the calculation itself. No member attribute enters
it.

1. **Pricing adequacy cannot be assessed.** Premium against cost measures the tier rate. Any
   loss-ratio conclusion is circular. **ACT / FIN**
2. **Premium and its relatives are barred as model features.** Enforced in section 6.
3. **Affordability remains a fair question**, since the charge ignores income entirely — 4.4.
"""),

code("""
# Plotting how far each member sits from the formula prediction
plt.figure(figsize=(10, 6))
plt.hist(premium_error, bins=60, color=BLUE)
plt.xlabel('Absolute difference between actual premium and formula prediction')
plt.ylabel('Members')
plt.title('The formula reproduces every premium to within half a cent')
plt.grid(axis='y', alpha=0.5)
save('v2_02_premium_formula_error')
plt.show()
"""),

md("""
Every member sits within half a cent of the prediction and the whole distribution is bunched
inside that range. There is no tail of members the formula fails on, which is what I would expect
to see if the premium had any genuine underwriting component my formula was missing.

For completeness, the monthly premium is nothing more than the annual figure divided by twelve, so
both premium fields carry exactly the same circular information.
"""),

code("""
# Confirming monthly premium carries nothing beyond annual / 12
bool(np.allclose(df['Monthly Premium'], df['Annual Premium'] / 12, atol=0.01))
"""),

md("""
---

## 4.2 What Actually Drives Cost?

Premium is circular, so the question becomes what genuinely moves cost — and therefore where
underwriting should look.

Most candidates are categorical, so correlation does not apply. I use **eta squared**: the share of
total cost variance falling between groups rather than within them, read as a percentage. 1% means
group membership accounts for 1% of why costs differ.

Eta squared drifts upward for factors with many levels, so **omega squared** is reported alongside
it as the bias-corrected counterpart.
"""),

code("""
# How much of the variation in cost each grouping explains
def eta_squared(frame, cat, val='Annual Medical Cost'):
    grouped = frame.groupby(cat, observed=True)[val]
    grand_mean = frame[val].mean()
    between = (grouped.count() * (grouped.mean() - grand_mean) ** 2).sum()
    total = ((frame[val] - grand_mean) ** 2).sum()
    return between / total
"""),

code("""
# The same thing, corrected for the number of groups a factor happens to have
def omega_squared(frame, cat, val='Annual Medical Cost'):
    grouped = frame.groupby(cat, observed=True)[val]
    grand_mean = frame[val].mean()
    k, n = frame[cat].nunique(), len(frame)
    ss_between = (grouped.count() * (grouped.mean() - grand_mean) ** 2).sum()
    ss_total = ((frame[val] - grand_mean) ** 2).sum()
    ms_error = (ss_total - ss_between) / (n - k)
    return max(0.0, (ss_between - (k - 1) * ms_error) / (ss_total + ms_error))
"""),

code("""
# Running both across every candidate segmentation I might want to use
CANDIDATES = ['Chronic Conditions Count', 'Smoker Status', 'Had Major Procedure',
              'Age Cohort (10y)', 'Glycemic Status', 'Employment Status', 'Household Size',
              'Education (Qualification)', 'Region', 'Urban / Rural', 'Sex',
              'Marital Status', 'Income Band', 'Plan Type', 'Network Tier', 'Alcohol Frequency']

effect = pd.DataFrame({
    'eta squared %': pd.Series({c: eta_squared(df, c) * 100 for c in CANDIDATES}),
    'omega squared %': pd.Series({c: omega_squared(df, c) * 100 for c in CANDIDATES}),
}).sort_values('eta squared %', ascending=False).round(3)
effect
"""),

md("""
The two columns agree on everything that matters. The same four variables lead on both measures
and every demographic stays at or below roughly 0.1% either way. With 96,639 members the bias was
never going to be the risk; running the correction simply removes the doubt.

There is a very sharp cliff in this list. Four variables clear 1%, and everything below them falls
away to fractions of a percent. Worth charting, because reading down a column makes 8.76 and 0.09
look like neighbours when one is roughly a hundred times the other.
"""),

code("""
# Plotting the effect sizes, colouring anything under 1% differently since it is not usable
league = effect['eta squared %']

plt.figure(figsize=(10, 6))
colours = [BLUE if v >= 1 else ORANGE for v in league.values][::-1]
bars = plt.barh(range(len(league)), league.values[::-1], color=colours)
plt.yticks(range(len(league)), league.index[::-1])

for bar, value in zip(bars, league.values[::-1]):
    plt.annotate(f'{value:.2f}%', (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                 xytext=(4, 0), textcoords='offset points', va='center', fontsize=10, color=INK_2)

plt.axvline(1, color='grey', ls='--')
plt.xlabel('Share of cost variation explained (%)')
plt.title('Only clinical burden, smoking, procedures and age move cost')
plt.xlim(0, 11)
plt.grid(axis='x', alpha=0.5)
save('v2_03_cost_drivers')
plt.show()
"""),

md("""
##### Interpretation

Four variables clear 1%: chronic condition count **8.8%**, smoking 2.6%, major procedure 2.2%, age
cohort 1.6%.

8.8% is the strongest thing here and still leaves 91% unexplained — normal for individual medical
cost, which is driven heavily by chance. Best available segmentation; no basis for predicting an
individual.

Below the line is almost everything a business would segment on. Employment **0.09%**, region
**0.008%** — a member's region leaves you no better placed to guess their cost than knowing nothing.
Income, sex, marital status, education, household size and plan type are effectively zero.

Region is not useless, it is not a *risk* factor: where members are, not what they cost. **UW / ACT**

One to carry: smoking is second strongest here, yet 4.5 shows smokers with near-identical disease
rates.
"""),

md("""
### Do the drivers stack, or are they the same thing measured four ways?

The league table above measures each variable on its own. That leaves an obvious objection: if the
four leaders are all capturing the same underlying thing, adding them together would explain no
more than the best of them alone, and the "demography is inert" verdict might be an artefact of
testing one variable at a time.

So I fit two models on log cost. The first uses only the four clinical and utilisation drivers;
the second adds every demographic field in the book at once.
"""),

code("""
# Fitting on log cost, because the raw amounts are lopsided enough to distort a straight-line fit
log_cost = np.log1p(df['Annual Medical Cost'])

DRIVER_COLS = ['Age', 'Chronic Conditions Count', 'Smoker Status',
               'Hospitalizations in Last 3 Years', 'Days Hospitalized in Last 3 Years']
DEMOGRAPHIC_COLS = ['Region', 'Urban / Rural', 'Sex', 'Education (Qualification)',
                    'Marital Status', 'Employment Status', 'Household Size', 'Income']


def linear_r2(columns):
    encoded = pd.get_dummies(df[columns], drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(encoded, log_cost, test_size=0.3, random_state=42)
    return r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))


driver_r2 = linear_r2(DRIVER_COLS)
full_r2 = linear_r2(DRIVER_COLS + DEMOGRAPHIC_COLS)

pd.Series({'Four clinical and utilisation drivers': round(driver_r2, 3),
           'The same four plus every demographic': round(full_r2, 3),
           'What the demographics add': round(full_r2 - driver_r2, 3)})
"""),

md("""
The four clinical and utilisation drivers jointly account for roughly 16% of the variation in log
cost. Adding every demographic field at once moves that by less than a point.

So the verdict holds when the variables are tested together, not just one at a time. And that
**16% is the honest ceiling** any cost model on this book should be judged against — I come back
to it in section 6 when a model scores 0.17 and somebody wants to know whether that is good.
"""),

md("""
---

## 4.3 Testing Every Relationship

Everything so far asked one question: does X move cost? That leaves a gap I am not comfortable
with, because it never tests whether variables relate to *each other*.

Different data types need different statistics, so I need three, all reporting on the same 0-to-1
scale: **Cramér's V** for two categories, the **correlation ratio** for a category against a
number, and ordinary correlation for two numbers.
"""),

code("""
# Cramer's V for two categorical fields, corrected for the bias small tables introduce
def cramers_v(a, b):
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2_corrected / max(1e-12, min(k_corrected - 1, r_corrected - 1)))
"""),

code("""
# The correlation ratio, for a categorical field against a numeric one
def correlation_ratio(cat, num):
    grouped = num.groupby(cat, observed=True)
    grand_mean = num.mean()
    between = (grouped.count() * (grouped.mean() - grand_mean) ** 2).sum()
    total = ((num - grand_mean) ** 2).sum()
    return np.sqrt(between / total) if total > 0 else np.nan
"""),

code("""
# Picking the right statistic automatically from the two field types.
# Testing the dtype by name rather than against `object`, because pandas 3 stores text as
# `str` and an `== object` check silently returns False for every text column.
def is_categorical(col):
    return str(df[col].dtype) in ('str', 'object', 'category') or df[col].nunique() <= 10


def association(a, b):
    cat_a, cat_b = is_categorical(a), is_categorical(b)
    if cat_a and cat_b:
        return cramers_v(df[a], df[b]), 'Cramers V'
    if cat_a:
        return correlation_ratio(df[a], df[b]), 'eta'
    if cat_b:
        return correlation_ratio(df[b], df[a]), 'eta'
    return abs(df[a].corr(df[b])), '|r|'
"""),

md("""
One thing has to be excluded first. Several columns are re-encodings of columns I already have —
`Age Cohort (10y)` is just `Age` in bands, `Income Band` is `Income` in quintiles. Testing `Age`
against `Age Cohort` would measure my own binning rather than the portfolio, and those tautologies
would dominate the top of the results.
"""),

code("""
# Dropping re-encodings of columns I already have, so I do not measure my own binning
DERIVED = ['Id', 'Plan Type (Full Name)', 'Age Groups', 'Age Cohort (10y)', 'Age_Life_Stage',
           'Glycemic Status', 'Diagnosis Status', 'Diabetes Clinical Quadrant',
           'Alcohol_Missing', 'Income Band', 'Top Decile', 'Loss Ratio',
           'Total Procedures', 'Monthly Premium', 'undiagnosed_htn_flag', 'early_retired_flag']

SCAN = [c for c in df.columns if c not in DERIVED]
len(SCAN)
"""),

code("""
# Testing every possible pair of the remaining fields
pairs = []
for a, b in itertools.combinations(SCAN, 2):
    value, stat = association(a, b)
    if value == value:                          # skipping any pair that comes back NaN
        pairs.append((a, b, stat, value))

scan = pd.DataFrame(pairs, columns=['A', 'B', 'stat', 'value']).sort_values('value', ascending=False)
len(scan)
"""),

md("""
1,326 pairs tested — every field against every other field.

Before looking at any individual pair I want the overall picture, so I am sorting them into bands.
All three measures run on the same 0-to-1 scale, where 0 means the two fields tell you nothing
about each other and 1 means one determines the other completely. The conventional reading is that
anything under 0.05 is noise, 0.10 to 0.15 is weak, 0.15 to 0.30 is moderate, and above 0.30 is
strong.
"""),

code("""
# Grouping the results into strength bands to see the overall picture
bands = pd.cut(scan.value, [-0.001, 0.05, 0.10, 0.15, 0.30, 1.001],
               labels=['under 0.05 (none)', '0.05-0.10 (trivial)', '0.10-0.15 (weak)',
                       '0.15-0.30 (modest)', 'over 0.30 (strong)'])
bands.value_counts().sort_index()
"""),

md("""
**988 of the 1,326 pairs — 75% of everything tested — come in under 0.05.** Three quarters of the
field pairs in this dataset tell you nothing whatsoever about each other. Only 47 pairs, about
3.5%, reach the 0.30 mark that counts as strong.

That is a sparse dataset. Most of what could relate to something else simply does not. Now let me
look at what those strong pairs actually are, because a strong number is not automatically an
interesting finding.
"""),

code("""
# The strongest relationships anywhere in the dataset
scan.head(12).reset_index(drop=True)
"""),

md("""
Almost every strong pair is two views of one thing rather than two things that move together:

- `Annual Medical Cost` ↔ `Annual Premium`, 0.965 — the formula from 4.1
- `Hospitalizations` ↔ `Days Hospitalized`, 0.89 — days follow admissions
- `Surgical Procedures` ↔ `Had Major Procedure`, 0.84 — the flag is derived from the count
- `Risk Score` ↔ `Is High Risk`, 0.83 — a thresholded version of the score
- `HbA1c Level` ↔ `Diabetes`, 0.79 — HbA1c is the diagnostic

The only pair describing two separate things is `Household Size` ↔ `Dependents` at 0.76.

Ranking by strength is therefore misleading, and the pairs need classifying by kind.
"""),

code("""
# Splitting the relationships by whether they are genuine or just circular
FINANCIAL = ['Annual Premium', 'Total Claims Paid', 'Average Claim Amount',
             'Annual Medical Cost', 'Risk Score', 'Is High Risk']

# Pairs where one field is computed from, or clinically defines, the other
DEFINITIONAL = [
    ({'Chronic Conditions Count'}, set(CONDITIONS)),
    ({'Hypertension'}, {'Systolic Blood Pressure', 'Diastolic Blood Pressure'}),
    ({'Diabetes'}, {'HbA1c Level'}),
    ({'Had Major Procedure'}, {'Surgical Procedures Count', 'Hospitalizations in Last 3 Years',
                               'Days Hospitalized in Last 3 Years'}),
    ({'Hospitalizations in Last 3 Years'}, {'Days Hospitalized in Last 3 Years'}),
    ({'Claims Count'}, {'Visits in Last Year'}),
]


def classify(a, b):
    if a in FINANCIAL and b in FINANCIAL:
        return 'financial (computed from each other)'
    for left, right in DEFINITIONAL:
        if (a in left and b in right) or (b in left and a in right):
            return 'definitional (a field and its own parts)'
    if a in FINANCIAL or b in FINANCIAL:
        return 'clinical or utilisation driving cost'
    return 'structural (a real relationship)'
"""),

code("""
# Classifying everything above the weak threshold
strong = scan[scan.value >= 0.10].copy()
strong['kind'] = [classify(a, b) for a, b in zip(strong.A, strong.B)]
strong.kind.value_counts()
"""),

md("""
**64 of 1,326 pairs — under 5% — describe a genuine connection between separately measured things.**
The rest is arithmetic, definition, or one financial field against another computed from it.

Those 64 fall into three groups: family composition, the body changing with age, and members with
more conditions using more care.

Nothing connects member demographics to product held. That absence closes several business
questions at once — 4.9.
"""),

md("""
### Putting specific business hypotheses to the test

The scan is thorough but abstract. What a reviewer will actually want to know is whether the
specific things they believe are true, so I wrote down the hypotheses somebody would reasonably
raise about a health book and measured each one.
"""),

code("""
# The hypotheses a business would actually ask about, written down to be tested rather than assumed
HYPOTHESES = [
    ('Smokers are more likely to have dependents', 'Smoker Status', 'Dependents'),
    ('Smokers have a higher BMI', 'Smoker Status', 'BMI'),
    ('Higher earners buy the richer tiers', 'Income Band', 'Network Tier'),
    ('Bigger households buy richer cover', 'Household Size', 'Network Tier'),
    ('The unemployed buy cheaper cover', 'Employment Status', 'Network Tier'),
    ('Rural members consume less care', 'Urban / Rural', 'Visits in Last Year'),
    ('Rural members cost less', 'Urban / Rural', 'Annual Medical Cost'),
    ('Better educated members are healthier', 'Education (Qualification)', 'Chronic Conditions Count'),
    ('Higher earners are healthier', 'Income Band', 'Chronic Conditions Count'),
    ('Married members claim more', 'Marital Status', 'Claims Count'),
    ('Diabetics choose lower deductibles', 'Diabetes', 'Deductible'),
    ('The chronically ill choose richer networks', 'Chronic Conditions Count', 'Network Tier'),
    ('Men and women differ in morbidity', 'Sex', 'Chronic Conditions Count'),
    ('Older members hold more conditions', 'Age', 'Chronic Conditions Count'),
    ('Household size tracks dependents', 'Household Size', 'Dependents'),
    ('Prior hospitalisation predicts cost', 'Hospitalizations in Last 3 Years', 'Annual Medical Cost'),
]
"""),

code("""
# Measuring each one against the same thresholds used above
verdicts = []
for label, a, b in HYPOTHESES:
    value, stat = association(a, b)
    verdict = ('structural' if value >= 0.30 else
               'supported' if value >= 0.15 else
               'weak' if value >= 0.10 else 'not supported')
    verdicts.append({'hypothesis': label, 'statistic': stat, 'value': round(value, 4), 'verdict': verdict})

pd.DataFrame(verdicts).sort_values('value', ascending=False).reset_index(drop=True)
"""),

md("""
##### What this table says

Almost every hypothesis a business would raise fails.

Two clear the bar. Household size tracks dependents at 0.76, which is one fact stated twice. Prior
hospitalisation predicts cost at 0.21 — moderate, actionable, with individual exceptions. That one
is built on in 4.10.

Smokers and dependents, the question that prompted the exercise: **0.000**.

The product hypotheses fail hardest. Higher earners buying richer tiers 0.003; bigger households
0.001; the unemployed buying cheaper cover 0.000 exactly. **Nothing about a member predicts which
product they hold.** **PROD / ACT**

The value of testing all 1,326 pairs is being able to say each of these came back empty, rather
than that it never occurred to me to look.
"""),

md("""
### Seeing the whole picture at once

A table of 1,326 rows is impossible to read as a whole, but a heatmap of the same numbers is
immediate. I am grouping the fields into families so the block structure is visible.
"""),

code("""
# Grouping fields into families so related blocks sit together on the heatmap
BLOCKS = {
    'Demographic': ['Age', 'Sex', 'Region', 'Urban / Rural', 'Income',
                    'Education (Qualification)', 'Marital Status', 'Employment Status',
                    'Household Size', 'Dependents'],
    'Lifestyle': ['BMI', 'Smoker Status', 'Alcohol Frequency'],
    'Clinical': CONDITIONS + ['Chronic Conditions Count', 'Systolic Blood Pressure',
                              'Diastolic Blood Pressure', 'LDL Cholesterol', 'HbA1c Level'],
    'Utilisation': ['Visits in Last Year', 'Hospitalizations in Last 3 Years',
                    'Days Hospitalized in Last 3 Years', 'Medication Count', 'Claims Count'],
    'Product': ['Plan Type', 'Network Tier', 'Deductible', 'Copay',
                'Policy Term (Years)', 'Policy Changes in Last 2 Years'],
    'Financial': ['Annual Medical Cost', 'Annual Premium', 'Total Claims Paid',
                  'Average Claim Amount', 'Risk Score'],
}

ORDER, boundaries, block_labels = [], [], []
for block, cols in BLOCKS.items():
    cols = [c for c in cols if c in SCAN]
    ORDER += cols
    boundaries.append(len(ORDER))
    block_labels.append((len(ORDER) - len(cols) / 2, block))

len(ORDER)
"""),

code("""
# Building the square matrix from the pair list
matrix = pd.DataFrame(np.nan, index=ORDER, columns=ORDER)
for a, b, stat, value in scan.itertuples(index=False):
    if a in matrix.index and b in matrix.index:
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

matrix.shape
"""),

code("""
# Drawing every tested relationship as one picture
plt.figure(figsize=(12, 11))
plt.imshow(matrix.values, cmap=SEQ, vmin=0, vmax=0.6)
plt.xticks(range(len(ORDER)), ORDER, rotation=90, fontsize=8)
plt.yticks(range(len(ORDER)), ORDER, fontsize=8)

# Marking where one family of fields ends and the next begins
for bound in boundaries[:-1]:
    plt.axhline(bound - 0.5, color=RED, lw=1.4)
    plt.axvline(bound - 0.5, color=RED, lw=1.4)
for pos, label in block_labels:
    plt.annotate(label, (pos - 0.5, -3), ha='center', fontsize=11, fontweight='bold', color=RED)

colourbar = plt.colorbar(fraction=0.03, pad=0.02)
colourbar.set_label('Strength of relationship')
plt.title('Every field against every other field, all 1,326 pairs')
save('v2_04_association_matrix')
plt.show()
"""),

md("""
##### Reading the heatmap

Each square is one pair; darker means stronger. Red lines separate the families.

The picture is **pale almost everywhere**. Darkness is confined to three patches: family
composition in the demographic block; the join between clinical and utilisation, where more
conditions means more visits, medication and procedures; and the financial block, dark because
those fields are computed from one another.

**The Product row and column are blank.** Not faint — blank. Nothing demographic, clinical or
financial relates to plan type, tier, deductible or copay.

One visible artefact that is not a finding: the mild `Age` ↔ `Employment Status` link is the
cleaning rule recoding over-70s to `Retired`. It is excluded from the count of 64.
"""),

md("""
### Does the alcohol imputation change anything?

Section 3 disclosed that all 29,092 `Non Alcoholic` members are imputed rather than observed. That
is a large enough assumption that I would rather test its consequences than carry it quietly, so:
do the recorded and imputed members differ, and does the field carry any signal either way?
"""),

code("""
# Comparing recorded against imputed members, then measuring the field's own signal both ways
recorded = df[~df['Alcohol_Missing']]
imputed = df[df['Alcohol_Missing']]


def alcohol_profile(frame):
    return {
        'Mean annual cost': frame['Annual Medical Cost'].mean(),
        'In the top cost decile (%)': frame['Top Decile'].mean() * 100,
        'Mean chronic conditions': frame['Chronic Conditions Count'].mean(),
        'Current smoker (%)': (frame['Smoker Status'] == 'Current').mean() * 100,
    }


pd.DataFrame({
    f'Recorded (n={len(recorded):,})': alcohol_profile(recorded),
    f'Imputed (n={len(imputed):,})': alcohol_profile(imputed),
}).round(2)
"""),

code("""
# And how much cost variation the field explains, with and without the imputed group
pd.Series({
    'eta squared %, imputed group included': round(eta_squared(df, 'Alcohol Frequency') * 100, 3),
    'eta squared %, recorded values only': round(eta_squared(recorded, 'Alcohol Frequency') * 100, 3),
})
"""),

md("""
Recorded and imputed members are statistically indistinguishable on cost, morbidity and smoking,
and the field explains close to nothing about cost whether the imputed group is included or not.

So no finding in this notebook depends on the alcohol field. That is a relief rather than a
vindication — the disclosure from section 3 still stands, and the field should not feed any
underwriting or care-management decision. **UW**
"""),

md("""
---

## 4.4 Who Is In This Book?

Section 4.2 showed none of these demographic cuts move cost, so I am not going to present them as
risk factors. But they still tell me who the members are, which matters for distribution and for
understanding the shape of the portfolio. I am reading them as **exposure, not as risk**.
"""),

code("""
# The age profile, and where the exposure actually concentrates
pd.concat([
    df['Age'].describe().round(1).rename('value').to_frame(),
    (df['Age Cohort (10y)'].value_counts(normalize=True).reindex(AGE_COHORT) * 100).round(1).rename('value').to_frame(),
])
"""),

md("""
A middle-aged book. Median age 48, running from 16 to 100. The 40s, 50s and 60s carry most of it —
about 63% of members between 40 and 69 — with a 10% youth cohort and just under 9% over 70. This is
a working-age book with a meaningful senior tail.
"""),

md("""
### Where members are, geographically
"""),

code("""
# How the book splits across regions
region_share = (df['Region'].value_counts(normalize=True) * 100).round(1)

plt.figure(figsize=(10, 5))
bars = plt.barh(range(len(region_share)), region_share.values[::-1], color=BLUE)
plt.yticks(range(len(region_share)), region_share.index[::-1])

for bar, value in zip(bars, region_share.values[::-1]):
    plt.annotate(f'{value}%', (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                 xytext=(4, 0), textcoords='offset points', va='center', fontsize=11, color=INK_2)

plt.xlabel('Share of the book (%)')
plt.title('South carries the most exposure, Central the least')
plt.xlim(0, 34)
plt.grid(axis='x', alpha=0.5)
save('v2_05_region_exposure')
plt.show()
"""),

md("""
South is the largest region at 28% of the book and Central the smallest at 12.1% — more than a
twofold difference in how much business each region carries. That is a real distribution fact and
it tells the sales side something about where the book is concentrated.

Before anyone reads a risk story into it, though, the regions need checking for what they cost.
"""),

code("""
# Checking whether the regions actually differ in cost or morbidity
df.groupby('Region').agg(
    members=('Id', 'size'),
    median_cost=('Annual Medical Cost', 'median'),
    mean_conditions=('Chronic Conditions Count', 'mean'),
).round(2)
"""),

md("""
The regions are essentially identical. Median cost spans about 46 across all five — on a median of
roughly 2,100, that is a 2% spread — and average chronic conditions differ in the second decimal
place.

So region tells you **where to sell, not what to charge**. A regional pricing or underwriting
strategy has nothing to stand on in this data, and I would not fund a regional risk initiative
from it. **NET / ACT**
"""),

md("""
### Employment, and the insured unemployed

A meaningful number of unemployed people hold cover here, which is interesting commercially — if
you can work out why they buy, you can sell to more of them.
"""),

code("""
# Comparing the unemployed cohort against the book average on everything I can measure
METRICS = {'Median age': ('Age', 'median'), 'Median income': ('Income', 'median'),
           'Mean conditions': ('Chronic Conditions Count', 'mean'),
           'Median cost': ('Annual Medical Cost', 'median'),
           'Mean risk score': ('Risk Score', 'mean'),
           'Mean visits': ('Visits in Last Year', 'mean')}

unemployed = df[df['Employment Status'] == 'Unemployed']

pd.DataFrame({
    'Members': df['Employment Status'].value_counts(),
    'Share (%)': (df['Employment Status'].value_counts(normalize=True) * 100).round(1),
})
"""),

code("""
# Indexing the unemployed cohort against the book, to see whether it differs on anything at all
pd.Series({label: round((unemployed[col].agg(how) / df[col].agg(how) - 1) * 100, 1)
           for label, (col, how) in METRICS.items()},
          name='Difference from book average (%)')
"""),

md("""
12,521 unemployed members hold cover, 13% of the book, and they sit within a few percent of the
book average on **every dimension measured** — age, income, morbidity, cost, utilisation.

A disappointing answer to a good question. Nothing recorded here explains why an unemployed person
buys cover; that needs acquisition data — channel, prior cover, purchase trigger — which is not in
this extract.

Better said plainly than dressed up as a 2% difference that is almost certainly noise.

(Retired members are 24.6%, inflated by the Rule 5 recode, so that figure is not read as a finding.)
"""),

md("""
### Can members afford this?

Section 4.1 established that premium is calculated from realised cost and ignores income
completely. That makes affordability a fair question: if the charge takes no account of what you
earn, who does it fall hardest on?
"""),

code("""
# Comparing what each income quintile earns against what they are charged
df.groupby('Income Band', observed=True).agg(
    members=('Id', 'size'),
    median_income=('Income', 'median'),
    median_premium=('Annual Premium', 'median'),
).round(0)
"""),

md("""
This is the whole finding in one table. Median income runs from 13,000 in the bottom quintile to
101,200 in the top — a **7.8 times** difference. Median premium runs from 467 to 466. It is
completely **flat**.

The consequence is arithmetic. Let me measure the burden per member rather than by dividing the
medians, because income is skewed within each quintile too.
"""),

code("""
# What share of income each member actually pays, taken as a median within each quintile
df['Burden %'] = df['Annual Premium'] / df['Income'] * 100
burden = df.groupby('Income Band', observed=True)['Burden %'].median()

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(burden)), burden.values, color=RED, width=0.6)
plt.xticks(range(len(burden)), burden.index, rotation=15)

for bar, value in zip(bars, burden.values):
    plt.annotate(f'{value:.2f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11, color=INK_2)

plt.ylabel('Premium as a share of income (%)')
plt.title('The poorest quintile pays 8.8 times the share of income the richest pays')
plt.grid(axis='y', alpha=0.5)
save('v2_06_affordability_burden')
plt.show()
"""),

md("""
##### What the burden chart shows

The poorest quintile pays **3.96% of income**; the richest **0.45%**. An 8.8-fold difference in
burden for essentially the same charge.

The mechanism is the 4.1 formula: premium keys off realised cost and tier, and income never enters
it.

Whether that is acceptable is policy, not analysis — some contribution schemes are deliberately
flat. It should be a deliberate choice, and product and compliance should know the structure has
this shape. **PROD / FIN**
"""),

md("""
---

## 4.5 What Does Demography Tell Us About Health?

This is the section I was most interested in going in. If the demographic attributes we collect
tell us anything about a member's health, underwriting can use them.

I am testing all ten recorded conditions against every demographic cut, rather than picking a
couple of conditions and a couple of cuts. I want the pattern, not an anecdote.
"""),

code("""
# How common each condition is across the whole book
(df[CONDITIONS].mean() * 100).sort_values(ascending=False).round(1)
"""),

md("""
Hypertension is by far the most common at 20.4%, then mental health at 13.1% and arthritis at
10.9%. Kidney and liver disease are rare at around 1.5% each.

Now the real question: does any demographic cut change these numbers? I am building a grid — every
condition against every demographic — and colouring by how far each group sits from the portfolio
baseline for that condition. If demography stratifies health, blocks of this grid should light up.
"""),

code("""
# The demographic cuts to test each condition against
CUTS = [('Age Cohort (10y)', AGE_COHORT),
        ('Urban / Rural', ['Urban', 'Suburban', 'Rural']),
        ('Region', ['North', 'South', 'East', 'West', 'Central']),
        ('Employment Status', EMPLOY),
        ('Sex', ['Female', 'Male']),
        ('Education (Qualification)', EDU),
        ('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed']),
        ('Smoker Status', ['Never', 'Former', 'Current']),
        ('Income Band', list(df['Income Band'].cat.categories))]

# Dropping the handful of retained minors, who are too few to read
adults = df[df['Age_Life_Stage'] != '0-15 (Minors)']
baseline = adults[CONDITIONS].mean() * 100

len(adults)
"""),

code("""
# One panel per demographic, ten conditions down the side, coloured by distance from baseline
fig, axes = plt.subplots(1, len(CUTS), figsize=(18, 6),
                         gridspec_kw={'width_ratios': [len(o) for _, o in CUTS], 'wspace': 0.1})
row_labels = [COND_SHORT.get(c, c) for c in CONDITIONS]

for ax, (cut, order) in zip(axes, CUTS):
    order = [o for o in order if o in adults[cut].astype(str).unique()]
    rates = (adults.groupby(cut, observed=True)[CONDITIONS].mean() * 100).reindex(order)
    counts = adults.groupby(cut, observed=True).size().reindex(order)
    relative = rates.sub(baseline, axis=1).div(baseline, axis=1) * 100
    relative = relative.mask(counts.lt(MIN_N), np.nan)          # suppressing thin groups

    im = ax.imshow(relative.T.values, cmap=DIV, aspect='auto',
                   norm=TwoSlopeNorm(vmin=-45, vcenter=0, vmax=45))
    ax.set_xticks(range(len(order)),
                  [str(o).split(' (')[0].replace(' lowest', '').replace(' highest', '') for o in order],
                  rotation=90, fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels if ax is axes[0] else [''] * len(row_labels), fontsize=10)
    ax.set_title(cut.split(' (')[0], fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

colourbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
colourbar.set_label("Difference from the condition's baseline (%)")
fig.suptitle('Ten conditions against every demographic cut', x=0.09, ha='left',
             fontsize=14, fontweight='bold', y=1.02)
save('v2_07_morbidity_grid')
plt.show()
"""),

md("""
##### Reading the grid

Rows are conditions, columns are groups, colour is distance from that condition's book-wide rate.
**Red above, blue below, pale means matching the book.**

Only the age panel shows a gradient — blue young, red old, on almost every condition. Settlement,
region, employment, sex, education, marital status and income band are all pale: each group within
a few percent of the book average on all ten conditions.

**Smoking is the exception worth stating.** Smokers show never-smoker rates across all ten,
including COPD, cardiovascular disease and cancer — yet 4.2 found smoking the second strongest cost
driver. Higher spending, no higher recorded disease. Both reported.

**Age is the only demographic worth using to stratify health risk. UW**
"""),

code("""
# Plotting all ten conditions across the age cohorts, highlighting the ones that actually climb
by_age = (adults.groupby('Age Cohort (10y)', observed=True)[CONDITIONS].mean() * 100).reindex(AGE_COHORT)

plt.figure(figsize=(10, 6))
x = range(len(AGE_COHORT))

for cond in CONDITIONS:
    values = by_age[cond].values
    steep = values[-1] / values[0] >= 1.25
    plt.plot(x, values, marker='o', markersize=5,
             color=BLUE if steep else '#c9c8c3', lw=2.5 if steep else 1.5, zorder=3 if steep else 1)
    if steep:
        plt.annotate(f'{COND_SHORT.get(cond, cond)}  {values[-1]:.0f}%', (len(x) - 1, values[-1]),
                     xytext=(6, 0), textcoords='offset points', va='center',
                     fontsize=10, color=BLUE, fontweight='bold')

plt.xticks(x, [c.split(' (')[0] for c in AGE_COHORT], rotation=20)
plt.xlabel('Age cohort')
plt.ylabel('Prevalence (%)')
plt.title('Conditions that climb with age (blue) against those that do not (grey)')
plt.xlim(-0.2, len(x) + 1.1)
plt.grid(alpha=0.5)
save('v2_08_age_morbidity_gradient')
plt.show()
"""),

md("""
Six of ten conditions rise with age. Hypertension 16.8% to 26.2%, arthritis 8.7% to 14.7%, mental
health 11.1% to 17.3%, diabetes 7.2% to 10.8%. The four grey lines — cancer, COPD, liver, kidney —
are recorded at much the same rate at both ends.

Size matters as much as direction. A 70-year-old is roughly **1.5 times** as likely to have
hypertension as someone in their twenties, not five or ten times: about one in six of the youngest
cohort against one in four of the oldest. Most older members still do not have it.

Which matches the 1.6% from 4.2. Age is usable and it is the only demographic stratifier available,
but it is a weak one.
"""),

md("""
### Do conditions cluster together?

In real populations, diseases co-occur. Diabetics develop kidney disease. People with COPD develop
cardiovascular problems. If that clustering exists here, then multi-morbid members are a
qualitatively different risk rather than just an arithmetic sum.
"""),

code("""
# How much more likely each condition is, given a member already has another one.
# A lift of 1.0 means the two conditions are completely independent of each other.
lift = np.zeros((len(CONDITIONS), len(CONDITIONS)))
for i, a in enumerate(CONDITIONS):
    for j, b in enumerate(CONDITIONS):
        lift[i, j] = np.nan if i == j else df.loc[df[a] == 1, b].mean() / df[b].mean()

pd.DataFrame(lift,
             index=[COND_SHORT.get(c, c) for c in CONDITIONS],
             columns=[COND_SHORT.get(c, c) for c in CONDITIONS]).round(2)
"""),

md("""
Each figure is a **lift** — how much more likely one condition is given another, against a member
picked at random. 1.00 means knowing the first tells you nothing about the second.

Every off-diagonal value sits at almost exactly 1.00. Diabetes does not raise kidney disease risk;
COPD does not raise cardiovascular risk. **The conditions occur independently.**

So multi-morbid members cost more because each condition adds its own cost, not because the
combination compounds. Size the multi-morbid group expecting cost to add rather than accelerate,
and do not price a fourth condition as more dangerous than the third. **ACT / CM**
"""),

md("""
### The silent risk: diabetics nobody has diagnosed

Section 2 built a field cross-referencing each member's HbA1c reading against whether they carry a
diabetes flag. Anyone at or above the 6.5% diagnostic threshold without a flag is diabetic on
paper but invisible administratively.
"""),

code("""
# How members split across the diabetes quadrants
undiagnosed = df['Diabetes Clinical Quadrant'].str.contains('UNDIAGNOSED')

pd.DataFrame({
    'Members': df['Diabetes Clinical Quadrant'].value_counts().sort_index(),
    'Share of book (%)': (df['Diabetes Clinical Quadrant'].value_counts(normalize=True).sort_index() * 100).round(2),
    'Share of spend (%)': (df.groupby('Diabetes Clinical Quadrant')['Annual Medical Cost'].sum()
                           / df['Annual Medical Cost'].sum() * 100).round(2),
})
"""),

md("""
**273 members — 0.28% of the book — sit at or above HbA1c 6.5% with no diabetes flag.** Diabetic on
their own file, invisible administratively.

Sized honestly: 0.28% of members, **0.3% of spend**. This does not move the portfolio's financial
position.

It is worth doing because it costs almost nothing. The lab value is already held, so identifying
them is a query, not a screening programme. A clinical-governance item with a list attached — not a
cost-saving initiative. **CM**
"""),

md("""
---

## 4.6 Where Does The Money Go?

Now to follow the money. This is the part with the most genuine signal, because cost concentration
and utilisation are properties of the claims process rather than of the demographic fields ruled
out earlier.
"""),

code("""
# The basic shape of medical cost
df['Annual Medical Cost'].describe().round(0)
"""),

md("""
Median **2,103**, mean **3,033**, maximum **65,725**.

The mean sits 44% above the median, which happens one way: a small group costs far more than
everyone else, lifting the total without moving the member in the middle. That maximum is 31 times
a typical member.

This changes a real decision. Budgeting 3,033 per member over-provides for most of the book and
still gets caught by the extremes. For a typical member the answer is 2,103. **FIN**
"""),

code("""
# Plotting cost on a log scale, since the tail makes a linear axis unreadable
plt.figure(figsize=(10, 6))
positive = df.loc[df['Annual Medical Cost'] > 0, 'Annual Medical Cost']
median_cost = df['Annual Medical Cost'].median()

plt.hist(positive, bins=np.logspace(1.5, 5, 70), color=BLUE)
plt.xscale('log')
plt.axvline(median_cost, color=ORANGE, lw=2)
plt.annotate(f'median {median_cost:,.0f}', (median_cost, plt.ylim()[1] * 0.9),
             xytext=(8, 0), textcoords='offset points', color=ORANGE, fontweight='bold')

plt.xlabel('Annual medical cost (log scale)')
plt.ylabel('Members')
plt.title('Medical cost is lognormal with a long right tail')
plt.grid(axis='y', alpha=0.5)
save('v2_09_cost_distribution')
plt.show()
"""),

md("""
##### Reading the distribution

The axis multiplies rather than adds, so 100-to-1,000 occupies the same width as 1,000-to-10,000.
Linear, the book would compress into the first inch.

One hump near 2,000, a thin thread running right, and the hump is roughly **symmetric** on this
axis — members are about as likely to cost half the typical amount as twice it.

The expensive members are therefore **not outliers to remove**; they continue the same pattern, so
they are ordinary members having an expensive year.

The shape also dictates model construction: fed raw, a model chases the largest numbers. Section 6
handles that, and the standard fix has a trap in it.
"""),

md("""
### How concentrated is the spending?

The tail matters more than its size suggests. Measuring exactly how much of the book's total spend
sits with how few members decides whether targeted intervention is worth doing at all.
"""),

code("""
# What share of total spend each cost decile carries
df['Cost Decile'] = pd.qcut(df['Annual Medical Cost'], 10, labels=False) + 1
(df.groupby('Cost Decile')['Annual Medical Cost'].sum() / df['Annual Medical Cost'].sum() * 100).round(1)
"""),

code("""
# The Lorenz curve, which shows the concentration as a whole
sorted_cost = np.sort(df['Annual Medical Cost'].values)
cumulative = np.cumsum(sorted_cost) / sorted_cost.sum()
share_of_members = np.arange(1, len(sorted_cost) + 1) / len(sorted_cost)
gini = 1 - 2 * np.trapezoid(cumulative, share_of_members)

plt.figure(figsize=(10, 6))
plt.plot(share_of_members * 100, cumulative * 100, color=BLUE, lw=2.5)
plt.plot([0, 100], [0, 100], color='grey', ls='--')
plt.fill_between(share_of_members * 100, cumulative * 100, share_of_members * 100, color=BLUE, alpha=0.12)

# Marking the points a business would actually care about
for pct in [50, 80, 90]:
    value = cumulative[int(len(sorted_cost) * pct / 100) - 1] * 100
    plt.scatter([pct], [value], color=ORANGE, zorder=5, s=50)
    plt.annotate(f'cheapest {pct}% of members\\ncarry {value:.0f}% of spend', (pct, value),
                 xytext=(-8, 12), textcoords='offset points', ha='right', fontsize=10, color=INK_2)

plt.annotate(f'Gini {gini:.2f}', (58, 25), fontsize=14, color=BLUE, fontweight='bold')
plt.xlabel('Cumulative share of members, cheapest first (%)')
plt.ylabel('Cumulative share of medical spend (%)')
plt.title('A tenth of the members carry a third of the spend')
plt.grid(alpha=0.5)
save('v2_10_cost_concentration')
plt.show()
"""),

md("""
##### Reading the Lorenz curve

Members accumulate left to right, cheapest first. The dashed line is perfect evenness; the gap to
the blue curve is the finding.

The **top decile carries 33.5%** of everything paid out and the top two deciles **50.5%**. The
cheaper half of the membership accounts for about 12%.

**Gini 0.45** summarises that gap on a 0-to-1 scale — firmly in heavily-concentrated territory.

This is the most actionable structural fact here: a programme reaching 10% of members addresses a
third of the cost, which is what makes targeted intervention fundable. Evenly spread spend would
leave no efficient target and only across-the-board price rises. **CM / FIN**

Whether that decile is identifiable *in advance* is Model B.
"""),

md("""
### Is cost driven by claiming often, or claiming big?

Total cost is frequency times severity. Which of the two drives it changes what you would do about
it — high frequency suggests managing routine utilisation, high severity suggests catastrophic
cover and reinsurance.
"""),

code("""
# How often members claim, and whether frequent claimers also claim larger amounts
claimants = df[df['Claims Count'] > 0]
severity = claimants.groupby('Claims Count').agg(members=('Id', 'size'),
                                                 median_claim=('Average Claim Amount', 'median'))
severity = severity[severity.members >= MIN_N]

plt.figure(figsize=(10, 6))
plt.plot(severity.index, severity.median_claim, marker='o', color=BLUE, lw=2.5)
plt.xlabel('Number of claims in the year')
plt.ylabel('Median size of each claim')
plt.title('Members who claim often claim small')
plt.grid(alpha=0.5)
save('v2_11_frequency_vs_severity')
plt.show()
"""),

md("""
##### Frequency or severity?

The line falls, so the two move in **opposite** directions. A member with one claim had a much
larger typical claim than one with eight.

That matches practice: one claim usually means one significant event, eight means routine
outpatient visits.

So cost here is driven by frequency rather than catastrophic single events, and the 38% who never
claim subsidise the rest. For care management that points at managing utilisation patterns rather
than chasing large claims. **CLM / CM**
"""),

md("""
### Does using more care actually cost more?

This sounds obvious, but I want to confirm it and see how steep each relationship is, because these
are the fields that would go into any predictive model — and unlike demographics, most of them are
observable before the year starts.
"""),

code("""
# The utilisation measures to test, each with a sensible cap so thin tails do not distort the line
UTILISATION = [('Visits in Last Year', 'GP and outpatient visits', 12),
               ('Hospitalizations in Last 3 Years', 'Hospitalisations (3 years)', 3),
               ('Days Hospitalized in Last 3 Years', 'Days hospitalised (3 years)', 14),
               ('Medication Count', 'Medications', 10),
               ('Total Procedures', 'Total procedures', 14),
               ('Chronic Conditions Count', 'Chronic conditions', 6)]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for ax, (col, label, cap) in zip(axes.ravel(), UTILISATION):
    grouped = df.groupby(col).agg(cost=('Annual Medical Cost', 'median'), n=('Id', 'size'))
    grouped = grouped[(grouped.index <= cap) & (grouped.n >= MIN_N)]

    ax.plot(grouped.index, grouped.cost, marker='o', color=BLUE, lw=2.5)
    ax.fill_between(grouped.index, 0, grouped.cost, color=BLUE, alpha=0.1)
    ax.set_title(f'{label}   x{grouped.cost.iloc[-1] / grouped.cost.iloc[0]:.1f}', fontsize=12)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Median cost', fontsize=10)
    ax.set_ylim(0, grouped.cost.max() * 1.2)
    ax.grid(alpha=0.5)

fig.suptitle('Every utilisation measure rises steadily with cost', x=0.02, ha='left',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('v2_12_utilisation_dose_response')
plt.show()
"""),

md("""
##### Reading the six panels

Each panel groups members by one utilisation measure and plots the median cost of each group.

All six climb, and climb **without reversals** — every step up in usage is a step up in cost. The
consistency matters more than the steepness: a relationship holding at every level can be relied on
for an individual member, not just on average. Days hospitalised is steepest.

Kept together because the point is the consistency across all six; separate charts would lose it.

These are the variables worth modelling. They carry real signal, unlike the demographics — but only
the prior-history ones are knowable before the year starts. That distinction is not visible here
and it matters a great deal in section 6.
"""),

md("""
---

## 4.7 Which Condition Is Actually High Risk?

If care management has to pick a condition to focus on, which one? The instinctive answer is
whichever costs the most per patient. I want to test that instinct, because I think it produces
the wrong priority list.
"""),

code("""
# What each condition costs a member, and what it costs the book overall
condition_rows = []
for cond in CONDITIONS:
    has, without = df[df[cond] == 1], df[df[cond] == 0]
    condition_rows.append({
        'condition': COND_SHORT.get(cond, cond),
        'members': len(has),
        'prevalence %': round(len(has) / len(df) * 100, 1),
        'extra cost per member': round(has['Annual Medical Cost'].median() - without['Annual Medical Cost'].median()),
        '% of all spend': round(has['Annual Medical Cost'].sum() / df['Annual Medical Cost'].sum() * 100, 1),
    })

conditions_cost = pd.DataFrame(condition_rows)
conditions_cost.sort_values('extra cost per member', ascending=False).reset_index(drop=True)
"""),

code("""
# The same ten conditions, ranked by their share of total portfolio spend instead
conditions_cost.sort_values('% of all spend', ascending=False).reset_index(drop=True)
"""),

md("""
The ordering has almost completely inverted. That divergence is the finding, so it goes on one
chart.
"""),

code("""
# Severity against burden, with each bubble sized by how common the condition is
plt.figure(figsize=(10, 6.5))
plt.scatter(conditions_cost['extra cost per member'], conditions_cost['% of all spend'],
            s=conditions_cost['prevalence %'] * 60, color=BLUE, alpha=0.55,
            edgecolor='white', linewidth=2, zorder=3)

# Nudging a couple of labels apart where the bubbles sit close together
NUDGE = {'Mental health': (-78, -4), 'Arthritis': (66, -6), 'Diabetes': (62, -4),
         'Asthma': (-66, -6), 'Cardiovascular': (78, -14), 'Cancer': (-34, 6), 'Kidney': (-16, 0)}

for _, r in conditions_cost.iterrows():
    dx, dy = NUDGE.get(r.condition, (0, 0))
    plt.annotate(f"{r.condition}\\n{r['prevalence %']}% of members",
                 (r['extra cost per member'], r['% of all spend']),
                 xytext=(dx, np.sqrt(r['prevalence %'] * 60) / 2 + 9 + dy),
                 textcoords='offset points', ha='center', fontsize=10)

plt.xlabel('Extra cost per affected member')
plt.ylabel("Share of the book's total spend (%)")
plt.title('The condition worst for a member is not the one worst for the book')
plt.xlim(650, 1180)
plt.ylim(-2, 31)
plt.grid(alpha=0.5)
save('v2_13_severity_vs_burden')
plt.show()
"""),

md("""
##### Severity against burden

Horizontal is **severity**, extra cost per affected member. Vertical is **burden**, share of the
book's spend. Bubble size is prevalence.

**Severity barely varies** — 761 to 1,039 across all ten conditions, a spread of **1.37 times**.
**Burden varies 13.5 times**, from kidney disease at 2.0% of spend to hypertension at 26.8%.

Liver disease is worst to have at 1,039 extra, affects 1.5% of members, accounts for 2.1% of spend.
Hypertension costs slightly less per member and 20.4% of the book has it, so it takes 26.8% of
everything paid out.

**Prioritise by prevalence × cost, not severity.** A liver programme cannot move the needle,
whatever its clinical merit. **CM / ACT**
"""),

md("""
---

## 4.8 Which Age Group Actually Needs Insurance?

Average cost is a poor way to answer this. It rises gently with age and implies the young barely
need cover, which misunderstands what insurance is for. Insurance absorbs the **bad year** — so the
honest measure is the gap between a normal year and a bad one.
"""),

code("""
# The typical year, the bad year and the catastrophic year, by age band
df['Age Band'] = pd.cut(df['Age'], [15, 29, 39, 49, 59, 69, 120],
                        labels=['16-29', '30-39', '40-49', '50-59', '60-69', '70+'])

exposure = df.groupby('Age Band', observed=True).agg(
    members=('Id', 'size'),
    income=('Income', 'median'),
    typical_year=('Annual Medical Cost', 'median'),
    bad_year=('Annual Medical Cost', lambda x: x.quantile(0.90)),
    catastrophic_year=('Annual Medical Cost', lambda x: x.quantile(0.99)))

# Stating the denominator explicitly: the catastrophic shortfall against the band's MEDIAN income
exposure['catastrophe gap'] = exposure.catastrophic_year - exposure.typical_year
exposure['catastrophe as % of median income'] = (exposure['catastrophe gap'] / exposure.income * 100).round(1)
exposure.round(0)
"""),

code("""
# Charting the spread between a normal year and a catastrophic one, by age
plt.figure(figsize=(10, 6))
x = np.arange(len(exposure))

plt.fill_between(x, exposure.typical_year, exposure.catastrophic_year, color=RED, alpha=0.12,
                 label='Catastrophic year (99th percentile)')
plt.fill_between(x, exposure.typical_year, exposure.bad_year, color=ORANGE, alpha=0.22,
                 label='Bad year (90th percentile)')
plt.plot(x, exposure.typical_year, marker='o', color=BLUE, lw=2.5, label='Typical year (median)')
plt.plot(x, exposure.catastrophic_year, marker='o', color=RED, lw=2)
plt.plot(x, exposure.bad_year, marker='o', color=ORANGE, lw=2)

for i in [0, len(x) - 1]:
    plt.annotate(f'{exposure.catastrophic_year.iloc[i]:,.0f}', (i, exposure.catastrophic_year.iloc[i]),
                 xytext=(0, 8), textcoords='offset points', ha='center',
                 fontsize=11, color=RED, fontweight='bold')

plt.xticks(x, exposure.index)
plt.xlabel('Age band')
plt.ylabel('Annual medical cost')
plt.title('What a year can cost, by age')
plt.legend()
plt.grid(alpha=0.5)
save('v2_14_bad_year_by_age')
plt.show()
"""),

md("""
##### Interpretation

**Median** is a normal year, **p90** a bad one, **p99** catastrophic. The shaded gaps are what
insurance absorbs — a member can budget the normal year, not the gap above it.

The median rises only **1.61 times** across the whole age range, which is the weak argument most
people reach for first.

A 70-year-old faces nearly 20,000 against a typical 2,725 — **47% of the band's median income**.

**The unexpected result is at the young end.** A 16-29 member has a median cost of 1,688 and a
catastrophic year of 12,215 — **29% of annual median income**. Young, mostly healthy, still unable
to absorb a bad year.

Marketing to the young on expected cost loses, because their expected cost really is low.
Marketing on **volatility** is honest and far stronger. **PROD**
"""),

md("""
---

## 4.9 Does the Product Ladder Do Anything?

Members can buy Bronze, Silver, Gold or Platinum. Section 4.3 already told me that nothing about a
member predicts which one they hold, which is odd in itself. Now the other side of it: what does a
member get for climbing the ladder?
"""),

code("""
# What each tier charges, against everything the member might be buying with it
ladder = df.groupby('Network Tier').agg(
    members=('Id', 'size'),
    median_premium=('Annual Premium', 'median'),
    provider_quality=('Provider Quality Rating', 'median'),
    tenure_years=('Policy Term (Years)', 'median'),
    policy_changes=('Policy Changes in Last 2 Years', 'mean'),
    visits=('Visits in Last Year', 'mean')).reindex(TIER)

# What share of their own medical spend each tier's members actually hand over
ladder['member pays %'] = (df.groupby('Network Tier')['Annual Premium'].sum()
                           / df.groupby('Network Tier')['Annual Medical Cost'].sum() * 100).reindex(TIER)
ladder['share of book %'] = (df['Network Tier'].value_counts(normalize=True) * 100).reindex(TIER)
ladder.round(2)
"""),

code("""
# Indexing everything to Bronze, so it is obvious which lines actually move
indexed = pd.DataFrame({
    'What the member pays': ladder['member pays %'],
    'Provider quality received': ladder.provider_quality,
    'Tenure (loyalty)': ladder.tenure_years,
    'Care actually used': ladder.visits})
indexed = indexed / indexed.iloc[0] * 100

plt.figure(figsize=(10, 6))
for i, col in enumerate(indexed.columns):
    emphasis = i == 0
    plt.plot(range(4), indexed[col], marker='o',
             color=RED if emphasis else CAT[i], lw=3 if emphasis else 2,
             zorder=5 if emphasis else 3, label=col)
    plt.annotate(f'{indexed[col].iloc[-1]:.0f}', (3, indexed[col].iloc[-1]),
                 xytext=(9, 0), textcoords='offset points', va='center',
                 fontsize=11, fontweight='bold', color=RED if emphasis else CAT[i])

plt.axhline(100, color='grey', ls='--')
plt.xticks(range(4), TIER)
plt.xlabel('Network tier')
plt.ylabel('Indexed to Bronze = 100')
plt.title('Climbing the ladder costs more and delivers nothing measurable')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
save('v2_15_tier_ladder')
plt.show()
"""),

md("""
##### Reading the ladder

Indexed to Bronze at 100. **One line moves.** Platinum members hand over **24.4%** of their own
medical spend against **16.6%** on Bronze — 1.47 times the rate.

Provider quality is identical at 3.6 across all four tiers. Care consumed is flat. Median tenure on
Platinum is a year *shorter* than the other three.

On every observable dimension the ladder is a **cost-share dial wearing the costume of a benefit
ladder**.

The question for product: what does Platinum deliver that Bronze does not? If it is something
unrecorded — faster pre-authorisation, wider hospital list — we should start recording it. **Either
attach a measurable benefit to the upper tiers or collapse them. PROD**
"""),

md("""
### Does where a member lives change what they use?

The last product-adjacent question is whether geography changes consumption, which would justify a
network strategy.
"""),

code("""
# The full utilisation basket across settlement types, as a deviation from the book average
UTIL_COLS = ['Visits in Last Year', 'Hospitalizations in Last 3 Years',
             'Days Hospitalized in Last 3 Years', 'Medication Count',
             'Imaging Procedures Count', 'Surgical Procedures Count',
             'Lab Procedures Count', 'Consultation Procedures Count', 'Claims Count']

settlement = df.groupby('Urban / Rural')[UTIL_COLS].mean().reindex(['Urban', 'Suburban', 'Rural'])
deviation = (settlement / df[UTIL_COLS].mean() - 1) * 100

plt.figure(figsize=(10, 6))
plt.axhspan(-10, 10, color=BLUE, alpha=0.08)
for i, place in enumerate(deviation.index):
    plt.plot(range(len(UTIL_COLS)), deviation.loc[place], marker='o', color=CAT[i], label=place)

plt.axhline(0, color=INK_2)
labels = [c.replace(' in Last Year', '').replace(' in Last 3 Years', ' (3y)')
           .replace(' Procedures Count', '').replace(' Count', '') for c in UTIL_COLS]
plt.xticks(range(len(UTIL_COLS)), labels, rotation=40, ha='right')
plt.ylabel('Deviation from the book average (%)')
plt.ylim(-14, 14)
plt.title('Urban, suburban and rural members consume almost identical care')
plt.legend()
plt.grid(axis='y', alpha=0.5)
save('v2_16_consumption_by_settlement')
plt.show()
"""),

md("""
Every measure sits inside the ±10% band. Those with enough volume to be reliable are flat to well
under 1%: visits differ by **0.86%** across settlement types, claims by **0.68%**, medication by
**0.14%**.

The two straying furthest — days hospitalised and surgical procedures, around 8% and 6% — both
average **under 0.4 per member**. On a base that small, a few hundredths of an event looks dramatic
as a percentage and is trivial in absolute terms.

**Where a member lives does not change the care they consume.** With the flat regional costs from
4.4, network and regional strategy cannot be built from this data; that needs provider-level
identifiers and finer geography than five regions. **NET**
"""),

md("""
---

## 4.10 Who Should We Care-Manage First?

Two things from earlier combine here. Cost is concentrated in a small group, and burden follows
prevalence rather than severity. Together they should tell me which cohorts are worth enrolling.

For each candidate group I want three numbers: how many members it reaches, how much of the book's
spend sits inside it, and how likely those members are to end up in the expensive decile.
"""),

code("""
# The cohorts a care-management team could realistically target
threshold = df['Annual Medical Cost'].quantile(0.90)

COHORTS = {
    'Hypertensive': df['Hypertension'] == 1,
    '2+ chronic conditions': df['Chronic Conditions Count'] >= 2,
    'Had a major procedure': df['Had Major Procedure'] == 1,
    '4+ visits last year': df['Visits in Last Year'] >= 4,
    'Mental health condition': df['Mental Health Condition'] == 1,
    'Current smoker': df['Smoker Status'] == 'Current',
    'Hospitalised in last 3 years': df['Hospitalizations in Last 3 Years'] > 0,
    'Aged 70+': df['Age'] >= 70,
    'Undiagnosed diabetic': undiagnosed,
}

targeting = pd.DataFrame([{
    'cohort': name,
    'members': int(mask.sum()),
    '% of book': round(mask.mean() * 100, 1),
    '% of spend': round(df.loc[mask, 'Annual Medical Cost'].sum() / df['Annual Medical Cost'].sum() * 100, 1),
    '% in top decile': round((df.loc[mask, 'Annual Medical Cost'] > threshold).mean() * 100, 1),
} for name, mask in COHORTS.items()]).sort_values('% of spend', ascending=False).reset_index(drop=True)
targeting
"""),

code("""
# Reach against return, so the trade-off is visible
plt.figure(figsize=(10, 6.5))
plt.scatter(targeting['% of book'], targeting['% of spend'],
            s=targeting['% in top decile'] * 24, color=BLUE, alpha=0.55,
            edgecolor='white', linewidth=2, zorder=3)

limit = max(targeting['% of book'].max(), targeting['% of spend'].max()) * 1.15
plt.plot([0, limit], [0, limit], color='grey', ls='--')
plt.annotate('no concentration\\n(spend share = member share)', (limit * 0.7, limit * 0.7),
             rotation=33, fontsize=10, color=INK_2, ha='center')

for _, r in targeting.iterrows():
    plt.annotate(r.cohort, (r['% of book'], r['% of spend']),
                 xytext=(0, np.sqrt(r['% in top decile'] * 24) / 2 + 8),
                 textcoords='offset points', ha='center', fontsize=10)

plt.xlabel('Reach — share of the membership in this cohort (%)')
plt.ylabel('Return — share of total medical spend it accounts for (%)')
plt.title('Cohorts above the line concentrate more spend than their headcount')
plt.xlim(0, limit)
plt.ylim(0, limit)
plt.grid(alpha=0.5)
save('v2_17_care_management_targeting')
plt.show()
"""),

md("""
##### Reach against return

Horizontal is **reach**, vertical is **return**, the dashed line is parity, bubble size is the
chance of landing in the top cost decile.

Every cohort sits above the line, which only says sick members cost more. What decides a budget is
how far above, and where each sits on reach.

**Hypertension is the volume play** — 20.4% of members, 26.8% of spend, the largest pool available.

**Prior hospitalisation is the precision play** — 9% of members, but **25% land in the most
expensive tenth**, two and a half times the base rate. Enrol a hundred and about twenty-five turn
out genuinely expensive, against ten at random.

**Build on prior hospitalisation and multi-morbidity; run hypertension separately for reach. CM**
"""),

md("""
---

## 4.11 Which Package Suits Which Member?

Bringing 4.8 and 4.10 together. Only age and clinical burden legitimately segment this book, so
those are the two axes. And 4.8 established that the right measure is downside exposure, not
average cost.
"""),

code("""
# Segments built from the only two variables that legitimately stratify this book
segment_frame = df[df['Age_Life_Stage'] != '0-15 (Minors)'].copy()
segment_frame['Burden'] = np.where(segment_frame['Chronic Conditions Count'] >= 2, '2+ conditions',
                          np.where(segment_frame['Chronic Conditions Count'] == 1, '1 condition', 'No condition'))

segments = segment_frame.groupby(['Age_Life_Stage', 'Burden'], observed=True).agg(
    members=('Id', 'size'),
    typical_year=('Annual Medical Cost', 'median'),
    bad_year=('Annual Medical Cost', lambda x: x.quantile(0.90)))
segments = segments[segments.members >= MIN_N]
segments['exposure gap'] = segments.bad_year - segments.typical_year
segments.round(0)
"""),

code("""
# The gap between a normal year and a bad one, for every segment
seg = segments.reset_index()
seg['stage_order'] = seg['Age_Life_Stage'].map({s: i for i, s in enumerate(LIFE_STAGE)})
seg = seg.dropna(subset=['stage_order']).sort_values(['stage_order', 'Burden'])
seg['label'] = seg['Age_Life_Stage'].str.split(' \\(').str[0] + '  ·  ' + seg['Burden']

plt.figure(figsize=(10, 7))
y = np.arange(len(seg))
plt.hlines(y, seg.typical_year, seg.bad_year, color=GRID_C, lw=7)
plt.scatter(seg.typical_year, y, color=BLUE, s=90, zorder=5, label='Typical year')
plt.scatter(seg.bad_year, y, color=RED, s=90, zorder=5, label='Bad year (90th percentile)')

for i, (lo, hi) in enumerate(zip(seg.typical_year, seg.bad_year)):
    plt.annotate(f'{lo:,.0f}', (lo, i), xytext=(-8, 0), textcoords='offset points',
                 ha='right', va='center', fontsize=10, color=BLUE, fontweight='bold')
    plt.annotate(f'{hi:,.0f}', (hi, i), xytext=(8, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=10, color=RED, fontweight='bold')

plt.yticks(y, seg.label)
plt.xlabel('Annual medical cost')
plt.title('The gap between a normal year and a bad one is what cover has to absorb')
plt.xlim(0, seg.bad_year.max() * 1.2)
plt.legend(loc='lower right')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.5)
save('v2_18_package_fit_by_segment')
plt.show()
"""),

md("""
##### What I would recommend from this

The bars widen down the chart, and the width is what the member cannot absorb.

A healthy under-30 has a typical year of 1,282 and a bad year of 3,627 — a gap near 2,300. A 65-plus
member with two or more conditions runs 3,828 to 10,894, a gap over 7,000.

- **Under 50, no condition:** gap small enough to part-self-insure. **High-deductible Bronze** is
  rational, since the formula charges only 1% of the deductible — raising it is nearly free.
- **Any age, one condition:** around 6,500 in a bad year. **Silver**.
- **50-plus, two or more conditions:** the widest gap in the book. **Low-deductible Gold or
  Platinum** earns its cost here.
- **Any prior hospitalisation:** three times more likely to land in the expensive decile whatever
  the plan. A care-management case, not a product one.

One caveat throughout: because premium is a fixed share of realised cost, **no tier is objectively
better value than another** here. This matches cover to exposure, which is real. It says nothing
about pricing efficiency, which this data cannot. **PROD / ACT**
"""),

md("""
---

## 4.12 Can We Trust Our Own Fields?

Before any of this feeds a model, I want to check the fields that look most useful, because the
ones correlating most strongly with cost are exactly the ones most likely to be circular.

`Risk Score` is the obvious candidate. If it is calculated from member attributes, it is a
legitimate underwriting input. If it is calculated from cost or claims, it is leakage.
"""),

code("""
# Testing whether Risk Score can be rebuilt from clinical attributes alone
OBSERVABLE = ['Age', 'Chronic Conditions Count', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
              'BMI', 'HbA1c Level', 'LDL Cholesterol', 'Visits in Last Year',
              'Hospitalizations in Last 3 Years', 'Medication Count'] + CONDITIONS

X_tr, X_te, y_tr, y_te = train_test_split(df[OBSERVABLE], df['Risk Score'], test_size=0.3, random_state=42)
round(r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te)), 3)
"""),

md("""
**0.86.** Age, chronic count, blood pressure and the other clinical fields rebuild 86% of `Risk
Score`.

So it is built from things observed before cost is incurred — a legitimate underwriting input, not
a field containing the answer.

It is also, on that evidence, nearly redundant: 86% of it already sits in fields a model would
have.
"""),

code("""
# How much each group of fields adds to a cost model, which is how leakage shows itself
def cost_model_r2(features):
    X_tr, X_te, y_tr, y_te = train_test_split(df[features], log_cost, test_size=0.3, random_state=42)
    return round(r2_score(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te)), 3)


pd.Series({
    'Observable clinical fields only': cost_model_r2(OBSERVABLE),
    'Plus Risk Score': cost_model_r2(OBSERVABLE + ['Risk Score']),
    'Plus the claims fields': cost_model_r2(OBSERVABLE + ['Risk Score', 'Total Claims Paid', 'Average Claim Amount']),
})
"""),

md("""
##### What the three models say

Clinical fields alone: **0.164** — matching the 16% ceiling from 4.2 by a different route.

Plus `Risk Score`: **0.190**. Small, as expected from a field 86% rebuildable from inputs already
present.

Plus the claims fields: **0.435**. No new information about any member arrived between those rows —
those fields are computed from the cost being predicted.

**Leakage is dangerous because it does not look like a bug, it looks like a better model.**

Rule for section 6: **exclude** premium, monthly premium, total claims paid, average claim amount
and loss ratio. **`Risk Score` may stay for reporting**, but models get its components instead.
"""),

md("""
---

## 4.13 What I Found

### Supported

1. **Pricing is arithmetic, not underwriting.** `200 + 0.01 × Deductible + tier_rate × Cost`
   reproduces 99.997% of premiums to the cent. Invalidates any pricing-adequacy analysis. **ACT / FIN**
2. **Cost is concentrated and clinically driven.** Top decile carries 33.5% of spend; membership
   goes with hospitalisation, multi-morbidity, procedures and age. **CM**
3. **Burden follows prevalence, not severity.** 1.37× spread in cost per member against 13.5× in
   share of spend. **CM / ACT**
4. **Insurance value is volatility.** Median cost rises 1.61× with age, but a 16-29 member faces a
   catastrophic year worth 29% of income. **PROD**
5. **Only age and clinical burden segment this book.** 64 of 1,326 pairs are genuine, all clinical,
   utilisation or family composition. **UW**

### Not supported by this data

| Question | Why not |
|:---|:---|
| Is the book priced adequately? | Premium is computed from realised cost |
| Which regions carry more risk? | Regional cost, morbidity and utilisation are flat |
| Why do unemployed people buy cover? | That cohort matches the book average on every field |
| Does anything drive plan choice? | Demographic-to-product association is about 0.01 |
| Do lifestyle factors drive morbidity? | Smokers and never-smokers have near-identical rates |
| Does multi-morbidity compound? | Conditions co-occur at a lift of 1.00 |
| Anything paediatric | Cleaning left 35 members under 18 |

### Recommended

1. **Underwrite on age, chronic count and prior utilisation** — the only variables that move cost.
2. **Stop segmenting on region, employment, education, household size, settlement.** Each under 1%.
3. **Build care management on prior hospitalisation and multi-morbidity**, hypertension separately.
4. **Escalate the 273 undiagnosed diabetics** — the lab value is already on file.
5. **Ask product what the upper tiers deliver.** 1.47× the contribution rate, nothing measurable.
6. **Flag the regressive contribution structure.** Poorest quintile pays 8.8× the income share.
7. **Drop premium and claims fields before modelling** — they lift R² from 0.16 to 0.44 by leaking.

### Worth confirming with whoever produced the extract

The ten conditions occur independently (lift 1.00 between every pair). Smoking drives cost but
shows no relationship to any recorded condition. Utilisation is identical across every geography.
Cancer history is as common at 25 as at 75. Premium is closed-form on realised cost.

Each is what the data says. None changes the method; several would change what the findings mean.
"""),

]
