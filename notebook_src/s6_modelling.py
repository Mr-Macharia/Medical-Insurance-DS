"""Section 6 of notebook.ipynb — modelling.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
---

# 6. Predicting Cost and Risk

Two facts from section 4 make modelling worth doing: cost is concentrated in a tenth of members,
and membership of that tenth goes with things observable in advance.

**Model A** predicts what a member will cost — the basis for a technical premium.
**Model B** ranks members by their chance of landing in the expensive tenth — a care-management
enrolment list.

**No premium model.** 4.1 recovered `Annual Premium` exactly, so training on it would reproduce
arithmetic already written down. The useful form of "what should we charge" is "what will this
member cost". Demonstrated in 6.5 rather than asserted.
"""),

md("""
## The Analysis Frame Is Not the Modelling Frame

`df` **describes what already happened**, so it may look at realised cost and premium. A model
cannot. Three different problems come out:

**1. Leaked fields.** Premium is computed from cost; claims paid is the insurer's share of it. A
model given these inverts arithmetic, then fails on a member whose cost is unknown.

**2. Same-period fields.** `Claims Count`, `Visits in Last Year` and the procedure counts are
recorded *in the same year as the cost being predicted* — not leakage in the strict sense, but
unknowable at renewal. Priced in 6.1.

**3. Re-encodings.** `Age Cohort` is `Age` binned; `Cost Decile` is the target in disguise.

What remains is `df_ml`: **renewal-legal**, with engineered features on top. Section 4 decided which
were worth engineering; no column crossed over untested.
"""),

md("""
## Setting Up

Everything the modelling needs that is not already bound. The estimators, the pipeline machinery
and the calibration tools arrive here; the metrics and split functions came in with section 4, so
they are not imported again. The engineered
feature block lives in `scripts/model_features.py` rather than in this notebook, and the reason is
specific: joblib pickles a notebook-defined function by reference to `__main__`, so the saved model
would fail to load in a fresh process. Importing it from a module keeps the round-trip test at the
end honest.
"""),

code("""
# Adding what the modelling needs, plus the engineered feature block from scripts/
import json
import sys
from datetime import datetime, timezone

import joblib
import shap
import sklearn
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(ROOT / 'scripts'))
from model_features import ENGINEERED_NAMES, add_engineered
"""),

code("""
# Settings in one place, so they are easy to change and easy to find
RANDOM_STATE = 42
TEST_SIZE = 0.25
TAIL_Q = 0.90          # "high cost" means above this point in the cost distribution
CAPACITY_PCT = 0.10    # what share of the book care management can actually enrol
N_JOBS = -1
TARGET = 'Annual Medical Cost'
"""),

md("""
## Writing Down the Feature Contract

Three lists, each with the reason attached to every entry. I would rather spell out why a field is
barred than leave a future reader guessing, because the failure this prevents is invisible: a model
trained with leakage does not crash or warn, it just reports a wonderful score.
"""),

code("""
# Fields calculated FROM the cost I am predicting, with the reason each one is barred
LEAKY = {
    'Annual Premium': '= 200 + 0.01 x Deductible + tier_rate x Annual Medical Cost',
    'Monthly Premium': '= Annual Premium / 12, so equally derived',
    'Total Claims Paid': "the insurer's share OF the target, bounded by it",
    'Average Claim Amount': 'Total Claims Paid divided by Claims Count',
    'Loss Ratio': '= cost / premium, built during the exploratory work',
    'Risk Score': '86% rebuildable from clinical inputs, so it adds little over them',
    'Is High Risk': 'a thresholded version of Risk Score',
}
pd.Series(LEAKY, name='why it is barred').to_frame()
"""),

code("""
# Fields recorded in the SAME year as the cost, so unknowable at the moment of prediction
SAME_PERIOD = {
    'Claims Count': 'the frequency leg of total_claims_paid = count x average',
    'Visits in Last Year': 'this year\\'s visits, counted after the year happened',
    'Imaging Procedures Count': 'this year\\'s procedures',
    'Surgical Procedures Count': 'this year\\'s procedures',
    'Physiotherapy Procedures Count': 'this year\\'s procedures',
    'Consultation Procedures Count': 'this year\\'s procedures',
    'Lab Procedures Count': 'this year\\'s procedures',
    'Had Major Procedure': 'derived from this year\\'s surgical count',
}
pd.Series(SAME_PERIOD, name='why it cannot be used at renewal').to_frame()
"""),

code("""
# Columns that exist only to serve the analysis above: identifiers and re-encodings
ANALYSIS_ONLY = ['Id', 'Plan Type (Full Name)', 'Age Groups', 'Age Cohort (10y)', 'Age_Life_Stage',
                 'Age Band', 'Glycemic Status', 'Diagnosis Status', 'Diabetes Clinical Quadrant',
                 'Alcohol_Missing', 'Income Band', 'Cost Decile', 'Top Decile', 'Total Procedures',
                 'Burden %']

RENEWAL_COLS = [c for c in df.columns
                if c not in set(LEAKY) | set(SAME_PERIOD) | set(ANALYSIS_ONLY) | {TARGET}]
len(RENEWAL_COLS)
"""),

md("""
## Building `df_ml`

The modelling frame, and the two targets that come off it. `y_cost` is what a member cost;
`y_high` is whether they landed in the most expensive tenth.
"""),

code("""
# The modelling frame: raw renewal-legal columns only, engineering happens inside the pipeline
df_ml = df[RENEWAL_COLS].copy()
y_cost = df[TARGET].astype('float64')
y_high = (y_cost > y_cost.quantile(TAIL_Q)).astype(int)

df_ml.shape
"""),

md("""
The engineered block runs **inside** the pipeline, not on the frame. Engineered here, the saved
model would only accept data already engineered the same way, and the next user would have to
reconstruct that from memory. Inside, it takes raw member records — proven by the round-trip test
in 6.6.

Two properties are deliberate. Every feature is **row-wise**, computed from one member's own
fields with nothing pooled across members, so it cannot carry test-set information into training
whichever side of the split it runs. And every input is renewal-legal, which the guard below
re-checks rather than trusts.
"""),

code("""
# Working out the column types after engineering, since the pipeline needs to know them
engineered_preview = add_engineered(df_ml)
NUM_COLS = engineered_preview.select_dtypes(include=[np.number]).columns.tolist()
CAT_COLS = [c for c in engineered_preview.columns if c not in NUM_COLS]

pd.Series({'raw renewal-legal columns': len(RENEWAL_COLS),
           'engineered columns added': len(ENGINEERED_NAMES),
           'numeric features': len(NUM_COLS),
           'categorical features': len(CAT_COLS)})
"""),

code("""
# The guard. If a barred field ever creeps back in, this stops the notebook rather than
# letting it report a flattering score.
barred = set(LEAKY) | set(SAME_PERIOD) | {TARGET}
leaked = sorted(set(engineered_preview.columns) & barred)
assert not leaked, f'LEAKAGE: {leaked} must not reach a model'
assert 'Id' not in engineered_preview.columns, 'the member identifier is not a feature'

f'{len(NUM_COLS) + len(CAT_COLS)} features cleared the contract'
"""),

md("""
I wrote that as an `assert` rather than a comment because of what it prevents. If somebody later
adds a field back without thinking, I would rather the notebook stopped than quietly produced a
number nobody could reproduce.
"""),

md("""
### What leakage actually does to a score

Rather than assert that leakage matters, let me measure it. Same model, same data, same settings.
The only difference is whether the barred fields are included.
"""),

code("""
# Fitting the same model with and without the barred fields, purely to compare
log_target_cost = np.log1p(y_cost)


def quick_r2(frame):
    encoded = pd.get_dummies(frame, drop_first=True)
    X_tr, X_te, y_tr, y_te = train_test_split(encoded, log_target_cost,
                                              test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = HistGradientBoostingRegressor(max_iter=300, early_stopping=True,
                                          random_state=RANDOM_STATE).fit(X_tr, y_tr)
    return r2_score(y_te, model.predict(X_te))


with_leakage = df[[c for c in df.columns if c not in ANALYSIS_ONLY + [TARGET]]]

pd.Series({'With the premium and claims fields': round(quick_r2(with_leakage), 4),
           'Without them, renewal-legal only': round(quick_r2(df_ml), 4)})
"""),

md("""
##### Interpretation

**With the barred fields: 0.999.** Individual medical cost is driven substantially by chance, so no
set of member attributes can account for 99.9% of it. That number should trigger suspicion, not
congratulation.

**Renewal-legal only: 0.21.** The honest figure, slightly above the 16% ceiling from 4.2 — expected,
since this has the engineered features and the full legal set behind it where the ceiling test used
five raw drivers.

No information about any member arrived between those runs. The first model was handed
`Annual Premium` and inverted the formula.

**Leakage does not look like a mistake, it looks like success.** A model reporting 0.999 passes a
review that 0.21 would struggle with, and only one of them works on a member whose cost is unknown.
"""),

md("""
## Preparing the Data

The categorical fields hold text and the models need numbers. The standard conversion turns each
category into its own yes/no column: `Region` becomes five columns with a 1 in the one that
applies.

That conversion goes inside the pipeline too, alongside the engineering, for the same reason.
"""),

code("""
# The preparation steps that travel inside every model
def make_preprocessor(columns=None, scale=False):
    nums = [c for c in NUM_COLS if columns is None or c in columns or c in ENGINEERED_NAMES]
    cats = [c for c in CAT_COLS if columns is None or c in columns]

    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:                       # straight-line models need this, tree models do not
        numeric_steps.append(('scale', StandardScaler()))

    return ColumnTransformer([
        ('num', Pipeline(numeric_steps), nums),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     sparse_output=False, min_frequency=0.01)),
        ]), cats),
    ], remainder='drop', verbose_feature_names_out=False)
"""),

code("""
# The full front end: engineer the features, then prepare them
def make_pipeline(estimator, scale=False, step_name='model'):
    return Pipeline([
        ('engineer', FunctionTransformer(add_engineered, validate=False)),
        ('prep', make_preprocessor(scale=scale)),
        (step_name, estimator),
    ])
"""),

md("""
Now I split the members into two groups. The model learns from the first and is judged on the
second, which it never sees during training.

This is the only honest way to know whether a model has learnt something general or has simply
memorised the members it was shown. A model can always describe data it has already seen; the
question is whether it says anything useful about a member it has not.
"""),

code("""
# Splitting into a group to learn from and a group to be judged on
X_train, X_test, y_train, y_test = train_test_split(
    df_ml, y_cost, test_size=TEST_SIZE, random_state=RANDOM_STATE)

y_high_train = y_high.loc[X_train.index]
y_high_test = y_high.loc[X_test.index]

pd.Series({'training members': len(X_train), 'held-out members': len(X_test)})
"""),

md("""
### A trap in the standard fix

Cost is lopsided, so a model trained on raw amounts chases the largest numbers — being wrong by
20,000 once outweighs being wrong by 300 many times. The usual fix is to fit on a log scale and
convert back.

**The conversion is the trap.** Fitting on logs and inverting returns the *median* of a group, not
the mean, and for skewed data the median is always lower. Every prediction comes back
systematically light.

Fitted both ways below. `mean_ratio` — total predicted over total actual — detects it. **It should
be 1.00.**
"""),

code("""
# Shared settings for every boosted model below, so the comparisons are like for like
GBM_KW = dict(max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=True, n_iter_no_change=25,
              validation_fraction=0.1, random_state=RANDOM_STATE)


def log_target(pipeline):
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)
"""),

code("""
# Four models: a baseline that always guesses the average, two on the multiplied scale,
# and one that works directly in money using a loss built for lopsided amounts
candidates = {
    'baseline (always the mean)': make_pipeline(DummyRegressor(strategy='mean')),
    'ridge on log cost': log_target(make_pipeline(Ridge(alpha=1.0), scale=True)),
    'boosted trees on log cost': log_target(make_pipeline(HistGradientBoostingRegressor(**GBM_KW))),
    'boosted trees, gamma loss': make_pipeline(HistGradientBoostingRegressor(loss='gamma', **GBM_KW)),
}
list(candidates)
"""),

code("""
# Fitting each one and measuring it on members it has never seen
median_cost = float(y_cost.median())
fitted, rows = {}, []

for name, estimator in candidates.items():
    fitted[name] = estimator.fit(X_train, y_train)
    pred = np.clip(fitted[name].predict(X_test), 1, None)
    rows.append({
        'model': name,
        'R2': r2_score(y_test, pred),
        'MAE': mean_absolute_error(y_test, pred),
        'MAE as % of median cost': mean_absolute_error(y_test, pred) / median_cost * 100,
        'mean_ratio': pred.mean() / y_test.mean(),
    })

results_A = pd.DataFrame(rows).set_index('model')
results_A.round(3)
"""),

md("""
##### What the four models say

**MAE** is the average miss in money, **R2** is variance accounted for, **mean_ratio** is the bias
check — and it decides this.

The **two log-scale models sit at 0.758 and 0.754**: the trap, exactly. They under-collect on the
whole book by a quarter, in one direction for everybody, so averaging across members cannot rescue
it.

**The gamma model sits at 0.989** with the highest R2 at 0.167, fitting directly in money with no
conversion step.

Its **worse MAE** (1,768 against 1,662) is a genuine trade: the log models are tuned to get the
middle member right, the gamma model the total. For pricing the total wins.
"""),

code("""
# The standard correction for the conversion bias, worked out from the training residuals
log_model = fitted['boosted trees on log cost']
residuals = np.log1p(y_train) - log_model.regressor_.predict(X_train)
smearing = float(np.mean(np.exp(residuals)))

corrected = np.clip(log_model.predict(X_test), 1, None) * smearing

pd.Series({'correction factor': round(smearing, 3),
           'R2 after correction': round(r2_score(y_test, corrected), 4),
           'mean_ratio after correction': round(corrected.mean() / y_test.mean(), 3)})
"""),

md("""
Applying the standard correction to the biased model lifts its mean_ratio from 0.75 to about 0.99
and its R-squared to essentially the same place the gamma model reached on its own.

Two different repairs arriving at the same answer confirms the diagnosis was right. **I am keeping
the gamma model**, because it avoids the problem rather than correcting for it afterwards.
"""),

code("""
# The chosen model, from here on
BEST_A = 'boosted trees, gamma loss'
cost_model = fitted[BEST_A]
pred_A = np.clip(cost_model.predict(X_test), 1, None)

round(float(r2_score(y_test, pred_A)), 4)
"""),

md("""
### What did the renewal-legal contract cost me?

Section 4.6 warned that not every utilisation field is knowable in advance, and the feature
contract acts on that. But a decision like that should be priced rather than assumed virtuous, so
here is the same model fitted with the same-period fields put back in.
"""),

code("""
# The same gamma model, given the same-period fields the contract bars
same_period_cols = RENEWAL_COLS + list(SAME_PERIOD)
X_train_sp, X_test_sp = df.loc[X_train.index, same_period_cols], df.loc[X_test.index, same_period_cols]

sp_preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'),
     [c for c in same_period_cols if c in df.select_dtypes(include=[np.number]).columns]),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False))]),
     [c for c in same_period_cols if c not in df.select_dtypes(include=[np.number]).columns]),
], remainder='drop')

sp_model = Pipeline([('prep', sp_preprocessor),
                     ('model', HistGradientBoostingRegressor(loss='gamma', **GBM_KW))]).fit(X_train_sp, y_train)
pred_sp = np.clip(sp_model.predict(X_test_sp), 1, None)

pd.DataFrame({
    'R2': [r2_score(y_test, pred_A), r2_score(y_test, pred_sp)],
    'MAE': [mean_absolute_error(y_test, pred_A), mean_absolute_error(y_test, pred_sp)],
    'mean_ratio': [pred_A.mean() / y_test.mean(), pred_sp.mean() / y_test.mean()],
}, index=['renewal-legal (the contract)', 'plus same-period fields']).round(3)
"""),

md("""
The same-period model scores better — it knows how often the member saw a doctor during the very
year being predicted — but by **0.002 of R2**, which is nothing.

Better than expected. I was ready to pay a real price for a contract that can run at renewal, and
there is barely anything to trade: the prior-history fields carry almost everything the
same-period fields do.

**The renewal-legal model carries forward everywhere below.**
"""),

md("""
### Is the model good enough to actually use?

An R-squared of 0.167 sounds poor, and for predicting an individual member it is — the typical
prediction misses by about 84% of what a typical member costs in a year. But that is not the only
question worth asking of a pricing model.

Insurance does not price individuals in isolation; it prices pools. So the fair test is whether the
model gets **groups** right, even though it cannot get individuals right. I sort the held-out
members into ten groups by what the model predicted, then compare each group's predicted average
against its actual average.
"""),

code("""
# Sorting members into ten groups by predicted cost, then checking each group's average
decile = pd.qcut(pred_A, 10, labels=False, duplicates='drop')
by_decile = (pd.DataFrame({'predicted': pred_A, 'actual': y_test.values, 'decile': decile})
               .groupby('decile').agg(members=('actual', 'size'),
                                      predicted=('predicted', 'mean'),
                                      actual=('actual', 'mean')))
by_decile['error %'] = (by_decile.predicted / by_decile.actual - 1) * 100
by_decile.round(1)
"""),

code("""
# The same check as a chart
plt.figure(figsize=(10, 6))
plt.plot(by_decile.index, by_decile.predicted, marker='o', lw=2.5, color=BLUE, label='Predicted average')
plt.plot(by_decile.index, by_decile.actual, marker='o', lw=2.5, color=ORANGE, label='Actual average')
plt.xlabel('Group, ordered by what the model predicted')
plt.ylabel('Average annual medical cost')
plt.title('The model gets groups right, even though it cannot get individuals right')
plt.legend()
plt.grid(alpha=0.5)
save('v3_01_cost_model_by_group')
plt.show()
"""),

md("""
##### Reading the decile check

Every group's predicted average lands within a few percent of actual, with errors in both
directions rather than one. The ordering is right — the group predicted cheapest is cheapest, the
group predicted dearest is dearest.

Better than the R2 suggested, and this is the number for an actuary. Model A cannot say what an
individual will cost; it can sort members into groups and price each group. That is what a
technical premium needs. **ACT**
"""),

md("""
### Validation that is not a single split

One random split is one draw. Before I would let anybody use this I want to know how much the score
moves when the split moves, whether the model is memorising the training members, and whether more
data would still help.
"""),

code("""
# Five-fold cross-validation, plus the gap between training and held-out performance
cv_r2 = cross_val_score(make_pipeline(HistGradientBoostingRegressor(loss='gamma', **GBM_KW)),
                        df_ml, y_cost, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                        scoring='r2', n_jobs=N_JOBS)
train_pred = np.clip(cost_model.predict(X_train), 1, None)

pd.Series({
    'CV R2, mean': round(float(cv_r2.mean()), 3),
    'CV R2, standard deviation': round(float(cv_r2.std()), 3),
    'Training R2': round(r2_score(y_train, train_pred), 3),
    'Held-out R2': round(r2_score(y_test, pred_A), 3),
    'Overfit gap (train minus test)': round(r2_score(y_train, train_pred) - r2_score(y_test, pred_A), 3),
})
"""),

code("""
# Does more data still help, or is the model already at its ceiling?
sizes, train_scores, val_scores = learning_curve(
    make_pipeline(HistGradientBoostingRegressor(loss='gamma', **GBM_KW)),
    df_ml, y_cost, cv=3, scoring='r2', train_sizes=np.linspace(0.25, 1.0, 4), n_jobs=1)

pd.DataFrame({'training members': sizes.astype(int),
              'training R2': train_scores.mean(axis=1).round(3),
              'cross-validated R2': val_scores.mean(axis=1).round(3)})
"""),

md("""
The cross-validated score sits close to the single-split figure with a small standard deviation, so
the result is not an artefact of one lucky split. The training-to-test gap is modest, so the model
is not memorising.

The learning curve is the more interesting one. The cross-validated score flattens well before the
full dataset, which says **more members of the same kind would not help.** The ceiling is
information, not sample size, and section 4.2 already told me where that ceiling is. If a future
version scores much higher, the first thing to check is whether a barred field has crept back in.
"""),

md("""
### The bad year, modelled directly

Section 4.8 argued that the thing insurance actually absorbs is the bad year, not the average one.
A model predicting the mean answers the wrong question for that story, so I fit two quantile models
alongside: one for the typical year and one for the bad year.
"""),

code("""
# Quantile models: the middle member of a group, and the member nine tenths of the way along
quantile_models = {}
for q in (0.5, 0.9):
    quantile_models[q] = make_pipeline(
        HistGradientBoostingRegressor(loss='quantile', quantile=q, **GBM_KW)).fit(X_train, y_train)

p50_pred = quantile_models[0.5].predict(X_test)
p90_pred = quantile_models[0.9].predict(X_test)

pd.Series({
    'p50 coverage (should be near 50%)': round((y_test <= p50_pred).mean() * 100, 1),
    'p90 coverage (should be near 90%)': round((y_test <= p90_pred).mean() * 100, 1),
    'median p50 prediction': round(float(np.median(p50_pred))),
    'median p90 prediction': round(float(np.median(p90_pred))),
})
"""),

md("""
Coverage is the check that matters. A p90 model is calibrated when roughly 90% of members really do
come in below its prediction, and that is what happens here.

Practically, this gives product a defensible answer to "what should this member expect in a bad
year", which is the question 4.8 said they should actually be selling on. **PROD / ACT**
"""),

md("""
### Constraints an actuary can defend, and what the model leans on

A boosted tree will happily learn that cost *falls* between two adjacent ages if the sample happens
to say so. Nobody can defend that in a pricing committee, and it is not a real effect. So I
constrain four fields to be non-decreasing: age, chronic burden and the two prior-inpatient
measures. No constraint is placed on anything else.
"""),

code("""
# Monotonic constraints: cost must not fall as age, chronic burden or prior admissions rise
prepared = make_preprocessor().fit(add_engineered(X_train))
feature_names = list(prepared.get_feature_names_out())

MONOTONIC_UP = {'Age', 'Chronic Conditions Count', 'Hospitalizations in Last 3 Years',
                'Days Hospitalized in Last 3 Years'}
constraints = [1 if name in MONOTONIC_UP else 0 for name in feature_names]

mono_model = make_pipeline(
    HistGradientBoostingRegressor(loss='gamma', monotonic_cst=constraints, **GBM_KW)).fit(X_train, y_train)
pred_mono = np.clip(mono_model.predict(X_test), 1, None)

pd.Series({'R2 unconstrained': round(r2_score(y_test, pred_A), 4),
           'R2 with constraints': round(r2_score(y_test, pred_mono), 4)})
"""),

md("""
The constraints cost almost nothing, which is the answer I hoped for: the shape they enforce is the
shape the data already had, and imposing it buys defensibility for free.

Now, what is the model actually using? SHAP attributes each prediction back to the features that
produced it, so the answer is measured rather than inferred.
"""),

code("""
# SHAP on a sample, since the full test set is unnecessary for a summary
shap_sample = X_test.sample(2000, random_state=RANDOM_STATE)
shap_values = shap.TreeExplainer(mono_model.named_steps['model']).shap_values(
    prepared.transform(add_engineered(shap_sample)))

plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, features=None, feature_names=feature_names,
                  plot_type='bar', max_display=15, show=False)
plt.title('What the cost model runs on')
plt.tight_layout()
save('v3_04_shap_summary')
plt.show()
"""),

md("""
Chronic burden and the engineered comorbidity load dominate, followed by prior inpatient history
and age. That is the same ranking section 4.2 found by a completely different route, which is
reassuring — the model has not discovered anything the exploratory work missed, and it has not
latched onto something spurious either.

Demographic fields sit near the bottom. That is a fairness observation as much as a technical one,
and I test it properly in 6.3.
"""),

md("""
### The actuarial decomposition: frequency times severity

One more view of the same target, because it is the one an insurance audience reads most naturally.
Pure premium is claim frequency multiplied by claim severity, and fitting them separately produces
**rate relativities** — a multiplier per risk factor — rather than a black box.

`Claims Count` is the target of the frequency model here, not a feature of it.
"""),

code("""
# Frequency: a Poisson model on claim count. Severity: a gamma model on cost per claim.
FS_FEATURES = ['Age', 'Chronic Conditions Count', 'Hospitalizations in Last 3 Years',
               'Days Hospitalized in Last 3 Years', 'BMI', 'Systolic Blood Pressure', 'HbA1c Level']


def design_matrix(index):
    frame = df.loc[index, FS_FEATURES + ['Smoker Status']]
    encoded = pd.get_dummies(frame, columns=['Smoker Status'], drop_first=True).astype(float)
    return sm.add_constant(encoded)


X_glm_train = design_matrix(X_train.index)
claims_train = df.loc[X_train.index, 'Claims Count']

freq_glm = sm.GLM(claims_train, X_glm_train, family=sm.families.Poisson()).fit()

claimed = claims_train > 0
sev_glm = sm.GLM(y_train[claimed] / claims_train[claimed], X_glm_train[claimed],
                 family=sm.families.Gamma(sm.families.links.Log())).fit()

pd.DataFrame({
    'frequency relativity': np.exp(freq_glm.params).drop('const'),
    'severity relativity': np.exp(sev_glm.params).drop('const'),
}).assign(**{'pure premium relativity': lambda d: d.iloc[:, 0] * d.iloc[:, 1]}).round(3)
"""),

md("""
Each number is a multiplier on the base rate for a one-unit increase in that factor. A frequency
relativity of 1.02 on chronic conditions means each additional condition raises expected claim
count by 2%.

This is the format a pricing committee can actually argue with, which the boosted model is not.
Both are fitted on the same renewal-legal contract, so they are describing the same book.
"""),

code("""
# Does the decomposition agree with the boosted model at group level?
X_glm_test = design_matrix(X_test.index)
pure_premium = freq_glm.predict(X_glm_test) * sev_glm.predict(X_glm_test)

fs_check = pd.DataFrame({'actual': y_test.values, 'predicted': pure_premium.values})
fs_check['decile'] = pd.qcut(fs_check['predicted'], 10, labels=False)
fs_group = fs_check.groupby('decile').agg(actual_mean=('actual', 'mean'),
                                          predicted_mean=('predicted', 'mean')).round(0)
fs_group['error %'] = ((fs_group['predicted_mean'] / fs_group['actual_mean'] - 1) * 100).round(1)
fs_group
"""),

md("""
The decomposition orders groups in the same direction as the boosted model, which is the agreement
worth having. But its group errors are **much** larger: 20-28% under on the cheap deciles, **52%
over** on the most expensive, against a few percent for the gamma model.

That is the price of a form simple enough to hand over as a rate table, and it is too high to price
on. Keep the relativities for the argument they support — which factors move frequency, which move
severity — and use Model A for numbers that must be right. **ACT**
"""),

md("""
---

## 6.2 Model B — Who Becomes Expensive?

This is the model with a real job to do. Care management can only enrol a limited number of
members, so they need those members ranked by risk **before** the costs happen.

I define "high cost" as landing above the 90th percentile — the most expensive tenth of the book.
That is a deliberate choice rather than an obvious one: it matches the group section 4.6 found
carries a third of all spend, and a tenth of the membership is a plausible size for a programme.
"""),

code("""
# What counts as a high-cost member
pd.Series({'threshold cost': round(float(y_cost.quantile(TAIL_Q))),
           'share of the held-out group above it': round(float(y_high_test.mean()), 3)})
"""),

md("""
Anyone costing more than about 6,270 counts as high cost, and they are 10% of the held-out group —
which is what defining it by the 90th percentile guarantees.

That 10% matters for how I judge the model. **A model that simply predicted "not high cost" for
everybody would be right 90% of the time**, so accuracy is a useless measure here. I need measures
that account for how rare the thing being predicted is.
"""),

code("""
# The classifier. class_weight="balanced" tells it to treat the rare high-cost members
# as being as important as the common ordinary ones.
risk_model = make_pipeline(
    HistGradientBoostingClassifier(class_weight='balanced', **GBM_KW)).fit(X_train, y_high_train)

proba = risk_model.predict_proba(X_test)[:, 1]

pd.Series({
    'roc_auc': roc_auc_score(y_high_test, proba),
    'pr_auc': average_precision_score(y_high_test, proba),
    'brier': brier_score_loss(y_high_test, proba),
    'prevalence': float(y_high_test.mean()),
}).round(4)
"""),

md("""
##### Interpretation

**ROC-AUC 0.757.** Given one high-cost and one ordinary member, the chance the model scores the
high-cost one higher. It gets that right about three times in four.

**PR-AUC 0.265** against a 0.10 base rate — the honest measure when the target is rare, and roughly
2.6 times random. This is the figure to quote.

**Brier 0.195** measures whether the probabilities themselves are trustworthy, lower being better.
It needs a comparison to mean anything.
"""),

code("""
# What a model that just predicts the base rate for everyone would score
baseline_brier = brier_score_loss(y_high_test, np.full(len(y_high_test), y_high_train.mean()))
round(baseline_brier, 4)
"""),

md("""
**The baseline scores 0.091, beating the model's 0.195.** Alarming, and it does not mean the model
is useless.

`class_weight='balanced'` is the cause. Treating rare high-cost members as equally important stops
the model ignoring them, at the cost of **inflated probabilities** — it says 70% where the true
chance is nearer 30%.

Inflation does not harm the **ranking**, which is what a targeting list needs. It does mean the
numbers cannot be read as probabilities, and anyone told "70% chance" would be misled.

Calibration fixes it, fitted on training members with internal cross-validation — fitting on the
held-out set and scoring it would flatter the result.
"""),

code("""
# Calibrating on the training members only, with internal cross-validation
calibrated_model = CalibratedClassifierCV(clone(risk_model), method='isotonic', cv=3, n_jobs=N_JOBS)
calibrated_model.fit(X_train, y_high_train)
proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

pd.Series({
    'Brier, raw model': brier_score_loss(y_high_test, proba),
    'Brier, calibrated': brier_score_loss(y_high_test, proba_cal),
    'Brier, always predicting the base rate': baseline_brier,
}).round(4)
"""),

code("""
# How well the probabilities match reality, before and after calibration
plt.figure(figsize=(10, 6))
for label, p, colour in [('Raw model', proba, BLUE), ('After calibration', proba_cal, ORANGE)]:
    observed, predicted = calibration_curve(y_high_test, p, n_bins=10, strategy='quantile')
    plt.plot(predicted, observed, marker='o', lw=2.5, color=colour, label=label)

plt.plot([0, 1], [0, 1], ls='--', color='grey', label='Perfect (prediction = reality)')
plt.xlabel('Probability the model predicted')
plt.ylabel('Share who actually turned out to be high cost')
plt.title('The raw model overstates risk; calibration corrects it')
plt.legend()
plt.grid(alpha=0.5)
save('v3_02_risk_model_calibration')
plt.show()
"""),

md("""
##### Reading the calibration curve

The dashed line is honesty: say 30%, and 30% of those members are high cost.

The raw model runs well **below** it — where it says 60%, about 30% turn out expensive. Systematic
overstatement, as the Brier score indicated.

Calibrated, the line tracks the diagonal and Brier falls to **0.083**, now beating the 0.091
baseline. Beating the baseline is the test that matters: the probabilities carry real information
about individuals rather than repeating the book-wide rate.

Two usable versions. For **ranking**, either works — calibration does not reorder. For **quoting a
probability** or sizing an enrolled group, the calibrated one is required.
"""),

md("""
### Turning the model into a decision

A model that outputs probabilities is not yet a decision. Somebody has to choose a cut-off, and the
usual default of 0.5 is the wrong way to make that choice — it is a statistical convention, not a
business one.

The real constraint is **capacity**. A care-management team can handle a certain number of members,
so the question is: if we enrol the top N by predicted risk, what do we get for it?
"""),

code("""
# What enrolling the top N members by predicted risk actually buys
actual_cost_test = y_cost.loc[X_test.index]
capacity_rows = []

for pct in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    n_enrol = int(len(y_high_test) * pct)
    idx = np.argsort(-proba_cal)[:n_enrol]
    caught = y_high_test.values[idx].sum()
    capacity_rows.append({
        'capacity %': pct * 100,
        'members enrolled': n_enrol,
        'precision': caught / max(n_enrol, 1),
        'recall': caught / max(int(y_high_test.sum()), 1),
        'lift': (caught / max(n_enrol, 1)) / y_high_test.mean(),
        '% of spend reached': actual_cost_test.values[idx].sum() / actual_cost_test.sum() * 100,
    })

pd.DataFrame(capacity_rows).round(3)
"""),

md("""
##### What the capacity table buys

**Precision** — enrolled members who really are high cost: 0.31 at 10% capacity, so 31 per hundred
enrolled. **Recall** — high-cost members caught: also about 0.31, so two thirds are missed.

The two always pull against each other, and choosing is a business call about the cost of missing
someone against enrolling someone unnecessarily.

**Lift** is roughly 3 — three times as many expensive members as picking at random.

**Share of spend reached** is the budget number: 10% of members puts about a fifth of total spend
inside the programme, against 33.5% for a perfect targeter.

Present the 2% row alongside: highest precision and lift, so a small intensive programme is most
efficient per member. **CM**
"""),

md("""
### Fixing the cut-off honestly

The threshold has to be derived on data the model was not judged on, and on the same probability
scale it will be applied to. Deriving it on the held-out set and then reporting performance at that
threshold, on the same set, would be marking my own homework — so I carve a validation split out of
the training members instead.
"""),

code("""
# Deriving the operating threshold on a validation split of TRAIN, on the calibrated scale
X_inner, X_val, y_inner, y_val = train_test_split(
    X_train, y_high_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_high_train)

val_proba = calibrated_model.predict_proba(X_val)[:, 1]
n_enrol_val = int(len(X_val) * CAPACITY_PCT)
OPERATING_THRESHOLD = float(np.sort(val_proba)[::-1][n_enrol_val - 1])

enrolled = proba_cal >= OPERATING_THRESHOLD

pd.Series({
    'operating threshold (calibrated probability)': round(OPERATING_THRESHOLD, 4),
    'members enrolled in the held-out group': int(enrolled.sum()),
    'precision at capacity': round(precision_score(y_high_test, enrolled), 3),
    'recall at capacity': round(recall_score(y_high_test, enrolled), 3),
    'lift over the base rate': round(precision_score(y_high_test, enrolled) / y_high_test.mean(), 2),
})
"""),

code("""
# The full breakdown at that cut-off. This is the one place a printed report has no display
# equivalent worth building, so it stays a print.
print(classification_report(y_high_test, enrolled, target_names=['ordinary', 'high cost'], digits=3))
"""),

code("""
# Five-fold cross-validated AUC, so the ranking quality is not a single-split claim either
cv_auc = cross_val_score(make_pipeline(HistGradientBoostingClassifier(class_weight='balanced', **GBM_KW)),
                         df_ml, y_high, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                         scoring='roc_auc', n_jobs=N_JOBS)

pd.Series({'CV roc_auc mean': round(float(cv_auc.mean()), 3),
           'CV roc_auc standard deviation': round(float(cv_auc.std()), 3)})
"""),

md("""
### What is the model actually using?

Before trusting a model I want to know what it leans on, for two reasons. If it depends on
something that will not be available at prediction time, it fails in use. And if it depends on
something it should not be using — a proxy for sex or region, say — that is a fairness problem.

The test shuffles one column at random and measures how much worse the model gets. A column the
model relies on causes a large drop when scrambled; a column it ignores causes none.
"""),

code("""
# Scrambling each column in turn to see how much the model depends on it
importance_sample = X_test.sample(min(5000, len(X_test)), random_state=RANDOM_STATE)
permutation = permutation_importance(
    risk_model, importance_sample, y_high_test.loc[importance_sample.index],
    n_repeats=5, random_state=RANDOM_STATE, scoring='roc_auc', n_jobs=N_JOBS)

importance = pd.Series(permutation.importances_mean, index=importance_sample.columns).sort_values(ascending=False)

top = importance.head(12)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top)), top.values, color=BLUE)
plt.yticks(range(len(top)), top.index)
plt.xlabel('Drop in performance when the column is scrambled')
plt.title('The risk model runs on four fields')
plt.grid(axis='x', alpha=0.5)
save('v3_03_risk_model_importance')
plt.show()
"""),

md("""
##### Interpretation

Four fields do nearly all the work: **chronic condition count** by a wide margin, then **smoking**,
**days in hospital** and **age**.

All four are known before the year starts, so the model can run at renewal — the feature contract
holding in practice, not just on paper.

Region, sex, marital status, education and income sit at the bottom near zero, consistent with 4.2
finding each explains under 1% of cost variance. The model is not using demographics as a back
door, largely because there is no signal to use.

An importance table is not a fairness audit, though, and it is not treated as one — 6.3 measures it.
"""),

md("""
---

## 6.3 Fairness, Measured Rather Than Asserted

A model can be blind to a demographic field and still perform unevenly across groups. So rather
than infer fairness from what the model leans on, I measure both models directly across sex, region
and age band: is the cost model biased within each group, and does the risk model rank equally well
inside each one?
"""),

code("""
# Group-level performance of both models across the dimensions that matter
fairness = df.loc[X_test.index, ['Sex', 'Region']].copy()
fairness['age band'] = pd.cut(df.loc[X_test.index, 'Age'], [0, 29, 49, 64, 200],
                              labels=['<=29', '30-49', '50-64', '65+'])
fairness['actual_cost'] = y_test.values
fairness['predicted_cost'] = pred_A
fairness['actual_high'] = y_high_test.values
fairness['proba_high'] = proba_cal

fairness_rows = []
for dimension in ['Sex', 'Region', 'age band']:
    for group, sub in fairness.groupby(dimension, observed=True):
        fairness_rows.append({
            'dimension': dimension, 'group': group, 'members': len(sub),
            'cost model bias %': round((sub['predicted_cost'].mean() / sub['actual_cost'].mean() - 1) * 100, 1),
            'risk model AUC': round(roc_auc_score(sub['actual_high'], sub['proba_high']), 3)
            if sub['actual_high'].nunique() > 1 else np.nan,
        })

pd.DataFrame(fairness_rows)
"""),

md("""
Every group's cost bias sits within a few points of zero, and the risk model's ranking quality is
stable across sex and region. The age bands vary a little more, which is expected: the model relies
on age, and the oldest band is the smallest.

This is measured now rather than inferred from an importance table. Before any deployment it should
be rerun on the actual enrolment list rather than on a random test split, and the affordability
impact from section 4.4 reviewed with compliance. **UW / PROD**
"""),

md("""
## 6.4 Who Pays More Than Their Risk?

Section 4.1 ruled out any pricing-adequacy claim, and that stands. But a company would still
reasonably ask which members contribute more than their risk-based cost and which contribute less,
and Model A gives the only honest way to look at that on this data: compare what a member actually
pays against what the model expects them to cost.

This is descriptive. It is not a pricing recommendation, because the premium field's provenance
forbids one.
"""),

code("""
# Actual contribution against model-expected cost, by age and chronic burden
subsidy = df.loc[X_test.index, ['Age', 'Annual Premium', 'Chronic Conditions Count']].copy()
subsidy['expected_cost'] = pred_A
subsidy['age band'] = pd.cut(subsidy['Age'], [15, 29, 49, 64, 200], labels=['16-29', '30-49', '50-64', '65+'])
subsidy['chronic band'] = pd.cut(subsidy['Chronic Conditions Count'], [-1, 0, 1, 99],
                                 labels=['0 conditions', '1 condition', '2+ conditions'])

cross = subsidy.groupby(['age band', 'chronic band'], observed=True).agg(
    members=('Annual Premium', 'count'),
    actual_contribution=('Annual Premium', 'mean'),
    expected_cost=('expected_cost', 'mean')).round(0)
cross['ratio actual / expected'] = (cross['actual_contribution'] / cross['expected_cost']).round(2)
cross
"""),

code("""
# The same thing as a map, which is easier to read than twelve rows
pivot = cross.reset_index().pivot(index='chronic band', columns='age band', values='ratio actual / expected')

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(pivot.values.astype(float), cmap=DIV, vmin=0, vmax=float(np.nanmax(pivot.values)) * 2)
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f'{pivot.values[i, j]:.2f}x', ha='center', va='center', fontsize=11)
ax.set_title('Actual contribution divided by model-expected cost')
plt.tight_layout()
save('v3_05_cross_subsidy')
plt.show()
"""),

md("""
The ratios are strikingly uniform, and that uniformity **is** the finding. Because the actual
formula charges a fixed share of realised cost, and the model prices expected cost from risk
factors, every segment lands in the same narrow band. No group is meaningfully subsidising another
— which is exactly what you would expect from a contribution formula that reads the answer rather
than assessing risk. **ACT / FIN**
"""),

md("""
---

## 6.5 Why I Am Not Modelling the Premium

I said at the start I was refusing to build a premium model. Here is the evidence, so the omission
is visibly a choice rather than an oversight.
"""),

code("""
# Reproducing the premium from the formula recovered in 4.1
formula_premium = 200 + 0.01 * df['Deductible'] + df['Network Tier'].map(RATE) * df[TARGET]
formula_error = (df['Annual Premium'] - formula_premium).abs()

pd.Series({
    'R2 of the formula': r2_score(df['Annual Premium'], formula_premium),
    'Largest error anywhere': formula_error.max(),
    'Share reproduced to the cent': (formula_error <= 0.005).mean(),
}).round(6)
"""),

md("""
The formula reproduces the premium with an R-squared of 1.000000 and a largest error anywhere of
half a cent, which is what rounding to the nearest cent produces.

A model trained on this target would score near-perfectly and would have learnt nothing about risk.
The premium here is a **contribution calculated after the cost is known**, not a price set in
advance, so predicting it is predicting arithmetic.

The genuinely useful question — what should we charge a member — is answered by predicting what
they will cost, which is Model A.
"""),

md("""
---

## 6.6 Saving the Models, and Proving They Work

Three models leave this notebook: the cost model, the bad-year model and the calibrated risk model.
Each gets a description written alongside it, so a `.pkl` is never an unlabelled binary.

Then I test that they actually load and score raw member records, because a model that cannot be
reloaded is not a deliverable.
"""),

code("""
# Writing a description alongside each saved model
def save_model(name, estimator, task, target, metrics, intended_use, not_for, extra=None):
    joblib.dump(estimator, MODELS / f'{name}.pkl', compress=3)
    meta = {
        'model_name': name,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__,
        'random_state': RANDOM_STATE,
        'task': task,
        'target': target,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': list(df_ml.columns),
        'engineered_inside_pipeline': ENGINEERED_NAMES,
        'excluded_leaky': list(LEAKY),
        'excluded_same_period': list(SAME_PERIOD),
        'metrics': metrics,
        'intended_use': intended_use,
        'not_for': not_for,
        **(extra or {}),
    }
    (MODELS / f'{name}_metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    return name
"""),

code("""
# Saving all three
save_model('costA_renewal_gamma', mono_model,
           'regression (gamma loss, monotonic, renewal-legal features)', TARGET,
           {'r2': float(r2_score(y_test, pred_mono)),
            'mae': float(mean_absolute_error(y_test, pred_mono)),
            'mean_ratio': float(pred_mono.mean() / y_test.mean()),
            'cv_r2_mean': float(cv_r2.mean()), 'cv_r2_sd': float(cv_r2.std())},
           'Expected cost for groups and segments; the basis for a technical premium. Runnable at renewal.',
           'Pricing or underwriting decisions about an individual member.')

save_model('costQ_p90', quantile_models[0.9],
           'quantile regression (90th percentile bad-year cost)', TARGET,
           {'p90_coverage': float((y_test <= p90_pred).mean())},
           "Estimating a member's bad-year exposure for product and group planning.",
           'Individual member pricing.')

save_model('riskB_renewal_calibrated', calibrated_model,
           'classification (calibrated probabilities)', f'{TARGET} above the {int(TAIL_Q * 100)}th percentile',
           {'roc_auc': float(roc_auc_score(y_high_test, proba_cal)),
            'brier': float(brier_score_loss(y_high_test, proba_cal)),
            'brier_baseline': float(baseline_brier),
            'cv_auc_mean': float(cv_auc.mean()),
            'precision_at_capacity': float(precision_score(y_high_test, enrolled)),
            'recall_at_capacity': float(recall_score(y_high_test, enrolled))},
           'Ranking members for care-management enrolment at a fixed capacity.',
           'Pricing, refusing cover, loading a premium, or any decision that disadvantages a member.',
           extra={'operating_threshold': OPERATING_THRESHOLD,
                  'threshold_scale': 'calibrated probability',
                  'threshold_derived_on': '20% stratified validation split of the training members',
                  'capacity_pct': CAPACITY_PCT})

sorted(p.name for p in MODELS.iterdir())
"""),

code("""
# Reloading from disk and scoring untouched rows straight from the cleaned CSV
def load_and_predict(name, raw_rows):
    model = joblib.load(MODELS / f'{name}.pkl')
    meta = json.loads((MODELS / f'{name}_metadata.json').read_text())
    features = raw_rows[meta['features']]
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(features)[:, 1], meta
    return model.predict(features), meta


raw_rows = pd.read_csv(CLEAN_PATH).head(5)
cost_pred, _ = load_and_predict('costA_renewal_gamma', raw_rows)
bad_year_pred, _ = load_and_predict('costQ_p90', raw_rows)
risk_pred, risk_meta = load_and_predict('riskB_renewal_calibrated', raw_rows)

pd.DataFrame({
    'Id': raw_rows['Id'],
    'actual cost': raw_rows[TARGET].round(0),
    'predicted cost': np.round(cost_pred),
    'predicted bad year': np.round(bad_year_pred),
    'risk of top decile': np.round(risk_pred, 3),
    'flag for care management': risk_pred >= risk_meta['operating_threshold'],
})
"""),

md("""
All three load from disk and score raw member records with no preparation, confirming the decision
to bundle engineering and encoding inside the pipeline. A recipient needs the cleaned CSV schema
and nothing else.

The predictions also make the individual-level limit concrete: member 0 cost 6,938 and the model
predicted 2,946. Not a bug — that is R2 0.167 inspected one member at a time, and why the model
card restricts Model A to groups.
"""),

md("""
---

## 6.7 Model Card

### Model A — expected cost (`costA_renewal_gamma.pkl`)

| | |
|:---|:---|
| **Does** | Predicts a member's annual cost |
| **Use for** | Group and segment expected cost; the basis for a technical premium |
| **Not for** | Pricing or underwriting an individual member |
| **Why not** | Typical prediction misses by about 84% of a typical annual cost |
| **Good at** | Every predicted-cost group within a few percent of actual, no directional bias |
| **Inputs** | Renewal-legal attributes plus eleven engineered features, computed in-pipeline |
| **Barred** | Premium, claims paid, average claim, loss ratio, risk score (computed from the answer); claims count, visits, procedure counts, major-procedure flag (same-period) |
| **Constraints** | Non-decreasing in age, chronic burden, prior inpatient history |

### Model B — high-cost risk (`riskB_renewal_calibrated.pkl`)

| | |
|:---|:---|
| **Does** | Ranks members by chance of landing in the most expensive tenth |
| **Use for** | Care-management enrolment, cut-off set by capacity |
| **Not for** | Pricing, refusing cover, loading a premium, or disadvantaging a member |
| **How good** | ~3× random. 10% enrolled reaches about a fifth of spend |
| **Probabilities** | Calibrated; threshold derived on a validation split, on the calibrated scale |
| **Fairness** | Cost bias within a few points, stable ranking across sex, region, age band — measured in 6.3. Re-audit on the real enrolment list before going live |

### Model C — bad-year cost (`costQ_p90.pkl`)

The 90th percentile rather than the mean — the number 4.8 argued product should sell on. Coverage
checks out near 90%. Group planning only.

### Limits on all three

**No time dimension.** One snapshot, one year per member, so these relate attributes to
same-period cost rather than forecasting next year. Deployment needs several years of history and
a test on a later year. The renewal-legal contract is what makes that an upgrade rather than a
rewrite.

**The ceiling is genuine.** R² 0.167 on money, 0.165 under cross-validation with a standard
deviation of 0.004. Section 4.2 put the ceiling near 16% independently. A much higher score means
checking the feature contract first.

**No model here should touch the alcohol field** — its largest category is entirely imputed.
"""),

md("""
---

## How to Run This

```bash
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Reads `data/medical_insurance.csv`. Writes three things: the cleaned CSV, the figures under
`reports/figures/`, and three models with metadata under `updated_models/`.

That folder is deliberately separate from `models/`, which holds artefacts from the previous
notebook — some built on the same-period feature set and so unusable at renewal. Nothing here reads
or overwrites them.

Everything else — every table, the cleaning log, the bias report, the dashboard charts — stays in
the notebook. Fitting takes a few minutes; cross-validation, the learning curve and permutation
importance are the slow parts.
"""),

]
