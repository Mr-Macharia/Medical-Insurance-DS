"""Section 5 of notebook.ipynb — dashboard.

Regenerated from notebook.ipynb. Edit here, then run:
    python3 notebook_src/build_notebook.py
"""

from common import md, code

CELLS = [

md("""
---

# 5. The Dashboard Picture

Section 4 is long, because working out what a book contains takes a lot of small checks. This
section is the opposite: the ten charts I would actually put in front of a room, drawn once,
interactive, in one place.

They are Plotly rather than matplotlib, which is a deliberate choice rather than a second style
for its own sake. Hovering to read an exact value matters far more in a meeting than it does
inside a written analysis, and these are the charts people will want to interrogate.

Everything here recomputes from the same frame the analysis used, and most of it reuses the tables
section 4 already built — `conditions_cost`, `targeting`, `exposure`, `effect`, `burden`. That is
the payoff for having one notebook rather than six: the dashboard cannot drift away from the
analysis, because there is only one set of numbers.

Nothing in this section is written to disk. These charts live in the notebook.
"""),

code("""
# Plotly, for charts a reader can hover over
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'plotly_mimetype+notebook_connected'
"""),

code("""
# A house style for this section, matching the colour-blind palette used above
PLOTLY_PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']


def style(fig, title, subtitle, height=480):
    fig.update_layout(
        template='plotly_white',
        title={'text': f'{title}<br><sup>{subtitle}</sup>', 'x': 0.02, 'xanchor': 'left'},
        height=height,
        margin={'l': 60, 'r': 30, 't': 80, 'b': 50},
        colorway=PLOTLY_PALETTE,
    )
    return fig
"""),

md("""
## 5.1 Cost distribution

The shape everything else follows from: right-skewed, long tailed, most members clustered around
2,100.
"""),

code("""
cost = df['Annual Medical Cost']
p50, p90 = cost.median(), cost.quantile(0.9)

# Pre-binning on a log grid, so 96,639 points become 60 readable bars
log_bins = np.logspace(np.log10(cost.min()), np.log10(cost.max()), 61)
counts, edges = np.histogram(cost, bins=log_bins)
centres = np.sqrt(edges[:-1] * edges[1:])          # geometric midpoints sit correctly on a log axis

fig = go.Figure(go.Bar(x=centres, y=counts, width=np.diff(edges),
                       marker_color=PLOTLY_PALETTE[0], name='Members'))
fig.update_xaxes(type='log', title='Annual medical cost (log scale)')
fig.update_yaxes(title='Members')
fig.add_vline(x=p50, line_dash='dash', line_color='#333333',
              annotation_text=f'median {p50:,.0f}', annotation_position='top left')
fig.add_vline(x=p90, line_dash='dash', line_color=PLOTLY_PALETTE[3],
              annotation_text=f'p90 {p90:,.0f}', annotation_position='top right')
style(fig, 'Cost Distribution',
      'Annual medical cost, log scale — the top decile carries 33.5% of total spend')
fig.show()
"""),

md("""
## 5.2 Spend concentration

The same fact stated as a Lorenz curve, which is the version that persuades a budget meeting: a
tenth of the members carry a third of the money.
"""),

code("""
sorted_costs = np.sort(df['Annual Medical Cost'].to_numpy(float))
cum_spend = np.concatenate([[0.0], np.cumsum(sorted_costs) / sorted_costs.sum()])
cum_pop = np.linspace(0, 1, len(cum_spend))

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect equality',
                         line=dict(dash='dash', color='#999999')))
# Decimating to a 1,001-point grid keeps the figure light without changing the curve
idx = np.unique(np.linspace(0, len(cum_pop) - 1, 1001).astype(int))
fig.add_trace(go.Scatter(x=cum_pop[idx], y=cum_spend[idx], mode='lines', name='Observed',
                         line=dict(color=PLOTLY_PALETTE[0], width=3), fill='tonexty',
                         fillcolor='rgba(0,114,178,0.12)'))
fig.add_annotation(x=0.35, y=0.75, text=f'Gini = {gini:.2f}', showarrow=False,
                   font=dict(size=18, color=PLOTLY_PALETTE[3]))
fig.update_xaxes(title='Cumulative share of members')
fig.update_yaxes(title='Cumulative share of spend')
style(fig, 'Spend Concentration', 'Lorenz curve of annual medical cost')
fig.show()
"""),

md("""
## 5.3 What actually drives cost

The variance league from 4.2, with the 1% line that separates the four usable variables from the
ten that are not.
"""),

code("""
league_sorted = effect['eta squared %'].sort_values()

fig = go.Figure(go.Bar(
    x=league_sorted.values, y=league_sorted.index, orientation='h',
    marker_color=[PLOTLY_PALETTE[3] if v >= 1 else PLOTLY_PALETTE[0] for v in league_sorted.values],
    text=[f'{v:.2f}%' for v in league_sorted.values], textposition='outside',
))
fig.update_xaxes(title='Share of cost variance explained (%)', range=[0, 11])
style(fig, 'What Actually Drives Cost',
      'Clinical burden, smoking, procedures and age clear 1%. Every demographic is inert', height=560)
fig.show()
"""),

md("""
## 5.4 Loss ratio by tier

Included with its caveat attached to the chart itself, because this is the chart somebody will
screenshot. The tier medians are simply `1 / tier_rate` from the formula in 4.1 — **arithmetic, not
evidence of underpricing.**
"""),

code("""
loss_ratio = df.groupby('Network Tier')['Loss Ratio'].median().reindex(TIER)

fig = go.Figure(go.Bar(
    x=loss_ratio.index, y=loss_ratio.values,
    marker_color=[PLOTLY_PALETTE[1], '#9E9E9E', '#D4AF37', '#7B7FD4'],
    text=[f'{v:.2f}x' for v in loss_ratio.values], textposition='outside',
))
fig.update_yaxes(title='Median loss ratio (cost / premium)')
style(fig, 'Loss Ratio by Tier',
      'CAVEAT: premium = 200 + 0.01xDeductible + tier_rate x Cost, so each bar is 1/tier_rate — arithmetic, not inadequacy')
fig.show()
"""),

md("""
## 5.5 Age and morbidity

The one demographic gradient that survives. Five conditions shown rather than all ten, because the
other five are flat and add nothing but ink.
"""),

code("""
GRADIENT_CONDITIONS = ['Hypertension', 'Diabetes', 'Arthritis',
                       'Mental Health Condition', 'Cardiovascular Disease']

fig = go.Figure()
for i, cond in enumerate(GRADIENT_CONDITIONS):
    fig.add_trace(go.Scatter(x=[c.split(' (')[0] for c in AGE_COHORT], y=by_age[cond],
                             mode='lines+markers', name=cond,
                             line=dict(color=PLOTLY_PALETTE[i], width=2.5)))
fig.update_yaxes(title='Prevalence (%)')
fig.update_layout(legend_title_text='Condition')
style(fig, 'Age-Morbidity Gradient',
      'Hypertension climbs 1.56x from 16-29 to 70+ — a real shift, but most older members still do not have it')
fig.show()
"""),

md("""
## 5.6 Severity against burden

The inversion that decides care-management priorities: the worst condition to *have* is not the
worst condition for the *book*.
"""),

code("""
fig = go.Figure(go.Scatter(
    x=conditions_cost['extra cost per member'], y=conditions_cost['% of all spend'],
    mode='markers+text', text=conditions_cost['condition'], textposition='top center',
    marker=dict(size=conditions_cost['prevalence %'] * 2.2 + 8, color=PLOTLY_PALETTE[0], opacity=0.65),
    hovertemplate='%{text}<br>+%{x:,.0f} per member<br>%{y:.1f}% of spend<extra></extra>',
))
fig.update_xaxes(title='Extra cost per member with the condition (median uplift)')
fig.update_yaxes(title='Share of total portfolio spend (%)')
style(fig, 'Severity vs Burden',
      'Bubble size is prevalence. Liver disease is worst per member; hypertension dominates the book by volume')
fig.show()
"""),

md("""
## 5.7 Care-management targeting

Which cohorts are worth enrolling, ranked by the share of spend they reach.
"""),

code("""
cohort_chart = targeting.sort_values('% of spend')

fig = go.Figure(go.Bar(
    x=cohort_chart['% of spend'], y=cohort_chart['cohort'], orientation='h',
    marker_color=PLOTLY_PALETTE[0],
    text=[f'{s:.1f}% of spend · {n:,} members'
          for s, n in zip(cohort_chart['% of spend'], cohort_chart['members'])],
    textposition='outside',
))
fig.update_xaxes(title='Share of total portfolio spend (%)',
                 range=[0, cohort_chart['% of spend'].max() * 1.4])
style(fig, 'Care-Management Targeting',
      'Prior hospitalisation reaches 9% of the book; the 273 undiagnosed diabetics are a governance item, not a financial one',
      height=520)
fig.show()
"""),

md("""
## 5.8 Affordability burden

Median income against the share of it that goes on premium. The bars rise 7.8 times; the line does
the opposite.
"""),

code("""
income_median = df.groupby('Income Band', observed=True)['Income'].median()

fig = go.Figure()
fig.add_trace(go.Bar(x=income_median.index.astype(str), y=income_median.values,
                     name='Median income', marker_color=PLOTLY_PALETTE[0],
                     text=[f'{v:,.0f}' for v in income_median.values], textposition='outside'))
fig.add_trace(go.Scatter(x=burden.index.astype(str), y=burden.values,
                         name='Premium burden (% of income)', yaxis='y2',
                         mode='lines+markers', line=dict(color=PLOTLY_PALETTE[3], width=3),
                         marker=dict(size=10)))
fig.update_layout(
    yaxis=dict(title='Median income'),
    yaxis2=dict(title='Premium burden (% of income)', overlaying='y', side='right',
                range=[0, burden.max() * 1.4]),
    legend=dict(x=0.02, y=0.98),
)
style(fig, 'Affordability Burden',
      'Median premium is flat near 465 while income rises 7.8x — the poorest quintile carries 8.8x more of its income')
fig.show()
"""),

md("""
## 5.9 Regional exposure

Where the book sits, with mean cost printed on each bar so nobody reads a risk story into a
distribution fact.
"""),

code("""
regions = (df.groupby('Region')
             .agg(members=('Id', 'count'), mean_cost=('Annual Medical Cost', 'mean'))
             .assign(share=lambda d: d['members'] / len(df) * 100)
             .sort_values('share', ascending=False))

fig = go.Figure(go.Bar(
    x=regions.index, y=regions['share'], marker_color=PLOTLY_PALETTE[0],
    text=[f'{s:.1f}%<br><sup>mean cost {c:,.0f}</sup>' for s, c in zip(regions['share'], regions['mean_cost'])],
    textposition='outside',
))
fig.update_yaxes(title='Share of members (%)', range=[0, regions['share'].max() * 1.3])
style(fig, 'Regional Exposure',
      'South holds 28% of the book against Central 12%, but mean cost is flat to about 2% — where to sell, not what to charge')
fig.show()
"""),

md("""
## 5.10 What a bad year costs, by age

The volatility story from 4.8, and the chart behind the bad-year model in section 6.
"""),

code("""
fig = go.Figure()
for series, name, colour, dash in [
        (exposure['typical_year'], 'Typical year (p50)', PLOTLY_PALETTE[0], 'solid'),
        (exposure['bad_year'], 'Bad year (p90)', PLOTLY_PALETTE[1], 'dash'),
        (exposure['catastrophic_year'], 'Catastrophic year (p99)', PLOTLY_PALETTE[3], 'dot')]:
    fig.add_trace(go.Scatter(x=exposure.index.astype(str), y=series, mode='lines+markers',
                             name=name, line=dict(color=colour, width=2.5, dash=dash)))
fig.update_yaxes(title='Annual medical cost')
fig.update_xaxes(title='Age band')
style(fig, 'Bad-Year Cost by Age',
      'A bad year costs roughly 3x a typical one at every age — the gap is what cover exists to absorb')
fig.show()
"""),

md("""
That is the whole story in ten charts: a concentrated, clinically driven book with a pricing
formula rather than a price, a regressive contribution structure, a product ladder that delivers
nothing measurable, and one clear care-management target.

Everything above describes what already happened. The remaining question is whether any of it can
be known **in advance**, which is what section 6 is for.
"""),

]
