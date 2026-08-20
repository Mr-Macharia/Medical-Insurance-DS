# notebook_src — the notebook as source

`notebook.ipynb` is 365 cells and 3 MB once executed. It is generated from the six section
modules here so it stays editable and reviewable.

```bash
python3 notebook_src/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

| Module | Notebook section |
|:---|:---|
| `s1_dictionary.py` | 1 — The fields I have been given |
| `s2_observation.py` | 2 — Looking at the raw extract |
| `s3_cleaning.py` | 3 — Cleaning |
| `s4_exploration.py` | 4 — Exploring the book |
| `s5_dashboard.py` | 5 — The dashboard picture |
| `s6_modelling.py` | 6 — Predicting cost and risk |

`common.py` holds the `md()` / `code()` cell constructors.

## Conventions the notebook holds to

- **Every module is imported exactly once across the whole notebook**, in the section that
  first needs it. `pandas` is not imported twice; neither is `sklearn.metrics`.
- **One idea per cell.** Median code cell is about ten lines.
- **No `print`** except `classification_report`, which has no display equivalent worth
  building. Cells end on a bare expression so Jupyter renders the value.
- **Writes only what is consumed outside**: the cleaned CSV, the figures under
  `reports/figures/`, and the models under `updated_models/`.

Rebuilding strips outputs, so always execute afterwards — `pagegen/extract.py` reads outputs
and yields nothing from an unexecuted notebook.
