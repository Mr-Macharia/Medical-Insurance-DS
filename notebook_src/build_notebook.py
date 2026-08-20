"""Assemble notebook.ipynb from the section modules.

    python3 notebook_src/build_notebook.py

Rebuilds the notebook WITHOUT outputs. Execute it afterwards:
    jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import s1_dictionary, s2_observation, s3_cleaning, s4_exploration, s5_dashboard, s6_modelling

SECTIONS = [s1_dictionary, s2_observation, s3_cleaning,
            s4_exploration, s5_dashboard, s6_modelling]

OUT = ROOT / 'notebook.ipynb'

cells = [c for mod in SECTIONS for c in mod.CELLS]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    'kernelspec': {'display_name': 'remediation-env', 'language': 'python',
                   'name': 'remediation-env'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, OUT)

n_code = sum(1 for c in cells if c.cell_type == 'code')
print(f'{len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown) -> {OUT}')
