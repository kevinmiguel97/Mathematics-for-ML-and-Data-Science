# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Course materials for the Mathematics for Machine Learning and Data Science specialization. Three subjects are covered, each organized by week:

- **Linear Algebra** (Course 1, prefix `C1`) — 4 weeks
- **Calculus** (Course 2, prefix `C2`) — 3 weeks
- **Probability** — in progress

## Environment Setup

A virtual environment is located at `.venv`. Activate it before running anything:

```powershell
.venv\Scripts\Activate.ps1
```

Key packages available: `numpy`, `jax`, `scipy`, `sympy`, `pandas`, `matplotlib`, `scikit-learn`, `jupyterlab`.

## Running Jupyter

```powershell
.venv\Scripts\jupyter lab
```

## Running Unit Tests

Each graded assignment folder contains a `w{n}_unittest.py` file. These are **not** standard pytest suites — the test functions take target functions or objects as arguments and are called from within the assignment notebook. To run them manually:

```powershell
# From the repo root, activate venv first, then:
python -c "
import sys
sys.path.insert(0, 'Calculus/w2/Graded lab')
import w2_unittest
# call individual test functions with your implementation
"
```

Or run a specific unittest file directly if it has a `__main__` block:

```powershell
python "Calculus\w2\Graded lab\w2_unittest.py"
```

## Repository Structure

```
Linear Algebra/
  w1/          # NumPy arrays intro, quiz
  w2/          # Linear systems, row echelon form — w2_unittest.py
  w3/          # Vector operations, graded assignment — w3_unittest.py + utils.py
  w4/          # Graded assignment — w4_unittest.py + utils.py

Calculus/
  w1/          # Differentiation lab, graded assignment (uses JAX) — w1_unittest.py
  w2/          # Graded lab — w2_unittest.py (numpy, pandas, sklearn)
  w3/          # Perceptron regression/classification, Newton's method labs (no graded lab)

Probability/   # Empty / in progress
```

## Assignment Conventions

- Graded assignment notebooks come in two versions: the working file (e.g., `C2_W2_Assignment.ipynb`) and a clean backup (`C2_W2_Assignment_clean.ipynb`). Edit the non-clean version.
- `utils.py` files in graded lab folders contain plotting helpers — do not modify them.
- `data/` folders hold CSV/dataset files consumed by the notebooks.
- Calculus notebooks use `jax.numpy` as `np` in place of standard `numpy` to support automatic differentiation.

## Notebook Naming

| Prefix | Subject |
|--------|---------|
| `C1_W{n}` | Linear Algebra week n |
| `C2_W{n}` | Calculus week n |
