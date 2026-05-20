# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course Overview

**Calculus for Machine Learning and Data Science** (Course 2, prefix `C2`). Covers differentiation, gradient-based optimization, and neural network fundamentals. 3 weeks.

## Week Structure

### Week 1 — Differentiation in Python
- **Lab:** `C2_W1_Lab_1_differentiation_in_python.ipynb` — compares symbolic (SymPy), numerical (NumPy), and automatic (JAX) differentiation
- **Graded Lab:** `Graded Lab/C2_W1_Assignment.ipynb` — portfolio optimization using JAX automatic differentiation
  - Data: `data/prices.csv` (50 rows, two supplier price series)
  - Tests: `w1_unittest.py` — validates `load_and_convert_data`, `f_of_omega`, `L_of_omega_array`, `dLdOmega_of_omega_array`

### Week 2 — Gradient Descent
- **Labs:** `Labs/` — gradient descent in one and two variables with interactive visualizations
  - `w2_tools.py` — gradient descent classes (`gradient_descent_one_variable`, `gradient_descent_two_variables`) and plotting functions; do not modify
- **Graded Lab:** `Graded lab/C2_W2_Assignment.ipynb` — linear regression via gradient descent applied to TV marketing data
  - Data: `data/tvmarketing.csv` (200 rows, TV spend vs. Sales)
  - Tests: `w2_unittest.py` — validates `load_data`, `pred_numpy`, `sklearn_fit`, `sklearn_predict`, `partial_derivatives` (dEdm, dEdb), `gradient_descent`

### Week 3 — Neural Networks and Newton's Method
- **Labs:** `Labs/` — perceptron regression, perceptron classification, Newton's method optimization
  - Data: `data/house_prices_train.csv` (large housing dataset), `data/tvmarketing.csv`
  - `images/` — neural network architecture diagrams referenced by notebooks
- **Graded Labs:** `Graded Labs/` — currently empty

## Library Notes

- Week 1 uses `jax.numpy as jnp` (not `numpy`) to enable automatic differentiation — do not substitute with standard numpy.
- Week 2 uses `matplotlib` widgets for interactive gradient descent animations; these require `%matplotlib widget` in notebooks.
- Week 2 graded lab uses both raw numpy and `sklearn.linear_model.LinearRegression` — both implementations are tested separately.
