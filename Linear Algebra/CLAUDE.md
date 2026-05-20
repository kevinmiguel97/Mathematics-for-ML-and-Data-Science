# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course Overview

**Linear Algebra for Machine Learning and Data Science** (Course 1, prefix `C1`). Covers NumPy arrays, linear systems, vector operations, transformations, and PCA. 4 weeks.

## Week Structure

### Week 1 — NumPy Arrays
- **Lab:** `C1_W1_Lab_1_introduction_to_numpy_arrays.ipynb` — array creation, indexing, slicing, broadcasting
- **Extras:** `Quizz.ipynb` + `quiz.py` — multiple choice quiz helper; `utils.py` — `plot_lines(M)` for visualizing 2×2 linear systems
- No graded assignment this week

### Week 2 — Linear Systems
- **Lab:** `C1W2_UGL_solving_linear_systems_3_variables.ipynb` — solving 3-variable systems via row operations
- **Graded Assignment:** `C1W2_Assignment.ipynb` — implement Gaussian elimination and back substitution
  - `utils.py` — `string_to_augmented_matrix(equations)` converts text equations to SymPy augmented matrix; used for input parsing in labs
  - Tests: `w2_unittest.py` — validates `row_echelon_form` (expects row-reduced matrix or `"Singular system"`) and `back_substitution`
  - Also contains `test.py` and `test1.py` — scratch test files, not the official grader

### Week 3 — Vector Operations and Neural Networks
- **Lab:** `C1W3_UGL_1_vector_operations.ipynb` — vector operations, dot products, orthogonality
- **Lab Assignment:** `Lab assignment/C1W3_Assignment.ipynb` — linear transformations and a single-layer neural network trained on a toy dataset
  - Data: `data/toy_dataset.csv`, `data/image.txt` (flattened image, ~16 MB)
  - `utils.py` — provides the full neural network scaffolding: `initialize_parameters`, `compute_cost`, `backward_propagation`, `update_parameters`, `train_nn`, `plot_transformation`; students implement core math only
  - Tests: `w3_unittest.py` — validates transformation functions (`T_stretch`, `T_hshear`, etc.)

### Week 4 — Eigenvalues and PCA
- **Extras:** `graded_quizz.ipynb` — quiz notebook
- **Lab Assignment:** `Lab assignment/C1W4_Assignment.ipynb` — PCA on grayscale cat images
  - Data: `data/cat (1).jpg` … `data/cat (55).jpg` (55 grayscale images loaded via `cv2` and `glob`)
  - `support_files/` — pre-computed `.npy` arrays (`expected_centered_data`, `expected_cov_mat`, `expected_eigvals`, `expected_eigvecs`, `expected_pca12`, `expected_pca2`, `imgs_flatten`) used by the unittest for comparison
  - `utils.py` — minimal; image loading is done inline in the notebook
  - Tests: `w4_unittest.py` — validates stochastic transition matrix `P`, initial state vectors `X0`/`X1`, and PCA outputs against the pre-computed `.npy` references

## Library Notes

- Week 2 uses `sympy` for symbolic equation parsing in `utils.py` — the `string_to_augmented_matrix` function converts string equations before passing to numpy.
- Week 4 uses `cv2` (OpenCV) for image loading and grayscale conversion, and `glob` for batch file discovery — both are used in the notebook itself, not in `utils.py`.
- `support_files/*.npy` in Week 4 are the ground-truth outputs — do not modify them; the grader compares student outputs against these files.
