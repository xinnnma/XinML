# XinML — Presentation Notes

Aim for about four minutes at a comfortable pace. The eight sections below
form the spoken script; the questions afterward are preparation material.
Keep the repository and saved notebooks open before starting.

## 1. Introduction — 0:00–0:25

My project is XinML, Machine Learning from Scratch, for CMOR 438 / INDE 577
at Rice University. I wanted to understand how two familiar models work by
building a small reusable package. I chose Linear Regression and binary
Logistic Regression, then connected them to real datasets, tests, and
notebook examples.

## 2. Repository and package structure — 0:25–0:50

The public repository is github.com/xinnnma/XinML. The reusable code is in
`src/xinml`, the tests are in `tests`, and the three demonstrations are in
`notebooks`. The README explains installation and usage. The commit history
records the setup, each algorithm, and the notebooks. Both models are imported
directly from `xinml`.

## 3. Linear Regression — 0:50–1:25

Linear Regression predicts a numerical target from a weighted combination
of features and an optional intercept. It chooses coefficients that minimize
the sum of squared errors. I use NumPy's `lstsq` solver rather than forming
an explicit matrix inverse. It also handles redundant features with a
minimum-norm solution. The class supports one or several features and provides
`fit`, `predict`, and an R-squared `score` method.

## 4. Logistic Regression and gradient descent — 1:25–2:05

Logistic Regression predicts a binary label. It converts a linear score into
a class-one probability using the sigmoid function, then applies a threshold
of one half. I implemented batch gradient descent: start with zero weights,
calculate the gradient of mean binary cross-entropy, and update the weights
using a learning rate. The sigmoid and loss calculations avoid exponential
overflow. The model also returns both class probabilities and records training
loss so I can inspect optimization progress.

## 5. Notebooks and datasets — 2:05–2:40

The first notebook explores the Diabetes data. The second uses all ten
Diabetes features for regression. The third classifies the Breast Cancer
data using thirty features. These datasets come bundled with scikit-learn,
but the trained models are my XinML implementations. I split the data before
fitting scalers or models, and fit them only on training rows. The fixed
random seed and stratified classification split make the examples reproducible.

## 6. Results and interpretation — 2:40–3:20

On 89 held-out Diabetes rows, Linear Regression achieved an R-squared of
0.4526 and an RMSE of about 53.85 target units. On 114 held-out Breast Cancer
rows, Logistic Regression classified 110 correctly, giving 96.49 percent
accuracy. These are different metrics for different tasks, so I do not compare
them directly. The confusion matrix includes one malignant case predicted as
benign. Across the three notebooks, eight figures help explain the data,
prediction errors, and training loss.

## 7. Tests and reproducibility — 3:20–3:55

All 152 pytest tests pass. They check known solutions, predictions, invalid
inputs, unfitted models, and numerical edge cases. Logistic tests also check
gradient updates and convergence to a known solution. The lockfile records
dependencies. I verified a fresh environment, reran all three notebooks
without dataset downloads, and checked both README examples. The package's
only runtime dependency is NumPy; notebook and test tools are development
dependencies.

## 8. Limitations and conclusion — 3:55–4:20

The models assume linear relationships or a linear decision boundary.
Logistic Regression is binary only, uses a fixed number of updates, and has
no regularization or automatic scaling. Each notebook uses one held-out
split, so the results have limited scope. XinML shows the full path from
mathematical ideas to reusable code, reliable tests, and interpretable examples.

## Five likely questions

### 1. What did you implement yourself if you used NumPy and scikit-learn?

I wrote the estimators, validation, prediction methods, scoring, and Logistic
Regression training loop. NumPy supplies array operations and the least-squares
solver. Scikit-learn supplies bundled datasets, splitting, scaling, and
evaluation utilities in the notebooks; neither model delegates training to a
scikit-learn estimator.

### 2. How did you prevent data leakage?

I split before exploration and preprocessing. Diabetes loads raw features
with `scaled=False`, so the loader does not normalize using the full dataset.
The scaler learns only training means and standard deviations, and the models
fit only training rows. Test labels are used for evaluation, with model
settings fixed beforehand.

### 3. What do R² of 0.4526 and accuracy of 96.49% mean?

R² is one minus the regression squared-error ratio relative to predicting the
test target mean. Here the model reduces that squared error by about 45.26%.
Accuracy is 110 correct binary predictions out of 114. They cannot rank these
two models against each other because the tasks and metrics differ. The
classification confusion matrix, `[[41, 1], [3, 69]]`, also exposes the kinds
of mistakes hidden by the overall accuracy.

### 4. How do you know gradient descent is correct and numerically stable?

Tests verify a calculated first update and a nonseparable problem whose
optimal probability is known. Separate sigmoid branches avoid exponential
overflow, and `logaddexp` computes cross-entropy from logits without taking
the logarithm of a rounded zero probability. Extreme-input tests exercise
these choices. In the classification notebook, training loss falls from
about 0.6931 to 0.0509; that supports optimization progress, not a claim of
global convergence for every dataset or test performance.

### 5. What are the main limitations, and what would you consider next?

The simple models have linear structure, no regularization, and no automatic
feature scaling. Logistic Regression handles only labels 0 and 1 and does not
stop early. A single split does not measure variation across splits or new
populations. If extending the project, I would first consider cross-validation
and regularization, while keeping the current implementation easy to explain.
