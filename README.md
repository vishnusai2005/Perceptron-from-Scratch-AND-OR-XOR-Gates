# Perceptron-from-Scratch-AND-OR-XOR-Gates
# 🧠 Perceptron from Scratch — AND | OR | XOR Gates

> A clean, ground-up implementation of the **Rosenblatt Perceptron** (1958) built entirely with NumPy — no ML frameworks. Demonstrates linear separability, the perceptron learning rule, decision boundary visualization, and the classic **XOR limitation** that motivated multi-layer neural networks.

---

## 📌 Overview

This project implements a **single-layer binary perceptron** from first principles and tests it on three classic logic gate classification tasks:

| Gate | Linearly Separable? | Perceptron Converges? |
|------|---------------------|-----------------------|
| AND  | ✅ Yes              | ✅ Yes                |
| OR   | ✅ Yes              | ✅ Yes                |
| XOR  | ❌ No               | ❌ No (by design)     |

The XOR experiment deliberately reveals the **fundamental limitation** of single-layer perceptrons — a historically significant result that paved the way for multi-layer networks and deep learning.

---

## 🧮 Mathematical Foundation

### The Perceptron Model

A perceptron is the simplest form of an artificial neuron. For an input vector **x** ∈ ℝⁿ, it computes:

```
z  =  w · x  +  b  =  Σᵢ (wᵢ · xᵢ)  +  b
ŷ  =  φ(z)
```

where **w** is the weight vector, **b** is the bias, and **φ** is the activation function.

### Activation Function — Heaviside Step

```
φ(z) = 1   if z ≥ 0
φ(z) = 0   if z < 0
```

This produces a hard binary output, making the perceptron a linear threshold unit (LTU).

### Perceptron Learning Rule

Weight and bias updates are driven by the error signal:

```
error  =  y  −  ŷ
w      ←  w  +  η · error · x
b      ←  b  +  η · error
```

where **η** (eta) is the **learning rate** (set to `0.1` in this implementation).

This rule is a special case of **gradient descent on the misclassification loss** for linearly separable data. The **Perceptron Convergence Theorem** guarantees that if the training data is linearly separable, this algorithm will converge in a finite number of steps.

---

## 🏗️ Implementation Architecture

The implementation is modular — each function handles exactly one responsibility:

```
perceptron/
│
├── intialize_weights(n_features)          → zeros init for w & b
├── step_function(z)                       → Heaviside activation  φ(z)
├── predict(x, weights, bias)              → single sample inference
├── update_weights(...)                    → one-step perceptron rule
├── train_perceptron(X, y, lr, epochs)     → full training loop
├── predict_all(X, weights, bias)          → batch inference
└── plot_decision_boundary(...)            → matplotlib visualization
```

### Module Breakdown

#### `intialize_weights(n_features)`
Initializes weight vector **w** as an all-zeros NumPy array and bias **b = 0**. Zero initialization is standard for the perceptron since the learning rule itself drives differentiation via training.

#### `step_function(z)`
Implements the Heaviside step activation. Returns `1` if the net input `z ≥ 0`, else `0`. This is the core non-linearity of a classical perceptron.

#### `predict(x, weights, bias)`
Computes the dot product `z = w · x + b` and passes it through `step_function`. Handles a single sample.

#### `update_weights(weights, bias, xi, target, prediction, learning_rate)`
Applies the perceptron update rule for a single training example:
- `error = target − prediction` (can be −1, 0, or +1)
- `weights += η × error × xᵢ`
- `bias += η × error`
- No update occurs when `error == 0` (correct prediction).

#### `train_perceptron(X, y, learning_rate=0.1, epochs=10)`
Runs the full **online learning loop** — for each epoch, iterates over every training sample individually and applies weight updates. Returns the final learned `weights` and `bias`.

> **Hyperparameters:** `learning_rate = 0.1`, `epochs = 10`

#### `predict_all(X, weights, bias)`
Applies `predict()` over all samples in **X** using a list comprehension — clean batch inference.

#### `plot_decision_boundary(X, y, weights, bias, title)`
Visualizes the learned linear decision boundary using `meshgrid` over the input space [−0.5, 1.5] × [−0.5, 1.5]. Fills regions with a `coolwarm` colormap and overlays training points color-coded by class.

---

## 🔬 Experiments & Results

### Experiment 1 — AND Gate

**Truth Table:**

| x₁ | x₂ | y (AND) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 0       |
| 1  | 0  | 0       |
| 1  | 1  | 1       |

AND is linearly separable — a single hyperplane cleanly divides the positive class `(1,1)` from the three negative examples. The perceptron converges and classifies all 4 points correctly.

**Decision Boundary:** A diagonal line separating the top-right quadrant from the rest.

---

### Experiment 2 — OR Gate

**Truth Table:**

| x₁ | x₂ | y (OR) |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 1  |

OR is linearly separable — only `(0,0)` is negative. The perceptron learns to separate it from the three positive examples. Converges cleanly.

**Decision Boundary:** A diagonal line isolating the origin.

---

### Experiment 3 — XOR Gate (The Limitation Demo)

**Truth Table:**

| x₁ | x₂ | y (XOR) |
|----|----|---------|
| 0  | 0  | 0       |
| 0  | 1  | 1       |
| 1  | 0  | 1       |
| 1  | 1  | 0       |

**XOR is NOT linearly separable.** The two positive examples `(0,1)` and `(1,0)` lie on a diagonal that cannot be divided from the two negative examples `(0,0)` and `(1,1)` by any single straight line.

The perceptron fails to converge — the decision boundary oscillates without ever correctly classifying all four points. This experiment deliberately demonstrates:

> "A single-layer perceptron cannot learn any function that is not linearly separable."
> — Minsky & Papert, *Perceptrons* (1969)

This historical result spurred research into **multi-layer perceptrons (MLPs)** and **backpropagation**, which can solve XOR by learning non-linear representations.

---

## 📊 Visualization

Decision boundaries are plotted using a fine `meshgrid` with `plt.contourf()`, giving an intuitive color-filled view of the learned hyperplane:

- 🔵 Blue points = Class 0 (output: 0)
- 🔴 Red points = Class 1 (output: 1)
- Background shading = Perceptron's predicted region for each class

---

## 🧪 Key Concepts Demonstrated

| Concept | Details |
|---------|---------|
| **Perceptron Learning Rule** | Online weight update: `w ← w + η·(y−ŷ)·x` |
| **Linear Separability** | Why AND/OR converge but XOR does not |
| **Decision Boundary** | The hyperplane `w·x + b = 0` learned by the perceptron |
| **Zero Initialization** | Weights & bias start at 0; learning rule differentiates them |
| **Step Activation** | Hard threshold — non-differentiable, output ∈ {0, 1} |
| **Perceptron Convergence Theorem** | Finite convergence guaranteed for linearly separable data |
| **XOR Problem** | Classic proof of single-layer perceptron's expressive limits |

---

## ⚙️ Tech Stack

| Library | Purpose |
|---------|---------|
| `numpy` | Weight math, dot products, array ops |
| `matplotlib` | Decision boundary plots |
| `pandas` | Tabular display of prediction results |
| `IPython.display` | Formatted DataFrame rendering in Jupyter |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib pandas notebook
```

### Run the Notebook

```bash
git clone https://github.com/<your-username>/perceptron-from-scratch.git
cd perceptron-from-scratch
jupyter notebook PERCEPTRON_IMPLEMENTATION.ipynb
```

Run all cells top to bottom. The notebook is fully self-contained — no external datasets needed.

---

## 📁 Project Structure

```
perceptron-from-scratch/
│
├── PERCEPTRON_IMPLEMENTATION.ipynb   # Main notebook (full implementation + experiments)
└── README.md                         # Project documentation
```

---

## 💡 Takeaways & Insights

1. **Simplicity of the perceptron rule** — just three lines of math can learn AND and OR from scratch in under 10 epochs.
2. **Zero initialization works** — unlike deep networks where symmetry breaking matters, a single perceptron learns fine from zeros because the error signal introduces asymmetry on the first mistake.
3. **XOR as a teaching tool** — intentionally breaking the perceptron with XOR is the cleanest way to motivate why hidden layers and non-linear activations are necessary.
4. **Online vs. batch learning** — this implementation uses online (sample-by-sample) updates, which is true to the original Rosenblatt perceptron formulation.

---

## 🔭 Extensions & Next Steps

- [ ] **Multi-Layer Perceptron (MLP)** — Add a hidden layer to solve XOR
- [ ] **Sigmoid activation** — Replace step function with sigmoid for differentiability
- [ ] **Backpropagation** — Implement gradient-based training for MLPs
- [ ] **Real dataset** — Apply a perceptron to a binary classification dataset (e.g., Iris, Breast Cancer)
- [ ] **Learning curve plot** — Track misclassification count per epoch to visualize convergence

---

## 📚 References

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review, 65(6), 386–408.
- Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.* MIT Press.
- Nielsen, M. A. (2015). *Neural Networks and Deep Learning.* Determination Press. [Online: neuralnetworksanddeeplearning.com]

---

