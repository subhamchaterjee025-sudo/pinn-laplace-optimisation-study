# PINN Laplace Optimisation Study
### Comparative Analysis of Adam and L-BFGS for Physics-Informed Neural Networks

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![PDE](https://img.shields.io/badge/Mathematics-Partial%20Differential%20Equations-purple)
![Status](https://img.shields.io/badge/Status-Experimental%20Study-orange)

## Overview

This project investigates the application of **Physics-Informed Neural Networks (PINNs)** to the numerical solution of the two-dimensional Laplace equation.

The primary objective is to examine how different optimisation algorithms influence the convergence behaviour and numerical accuracy of PINNs.

Two optimisation algorithms are investigated:

- **Adam:** An adaptive gradient-based optimisation algorithm.
- **L-BFGS:** A limited-memory quasi-Newton optimisation algorithm that incorporates approximate curvature information.

The predicted solutions are validated against an analytical solution, allowing the comparison to extend beyond training loss curves to actual numerical approximation errors.

The project is implemented using Python and PyTorch, with automatic differentiation used to calculate the second-order spatial derivatives required by the governing PDE.

---

## 1. Mathematical Formulation

The governing equation is the two-dimensional Laplace equation:

\[
\boxed{
\frac{\partial^2u}{\partial x^2}
+
\frac{\partial^2u}{\partial y^2}=0
}
\]

over the computational domain:

\[
\Omega=[0,1]\times[0,1].
\]

### Boundary Conditions

The following Dirichlet boundary conditions are imposed:

\[
u(0,y)=0
\]

\[
u(1,y)=0
\]

\[
u(x,0)=0
\]

\[
u(x,1)=\sin(\pi x)
\]

### Analytical Solution

The exact solution for this boundary-value problem is:

\[
\boxed{
u(x,y)=
\frac{\sinh(\pi y)}{\sinh(\pi)}
\sin(\pi x)
}
\]

This analytical expression is used as the reference solution for evaluating the accuracy of the PINN predictions.

---

## 2. Neural Network Architecture

A fully connected feedforward neural network is implemented using PyTorch.

**Architecture:**

```text
Input Layer:       2 neurons (x, y)
Hidden Layer 1:   32 neurons + Tanh
Hidden Layer 2:   32 neurons + Tanh
Hidden Layer 3:   32 neurons + Tanh
Output Layer:      1 neuron (u)
```

The network approximates the solution:

\[
u_\theta(x,y)
\]

where \(\theta\) represents the trainable network parameters.

### Automatic Differentiation

PyTorch's automatic differentiation framework is used to calculate:

\[
u_x,\quad u_y,\quad u_{xx},\quad u_{yy}.
\]

The PDE residual is constructed as:

\[
r_\theta(x,y)=u_{xx}+u_{yy}.
\]

This residual is incorporated into the training objective, allowing the network to learn from the governing differential equation rather than relying exclusively on labelled solution data.

---

## 3. Physics-Informed Loss Function

The total loss consists of two components.

### PDE Loss

\[
L_{\mathrm{PDE}}=
\frac{1}{N_i}
\sum_{j=1}^{N_i}
\left[
u_{xx}(x_j,y_j)+u_{yy}(x_j,y_j)
\right]^2
\]

### Boundary Loss

\[
L_{\mathrm{BC}}=
\frac{1}{N_b}
\sum_{j=1}^{N_b}
\left[
u_\theta(X_j)-U_j
\right]^2
\]

### Total Loss

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{PDE}}+10L_{\mathrm{BC}}
}
\]

The boundary loss weighting coefficient is fixed at 10 for the reported experiments.

---

## 4. Dataset Configuration

The training dataset consists of 10,000 points.

| Dataset component | Number of points |
|---|---:|
| Interior collocation points | 8,000 |
| Left boundary | 500 |
| Right boundary | 500 |
| Bottom boundary | 500 |
| Top boundary | 500 |
| **Total** | **10,000** |

Interior collocation points are used to evaluate the PDE residual, while boundary points enforce the prescribed Dirichlet conditions.

An independent \(100\times100\) evaluation grid is used for numerical validation.

---

## 5. Optimisation Algorithms

### Adam

Adam uses adaptive estimates of the first and second moments of gradients.

Experimental configuration:

| Parameter | Value |
|---|---:|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Epochs | 10,000 |

### L-BFGS

L-BFGS is a quasi-Newton optimisation method that approximates inverse-Hessian information using a limited history of gradient and parameter updates.

Experimental configuration:

| Parameter | Value |
|---|---:|
| Optimiser | L-BFGS |
| Learning-rate parameter | 1.0 |
| Outer iterations | 10,000 |
| History size | 50 |
| Line search | Strong Wolfe |
| Maximum internal iterations | 1 |

**Note:** Adam's learning rate and L-BFGS's step-size parameter are not directly equivalent. L-BFGS also performs potentially multiple closure evaluations during line search.

---

## 6. Experimental Results

### 6.1 L-BFGS: Solution and Error Distribution

The following plots compare the analytical solution, the PINN prediction and their absolute difference.

![L-BFGS Solution and Error](figures/lbfgs_solution_error.png)

### 6.2 Adam: Solution and Error Distribution

![Adam Solution and Error](figures/adam_solution_error.png)

### 6.3 L-BFGS Convergence

The L-BFGS convergence curve demonstrates rapid initial loss reduction followed by an extended plateau.

![L-BFGS Training Curve](figures/lbfgs_convergence.png)

### 6.4 Adam Convergence

Adam exhibits gradual overall loss reduction, accompanied by recurring local oscillations and spikes.

![Adam Training Curve](figures/adam_convergence.png)

---

## 7. Numerical Validation

The predicted solutions are evaluated against the analytical solution using three error metrics.

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{N}\sum_{i=1}^{N}
|\hat u_i-u_i|
\]

**Maximum Absolute Error**

\[
E_{\max}=\max_i|\hat u_i-u_i|
\]

**Relative \(L^2\) Error**

\[
E_{L^2}=
\frac{\|\hat{\mathbf u}-\mathbf u\|_2}
{\|\mathbf u\|_2}
\]

### Quantitative Comparison

| Validation Metric | Adam (0.001) | L-BFGS (1.0) |
|---|---:|---:|
| Mean Absolute Error | 0.0010247309 | **0.00057760574** |
| Maximum Absolute Error | **0.0071077373** | 0.0077121293 |
| Relative \(L^2\) Error | 0.0048162495 | **0.0031214966** |

### Observations

The reported experiments reveal the following:

1. L-BFGS achieved approximately 43.63% lower MAE than Adam.
2. L-BFGS achieved approximately 35.20% lower relative \(L^2\) error.
3. Adam produced a slightly lower maximum absolute error.
4. Adam exhibited gradual convergence with recurring loss oscillations.
5. L-BFGS demonstrated rapid initial convergence followed by an extended plateau.

These results suggest that L-BFGS achieved better average and relative global accuracy in the reported experiments, while Adam retained a small advantage in maximum pointwise accuracy.

The results are specific to the tested configurations and do not establish universal superiority of either optimiser.

---

## 8. Key Questions Investigated

The project developed around several computational and mathematical questions:

- How can automatic differentiation be used to construct a second-order differential operator?
- Why do PINN training losses exhibit oscillations during optimisation?
- How does learning-rate selection influence convergence?
- Does increasing the number of collocation points necessarily improve numerical accuracy?
- How do adaptive gradient methods differ from quasi-Newton methods?
- Why can L-BFGS stagnate under one configuration but converge rapidly under another?
- Why should solution error be evaluated independently of training loss?

These investigations form the basis of the comparative analysis.

---

## 9. Limitations and Future Work

The current study is an initial numerical investigation. Several improvements remain possible:

- Repeat experiments using multiple random seeds.
- Use identical boundary sampling and network initialisation for a controlled optimiser comparison.
- Compare wall-clock training time and objective evaluations.
- Investigate different collocation-point densities.
- Study the relationship between PDE residual minimisation and solution accuracy.
- Explore hybrid optimisation strategies combining L-BFGS and Adam.

---

## 10. Technologies Used

- **Python:** Scientific computing and implementation.
- **PyTorch:** Neural-network construction and automatic differentiation.
- **NumPy:** Numerical operations.
- **Matplotlib:** Convergence curves and contour visualisations.
- **LaTeX:** Mathematical documentation and project reporting.

---

## 11. Repository Structure

```text
pinn-laplace-optimisation-study/
│
├── figures/
│   ├── adam_convergence.png
│   ├── adam_solution_error.png
│   ├── lbfgs_convergence.png
│   └── lbfgs_solution_error.png
│
├── adam_pinn.py
├── lbfgs_pinn.py
│
├── requirements.txt
│
├── report/
│   └── main.tex
│
└── README.md
```

*The filenames above are suggested repository organisation. Adjust them to match the files actually committed.*

---

## 12. Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pinn-laplace-optimisation-study.git

cd pinn-laplace-optimisation-study
```

Install dependencies:

```bash
pip install torch numpy matplotlib
```

Run the Adam and L-BFGS implementations separately:

```bash
python adam_pinn.py
```

```bash
python lbfgs_pinn.py
```

---

## Conclusion

This project provides a computational investigation of Physics-Informed Neural Networks for solving an elliptic partial differential equation.

By comparing Adam and L-BFGS, the study examines how optimisation mechanisms influence convergence behaviour and numerical approximation accuracy.

The analytical solution provides a mathematical reference for validating the network predictions, while the loss histories and error distributions provide complementary perspectives on model performance.

The overall objective is to understand the interaction between differential equations, neural-network approximation and numerical optimisation through a reproducible computational experiment.

**Author:** Subham

**Project Domain:** Scientific Machine Learning · Numerical Analysis · Partial Differential Equations
