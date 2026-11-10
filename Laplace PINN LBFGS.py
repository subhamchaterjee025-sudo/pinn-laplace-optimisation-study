
# ============================================================
# LAPLACE EQUATION SOLVER USING A PHYSICS-INFORMED NEURAL NETWORK
#
# PDE: u_xx + u_yy = 0
# Domain: [0,1] x [0,1]
#
# Boundary conditions:
# u(0,y) = 0
# u(1,y) = 0
# u(x,0) = 0
# u(x,1) = sin(pi*x)
#
# Neural network:
# 2 -> 32 -> 32 -> 32 -> 1
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. REPRODUCIBILITY AND DEVICE
# ------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ------------------------------------------------------------
# 2. GENERATE TRAINING DATA
# ------------------------------------------------------------

N_interior = 8000
N_edge = 500

# Interior collocation points
X_interior = torch.rand(
    N_interior, 2, device=device
)

# Boundary parameter
s = torch.linspace(
    0, 1, N_edge, device=device
).reshape(-1, 1)

# Four boundary coordinate sets

left = torch.cat(
    [torch.zeros_like(s), s], dim=1
)

right = torch.cat(
    [torch.ones_like(s), s], dim=1
)

bottom = torch.cat(
    [s, torch.zeros_like(s)], dim=1
)

top = torch.cat(
    [s, torch.ones_like(s)], dim=1
)

# Combine boundary coordinates
X_boundary = torch.cat(
    [left, right, bottom, top], dim=0
)

# Corresponding boundary values

zeros = torch.zeros_like(s)

top_values = torch.sin(torch.pi * s)

U_boundary = torch.cat(
    [zeros, zeros, zeros, top_values],
    dim=0
)

print("Interior shape:", X_interior.shape)
print("Boundary shape:", X_boundary.shape)
print("Boundary values shape:", U_boundary.shape)


# ------------------------------------------------------------
# 3. DEFINE NEURAL NETWORK
# ------------------------------------------------------------

class LaplacePINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 1)

        )

    def forward(self, X):
        return self.network(X)


model = LaplacePINN().to(device)


# ------------------------------------------------------------
# 4. CALCULATE LAPLACIAN / PDE RESIDUAL
# ------------------------------------------------------------

def calculate_residual(model, X):

    # Create independent differentiable coordinates
    x = X[:, 0:1].detach().clone().requires_grad_(True)
    y = X[:, 1:2].detach().clone().requires_grad_(True)

    coordinates = torch.cat([x, y], dim=1)

    # Forward pass
    u = model(coordinates)

    # First derivatives
    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_y = torch.autograd.grad(
        outputs=u,
        inputs=y,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivatives
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_yy = torch.autograd.grad(
        outputs=u_y,
        inputs=y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0]

    # Laplace equation residual
    residual = u_xx + u_yy

    return residual


# ------------------------------------------------------------
# 5. DEFINE LOSS FUNCTION
# ------------------------------------------------------------

lambda_bc = 10.0

def calculate_loss(model):

    # PDE loss
    residual = calculate_residual(
        model, X_interior
    )

    loss_pde = torch.mean(residual**2)

    # Boundary loss
    u_pred_boundary = model(X_boundary)

    loss_boundary = torch.mean(
        (u_pred_boundary - U_boundary)**2
    )

    # Total loss
    total_loss = (
        loss_pde + lambda_bc * loss_boundary
    )

    return total_loss, loss_pde, loss_boundary


# ------------------------------------------------------------
# 6. OPTIMISATION USING L-BFGS
# ------------------------------------------------------------

# Number of outer L-BFGS steps.
# Note: one outer step can evaluate the closure multiple times
# because the line search may test several step lengths.
iterations = 10000

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=1,
    max_eval=20,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    history_size=50,
    line_search_fn="strong_wolfe"
)

# Store loss history
history_total = []
history_pde = []
history_bc = []


def closure():
    """
    L-BFGS repeatedly calls this function while choosing
    a parameter update. It must recompute the same objective
    on the same fixed collocation and boundary points.
    """
    optimizer.zero_grad(set_to_none=True)

    total_loss, loss_pde, loss_bc = calculate_loss(model)

    total_loss.backward()

    return total_loss


print("\\nStarting L-BFGS training...\\n")

for iteration in range(iterations):

    # Performs one outer quasi-Newton step.
    # The strong-Wolfe line search may call closure multiple times.
    optimizer.step(closure)

    # Re-evaluate losses at the UPDATED parameters for logging.
    # Gradients are enabled because calculate_residual uses autograd
    # to compute spatial derivatives, even though we do not call backward.
    total_loss, loss_pde, loss_bc = calculate_loss(model)

    total_value = total_loss.detach().item()
    pde_value = loss_pde.detach().item()
    bc_value = loss_bc.detach().item()

    history_total.append(total_value)
    history_pde.append(pde_value)
    history_bc.append(bc_value)

    if iteration % 500 == 0 or iteration == iterations - 1:
        print(
            f"Iteration: {iteration + 1:5d} | "
            f"Total Loss: {total_value:.6e} | "
            f"PDE Loss: {pde_value:.6e} | "
            f"Boundary Loss: {bc_value:.6e}"
        )

print("\\nL-BFGS training completed!")


# ------------------------------------------------------------

# 7. ANALYTICAL SOLUTION
# ------------------------------------------------------------

def exact_solution(x, y):

    denominator = torch.sinh(
        torch.tensor(
            torch.pi,
            dtype=y.dtype,
            device=y.device
        )
    )

    return (
        torch.sinh(torch.pi * y) / denominator
    ) * torch.sin(torch.pi * x)



# ------------------------------------------------------------
# 8. GENERATE INDEPENDENT TEST GRID
# ------------------------------------------------------------

n = 100

x_test = torch.linspace(
    0, 1, n, device=device
)

y_test = torch.linspace(
    0, 1, n, device=device
)

X_grid, Y_grid = torch.meshgrid(
    x_test,
    y_test,
    indexing="ij"
)

test_points = torch.cat(
    [
        X_grid.reshape(-1, 1),
        Y_grid.reshape(-1, 1)
    ],
    dim=1
)


# ------------------------------------------------------------
# 9. PREDICTION AND ERROR
# ------------------------------------------------------------

model.eval()

# No gradient graph needed for solution comparison
with torch.no_grad():

    prediction = model(test_points)

    exact = exact_solution(
        test_points[:, 0:1],
        test_points[:, 1:2]
    )

    absolute_error = torch.abs(
        prediction - exact
    )

# Reshape for contour plotting
pred = prediction.reshape(n, n).cpu().numpy()

true = exact.reshape(n, n).cpu().numpy()

error = absolute_error.reshape(n, n).cpu().numpy()

X_plot = X_grid.cpu().numpy()
Y_plot = Y_grid.cpu().numpy()


# ------------------------------------------------------------
# 10. NUMERICAL ERROR METRICS
# ------------------------------------------------------------

mae = np.mean(error)
max_error = np.max(error)

relative_l2 = (
    np.linalg.norm(pred - true)
    / np.linalg.norm(true)
)

print("\nVALIDATION RESULTS")
print("------------------")
print("Mean Absolute Error:", mae)
print("Maximum Absolute Error:", max_error)
print("Relative L2 Error:", relative_l2)


# ------------------------------------------------------------
# 11. VISUALISE SOLUTION AND ERROR
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3, figsize=(17, 5)
)

# Analytical solution
c1 = axes[0].contourf(
    X_plot, Y_plot, true, levels=40
)
axes[0].set_title("Analytical Solution")
fig.colorbar(c1, ax=axes[0])

# PINN prediction
c2 = axes[1].contourf(
    X_plot, Y_plot, pred, levels=40
)
axes[1].set_title("PINN Prediction")
fig.colorbar(c2, ax=axes[1])

# Absolute error
c3 = axes[2].contourf(
    X_plot, Y_plot, error, levels=40
)
axes[2].set_title("Absolute Error")
fig.colorbar(c3, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. PLOT LOSS CONVERGENCE
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.semilogy(
    history_total,
    label="Total Loss"
)

plt.semilogy(
    history_pde,
    label="PDE Loss"
)

plt.semilogy(
    history_bc,
    label="Boundary Loss"
)

plt.xlabel("L-BFGS outer iteration")
plt.ylabel("Loss (log scale)")
plt.title("PINN Training Convergence (L-BFGS)")

plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# END OF LAPLACE PINN IMPLEMENTATION
# ============================================================
