import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.ticker import MaxNLocator


def f(x):
    return 0.5 * x @ A @ x - b @ x


def create_mesh(f):
    x = np.arange(-5, 5, 0.025)
    y = np.arange(-5, 5, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))

    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = f(np.array([x_coor, y_coor]))

    return X, Y, Z


def plot_contour(ax, X, Y, Z):
    ax.set(title="Path during optimization", xlabel="x1", ylabel="x2")
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize="smaller", fmt="%1.2f")
    ax.axis("square")
    return ax


def plot_conjugateGradient_solution(sol):
    fig, ax = plt.subplots(figsize=(6, 6))
    X, Y, Z = create_mesh(f)
    ax = plot_contour(ax, X, Y, Z)
    ax.plot(sol[:, 0], sol[:, 1], linestyle="--", marker="o", color="orange")
    ax.plot(sol[-1, 0], sol[-1, 1], "ro")
    plt.show()


def plot_conjugateGradientNL_solution(sol_x, sol_y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Conjugate Non-Linear Gradient")

    ax1.plot(sol_x[:, 0], sol_x[:, 1], linestyle="--", marker="o", color="orange")
    ax1.plot(sol_x[-1, 0], sol_x[-1, 1], "ro")
    ax1.set(title="Path During Optimization", xlabel="x1", ylabel="x2")
    
    x = np.arange(-5, 5, 0.025)
    y = np.arange(-5, 5, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, fontsize="smaller", fmt="%1.2f")
    ax1.axis("square")

    ax2.plot(sol_y, linestyle="--", marker="o", color="orange")
    ax2.plot(len(sol_y) - 1, sol_y[-1], "ro")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title="OF Value During Optimization", xlabel="Iterations", ylabel="OF Value"
    )
    ax2.legend(["Armijo line search algorithm"])

    plt.tight_layout()
    plt.show()


def conjugateGradient(A, b, x_0, eps=1e-6):
    x_k = x_0
    r_k = A @ x_k - b
    d_k = -r_k
    r_kr_k = np.dot(r_k, r_k)
    r_k_norm = np.linalg.norm(r_k)
    curve_x = [x_k]
    count = 0

    while r_k_norm > eps:
        alpha = (r_k @ r_k) / (d_k @ A @ d_k)
        x_k = x_k + alpha * d_k
        r_k = r_k + alpha * A @ d_k
        beta = (r_k @ r_k) / r_kr_k
        d_k = -r_k + beta * d_k

        count += 1
        curve_x.append(x_k)
        r_k_norm = np.linalg.norm(r_k)

    return np.array(curve_x)


def armijoLineSearch(F, F_nabla, x_k, alpha, d, y, rho=0.5, eps=1e-4):
    dy = F_nabla(x_k).T @ d
    y_k = F(x_k + alpha * d)

    while not y_k <= y + eps * alpha * dy:
        alpha = rho * alpha
        y_k = F(x_k + alpha * d)

    return alpha, y_k


# we use Armijo-Goldstein-condition for this algorithm
def conjugateGradientNL(F, F_nabla, x_0, alpha=1.0, eps=1e-6, max_iter=100):
    x = x_0
    y = F(x)
    dF = F_nabla(x)
    dF_norm = np.linalg.norm(dF)
    d = -dF

    count = 0
    curve_x = [x]
    curve_y = [y]

    while dF_norm > eps and count < max_iter:
        alpha, y_k = armijoLineSearch(F, F_nabla, x, alpha, d, y)
        x_k = x + alpha * d
        dF_k = F_nabla(x_k)

        beta = (dF_k @ dF_k) / (dF @ dF)
        error = y - y_k
        x = x_k
        y = y_k
        dF = dF_k
        d = -dF + beta * d
        dF_norm = np.linalg.norm(dF)

        if dF_norm <= eps:
            print("Anzahl der Iterationen für eine Genauigkeit von 1e-6: " + str(count))

        count += 1
        curve_x.append(x)
        curve_y.append(y)

    return np.array(curve_x), np.array(curve_y)


# symmetric, positive definite matrix A
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
x_0 = np.array([-2, 2])

sol = conjugateGradient(A, b, x_0)
plot_conjugateGradient_solution(sol)

# function and its gradient
F = lambda x: (x[0] + x[1]) ** 2 + 3 * np.sin(x[1])
F_nabla = lambda x: np.array([2 * (x[0] + x[1]), 2 * (x[0] + x[1]) + 3 * np.cos(x[1])])
x_0 = np.array([-5, -5])
alpha_0 = 1

sol_x, sol_y = conjugateGradientNL(F, F_nabla, x_0, alpha_0)
plot_conjugateGradientNL_solution(sol_x, sol_y)
