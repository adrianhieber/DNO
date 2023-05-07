# Authors: Adrian Hieber, Luciano Melodia

import matplotlib.pyplot as plt
import numpy as np
import random as rd

from matplotlib import cm

# funtions
f1 = lambda x: 5 * (x[0] ** 2) - 6 * x[0] * x[1] + 5 * (x[1] ** 2)
f2 = lambda x: 100 * (x[1] - (x[0] ** 2)) ** 2 + (1 - x[0]) ** 2

# gradients
f1_nabla = lambda x: [10 * x[0] + -6 * x[1], -6 * x[0] + 10 * x[1]]
f2_nabla = lambda x: [400 * (x[0] ** 3 - x[0] * x[1]), 200 * (x[1] - x[0] ** 2)]

# norm function
norm = lambda f: np.sqrt(f[0] ** 2 + f[1] ** 2)


# Plot two and three dimensional graphs
def plotGraphs2D3D():
    fig = plt.figure()

    ax_f1_3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_f1_2d_x = fig.add_subplot(2, 3, 2)
    ax_f1_2d_y = fig.add_subplot(2, 3, 3)

    ax_f2_3d = fig.add_subplot(2, 3, 4, projection="3d")
    ax_f2_2d_x = fig.add_subplot(2, 3, 5)
    ax_f2_2d_y = fig.add_subplot(2, 3, 6)

    # Make data
    x = np.linspace(-10, 10)
    y = np.linspace(-10, 10)
    x_mg, y_mg = np.meshgrid(x, y)
    z_f1 = f1([x_mg, y_mg])
    z_f2 = f2([x_mg, y_mg])

    # Plot the surface
    ax_f1_3d.plot_surface(x_mg, y_mg, z_f1)
    ax_f1_3d.set(xlabel="x", ylabel="y", zlabel="f1(x,y)", title="F1 3d")

    ax_f1_2d_x.plot(x, f1([x, y]))
    ax_f1_2d_x.set(xlabel="x", ylabel="f1(x,y)", title="F1 2d x-axis")

    ax_f1_2d_y.plot(y, f1([x, y]))
    ax_f1_2d_y.set(xlabel="y", ylabel="f1(x,y)", title="F1 2d y-axis")

    ax_f2_3d.plot_surface(x_mg, y_mg, z_f2)
    ax_f2_3d.set(xlabel="x", ylabel="y", zlabel="f1(x,y)", title="F2 3d")

    ax_f2_2d_x.plot(x, f2([x, y]))
    ax_f2_2d_x.set(xlabel="x", ylabel="f2(x,y)", title="F2 2d x-axis")

    ax_f2_2d_y.plot(y, f2([x, y]))
    ax_f2_2d_y.set(xlabel="y", ylabel="f2(x,y)", title="F2 2d y-axis")

    fig.tight_layout()
    plt.show()


# check user input
def check(F, F_nabla, x_0):
    if not callable(F):
        raise Exception("F must be callable")
    if not callable(F_nabla):
        raise Exception("F_nabla must be callable")
    if len(x_0) != 2:
        raise Exception("X_0 must be 2-dim vector")


def checkBlocks(x_0, s):
    if s[0] < 0 or s[0] > s[1]:
        raise Exception("Invalid length of coordinate blocks.")
    if s[1] > len(x_0):
        raise Exception("Length of coordinate blocks is too large.")


# plot gradient
def plotter(val):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Make the grid
    x, y, z = (
        val[0][0 : len(val[0]) - 1],
        val[1][0 : len(val[0]) - 1],
        val[2][0 : len(val[0]) - 1],
    )

    xl = np.linspace(-10, 10)
    yl = np.linspace(-10, 10)
    x_mg, y_mg = np.meshgrid(xl, yl)
    z_f1 = f1([x_mg, y_mg])

    # Make the direction data for the arrows
    u = val[0][1 : len(val[0])]
    v = val[1][1 : len(val[0])]
    w = val[2][1 : len(val[0])]

    ax.quiver(x, y, z, u, v, w, color="red", normalize=True, length=3)
    ax.plot_surface(x_mg, y_mg, z_f1, cmap=cm.coolwarm, antialiased=False)
    plt.show()


# Gradientenastiegsverfahren
def gradientDescent(F, F_nabla, x_0, a_0=0.01, sig=0.01, eps=0.01):
    check(F, F_nabla, x_0)

    # Initialisierung
    a_k = a_0
    x_k = x_0
    x_k1 = x_0
    f_x_k1 = float("inf")
    memory = [[], [], []]

    # Iteriere bis gewünschte Genauigkeit erreicht ist
    while abs(f_x_k1 - F(x_k)) > eps:
        while F([x_k[i] - a_k * F_nabla(x_k)[i] for i in range(len(x_0))]) / norm(
            F_nabla(x_k)
        ) > F(x_k):
            # Verringere Schrittweise um Faktor sig
            a_k = sig * a_k

        # Update in Richtung des größten Gradientenabstiegs
        x_k = x_k1
        x_k1 = x_k - ([a_k * F_nabla(x_k)[i] for i in range(len(x_0))]) / norm(
            F_nabla(x_k)
        )
        f_x_k1 = F(x_k1)

        memory[0].append(x_k1[0])
        memory[1].append(x_k1[1])
        memory[2].append(f_x_k1)

    print(f"End with x={x_k1} and F(x)={F(x_k1)}")
    plotter(memory)

    # Ausgabe des letzten Punktes
    return x_k1, F(x_k1)


# Koordinatenabstiegsverfahren
def coordinateDescent(F, F_nabla, x_0, a_0=0.01, sig=0.01, eps=0.01, s=[0, 2]):
    check(F, F_nabla, x_0)
    checkBlocks(x_0, s)

    # Initialisierung
    a_k = a_0
    x_k, x_k1 = x_0, x_0
    s1, s2 = s[0], s[1]
    f_x_k1 = float("inf")

    # Iteriere bis gewünschte Genauigkeit erreicht ist
    while abs(f_x_k1 - F(x_k)) > eps:
        while F([x_k[i] - a_k * F_nabla(x_k)[i] for i in range(len(x_0))]) / norm(
            F_nabla(x_k)
        ) > F(x_k):
            # Verringere Schrittweise um Faktor sig
            a_k = sig * a_k

        # Update in Richtung des größten Gradientenabstiegs
        x_k = x_k1
        x_k1 = x_k - ([a_k * F_nabla(x_k)[i] for i in range(s1, s2)]) / norm(
            F_nabla(x_k)
        )
        f_x_k1 = F(x_k1)

    print(f"End with x={x_k1} and F(x)={F(x_k1)}")

    # Ausgabe des letzten Punktes
    return x_k1, F(x_k1)


def stochasticGradientDescent(F, F_nabla, x_0, a_0=0.01, sig=0.01, eps=0.01):
    check(F, F_nabla, x_0)

    # Initialisierung
    a_k = a_0
    x_k, x_k1 = x_0, x_0
    f_x_k1 = float("inf")
    memory = [[], [], []]

    # Iteriere bis gewünschte Genauigkeit erreicht ist
    while abs(f_x_k1 - F(x_k)) > eps:
        while F([x_k[i] - a_k * F_nabla(x_k)[i] for i in range(len(x_0))]) / norm(
            F_nabla(x_k)
        ) > F(x_k):
            # Verringere Schrittweise um Faktor sig
            a_k = sig * a_k

        # Update in Richtung des größten Gradientenabstiegs
        s1 = rd.randrange(0, len(x_0) - 1)
        s2 = rd.randrange(s1 + 1, len(x_0) + 1)
        x_k = x_k1
        x_k1 = x_k - ([a_k * F_nabla(x_k)[i] for i in range(s1, s2)]) / norm(
            F_nabla(x_k)
        )
        f_x_k1 = F(x_k1)

        memory[0].append(x_k1[0])
        memory[1].append(x_k1[1])
        memory[2].append(f_x_k1)

    print(f"End with x={x_k1} and F(x)={F(x_k1)}")
    plotter(memory)

    # Ausgabe des letzten Punktes
    return x_k1, F(x_k1)


if __name__ == "__main__":
    # Plot two dimensional and three dimensional graph of functions
    plotGraphs2D3D()

    f1_values = [[-5, 5], [-3, 2]]
    f2_values = [[0, 3], [2, 1]]

    for v in f1_values:
        gradientDescent(F=f1, F_nabla=f1_nabla, x_0=v)
        coordinateDescent(F=f1, F_nabla=f1_nabla, x_0=v)
        stochasticGradientDescent(F=f1, F_nabla=f1_nabla, x_0=v)
    for v in f2_values:
        gradientDescent(F=f2, F_nabla=f2_nabla, x_0=v)
        coordinateDescent(F=f2, F_nabla=f2_nabla, x_0=v)
        stochasticGradientDescent(F=f2, F_nabla=f2_nabla, x_0=v)
