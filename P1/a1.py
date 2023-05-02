# author Adrian Hieber

import matplotlib.pyplot as plt
import numpy as np

# funtions
f1 = lambda x: 5 * (x[0] ** 2) - 6 * x[0] * x[1] + 5 * (x[1] ** 2)
f2 = lambda x: 100 * (x[1] - (x[0] ** 2)) ** 2 + (1 - x[0]) ** 2
f1_nabla = lambda x: [10 * x[0] + -6 * x[1], -6 * x[0] + 10 * x[1]]
f2_nabla = lambda x: [400 * (x[0] ** 3 - x[0] * x[1]), 200 * (x[1] - x[0] ** 2)]

norm = lambda f: np.sqrt(f[0] ** 2 + f[1] ** 2)


# plot all graphs
def plot():
    # init
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
    # TODO check for numbers important (a,sig,eps)?


# plot gradient
def plotter(val):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Make the grid
    # TODO not just 20, just for test
    x, y, z = val[0][0:20], val[1][0:20], val[2][0:20]

    xl = np.linspace(-10, 10)
    yl = np.linspace(-10, 10)
    x_mg, y_mg = np.meshgrid(xl, yl)
    z_f1 = f1([x_mg, y_mg])

    # Make the direction data for the arrows
    u = val[0][1:21]
    v = val[1][1:21]
    w = val[2][1:21]

    ax.quiver(x, y, z, u, v, w, color="Red", normalize=True, length=3)
    ax.plot_surface(x_mg, y_mg, z_f1)
    plt.show()


# excs a1 i)
def gradientDescent(F, F_nabla, x_0, a_0=2, sig=0.2, eps=0.01):
    check(F, F_nabla, x_0)

    # init
    a_k = a_0
    x_k = x_0
    x_k1 = x_0
    f_x_k1 = float("inf")
    j = 0
    maxi = 50  # TODO just for test, no maxi!
    plot = [np.empty(maxi) for i in range(3)]

    # iterate til eps
    while abs(f_x_k1 - F(x_k)) > eps and j < maxi:
        # print(abs(f_x_k1 - F(x_k)))
        plot[0][j] = x_k[0]
        plot[1][j] = x_k[1]
        plot[2][j] = F(x_k)
        j = j + 1
        # print(abs(f_x_k1 - F(x_k)))
        while F([x_k[i] - a_k * F_nabla(x_k)[i] for i in range(2)]) / norm(
            F_nabla(x_k)
        ) > F(x_k):
            # adapt with sigma
            a_k = sig * a_k
        # update with largest
        x_k = x_k1
        x_k1 = x_k - ([a_k * F_nabla(x_k)[i] for i in range(2)]) / norm(F_nabla(x_k))
        f_x_k1 = F(x_k1)

    print(f"End with x={x_k} and F(x)={F(x_k)}")
    plotter(plot)


if __name__ == "__main__":
    # plot()
    gradientDescent(F=f1, F_nabla=f1_nabla, x_0=[-5, 5])
