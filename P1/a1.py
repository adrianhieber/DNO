# author Adrian Hieber

import matplotlib.pyplot as plt
import numpy as np

# funtions
f1 = lambda x, y: 5 * (x ** 2) - 6 * x * y + 5 * (y ** 2)
f2 = lambda x, y: 100 * (y - (x ** 2)) ** 2 + (1 - x) ** 2


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
    z_f1 = f1(x_mg, y_mg)
    z_f2 = f2(x_mg, y_mg)

    # Plot the surface
    ax_f1_3d.plot_surface(x_mg, y_mg, z_f1)
    ax_f1_3d.set(xlabel="x", ylabel="y", zlabel="f1(x,y)", title="F1 3d")

    ax_f1_2d_x.plot(x, f1(x, y))
    ax_f1_2d_x.set(xlabel="x", ylabel="f1(x,y)", title="F1 2d x-axis")

    ax_f1_2d_y.plot(y, f1(x, y))
    ax_f1_2d_y.set(xlabel="y", ylabel="f1(x,y)", title="F1 2d y-axis")

    ax_f2_3d.plot_surface(x_mg, y_mg, z_f2)
    ax_f2_3d.set(xlabel="x", ylabel="y", zlabel="f1(x,y)", title="F2 3d")

    ax_f2_2d_x.plot(x, f2(x, y))
    ax_f2_2d_x.set(xlabel="x", ylabel="f2(x,y)", title="F2 2d x-axis")

    ax_f2_2d_y.plot(y, f2(x, y))
    ax_f2_2d_y.set(xlabel="y", ylabel="f2(x,y)", title="F2 2d y-axis")

    fig.tight_layout()
    plt.show()

def check(F, F_nabla , x_0):
	if not callable(F):
		raise Exception("F must be callable")
	if not callable(F_nabla):
		raise Exception("F_nabla must be callable")
	if len(x_0)!=2:
		raise Exception("X_0 must be 2-dim vector")
	#TODO check for numbers important (a,sig,eps)?


def gradientDescent(F, F_nabla , x_0, a_0, sig, eps ):
	check(F, F_nabla , x_0)
	


if __name__ == "__main__":
    plot()
    
