# authors: Adrian Hieber, Luciano Melodia

# imports
import numpy as np

# var
F = lambda x, y: (x + y) ** 2 + 3 * np.sin(y)

# func
def check(f, grad_f, x_0, sig):
    if not callable(F):
        raise Exception("F must be callable")
    if not callable(F_nabla):
        raise Exception("F_nabla must be callable")
    if len(x_0) != 2:
        raise Exception("X_0 must be 2-dim vector")
    if sig <= 0 or sig >= 1:
        raise Exception("0<sigma<1")


def bfgs(f, grad_f, x_0, H, alp, sig):
    check(f, grad_f, x_0, sig)  # TODO more checks
    pass


def test():
    bfgs(F, grad_F, [-5, 5], 1)
