import numpy as np

F = (
    lambda x: (x[0] + x[1]) ** 2 + 3 * np.sin(x[1])
    if not np.isinf(x[0]) and not np.isinf(x[1])
    else np.inf
)
nablaF = (
    lambda x: np.array([2 * (x[0] + x[1]), 2 * (x[0] + x[1]) + 3 * np.cos(x[1])])
    if not np.isinf(x[0]) and not np.isinf(x[1])
    else np.inf
)


def BFGS(F, grad, x_0, H_0=np.identity(2), alpha_0=1, sig=0.15, eps=10**-6, plot=True):
    # initialize values
    x_k = x_0
    H_k = H_0
    alpha_k = alpha_0
    p_0 = -np.inner(H_0, grad(x_0))
    x_k1 = x_0 + alpha_0 * p_0

    x_store = []

    while abs(F(x_k1) - F(x_k)) > eps:
        if plot == True:
            x_store.append(F(x_k))

        s_k = x_k1 - x_k
        y_k = grad(x_k1) - grad(x_k)
        rho_k = 1 / y_k.dot(s_k)

        # compute H_k1
        t_2 = H_k @ np.outer((rho_k * y_k), s_k.T)
        t_3 = rho_k * (np.outer(s_k, y_k.T) @ H_k)
        t_4 = rho_k * (np.outer(s_k, y_k.T) @ H_k @ np.outer((rho_k * y_k), s_k.T))
        t_5 = rho_k * np.outer(s_k, s_k.T)
        H_k = H_k - t_2 - t_3 + t_4 + t_5

        # update values
        x_k = x_k1
        p_k = -H_k @ grad(x_k)

        # adapt amount of change
        while (F(x_k + alpha_k * p_k) - F(x_k)) > 0:
            alpha_k = alpha_k * sig

        # update step
        x_k1 = x_k + alpha_k * p_k

        # reset change
        alpha_k = alpha_0

    if plot == True:
        print(x_store)
    return x_k1, F(x_k1)


BFGS(F, nablaF, np.array([-5, -5]))
