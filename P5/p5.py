import numpy as np


def imEuler(u_0, F, tau, steps):
    if tau <= 0:
        raise Exception("tau has to be larger than 0")

    u = []
    u_tk = u_0
    u.append(u_tk)

    for k in range(1, steps):
        tk = k * tau
        tosolve = lambda u_tk1: u_tk1 - u_tk - tau * F(tk, u_tk1)
        u_tk = fsolve(tosolve, np.array([0, 0]))
        u.append(u_tk)

    return u


# x'_k
norm = lambda a: np.sqrt((a[0]) ** 2 + (a[1]) ** 2)

x_der = lambda t, F: [
    sum(
        [
            F(norm(x[j](t) - x[k](t))) * (x[j](t) - x[k](t)) / abs(x[j](t) - x[k](t))
            for j in range(N)
            if j != k
        ]
    )
    / (-N)
    for k in range(N)
]

# Interaktion der Tier
a = b = np.NaN
check_F = lambda a, b: a > 0 and -np.tanh(a) < b and b < 1
F = lambda r: np.tanh((1 - r) * a) + b

# Stammfkt von F
G = lambda a, b: b * x - np.lpg(np.cosh(a * (x - 1))) / a

# Gesamtenergie
Eps = lambda x, t: sum([G(norm(x[j](t) - x[k](t))) for j in range(N) if j != k]) / (
    N**2
)

# Schwerpunkt
S = lambda x, t: sum([x[j] for j in range(N)]) / N
