import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


def exEuler(u_0, F, tau, steps):
    if tau <= 0:
        raise Exception("tau has to be larger than 0")

    u = []
    u_tk = u_0
    u.append(u_tk)

    for k in range(1, steps):
        # print(f"u_tk: {u_tk}")
        tk = k * tau
        # print("tk:",tk)
        # F=F(tk, u_tk)
        Fau = F(tk, u_tk)
        # cprint(f"F: {Fau}")
        u_tk = u_tk + tau * Fau
        u.append(u_tk)

    return u


def imEuler():
    # TODO
    pass


# test functions
a_du = lambda u_t: np.array([[0, 1], [-1, 0]]) @ u_t
a_u0 = np.array([0, 1])

b_du = lambda u_t: np.array([[0.5, -1], [1, -1]]) @ u_t
b_u0 = np.array([0, 1])

c_du = lambda u_t: np.array([np.sqrt(u_t[1]), -2 * u_t[0] * np.sqrt(u_t[1])])
c_u0 = np.array([0, 1])


def tester():
    steps = 100
    tau = 0.1

    # a
    a_F = lambda t, u_t: a_du(u_t)
    a_an_u = np.array([np.array([np.sin(t), np.cos(t)]) for t in range(steps)])
    a_ex_u = exEuler(a_u0, a_F, tau, steps)
    # TODO a_im_u = imEuler(...)
    plot("Aufgabe a", a_an_u, a_ex_u, steps)

    # b
    b_F = lambda t, u_t: b_du(u_t)
    b_an_u = np.array(
        [
            np.array(
                [
                    -4 * np.sin(np.sqrt(7) * t / 4) / (np.sqrt(7) * np.exp(t / 4)),
                    (
                        np.cos(np.sqrt(7) * t / 4)
                        - 3 * np.sin(np.sqrt(7) * t / 4) / np.sqrt(7)
                    )
                    / np.exp(t / 4),
                ]
            )
            for t in range(steps)
        ]
    )
    b_ex_u = exEuler(b_u0, b_F, tau, steps)
    # TODO b_im_u = imEuler(...)
    plot("Aufgabe b", b_an_u, b_ex_u, steps)

    # c
    c_F = lambda t, u_t: c_du(u_t)
    c_an_u = np.array([np.array([np.sin(t), np.cos(t) ** 2]) for t in range(steps)])
    c_ex_u = exEuler(c_u0, c_F, tau, steps)
    # TODO c_im_u = imEuler(...)
    plot("Aufgabe c", c_an_u, c_ex_u, steps)


def plot(title, an, ex, steps):
    fig = plt.figure(label=title)
    ax = fig.add_subplot(projection="3d")

    t = np.array(list(range(steps)))
    x_an = np.array([an[t][0] for t in range(steps)])
    y_an = np.array([an[t][1] for t in range(steps)])
    data_an = np.array([x_an, y_an, t])

    x_ex = np.array([ex[t][0] for t in range(steps)])
    y_ex = np.array([ex[t][1] for t in range(steps)])
    data_ex = np.array([x_ex, y_ex, t])

    (line_an,) = ax.plot(
        data_an[0, 0:1], data_an[1, 0:1], data_an[2, 0:1], label="Analytisch"
    )
    (line_ex,) = ax.plot(
        data_ex[0, 0:1], data_ex[1, 0:1], data_ex[2, 0:1], label="Explizit"
    )

    def update(num, data, line):
        line_ex.set_data(data_ex[:2, :num])
        line_ex.set_3d_properties(data_ex[2, :num])

        line_an.set_data(data_an[:2, :num])
        line_an.set_3d_properties(data_an[2, :num])

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel("X")

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel("Y")

    ax.set_zlim3d([0.0, 100.0])
    ax.set_zlabel("t")
    ax.legend()

    ani = animation.FuncAnimation(
        fig, update, steps, fargs=(data_an, line_an), interval=10000 / steps, blit=False
    )
    plt.show()


if __name__ == "__main__":
    tester()
