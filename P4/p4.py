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
        tk = k*tau
        u_tk = u_tk + tau * F(tk, u_tk)
        u.append(u_tk)
        
    return u
    
    
def imEuler():
    #TODO
    pass


#test functions  
a_du = lambda u_t: np.array([[0,1],[-1,0]]) * u_t
a_u0 = np.array([0,1])

b_du = lambda u_t: np.array([[0.5,-1],[1,-1]]) * u_t
b_u0 = np.array([0,1])

c_du = lambda u_t: np.array([np.sqrt(u_t[1]), -2 * u_t[0] * np.sqrt(u_t[1])])
c_u0 = np.array([0,1])
        
        
def tester():
    steps = 100
    tau = 0.1
    
    #a
    a_F = lambda t,u_t: a_du(u_t) 
    #TODO a_an_u = ...
    a_ex_u = exEuler(a_u0, a_F, tau, steps)
    #TODO a_im_u = imEuler(...)
    
    #b 
    b_F = lambda t,u_t: b_du(u_t) 
    #TODO b_an_u = ...
    b_ex_u = exEuler(b_u0, b_F, tau, steps)
    #TODO b_im_u = imEuler(...)
    
    #c
    c_F = lambda t,u_t: c_du(u_t) 
    c_an_u = [ np.array([np.sin(t), np.cos(t)**2]) for t in range(steps)]
    c_ex_u = exEuler(c_u0, c_F, tau, steps)
    #TODO c_im_u = imEuler(...)
    plot("Aufgabe c", c_an_u, c_ex_u, steps)
    
def plot(title, an, ex, steps):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    t=np.array(list(range(steps)))
    x_an=np.array([an[t][0] for t in range(steps)])
    y_an=np.array([an[t][1] for t in range(steps)])
    data_an = np.array([x_an, y_an, t])
    
    x_ex=np.array([ex[t][0] for t in range(steps)])
    y_ex=np.array([ex[t][1] for t in range(steps)])
    
    ax.plot(x_ex, y_ex, t, label="instant ex")
    #ax.plot(x_an, y_an, t, label="instant an")
    

    
    def update(num, data, line):
        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    line_an, = ax.plot(data_an[0, 0:1], data_an[1, 0:1], data_an[2, 0:1], label="An")

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 100.0])
    ax.set_zlabel('t')

    ani = animation.FuncAnimation(fig, update, steps, fargs=(data_an, line_an), interval=10000/steps, blit=False)
    #ani.save('matplot003.gif', writer='imagemagick')
    plt.show()
    
    
if __name__ == "__main__":
    tester()
    
