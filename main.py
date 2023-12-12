import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from quad_sim import simulate_quadrotor
from quadrotor import Quadrotor
from create_animation import create_animation



# Weights of MPC cost
R  = np.eye(2)
Q  = np.diag([10, 10, 1, 1, 1, 1])
Qf = np.diag([30, 30, 1, 1, 1, 1])


# End time of the simulation
tf = 10

u_d = np.array([9.81/2, 9.81/2])
x_d = np.array([3, 3, 0, 0, 0, 0])

# Construct our quadrotor controller 
quadrotor = Quadrotor(Q, R, Qf, x_d, u_d)

x0 = np.array([0, -0.5, 0, 0, 0, 0])
x, u, t, x_pred = simulate_quadrotor(x0, tf, quadrotor)

anim, fig = create_animation(x, tf, quadrotor.x_d(), x_pred)

writer = PillowWriter(fps=15)
anim.save('demo/MPC.gif', writer=writer)

plt.show()