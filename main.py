import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from quad_sim import simulate_quadrotor
from quadrotor import Quadrotor
from create_animation import create_animation


R = np.eye(2)
# Q = np.diag([10, 10, 1, 10, 10, 10])
Q = np.diag([10, 10, 1, 1, 1, 1])
Qf = Q
tf = 30
x_d = np.array([16, 14, 0, 0, 0, 0])

quadrotor = Quadrotor(x_d, Q, R, Qf)


x0 = np.array([0.5, 0.5, 0, -1, -1, 0])
x, u, t, x_pred = simulate_quadrotor(x0, tf, quadrotor)


anim, fig = create_animation(x, x_d, x_pred, tf)

writer = PillowWriter(fps=15)
anim.save('demo/MPC.gif', writer=writer)

plt.show()
