import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables
from scipy.signal import cont2discrete


class Quadrotor(object):
  def __init__(self, Q, R, Qf):
    self.g = 9.81
    self.m = 1
    self.a = 0.25
    self.I = 0.0625
    self.Q = Q
    self.R = R
    self.Qf = Qf

    # Input limits
    self.umin = 0
    self.umax = 5.5

    self.n_x = 6
    self.n_u = 2
    self.S = np.zeros((6, 6))
    try:
      self.S = np.load('S_sol.npy')
      # For the SOS problem in this homework, we maximized the rho level-set 
      # where rho = 1
      self.rho = 1.0
    except:
      print("Warning: S_sol.npy does not exist. CLF-based controllers (Problem 3) will not work")    
      print("To generate S_sol.npy, please complete Problem 2 and run stability_analysis.py")

    # Set use_experimental_inputs to True to test the CLF QP boundary controller
    # Only works after Problem 3.2.b and 3.3.b are completed
    self.use_experimental_inputs = False

  def x_d(self):
    # Nomial state
    return np.array([1, 1, 0, 0, 0, 0])

  def u_d(self):
    # Nomial input
    return np.array([self.m*self.g/2, self.m*self.g/2])

  def continuous_time_full_dynamics(self, x, u):
    # Dynamics for the quadrotor
    g = self.g
    m = self.m
    a = self.a
    I = self.I

    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]
    u0 = u[0]
    u1 = u[1]

    xdot = np.array([ydot,
                     zdot,
                     thetadot,
                     -sin(theta) * (u0 + u1) / m,
                     -g + cos(theta) * (u0 + u1) / m,
                     a * (u0 - u1) / I])
    return xdot

  def continuous_time_linearized_dynamics(self):
    # Dynamics linearized at the fixed point
    # This function returns A and B matrix

    # A = np.zeros((6,6))
    # A[:3, -3:] = np.identity(3)
    # A[3, 2] = -self.g

    # B = np.zeros((6,2))
    # B[4,0] = 1/self.m
    # B[4,1] = 1/self.m
    # B[5,0] = self.a/self.I
    # B[5,1] = -self.a/self.I


    A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, -cos(self.x_d()[2]) * (self.u_d()[0] + self.u_d()[1]) / self.m, 0, 0, 0],
              [0, 0, -sin(self.x_d()[2]) * (self.u_d()[0] + self.u_d()[1]) / self.m, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [-sin(self.x_d()[2]) / self.m, -sin(self.x_d()[2]) / self.m],
                  [cos(self.x_d()[2]) / self.m, cos(self.x_d()[2]) / self.m],
                  [self.a / self.I, -self.a / self.I]])

    return A, B

  def discrete_time_linearized_dynamics(self, T):
    # Discrete time version of the linearized dynamics at the fixed point
    # This function returns A and B matrix of the discrete time dynamics

    A_c, B_c = self.continuous_time_linearized_dynamics()

    # A_d = np.identity(6) + A_c * T
    # B_d = B_c * T


    C = np.eye(A_c.shape[0])
    D = np.zeros((A_c.shape[0],))
    [A_d, B_d, _, _, _] = cont2discrete((A_c, B_c, C, D), T)

    return A_d, B_d

  def add_initial_state_constraint(self, prog, x, x_current):
    # TODO: impose initial state constraint.
    # Use AddBoundingBoxConstraint
    prog.AddBoundingBoxConstraint(x_current-self.x_d(), x_current-self.x_d(), x[0])

  def add_input_saturation_constraint(self, prog, x, u, N):
    # TODO: impose input limit constraint.
    # Use AddBoundingBoxConstraint
    # The limits are available through self.umin and self.umax
    for i in range(N-1):
      prog.AddBoundingBoxConstraint(self.umin - self.u_d(), self.umax - self.u_d(), u[i])

  def add_dynamics_constraint(self, prog, x, u, N, T):
    # TODO: impose dynamics constraint.
    # Use AddLinearEqualityConstraint(expr, value)
    A, B = self.discrete_time_linearized_dynamics(T)
    for i in range(N-1):
      prog.AddLinearEqualityConstraint(A @ x[i] + B @ u[i] - x[i+1], np.zeros(6))

  def add_cost(self, prog, x, u, N):
    # TODO: add cost.
    for i in range(N-1):
      prog.AddQuadraticCost(u[i] @ self.R @ u[i])
      prog.AddQuadraticCost(x[i] @ self.Q @ x[i])
    prog.AddQuadraticCost(x[N-1] @ self.Qf @ x[N-1])

  def compute_mpc_feedback(self, x_current, use_clf=False):
    '''
    This function computes the MPC controller input u
    '''

    # Parameters for the QP
    N = 10
    T = 0.1

    # Initialize mathematical program and decalre decision variables
    prog = MathematicalProgram()
    x = np.zeros((N, 6), dtype="object")
    for i in range(N):
      x[i] = prog.NewContinuousVariables(6, "x_" + str(i))
    u = np.zeros((N-1, 2), dtype="object")
    for i in range(N-1):
      u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

    # Add constraints and cost
    self.add_initial_state_constraint(prog, x, x_current)
    self.add_input_saturation_constraint(prog, x, u, N)
    self.add_dynamics_constraint(prog, x, u, N, T)
    self.add_cost(prog, x, u, N)

    # Solve the QP
    solver = OsqpSolver()
    result = solver.Solve(prog)

    u_mpc = np.zeros(2)
    # TODO: retrieve the controller input from the solution of the optimization problem
    # and use it to compute the MPC input u
    # You should make use of result.GetSolution(decision_var) where decision_var
    # is the variable you want

    # Begin solution code
    u0 = result.GetSolution(u[0])
    u_mpc = self.u_d() + u0
    # End solution code

    x_series = result.GetSolution(x)
    x_mpc = x_series + self.x_d()

    return u_mpc, x_mpc

  def compute_lqr_feedback(self, x):
    '''
    Infinite horizon LQR controller
    '''
    A, B = self.continuous_time_linearized_dynamics()
    S = solve_continuous_are(A, B, self.Q, self.R)
    K = -inv(self.R) @ B.T @ S
    u = self.u_d() + K @ x
    return u

  def dynamics_cubic_approximation(self, x, u):
    '''
    Approximated Dynamics for the quadrotor.
    We substitute
      sin(theta) = theta - (theta**3)/6
      cos(theta) = 1 - (theta**2)/2
    into the full dynamics.
    '''
    g = self.g
    m = self.m
    a = self.a
    I = self.I

    theta = x[2]
    ydot = x[3]
    zdot = x[4]
    thetadot = x[5]
    u0 = u[0]
    u1 = u[1]

    xdot = np.array([ydot,
                     zdot,
                     thetadot,
                     -(theta - (theta**3)/6) * (u0 + u1) / m,
                     -g + (1 - (theta**2)/2) * (u0 + u1) / m,
                     a * (u0 - u1) / I])
    return xdot

  def closed_loop_dynamics_cubic_approximation(self, x):
    # Closed-loop dynamics with infinite horizon LQR
    u = self.compute_lqr_feedback(x)
    return self.dynamics_cubic_approximation(x, u)
    