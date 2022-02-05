import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit

def get_solution(c:np.array, A: np.array, h: np.array) -> np.array:
    num_of_hours = A.shape[1]
    work_intervals = list(range(4,9))
    num_of_intervals = A.shape[0]
    
    solver = pywraplp.Solver.CreateSolver('GLOP')
    infinity = solver.infinity()

    #create vars
    x = []
    for i in range(num_of_intervals):
        x.append(solver.IntVar(0.0, infinity, 'x'))

    #constrains
    ct = []
    for j in range(num_of_hours):
        ct.append(solver.Constraint(h[j], infinity, f'ct_{j}'))
        for i in range(num_of_intervals):
            ct[j].SetCoefficient(x[i], A[i,j])

    #objective
    objective = solver.Objective()
    for i in range(num_of_intervals):
        objective.SetCoefficient(x[i], c[i])
    objective.SetMinimization()

    #optimize
    solver.Solve()

    #get vars
    x_vars = np.zeros(num_of_intervals)
    for i in range(num_of_intervals):
        x_vars[i] = x[i].solution_value()
        
    return x_vars
