from typing import Callable, List, Tuple

import numpy as np

from comp_math.linear_algebra.objects.vector import Vector
from comp_math.ode.implicit.base_rosenbrock_solver import BaseRosenbrockSolver


class Rosenbrock1Solver(BaseRosenbrockSolver):
    """Метод Розенброка 1-го порядка (неявный Эйлер)"""
    
    def __init__(self):
        super().__init__()
        self.order = 1
        self.gamma = 1.0
        self.a = [[0.0]]
        self.b = [1.0]
        self.alpha = [1.0]
        self.gamma_matrix = [[0.0]]


class Rosenbrock2Solver(BaseRosenbrockSolver):
    """Метод Розенброка 2-го порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 2
        self.gamma = 1.0 - np.sqrt(2)/2
        
        self.a = [
            [0.0, 0.0],
            [2.0, 0.0]
        ]
        
        self.b = [1.0, 1.0]
        self.alpha = [0.0, 1.0]
        
        self.gamma_matrix = [
            [0.0, 0.0],
            [0.0, 0.0]
        ]


class Rosenbrock3Solver(BaseRosenbrockSolver):
    """Метод Розенброка 3-го порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 3
        self.gamma = 0.435866521508459
        
        self.a = [
            [0.0, 0.0, 0.0],
            [0.4358665215, 0.0, 0.0],
            [0.2820667392, 0.2820667392, 0.0]
        ]
        
        self.b = [1.208496649, -0.644363171, 1.435866522]
        self.alpha = [0.0, 0.4358665215, 1.0]
        
        self.gamma_matrix = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]


class Rosenbrock4Solver(BaseRosenbrockSolver):
    """Метод Розенброка 4-го порядка"""
    
    def __init__(self):
        super().__init__()
        self.gamma = 0.57282
        
        self.a = [
            [],
            [2.0],
            [4.0/3.0, 2.0/3.0]
        ]
        
        self.b = [2.0/3.0, 1.0/3.0, 1.0/3.0]
        self.alpha = [0.0, 1.0, 0.5]
        
        self.gamma_matrix = [
            [],
            [-5.0],
            [1.0, 1.0]
        ]
        
        self.order = 4
    