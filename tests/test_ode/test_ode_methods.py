import unittest
import numpy as np
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.ode.ode_registry import ODERegistry


class TestAdamsBashforth(unittest.TestCase):
    
    def setUp(self):
        """Создаем все солверы"""
        self.solvers = {
            1: ODERegistry.create_solver("adam1"),
            2: ODERegistry.create_solver("adam2"),
            3: ODERegistry.create_solver("adam3"),
            4: ODERegistry.create_solver("adam4"),
            5: ODERegistry.create_solver("euler"),
            6: ODERegistry.create_solver("kutta"),
            7: ODERegistry.create_solver("heun"),
            8: ODERegistry.create_solver("rk4"),
            9: ODERegistry.create_solver("bdf1"),
            10: ODERegistry.create_solver("bdf2"),
            11: ODERegistry.create_solver("bdf3"),
            12: ODERegistry.create_solver("bdf4"),
            13: ODERegistry.create_solver("gauss_legendre_2"),
            14: ODERegistry.create_solver("gauss_legendre_4"),
            15: ODERegistry.create_solver("rado"),
            16: ODERegistry.create_solver("lobatto")
        }

    def test_linear_equation(self):
        """Тест на линейном уравнении для всех солверов"""
        def f(t, y):
            return y
        
        for order, solver in self.solvers.items():
            t, y = solver.solve(f, 1, (0, 2), [1.0], 0.00001)
            
            y_exact = np.exp(t)
            y_numeric = np.array([y[i][0] for i in range(len(y))])
            error = np.max(np.abs(y_numeric - y_exact))
            
            self.assertLess(error, 1e-4)
    
    def test_system(self):
        """Тест на системе уравнений для всех солверов"""
        def f(t, y_vec):
            return Vector([y_vec[1], -y_vec[0]])
        
        for order, solver in self.solvers.items():
            t, y = solver.solve(f, 2, (0, 2*np.pi), [1.0, 0.0], 0.0001)
            
            x_numeric = np.array([y[i][0] for i in range(len(y))])
            y_numeric = np.array([y[i][1] for i in range(len(y))])
            x_exact = np.cos(t)
            y_exact = -np.sin(t)
            
            error_x = np.max(np.abs(x_numeric - x_exact))
            error_y = np.max(np.abs(y_numeric - y_exact))
            
            self.assertLess(error_x, 1e-3)
            self.assertLess(error_y, 1e-3)