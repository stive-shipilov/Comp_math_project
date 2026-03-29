import numpy as np
from typing import Callable, Tuple
from comp_math.bvp.base_bvp_solver import BaseBVPSolver
from comp_math.bvp.model.first_order_system import FirstOrderSystem
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.linear_algebra.sla_solvers.SLA_solvers_registry import SLASolverRegistry
from comp_math.ode.ode_registry import ODERegistry


class ShootingSolver(BaseBVPSolver):
    """Метод стрельбы для уравнения второго порядка"""
    
    def __init__(self, problem: FirstOrderSystem, ode_solver_name: str = "rk4", sla_solver_name: str = "gauss"):
        super().__init__(problem)
        
        if not isinstance(problem, FirstOrderSystem):
            raise TypeError(
                f"ShootingSolver требует FirstOrderSystem, "
                f"получен {problem.get_type()}"
            )
        
        self.system = problem.get_system()
        self.bc = problem.get_bc()
        self.n_vars = problem.get_n_vars()

        self.t_span = self.a, self.b
        self.vec_dim = self.n_vars * (self.n_vars + 1)

        self.jacobian = NumericalJacobian()
        self.ode_solver = ODERegistry.create_solver(ode_solver_name)
        self.sla_solver = SLASolverRegistry.create_solver(sla_solver_name)

        # для графика невязок от итерации
        self.alpha_iters = []
        self.alphas = []
        self.last_iteration = 0
    
    def _get_expected_type(self) -> str:
        return "first_order_system"

    def get_system_jacobian_a(self, x0, y0, h_jacobi):
        """Якобиан d(system)/d(y) в точке x0, y0"""
        def temp_system(y):
            return self.system(x0, y)
        return self.jacobian.differentiate(temp_system, y0, h_jacobi)
    
    def get_bc_jacobians(self, ya: np.ndarray, yb: np.ndarray, h_jacobi: float) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Якобианы d(bc)/d(y) в точках y(a) и y(b)"""

        def temp_bc_a(y_param: np.ndarray):
            return self.bc(y_param, yb)
        
        def temp_bc_b(y_param: np.ndarray):
            return self.bc(ya, y_param)
        
        return self.jacobian.differentiate(temp_bc_a, ya, h_jacobi), \
            self.jacobian.differentiate(temp_bc_b, yb, h_jacobi)

    def arr_to_matrix(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(self.n_vars, self.n_vars)
    
    def matrix_to_arr(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.flatten()
    
    def get_y_from_vec(self, vec: np.ndarray) -> np.ndarray:
        return vec[:self.n_vars]
    
    def get_v_from_vec(self, vec: np.ndarray) -> np.ndarray:
        """Достаёт v из vec = [y, v] и возвращает в виде матрицы nxn"""
        return self.arr_to_matrix(vec[self.n_vars:])
    
    def get_initial_v(self) -> np.ndarray:
        # это всегда единичная матрица на левом конце
        return np.eye(self.n_vars)
    
    def get_alphas(self) -> Tuple[list[float], list[Vector]]:
        return self.alpha_iters, self.alphas
    
    def reset_alphas(self):
        self.alpha_iters = []
        self.alphas = []
        self.last_iteration = 0
    
    def solve(self, a0: np.ndarray, h: float = 1e-3, h_jacobi = 1e-6, tol: float = 1e-6, max_iter: int = 50, \
              record_alpha: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Решает методом стрельбы."""

        self.reset_alphas()

        # для начала преобразуем систему, чтобы на каждой итерации решать всего одну задачу Коши
        # изначальная система y' = f
        # вводим v = dy / da, (матрица nxn)
        # тогда новая система - это
        # y' = f 
        # v' = f' * v
        # первая часть размерности n, вторая часть разбивается на n систем длиной n
        # итого размерность n(n + 1) 

        # новая система
        def new_system(x: float, vec: Vector) -> Vector:
            # vec = [y, v]
            vec_nd = vec.to_numpy()
            y = self.get_y_from_vec(vec_nd)
            v = self.get_v_from_vec(vec_nd) # nxn
            dif_y: np.ndarray = self.system(x, y)
            dif_f: np.ndarray = self.get_system_jacobian_a(x, y, h_jacobi) # nxn
            dif_v: np.ndarray = dif_f @ v # nxn

            return Vector(np.append(dif_y, self.matrix_to_arr(dif_v)))
        
        a = Vector(a0)

        for i in range(0, max_iter):
            # vec = [y, v]

            # сигнатура вызываемой фнукции:
            # def solve(self, f, dim, t_span: Tuple[float, float], y0: List[float], h: float)
            vec_a = np.append(a.to_numpy(), self.matrix_to_arr(self.get_initial_v()))
            x, y_vec = self.ode_solver.solve(new_system, self.vec_dim, self.t_span, vec_a.tolist(), h)

            ya = self.get_y_from_vec(y_vec[0])
            yb = self.get_y_from_vec(y_vec[len(y_vec) - 1])

            psi = self.bc(ya, yb)
            
            if (np.max(np.abs(psi)) < tol):
                y = list(map(lambda yn_vec: self.get_y_from_vec(yn_vec), y_vec))
                return x, y
            
            va = self.get_v_from_vec(y_vec[0]) # nxn
            vb = self.get_v_from_vec(y_vec[len(y_vec) - 1]) # nxn

            ga_jacobi, gb_jacobi = self.get_bc_jacobians(ya, yb, h_jacobi)

            psi_1 = ga_jacobi @ va + gb_jacobi @ vb # psi' матрица nxn

            # сигнатура вызываемой фнукции:
            # def solve(self, A: Matrix, b: Vector) -> Vector:
            da = self.sla_solver.solve(Matrix(psi_1), (-1) * Vector(psi))

            a = a + da

            if record_alpha:
                self.alpha_iters.append(i)
                self.alphas.append(a)
                self.last_iteration = i

        raise RuntimeError("Решение с точностью " + str(tol) + " за " + str(max_iter) + " итераций не достигнуто:(")