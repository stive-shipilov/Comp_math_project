import numpy as np
from typing import Callable, Tuple
from comp_math.bvp.base_bvp_solver import BaseBVPSolver
from comp_math.bvp.model.second_order_equation import SecondOrderEquation
from comp_math.differentiation.numerical.numericalJacobian import NumericalJacobian


class QuasilinearizationSolver(BaseBVPSolver):
    """Метод квазилинеаризации для уравнения второго порядка"""
    
    def __init__(self, problem: SecondOrderEquation, n_points: int = 100):
        super().__init__(problem)
        
        if not isinstance(problem, SecondOrderEquation):
            raise TypeError(
                f"QuasilinearizationSolver требует SecondOrderEquation, "
                f"получен {problem.get_type()}"
            )
        
        self.f = problem.get_f()
        self.alpha = problem.get_alpha()
        self.beta = problem.get_beta()
        
        self.n_points = n_points
        self.h = (self.b - self.a) / (n_points - 1)
        self.x = np.linspace(self.a, self.b, n_points)
        
        # Якобиан для численного дифференцирования
        self.jacobian = NumericalJacobian()
    
    def _get_expected_type(self) -> str:
        return "second_order_equation"
    
    def _initial_guess(self) -> np.ndarray:
        """Линейное начальное приближение"""
        return self.alpha + (self.beta - self.alpha) * (self.x - self.a) / (self.b - self.a)
    
    def _numerical_derivatives(self, x: float, y: float, yp: float, eps: float = 1e-6) -> Tuple[float, float]:
        """Вычисляет df/dy и df/dyp"""
        def F(vec):
            return np.array([self.f(x, vec[0], vec[1])])
        
        vec = np.array([y, yp])
        J = self.jacobian.differentiate(F, vec, h=eps)
        
        return J[0, 0], J[0, 1]
    
    def _tridiagonal_solve(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Метод прогонки для трехдиагональной системы.
        A, B, C, D — массивы numpy.
        Для скалярного случая используем numpy (быстрее и проще).
        """
        n = len(A)
        P = np.zeros(n)
        Q = np.zeros(n)
        
        # Прямой ход
        for i in range(1, n-1):
            denom = B[i] + A[i] * P[i-1]
            if abs(denom) < 1e-12:
                raise RuntimeError(f"Нулевой знаменатель в прогонке на шаге {i}")
            P[i] = -C[i] / denom
            Q[i] = (D[i] - A[i] * Q[i-1]) / denom
        
        # Обратный ход
        y = np.zeros(n)
        y[-1] = self.beta
        for i in range(n-2, -1, -1):
            y[i] = P[i] * y[i+1] + Q[i]
        
        return y
    
    def solve(self, tol: float = 1e-6, max_iter: int = 50, verbose: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Решает методом квазилинеаризации."""
        y = self._initial_guess()
        
        for it in range(max_iter):
            y_old = y.copy()
            yp = np.gradient(y, self.h)
            
            # Вычисляем коэффициенты
            p = np.zeros(self.n_points)  # df/dyp
            q = np.zeros(self.n_points)  # df/dy
            r = np.zeros(self.n_points)  # f - p*yp - q*y
            
            for i in range(self.n_points):
                dfdy, dfdyp = self._numerical_derivatives(self.x[i], y[i], yp[i])
                q[i] = dfdy
                p[i] = dfdyp
                f0 = self.f(self.x[i], y[i], yp[i])
                r[i] = f0 - p[i] * yp[i] - q[i] * y[i]
            
            # Строим трехдиагональную систему
            A = np.zeros(self.n_points)
            B = np.zeros(self.n_points)
            C = np.zeros(self.n_points)
            D = np.zeros(self.n_points)
            
            for i in range(1, self.n_points - 1):
                A[i] = 1 + p[i] * self.h / 2
                B[i] = -(2 + q[i] * self.h**2)
                C[i] = 1 - p[i] * self.h / 2
                D[i] = r[i] * self.h**2
            
            # Краевые условия
            D[1] -= A[1] * self.alpha
            D[self.n_points - 2] -= C[self.n_points - 2] * self.beta
            A[1] = 0
            C[self.n_points - 2] = 0
            
            # Решаем систему
            y_new = self._tridiagonal_solve(A, B, C, D)
            y_new[0] = self.alpha
            y_new[-1] = self.beta
            
            error = np.max(np.abs(y_new - y))
            y = y_new
            
            if verbose and it % 5 == 0:
                print(f"Итерация {it}: ошибка = {error:.2e}")
            
            if error < tol:
                if verbose:
                    print(f"Сошлось за {it+1} итераций")
                return self.x, y
        
        print(f"Не сошлось за {max_iter} итераций")
        return self.x, y