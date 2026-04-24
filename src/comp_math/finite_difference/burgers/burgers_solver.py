
import numpy as np

from comp_math.finite_difference.burgers.burgers_problem import BurgersProblem

class BurgersSolver:
    """
    Решить уравнение Бюргерса: 
    du/dt + c du/dx = mu d2u/dx2
    """

    def __init__(self, problem: BurgersProblem):
        self.problem = problem

    def _tridiagonal_solve(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Метод прогонки для трехдиагональной системы.
        A, B, C, D — массивы numpy.
        Для скалярного случая используем numpy (быстрее и проще).
        """
        n = len(A)

        P = np.zeros(n)
        Q = np.zeros(n)

        P[0] = -C[0] / B[0]
        Q[0] = D[0] / B[0]
        
        # Прямой ход
        for i in range(1, n):
            denom = B[i] + A[i] * P[i-1]
            if abs(denom) < 1e-12:
                raise RuntimeError(f"Нулевой знаменатель в прогонке на шаге {i}")
            P[i] = -C[i] / denom if i < n - 1 else 0.0
            Q[i] = (D[i] - A[i] * Q[i-1]) / denom
        
        # Обратный ход
        y = np.zeros(n)
        y[-1] = Q[-1]
        for i in range(n-2, -1, -1):
            y[i] = P[i] * y[i+1] + Q[i]
        
        return y

    def solve(self, tau: float, h: float) -> np.ndarray:
        self.p = self.problem.get_c() * tau / (4 * h)
        self.q = self.problem.get_mu() * tau / (2 * h ** 2)

        t_num = int(self.problem.get_T() / tau)
        x_num = int(self.problem.get_L() / h)

        U0t = self.problem.get_U0t()
        ULt = self.problem.get_ULt()
        Ux0 = self.problem.get_Ux0()
        
        u = np.zeros(shape=(t_num, x_num))
        
        for j in range(x_num):
            u[0][j] = Ux0(j * h)

        for n in range(1, t_num):
            u[n] = self.step(u[n - 1])
            u[n][0] = U0t(n * tau)
            u[n][x_num - 1] = ULt(n * tau)

        return u
    
    def step(self, u: np.ndarray) -> np.ndarray:
        n = len(u)
        n_int = n - 2

        A = np.full(n_int, -self.p - self.q)
        B = np.full(n_int, 1.0 + 2.0 * self.q)
        C = np.full(n_int, self.p - self.q)

        D = ((self.p + self.q) * u[:-2]
            + (1.0 - 2.0 * self.q) * u[1:-1]
            + (-self.p + self.q) * u[2:])

        # граничные условия
        D[0] -= A[0] * u[0]
        D[-1] -= C[-1] * u[-1]

        u_int = self._tridiagonal_solve(A, B, C, D)

        u_new = np.empty(n)
        u_new[0] = u[0]          
        u_new[-1] = u[-1]       
        u_new[1:-1] = u_int

        return u_new

