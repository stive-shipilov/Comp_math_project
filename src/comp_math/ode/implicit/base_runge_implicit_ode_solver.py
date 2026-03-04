import math
from typing import Tuple, List
import numpy as np
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from scipy.optimize import fsolve


class BaseRungeImplicitODESolver:
    def __init__(self, a: Matrix, b: Vector, c: Vector):
        self.check_abc(a, b, c, 0.0001)

        self.method_dim = c.dim
        self.a = a
        self.b = b
        self.c = c
        self.epsilon = 0.00001

    def check_abc(self, a: Matrix, b: Vector, c: Vector, epsilon: float):
        if c.dim != b.dim or c.dim != a.shape[0] or a.shape[0] != a.shape[1]:
            raise ValueError("Неверные размерности у a, b, c из матрицы Бутчера")
        
        dim = c.dim

        for i in range(dim):
            sum = 0.0
            for j in range(dim):
                sum += a[i][j]
            if not math.isclose(sum, c[i], abs_tol=epsilon):
                raise ValueError("Сумма а по строке i должна равняться с[i]: " \
                                 + str(sum) + " != " + str(c[i]))
        
        for i in range(dim):
            if not math.isclose(b.to_numpy().sum(), 1.0):
                raise ValueError("Сумма b должна быть 1")

    def solve(self, f, t_span: Tuple[float, float], y0: List[float], h: float) \
    -> Tuple[np.ndarray, np.ndarray]:
    
        if h <= 0:
            raise ValueError(f"Шаг должен быть положительным: {h}")
        
        t0, t_end = t_span
        if t0 >= t_end:
            raise ValueError(f"Начало интервала должно быть меньше конца: {t_span}")
        
        yn = Vector(y0)
        n_steps = int((t_end - t0) / h) + 1
        
        t = np.empty(n_steps, dtype=float)
        y = np.empty(n_steps, dtype=object)
        
        t[0] = t0
        y[0] = yn
        dim = yn.dim

        print(n_steps)
        
        for n in range(n_steps - 1):
            k_list = self.calc_k(f, dim, t[n], y[n], h)
            bk = Vector(np.zeros(dim))
            for i in range(self.method_dim):
                bk += self.b[i] * k_list[i]
            t[n + 1] = t[n] + h
            y[n + 1] = y[n] + h * bk

        return t, y

    def calc_k(self, f, dim, t, yn, h) -> List[Vector]:
        # k_list - набор векторов, записанных подряд как 
        # k11 k12 ... k1n, k21, k22, ... k2n, k31 ... , kmn

        def func(k_arr: np.ndarray) -> np.ndarray:
            k_list = self.to_vectors(k_arr, dim)
            f_vectors = []
            for i in range(self.method_dim):
                ak = Vector(np.zeros(dim))
                for j in range(self.method_dim):
                    ak += self.a[i][j] * k_list[j]
                f_vectors.append(f(t + self.c[i] * h, yn + h * ak) - k_list[i])
            return self.to_one_ndarray(f_vectors)

        k_arr0 = np.ones(dim * self.method_dim)

        return self.to_vectors(fsolve(func, k_arr0), dim)
    
    def to_one_ndarray(self, list: List[Vector]) -> np.ndarray:
        list_len = len(list)
        dim = list[0].dim
        arr = np.zeros(list_len * dim)
        for i in range(list_len):
            for j in range(dim):
                arr[i * dim + j] = list[i][j]
        return arr
    
    def to_vectors(self, arr: np.ndarray, dim: int) -> list[Vector]:
        list = []
        for i in range(int(len(arr) / dim)):
            list.append(Vector(arr[dim * i : dim * (i + 1)]))
        return list
