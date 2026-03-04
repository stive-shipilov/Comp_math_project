import math
from typing import Tuple, List
import numpy as np
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector


class BaseRungeExplicitODESolver:
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
            for j in range(i, dim):
                if a[i][j] != 0:
                    raise ValueError("Матрица А для явных методов Рунге-Кутты \
                                     должна быть нижней треугольной")
        
        for i in range(dim):
            sum = 0.0
            for j in range(i):
                sum += a[i][j]
            if not math.isclose(sum, c[i]):
                raise ValueError("Сумма а по строке i должна равняться с[i]")
        
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
        
        for n in range(n_steps - 1):
            k_list = self.calc_k(f, dim, t[n], y[n], h)
            bk = Vector(np.zeros(dim))
            for i in range(self.method_dim):
                bk += self.b[i] * k_list[i]
            
            t[n + 1] = t[n] + h
            y[n + 1] = y[n] + h * bk

        return t, y

    def calc_k(self, f, dim, t, yn, h) -> List[Vector]:
        k_list = []
        k_list.append(f(t + self.c[0] * h, yn))

        for i in range(1, self.method_dim):
            ak = Vector(np.zeros(dim))
            for j in range(i):
                ak += self.a[i][j] * k_list[j]
            k_list.append(f(t + self.c[i] * h, yn + h * ak))

        return k_list


