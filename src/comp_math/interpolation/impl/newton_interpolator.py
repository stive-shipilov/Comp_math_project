from ..base_interpolator import BaseInterpolator
from typing import Optional
import numpy as np
from numpy.typing import NDArray

class NewtonInterpolator(BaseInterpolator):
    """Класс реализующий метод Ньютона для интерполяции"""
    def __init__(self):
        super().__init__()
        self.divided_diffs: Optional[NDArray[np.float64]] = None
        
    def _compute_devided_diifs(self):
        n = len(self.x)

        table = np.zeros((n, n))
        table[:, 0] = self.y
        
        # Разделенные разности
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (self.x[i + j] - self.x[i])
        
        self.divided_diffs = table[0, :]

    def _evaluate(self, x_query):
        self._compute_devided_diifs()
        if self.divided_diffs is None:
            raise ValueError("Сначала нужно вычислить разделнные разности. Вызовите метод fit()")
            
        x_query = np.asarray(x_query)
        result = np.zeros_like(x_query, dtype=np.float64)
        
        for k, x_point in enumerate(x_query):
            value = self.divided_diffs[0]
            
            product_term = 1.0
            for i in range(1, len(self.x)):
                product_term *= (x_point - self.x[i - 1])
                value += self.divided_diffs[i] * product_term
                
            result[k] = value
            
        return result