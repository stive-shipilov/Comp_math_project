from ...base_SLA_solver import SLASolver


class GaussSolver(SLASolver):
    """Решение СЛАУ методом Гаусса"""
    
    # Заглушка
    def solve(self, A, b):
        return 11
