from typing import Dict, Type, Any
from .solvers.direct.gauss_SLA_solver import GaussSolver
from .base_SLA_solver import SLASolver


class SLASolverRegistry:
    """Фабрика для создания экземпляров решателей СЛАУ"""
    
    _solvers: Dict[str, Type[SLASolver]] = {
        "gauss": GaussSolver,
    }

    @classmethod
    def create_solver(cls, method: str, **kwargs: Any) -> SLASolver:
        """Создает экземпляр решателя по имени метода"""
        if method not in cls._solvers:
            available = ", ".join(cls._solvers.keys())
            raise ValueError(f"Неизвестный метод '{method}'. Доступны только: {available}")
        
        solver_class = cls._solvers[method]
        return solver_class(**kwargs)

    @classmethod
    def get_available_solvers(cls) -> list:
        """Возвращает список доступных методов"""
        return list(cls._solvers.keys())

