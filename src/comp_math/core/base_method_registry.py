from abc import ABC, abstractmethod
from typing import Any
from ..core.base_solver import BaseNumericalMethod

class BaseMethodRegistry(ABC):
    
    @classmethod
    def create_solver(cls, method: str, **kwargs: Any) -> BaseNumericalMethod:
        """Создает экземпляр решателя по имени метода"""
        if method not in cls._solvers:
            raise ValueError(f"Неизвестный метод '{method}'. Доступны только: {cls._solvers.keys()}")
        
        solver_class = cls._solvers[method]
        return solver_class(**kwargs)

    @classmethod
    def get_available_solvers(cls) -> list:
        """Возвращает список доступных методов"""
        return list(cls._solvers.keys())
