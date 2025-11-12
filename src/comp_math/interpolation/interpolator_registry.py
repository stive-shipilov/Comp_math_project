from typing import Dict, Type, Any
from ..interpolation.base_interpolator import BaseInterpolator
from ..interpolation.impl.newton_interpolator import NewtonInterpolator

class InterpolatorRegistry:
    """Фабрика для создания экземпляров интерполяторов СЛАУ"""
    
    _solvers: Dict[str, Type[BaseInterpolator]] = {
        "newton": NewtonInterpolator,
    }

    @classmethod
    def create_solver(cls, method: str, **kwargs: Any) -> BaseInterpolator:
        """Создает экземпляр решателя по имени метода"""
        if method not in cls._solvers:
            raise ValueError(f"Неизвестный метод '{method}'. Доступны только: {cls._solvers.keys()}")
        
        solver_class = cls._solvers[method]
        return solver_class(**kwargs)

    @classmethod
    def get_available_solvers(cls) -> list:
        """Возвращает список доступных методов"""
        return list(cls._solvers.keys())

