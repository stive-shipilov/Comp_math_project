from typing import Dict, Type
from ..interpolation.base_interpolator import BaseInterpolator
from ..interpolation.impl.newton_interpolator import NewtonInterpolator
from ..core.base_method_registry import BaseMethodRegistry


class InterpolatorRegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров интерполяторов СЛАУ"""
    
    _solvers: Dict[str, Type[BaseInterpolator]] = {
        "newton": NewtonInterpolator,
    }
    