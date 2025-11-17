from typing import Dict, Type
from ..interpolation.base_interpolator import BaseInterpolator
from ..interpolation.impl.newton_interpolator import NewtonInterpolator
from ..interpolation.impl.spline_interpolator import CubicSpline
from ..interpolation.impl.lsq_interpolator import UniversalLSQ
from ..core.base_method_registry import BaseMethodRegistry


class InterpolatorRegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров интерполяторов СЛАУ"""
    
    _solvers: Dict[str, Type[BaseInterpolator]] = {
        "newton": NewtonInterpolator,
        "cubic_spline": CubicSpline,
        "lsq": UniversalLSQ
    }
    