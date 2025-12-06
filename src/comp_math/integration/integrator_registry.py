from typing import Dict, Type
from ..core.base_method_registry import BaseMethodRegistry
from .impl.simple_integrator import RectangleIntegrator, TrapezoidalIntegrator, SimpsonIntegrator
from .impl.gauss_integrator import GaussIntegrator
from .impl.monte_carlo_integrator import MonteCarloIntegrator
from .base_integrator import BaseIntegrator

class IntegratorRegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров интегратора"""
    
    _solvers: Dict[str, Type[BaseIntegrator]] = {
        "rectangle": RectangleIntegrator,
        "trapezoida": TrapezoidalIntegrator,
        "simpson": SimpsonIntegrator,
        "gauss": GaussIntegrator,
        "monte_carlo": MonteCarloIntegrator
    }
    