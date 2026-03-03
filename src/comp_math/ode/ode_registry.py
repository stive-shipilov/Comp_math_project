from typing import Dict, Type
from ..core.base_method_registry import BaseMethodRegistry
from .base_ode_solver import BaseODESolver

class ODERegistry(BaseMethodRegistry):
    """Фабрика для создания экземпляров решателей ОДУ"""
    
    _solvers: Dict[str, Type[BaseODESolver]] = {
    }
    