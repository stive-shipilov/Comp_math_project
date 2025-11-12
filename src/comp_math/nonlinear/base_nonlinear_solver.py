from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
import numpy as np

from ..linear_algebra.objects.matrix import Matrix
from ..linear_algebra.objects.vector import Vector
from ..core.base_solver import BaseNumericalMethod


class NonlinearSolver(BaseNumericalMethod):
    """Абстрактный базовый класс для решателей нелинейных уравнений"""
    
    def __init__(self, max_iterations=1000, tolerance=1e-10, verbose=False):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self._iterations = 0
        self._errors = []
    
    def _add_iteration(self, error: float):
        self._iterations += 1
        self._errors.append(error)
        
    @property
    def iterations_count(self) -> int:
        return self._iterations
    
    @property
    def errors(self) -> List[float]:
        return self._errors
    
    @property
    def last_error(self) -> float:
        if self._errors:
            return self._errors[-1] 
        else:
            return 0.0
        
    def _prepare_solver(self):
        self._iterations = 0
        self._errors = []
    

class NonlinearSolver1D(NonlinearSolver):
    """Базовый класс для 1D методов (скалярные уравнения f(x) = 0)"""
    
    def solve(self, f: Callable[[float], float], 
          search_area: Tuple[float, float] = (-1e8, 1e8),
          grid_points: int = 1000,
          x_initial: List[float] = None) -> List[float]:
        """Решает скалярное уравнение f(x) = 0"""
        roots = []
        intervals = self.find_root_intervals(f, search_area, grid_points)
        if x_initial is None:
            x_initial = np.zeros(len(intervals))
        for i, interval in enumerate(intervals):
            self._validate_input_1d(f, interval)
            self._prepare_solver()
            roots.append(self._solve_implementation_1d(f, interval, x_initial[i]))
            
        return roots

    
    @abstractmethod
    def _solve_implementation_1d(self, f: Callable[[float], float],
                                interval: Tuple[float, float],
                                x_initial: float = None) -> float:
        pass

    def find_root_intervals(self, F: Callable[[np.ndarray], np.ndarray],
                           search_area: Tuple[float, float],
                           grid_points: int) -> List[Tuple[float, float]]:
        """Локализует корни"""
        intervals = []
    
        a, b = search_area
        points = np.linspace(a, b, grid_points)
        
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i+1]
            f1, f2 = F(np.array([x1])), F(np.array([x2]))
            
            if np.sign(f1[0]) != np.sign(f2[0]):
                intervals.append((x1, x2))
        
        return intervals
        
    def _validate_input_1d(self, f: Callable[[float], float],
                          interval: Tuple[float, float]):
        if not callable(f):
            raise ValueError("f должна быть функцией")

class NonlinearSolverND(NonlinearSolver):
    """Базовый класс для многомерных методов (системы F(x) = 0)"""
    
    def solve(self, F: Callable[[np.ndarray], np.ndarray],
              x0: np.ndarray,
              J: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Решает систему уравнений F(x) = 0"""
        self._validate_input_nd(F, x0, J)
        self._prepare_solver()
        return self._solve_implementation_nd(F, x0, J)
    
    @abstractmethod
    def _solve_implementation_nd(self, F: Callable[[np.ndarray], np.ndarray],
                               x0: np.ndarray,
                               J: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        pass
    
    def _validate_input_nd(self, F: Callable[[np.ndarray], np.ndarray],
                          x0: np.ndarray,
                          J: Optional[Callable[[np.ndarray], np.ndarray]]):
        x0 = np.asarray(x0)
        if x0.ndim != 1:
            raise ValueError("x0 должен быть вектором")


class VariationalSolver1D(NonlinearSolver):
    """Базовый класс для 1D вариационных методов (минимизация функционалов)"""
    
    def solve(self, functional: Callable[[float], float], 
              search_area: Tuple[float, float] = (-1e8, 1e8),
              grid_points: int = 1000,
              basis_functions: Optional[List[Callable]] = None,
              boundary_conditions: Optional[dict] = None) -> List[float]:
        """Решает вариационную задачу минимизации функционала"""
        
        # Если базисные функции не заданы, используем простой полиномиальный базис
        if basis_functions is None:
            basis_functions = self._default_basis_functions()
        
        if boundary_conditions is None:
            boundary_conditions = {}
            
        roots = []
        # Для вариационных методов ищем минимумы функционала
        intervals = self.find_root_intervals(functional, search_area, grid_points)
        
        if self.verbose:
            print(f"Найдены интервалы с минимумами: {intervals}")
            
        for i, interval in enumerate(intervals):
            self._validate_input_variational(functional, interval, basis_functions)
            self._prepare_solver()
            root = self._solve_implementation_variational(
                functional, interval, basis_functions, boundary_conditions
            )
            roots.append(root)
            
        return roots
    
    def _default_basis_functions(self) -> List[Callable]:
        """Возвращает базисные функции по умолчанию"""
        return [
            lambda x: 1,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3
        ]
    
    @abstractmethod
    def _solve_implementation_variational(self, 
                                        functional: Callable[[float], float],
                                        interval: Tuple[float, float],
                                        basis_functions: List[Callable],
                                        boundary_conditions: dict) -> float:
        """Метод для реализации вариационного алгоритма"""
        pass

    def find_root_intervals(self, F: Callable[[np.ndarray], np.ndarray],
                           search_area: Tuple[float, float],
                           grid_points: int) -> List[Tuple[float, float]]:
        """Локализует корни"""
        intervals = []
    
        a, b = search_area
        points = np.linspace(a, b, grid_points)
        
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i+1]
            f1, f2 = F(np.array([x1])), F(np.array([x2]))
            
            if np.sign(f1[0]) != np.sign(f2[0]):
                intervals.append((x1, x2))
        
        return intervals
    
    def _validate_input_variational(self, 
                                  functional: Callable[[float], float],
                                  interval: Tuple[float, float],
                                  basis_functions: List[Callable]):
        """Валидация входных данных для вариационных методов"""
        if not callable(functional):
            raise ValueError("functional должна быть функцией")
