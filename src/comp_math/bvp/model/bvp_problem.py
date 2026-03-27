import numpy as np
from scipy.integrate import solve_bvp
from dataclasses import dataclass
from typing import List, Tuple, Union, Callable, Dict
import matplotlib.pyplot as plt
import inspect

from comp_math.bvp.model.bvp_result import BVPResult
from comp_math.bvp.model.model_bvp import BoundaryEquation, Derivative, Equation, Variable


class BVProblem:
    def __init__(self, equations: List[Equation], conditions: List, domain: Tuple[float, float], n_points: int = 50):
        self.equations = equations
        self.conditions = conditions
        self.domain = domain
        self.n_points = n_points
        
        self.variables = self._extract_variables()
        self.max_order = self._compute_max_orders()
        self.system_vars, self.system_eqs = self._reduce_to_first_order()
        
        for i, var in enumerate(self.system_vars):
            var._index = i
        
        self._system_func = self._build_system()
        self._bc_func = self._build_bc()
    
    def _extract_variables(self) -> List[Variable]:
        vars_set = set()
        def add(obj):
            if isinstance(obj, Variable):
                vars_set.add(obj)
            elif isinstance(obj, Derivative):
                add(obj.var)
            elif isinstance(obj, Equation):
                add(obj.lhs)
                add(obj.rhs)
            elif isinstance(obj, BoundaryEquation):
                add(obj.bc.var)
        
        for eq in self.equations:
            add(eq)
        for cond in self.conditions:
            if isinstance(cond, BoundaryEquation):
                add(cond)
        return list(vars_set)
    
    def _compute_max_orders(self) -> Dict[Variable, int]:
        max_order = {var: 0 for var in self.variables}
        for eq in self.equations:
            if isinstance(eq.lhs, Derivative):
                max_order[eq.lhs.var] = max(max_order[eq.lhs.var], eq.lhs.order)
        return max_order
    
    def _reduce_to_first_order(self) -> Tuple[List[Variable], List[Equation]]:
        system_vars = []
        system_eqs = []
        var_components = {}
        
        for var in self.variables:
            components = []
            y0 = Variable(var.name)
            components.append(y0)
            system_vars.append(y0)
            
            for order in range(1, self.max_order[var]):
                yi = Variable(f"{var.name}_{order}")
                components.append(yi)
                system_vars.append(yi)
                if order < self.max_order[var] - 1:
                    yi_next = Variable(f"{var.name}_{order+1}")
                    system_eqs.append(Equation(Derivative(yi, 1), yi_next))
            
            var_components[var] = components
        
        for eq in self.equations:
            if isinstance(eq.lhs, Derivative):
                var = eq.lhs.var
                order = eq.lhs.order
                if order <= self.max_order[var]:
                    lhs_var = var_components[var][order - 1]
                    rhs_expr = self._substitute(eq.rhs, var_components)
                    system_eqs.append(Equation(Derivative(lhs_var, 1), rhs_expr))
        
        return system_vars, system_eqs
    
    def _substitute(self, expr, var_components):
        if isinstance(expr, Variable):
            for var, comps in var_components.items():
                if var.name == expr.name:
                    return comps[0]
            return expr
        if isinstance(expr, Derivative):
            var = expr.var
            if var in var_components and expr.order <= len(var_components[var]):
                return var_components[var][expr.order]
            return expr
        if isinstance(expr, (int, float, Callable)):
            return expr
        return expr
    
    def _build_system(self) -> Callable:
        eq_map = {}
        for eq in self.system_eqs:
            if isinstance(eq.lhs, Derivative):
                eq_map[eq.lhs.var] = eq.rhs
        
        def system(x, y):
            values = {var: y[var._index] for var in self.system_vars}
            dy = np.zeros_like(y)
            for var in self.system_vars:
                if var in eq_map:
                    dy[var._index] = self._eval(eq_map[var], x, values)
            return dy
        return system
    
    def _eval(self, expr, x, values):
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, Variable):
            return values.get(expr, 0.0)
        if callable(expr):
            sig = inspect.signature(expr)
            args = []
            for name in sig.parameters:
                if name == 'x':
                    args.append(x)
                else:
                    found = False
                    for var, val in values.items():
                        if var.name == name or var.name.startswith(name):
                            args.append(val)
                            found = True
                            break
                    if not found:
                        args.append(0.0)
            return expr(*args)
        return 0.0
    
    def _build_bc(self) -> Callable:
        a, b = self.domain
        def bc(ya, yb):
            residuals = []
            for cond in self.conditions:
                if isinstance(cond, BoundaryEquation):
                    bc_obj = cond.bc
                    target = bc_obj.var
                    if bc_obj.derivative_order > 0:
                        for sv in self.system_vars:
                            if sv.name == f"{bc_obj.var.name}_{bc_obj.derivative_order}":
                                target = sv
                                break
                    idx = target._index
                    if abs(bc_obj.point - a) < 1e-12:
                        val = ya[idx]
                    else:
                        val = yb[idx]
                    residuals.append(val - cond.value)
            return np.array(residuals)
        return bc
    
    def _make_initial_guess(self):
        a, b = self.domain
        x = np.linspace(a, b, self.n_points)
        y_guess = np.zeros((len(self.system_vars), self.n_points))
        
        left_val, right_val = {}, {}
        for cond in self.conditions:
            if isinstance(cond, BoundaryEquation):
                bc_obj = cond.bc
                target = bc_obj.var
                if bc_obj.derivative_order > 0:
                    for sv in self.system_vars:
                        if sv.name == f"{bc_obj.var.name}_{bc_obj.derivative_order}":
                            target = sv
                            break
                if abs(bc_obj.point - a) < 1e-12:
                    left_val[target] = cond.value
                else:
                    right_val[target] = cond.value
        
        for var in self.system_vars:
            left = left_val.get(var, 0.0)
            right = right_val.get(var, 1.0)
            y_guess[var._index] = left + (right - left) * (x - a) / (b - a)
        
        return x, y_guess
    
    def solve(self, **kwargs):
        x, y_guess = self._make_initial_guess()
        sol = solve_bvp(self._system_func, self._bc_func, x, y_guess, **kwargs)
        if not sol.success:
            raise RuntimeError(f"Решение не найдено: {sol.message}")
        return BVPResult(sol, self.variables, self.system_vars)

