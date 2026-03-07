from comp_math.ode.implicit.base_gear_solver import BaseGearSolver


class Gear1Solver(BaseGearSolver):
    """Метод Гира 1-го порядка (неявный Эйлер)"""
    def __init__(self):
        super().__init__()
        self.order = 1
        self.alpha = [1.0]
        self.beta = 1.0


class Gear2Solver(BaseGearSolver):
    """Метод Гира 2-го порядка"""
    def __init__(self):
        super().__init__()
        self.order = 2
        self.alpha = [4.0/3.0, -1.0/3.0]
        self.beta = 2.0/3.0


class Gear3Solver(BaseGearSolver):
    """Метод Гира 3-го порядка"""
    def __init__(self):
        super().__init__()
        self.order = 3
        self.alpha = [18.0/11.0, -9.0/11.0, 2.0/11.0]
        self.beta = 6.0/11.0


class Gear4Solver(BaseGearSolver):
    """Метод Гира 4-го порядка"""
    def __init__(self):
        super().__init__()
        self.order = 4
        self.alpha = [48.0/25.0, -36.0/25.0, 16.0/25.0, -3.0/25.0]
        self.beta = 12.0/25.0