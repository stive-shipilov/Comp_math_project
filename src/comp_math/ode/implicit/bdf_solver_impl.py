from comp_math.ode.implicit.base_bdf_solver import BaseBDFSolver


class BDF1Solver(BaseBDFSolver):
    """Метод ФДН 1 порядка (неявный Эйлер)"""
    
    def __init__(self):
        super().__init__()
        self.order = 1
        self.alpha = [1, -1]
        self.beta = 1.0


class BDF2Solver(BaseBDFSolver):
    """Метод ФДН 2 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 2
        self.alpha = [1, -4/3, 1/3]
        self.beta = 2/3


class BDF3Solver(BaseBDFSolver):
    """Метод ФДН 3 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 3
        self.alpha = [1, -18/11, 9/11, -2/11]
        self.beta = 6/11


class BDF4Solver(BaseBDFSolver):
    """Метод ФДН 4 порядка"""
    
    def __init__(self):
        super().__init__()
        self.order = 4
        self.alpha = [1, -48/25, 36/25, -16/25, 3/25]
        self.beta = 12/25
