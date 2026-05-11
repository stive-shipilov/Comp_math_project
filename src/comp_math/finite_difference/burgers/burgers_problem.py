from typing import Callable

class BurgersProblem:
    """
    Задача конвекции-диффузии:
        du/dt + c * du/dx = mu * d^2u/dx^2

    Параметры:
        U0t   : u(0, t) — левое граничное условие
        ULt   : u(L, t) — правое граничное условие
        Ux0   : u(x, 0) — начальное условие
        T     : конечное время
        L     : длина отрезка
        c     : скорость переноса
        mu    : коэффициент диффузии
    """
    
    def __init__(self, *,
                 U0t: Callable[[float], float],
                 ULt: Callable[[float], float],
                 Ux0: Callable[[float], float],
                 T: float,
                 L: float,
                 c: float,
                 mu: float):
        self.U0t = U0t
        self.ULt = ULt
        self.Ux0 = Ux0
        self.T = T
        self.L = L
        self.c = c
        self.mu = mu

    def get_T(self) -> float:
        return self.T

    def get_L(self) -> float:
        return self.L

    def get_Ux0(self) -> Callable[[float], float]:
        return self.Ux0

    def get_U0t(self) -> Callable[[float], float]:
        return self.U0t

    def get_ULt(self) -> Callable[[float], float]:
        return self.ULt

    def get_c(self) -> float:
        return self.c

    def get_mu(self) -> float:
        return self.mu