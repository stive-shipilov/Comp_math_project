import math
import numpy as np
import pytest
from comp_math.differentiation.numerical.numericalDifferentiator import NumericalDifferentiator


class TestNumericalDiffentiator:
    """Тесты для численного дифференцирования"""

    def test_numerical_diff(self):
        """Тест для тестированяи разлчиных 
        типов численного диффернцирования
        """
        def x2 (x): return x**2

        #Правое дифференцирование
        assert 19.5 <=NumericalDifferentiator.rightDifferentiate(x2, 10, 0.1) <= 20.5

        #Левое дифференцирование
        assert 19.5 <=NumericalDifferentiator.leftDifferentiate(x2, 10, 0.1) <= 20.5

        #Двусторонее дифференцирование
        assert 19.5 <=NumericalDifferentiator.doubleSideDifferentiate(x2, 10, 0.1) <= 20.5

        #4-х узловое дифференцирование
        assert 19.5 <=NumericalDifferentiator.fourNodeDifferentiate(x2, 10, 0.1) <= 20.5

        #6-и узловое дифференцирование
        assert 19.5 <=NumericalDifferentiator.sixNodeDifferentiate(x2, 10, 0.1) <= 20.5
