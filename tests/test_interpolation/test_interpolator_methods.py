import numpy as np
import pytest
from comp_math.linear_algebra.objects.matrix import Matrix
from comp_math.linear_algebra.objects.vector import Vector
from comp_math.interpolation.interpolator_registry import InterpolatorRegistry

def test_interpolators():
    """Тесты методов интерполяции"""
    
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 3.0])
    
    interpolator = InterpolatorRegistry.create_solver("newton")
    interpolator.fit(x, y)
    
    # Проверяем в промежуточных точках
    x_test = np.array([0.5, 0.25, 0.75])
    y_expected = 2 * x_test + 1
    y_actual = np.zeros(x_test.shape[0])
    y_actual = interpolator(x_test)
    
    assert np.allclose(y_actual, y_expected, rtol=1e-10)