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
    
    assert np.allclose(y_actual.to_numpy(), y_expected, rtol=1e-10)

    interpolator = InterpolatorRegistry.create_solver("cubic_spline")
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    
    interpolator.fit(x, y)
    result = interpolator([0.5, 1.5])
    
    expected = np.array([1.5, 2.5])
    assert np.allclose(result.to_numpy(), expected, rtol=1e-10)

    # Тест на точное воспроизведение квадратичной функции
    x = np.array([0, 1, 2, 3, 4])
    y = 2*x**2 + 3*x + 1
    
    interpolator = InterpolatorRegistry.create_solver("lsq")
    interpolator.fit(x, y)
    
    x_test = np.array([0.5, 1.5, 2.5])
    y_pred = interpolator(x_test)
    y_true = 2*x_test**2 + 3*x_test + 1
    
    np.testing.assert_allclose(y_pred.to_numpy(), y_true, rtol=1e-10)
    print("✅ МНК точно воспроизводит полиномиальную функцию")