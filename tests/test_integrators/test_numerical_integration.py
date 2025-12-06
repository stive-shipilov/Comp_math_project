import numpy as np
from comp_math.integration.integrator_registry import IntegratorRegistry

def test_integrators():
    """Тесты для методов численного интегрирования"""
        
    def f1(x):
        """f(x) = x^2"""
        return x**2
    
    def f2(x):
        """f(x) = 1/(1 + x^2)"""
        return 1 / (1 + x**2)
    
    rectangle_interpolator = IntegratorRegistry.create_solver("rectangle")
    trapezoida_interpolator = IntegratorRegistry.create_solver("trapezoida")
    simpson_interpolator = IntegratorRegistry.create_solver("simpson")
    gauss_interpolator = IntegratorRegistry.create_solver("gauss")
    monte_carlo_interpolator = IntegratorRegistry.create_solver("monte_carlo")

    result1 = rectangle_interpolator.integrate(f1, 0, 10)
    result2 = trapezoida_interpolator.integrate(f1, 0, 10)
    result3 = simpson_interpolator.integrate(f1, 0, 10)
    result4 = gauss_interpolator.integrate(f1, 0, 10)
    result5 = monte_carlo_interpolator.integrate(f1, 0, 10)

    # Проверки для теста 1
    exact1 = 333.3333333
    
    assert abs(result1 - exact1) < 0.01
    assert abs(result2 - exact1) < 0.005
    assert abs(result3 - exact1) < 0.0001
    assert abs(result4 - exact1) < 0.0001
    assert abs(result5 - exact1) < exact1/100
    
    result1 = rectangle_interpolator.integrate(f2, 0, 1)
    result2 = trapezoida_interpolator.integrate(f2, 0, 1)
    result3 = simpson_interpolator.integrate(f2, 0, 1)
    result4 = gauss_interpolator.integrate(f2, 0, 1)
    result5 = monte_carlo_interpolator.integrate(f2, 0, 1)
    
    # Проверки для теста 2
    exact2 = np.pi / 4
    
    assert abs(result1 - exact2) < 0.001
    assert abs(result2 - exact2) < 0.0005
    assert abs(result3 - exact2) < 1e-6
    assert abs(result4 - exact2) < 1e-2
    assert abs(result5 - exact2) < 0.01


    

