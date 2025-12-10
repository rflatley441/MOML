"""
Loss Surface Generators for Optimizer Visualization

This module provides various loss surfaces with different properties:
- Convex surfaces (guaranteed global minimum)
- Multi-local-minimum surfaces (tests optimizer escape ability)
- Non-convex surfaces (challenging optimization landscapes)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Callable


class LossSurface(ABC):
    """Abstract base class for loss surfaces."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """Compute loss at point x."""
        pass
    
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient at point x."""
        pass
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) for visualization."""
        return (-5, 5, -5, 5)
    
    def get_optimal(self) -> np.ndarray:
        """Return the global minimum location (if known)."""
        return np.array([0.0, 0.0])


class QuadraticBowl(LossSurface):
    """
    Convex quadratic loss surface: f(x,y) = a*x² + b*y²
    
    Properties:
    - Strictly convex
    - Single global minimum at origin
    - Condition number controlled by a/b ratio
    """
    
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super().__init__(
            "Quadratic Bowl",
            f"Convex quadratic: {a}x² + {b}y² (condition number: {max(a,b)/min(a,b):.1f})"
        )
        self.a = a
        self.b = b
    
    def __call__(self, x: np.ndarray) -> float:
        return self.a * x[0]**2 + self.b * x[1]**2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2 * self.a * x[0], 2 * self.b * x[1]])
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-3, 3, -3, 3)


class EllipticalBowl(LossSurface):
    """
    Rotated elliptical quadratic - tests optimizer behavior with correlated dimensions.
    
    f(x,y) = (x*cos(θ) + y*sin(θ))²/a² + (-x*sin(θ) + y*cos(θ))²/b²
    """
    
    def __init__(self, a: float = 1.0, b: float = 10.0, theta: float = np.pi/4):
        super().__init__(
            "Elliptical Bowl",
            f"Rotated ellipse with eccentricity {max(a,b)/min(a,b):.1f}"
        )
        self.a = a
        self.b = b
        self.theta = theta
        self.cos_t = np.cos(theta)
        self.sin_t = np.sin(theta)
    
    def __call__(self, x: np.ndarray) -> float:
        u = x[0] * self.cos_t + x[1] * self.sin_t
        v = -x[0] * self.sin_t + x[1] * self.cos_t
        return (u**2 / self.a**2) + (v**2 / self.b**2)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        u = x[0] * self.cos_t + x[1] * self.sin_t
        v = -x[0] * self.sin_t + x[1] * self.cos_t
        
        du_dx = self.cos_t
        du_dy = self.sin_t
        dv_dx = -self.sin_t
        dv_dy = self.cos_t
        
        df_du = 2 * u / self.a**2
        df_dv = 2 * v / self.b**2
        
        return np.array([
            df_du * du_dx + df_dv * dv_dx,
            df_du * du_dy + df_dv * dv_dy
        ])
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-4, 4, -4, 4)


class Rastrigin(LossSurface):
    """
    Rastrigin function - highly multimodal with many local minima.
    
    f(x) = An + Σ[xᵢ² - A*cos(2πxᵢ)]
    
    Properties:
    - Global minimum at origin
    - Many local minima arranged in a regular lattice
    - Tests optimizer's ability to escape local minima
    """
    
    def __init__(self, A: float = 10.0):
        super().__init__(
            "Rastrigin",
            "Multi-modal with regular local minima grid"
        )
        self.A = A
    
    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        return self.A * n + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-5.12, 5.12, -5.12, 5.12)


class Ackley(LossSurface):
    """
    Ackley function - nearly flat outer region with a deep hole at center.
    
    Properties:
    - Global minimum at origin
    - Many local minima
    - Nearly flat outer regions (vanishing gradients)
    """
    
    def __init__(self, a: float = 20, b: float = 0.2, c: float = 2*np.pi):
        super().__init__(
            "Ackley",
            "Deep central minimum with flat outer regions"
        )
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: np.ndarray) -> float:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        
        return term1 + term2 + self.a + np.e
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        
        sqrt_term = np.sqrt(sum_sq / n) if sum_sq > 0 else 1e-10
        
        # Gradient of term1
        grad1 = self.a * self.b * np.exp(-self.b * sqrt_term) * x / (n * sqrt_term)
        
        # Gradient of term2
        grad2 = np.exp(sum_cos / n) * self.c * np.sin(self.c * x) / n
        
        return grad1 + grad2
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-5, 5, -5, 5)


class Rosenbrock(LossSurface):
    """
    Rosenbrock function - the classic "banana" function.
    
    f(x,y) = (a-x)² + b(y-x²)²
    
    Properties:
    - Global minimum at (a, a²)
    - Long, narrow, parabolic valley
    - Finding the valley is easy, converging to minimum is hard
    """
    
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__(
            "Rosenbrock",
            "Banana-shaped valley - easy to find, hard to converge"
        )
        self.a = a
        self.b = b
    
    def __call__(self, x: np.ndarray) -> float:
        val = (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2
        return np.clip(val, -1e10, 1e10)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        dx = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0]**2)
        dy = 2 * self.b * (x[1] - x[0]**2)
        grad = np.array([dx, dy])
        # Clip gradients to prevent numerical overflow
        return np.clip(grad, -1e6, 1e6)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-2, 2, -1, 3)
    
    def get_optimal(self) -> np.ndarray:
        return np.array([self.a, self.a**2])


class Beale(LossSurface):
    """
    Beale function - flat regions and sharp valleys.
    
    Properties:
    - Global minimum at (3, 0.5)
    - Flat regions that cause vanishing gradients
    - Sharp valleys
    """
    
    def __init__(self):
        super().__init__(
            "Beale",
            "Flat regions with sharp valleys"
        )
    
    def __call__(self, x: np.ndarray) -> float:
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        val = term1 + term2 + term3
        return np.clip(val, -1e10, 1e10)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        t1 = 1.5 - x[0] + x[0]*x[1]
        t2 = 2.25 - x[0] + x[0]*x[1]**2
        t3 = 2.625 - x[0] + x[0]*x[1]**3
        
        dx = 2*t1*(x[1]-1) + 2*t2*(x[1]**2-1) + 2*t3*(x[1]**3-1)
        dy = 2*t1*x[0] + 2*t2*2*x[0]*x[1] + 2*t3*3*x[0]*x[1]**2
        
        grad = np.array([dx, dy])
        # Clip gradients to prevent numerical overflow
        return np.clip(grad, -1e6, 1e6)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-4.5, 4.5, -4.5, 4.5)
    
    def get_optimal(self) -> np.ndarray:
        return np.array([3.0, 0.5])


class SaddlePoint(LossSurface):
    """
    Simple saddle point surface: f(x,y) = x² - y²
    
    Properties:
    - Saddle point at origin
    - Tests optimizer behavior at saddle points
    - No global minimum (unbounded below)
    """
    
    def __init__(self):
        super().__init__(
            "Saddle Point",
            "x² - y² with saddle at origin"
        )
    
    def __call__(self, x: np.ndarray) -> float:
        return x[0]**2 - x[1]**2
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2*x[0], -2*x[1]])
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-3, 3, -3, 3)
    
    def get_optimal(self) -> np.ndarray:
        return np.array([0.0, 0.0])  # Saddle point, not minimum


class SixHumpCamel(LossSurface):
    """
    Six-Hump Camel function - multiple local minima with two global minima.
    
    Properties:
    - Two global minima at approximately (±0.0898, ∓0.7126)
    - Four additional local minima
    - Tests optimizer's exploration vs exploitation
    """
    
    def __init__(self):
        super().__init__(
            "Six-Hump Camel",
            "Six local minima, two global minima"
        )
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        term1 = (4 - 2.1*x1**2 + x1**4/3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4*x2**2) * x2**2
        val = term1 + term2 + term3
        return np.clip(val, -1e10, 1e10)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x[0], x[1]
        
        dx1 = 8*x1 - 8.4*x1**3 + 2*x1**5 + x2
        dx2 = x1 - 8*x2 + 16*x2**3
        
        grad = np.array([dx1, dx2])
        # Clip gradients to prevent numerical overflow
        return np.clip(grad, -1e6, 1e6)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-3, 3, -2, 2)
    
    def get_optimal(self) -> np.ndarray:
        return np.array([0.0898, -0.7126])


# Factory function to get loss surfaces by category
def get_loss_surfaces_by_category():
    """Return loss surfaces organized by their properties."""
    return {
        "convex": [
            QuadraticBowl(a=1.0, b=1.0),
            QuadraticBowl(a=1.0, b=10.0),  # Ill-conditioned
            EllipticalBowl(a=1.0, b=10.0, theta=np.pi/4),
        ],
        "multi_local_min": [
            Rastrigin(A=10.0),
            Ackley(),
            SixHumpCamel(),
        ],
        "non_convex": [
            Rosenbrock(a=1.0, b=100.0),
            Beale(),
            SaddlePoint(),
        ]
    }


def get_all_loss_surfaces():
    """Return a list of all available loss surfaces."""
    categories = get_loss_surfaces_by_category()
    surfaces = []
    for category_surfaces in categories.values():
        surfaces.extend(category_surfaces)
    return surfaces


if __name__ == "__main__":
    # Test all loss surfaces
    print("Testing Loss Surfaces\n" + "="*50)
    
    for surface in get_all_loss_surfaces():
        x = np.array([1.0, 1.0])
        loss = surface(x)
        grad = surface.gradient(x)
        
        print(f"\n{surface.name}")
        print(f"  Description: {surface.description}")
        print(f"  f([1,1]) = {loss:.4f}")
        print(f"  ∇f([1,1]) = [{grad[0]:.4f}, {grad[1]:.4f}]")
        print(f"  Optimal: {surface.get_optimal()}")

