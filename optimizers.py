"""
Gradient Descent Optimizers

This module implements various optimization algorithms for gradient descent:
- SGD (Stochastic Gradient Descent)
- Momentum
- Nesterov Accelerated Gradient
- AdaGrad
- RMSprop
- Adam
- AdamW (Adam with weight decay)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OptimizationStep:
    """Record of a single optimization step."""
    position: np.ndarray
    loss: float
    gradient: np.ndarray
    step_num: int


@dataclass
class OptimizationHistory:
    """Complete history of an optimization run."""
    optimizer_name: str
    initial_position: np.ndarray
    steps: List[OptimizationStep] = field(default_factory=list)
    diverged: bool = False  # True if optimization went out of bounds
    
    @property
    def positions(self) -> np.ndarray:
        """Return array of all positions visited."""
        return np.array([step.position for step in self.steps])
    
    @property
    def losses(self) -> np.ndarray:
        """Return array of all loss values."""
        return np.array([step.loss for step in self.steps])
    
    @property
    def gradients(self) -> np.ndarray:
        """Return array of all gradients."""
        return np.array([step.gradient for step in self.steps])
    
    @property
    def final_position(self) -> np.ndarray:
        """Return the final position."""
        return self.steps[-1].position if self.steps else self.initial_position
    
    @property
    def final_loss(self) -> float:
        """Return the final loss value."""
        return self.steps[-1].loss if self.steps else float('inf')


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.01, name: str = "Optimizer"):
        self.learning_rate = learning_rate
        self.name = name
    
    @abstractmethod
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Perform one optimization step."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Return optimizer parameters for display."""
        return {"learning_rate": self.learning_rate}


class GradientDescent(Optimizer):
    """
    Vanilla Gradient Descent
    
    Update rule: x_{t+1} = x_t - lr * ∇f(x_t)
    
    The simplest optimizer - directly follows the negative gradient.
    No momentum, no adaptive learning rates.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate, "Gradient Descent")
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return x - self.learning_rate * gradient
    
    def reset(self):
        pass  # GD has no state


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    Update rule: x_{t+1} = x_t - lr * ∇f(x_t)
    
    The simplest optimizer - directly follows the negative gradient.
    (In this context, same as GD since we're on loss surfaces, not batches)
    """
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate, "SGD")
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return x - self.learning_rate * gradient
    
    def reset(self):
        pass  # SGD has no state


class Momentum(Optimizer):
    """
    SGD with Momentum
    
    Update rule:
        v_{t+1} = β * v_t + ∇f(x_t)
        x_{t+1} = x_t - lr * v_{t+1}
    
    Accumulates velocity in directions of persistent gradient.
    Helps escape local minima and speeds up convergence.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate, "Momentum")
        self.beta = beta
        self.velocity = None
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        
        self.velocity = self.beta * self.velocity + gradient
        return x - self.learning_rate * self.velocity
    
    def reset(self):
        self.velocity = None
    
    def get_params(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "beta": self.beta}


class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG)
    
    Update rule:
        v_{t+1} = β * v_t + ∇f(x_t - lr * β * v_t)  [look-ahead gradient]
        x_{t+1} = x_t - lr * v_{t+1}
    
    Computes gradient at the "look-ahead" position.
    Often converges faster than standard momentum.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        super().__init__(learning_rate, "Nesterov")
        self.beta = beta
        self.velocity = None
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        

        self.velocity = self.beta * self.velocity + gradient
        return x - self.learning_rate * self.velocity
    
    def get_lookahead_position(self, x: np.ndarray) -> np.ndarray:
        """Get the look-ahead position for gradient computation."""
        if self.velocity is None:
            return x
        return x - self.learning_rate * self.beta * self.velocity
    
    def reset(self):
        self.velocity = None
    
    def get_params(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "beta": self.beta}


class AdaGrad(Optimizer):
    """
    Adaptive Gradient Algorithm
    
    Update rule:
        G_{t+1} = G_t + ∇f(x_t)²
        x_{t+1} = x_t - lr * ∇f(x_t) / (√G_{t+1} + ε)
    
    Adapts learning rate per-parameter based on historical gradients.
    Good for sparse gradients, but learning rate can decay too fast.
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate, "AdaGrad")
        self.epsilon = epsilon
        self.G = None 
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(x)
        
        self.G += gradient ** 2
        adjusted_lr = self.learning_rate / (np.sqrt(self.G) + self.epsilon)
        return x - adjusted_lr * gradient
    
    def reset(self):
        self.G = None
    
    def get_params(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "epsilon": self.epsilon}


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation
    
    Update rule:
        E[g²]_t = β * E[g²]_{t-1} + (1-β) * ∇f(x_t)²
        x_{t+1} = x_t - lr * ∇f(x_t) / (√E[g²]_t + ε)
    
    Uses exponential moving average of squared gradients.
    Fixes AdaGrad's aggressive learning rate decay.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate, "RMSprop")
        self.beta = beta
        self.epsilon = epsilon
        self.E_g2 = None  # exponential moving average of squared gradients
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.E_g2 is None:
            self.E_g2 = np.zeros_like(x)
        
        self.E_g2 = self.beta * self.E_g2 + (1 - self.beta) * gradient ** 2
        adjusted_lr = self.learning_rate / (np.sqrt(self.E_g2) + self.epsilon)
        return x - adjusted_lr * gradient
    
    def reset(self):
        self.E_g2 = None
    
    def get_params(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "beta": self.beta}


class Adam(Optimizer):
    """
    Adaptive Moment Estimation
    
    Update rule:
        m_t = β₁ * m_{t-1} + (1-β₁) * ∇f(x_t)           [1st moment]
        v_t = β₂ * v_{t-1} + (1-β₂) * ∇f(x_t)²          [2nd moment]
        m̂_t = m_t / (1 - β₁^t)                          [bias correction]
        v̂_t = v_t / (1 - β₂^t)                          [bias correction]
        x_{t+1} = x_t - lr * m̂_t / (√v̂_t + ε)
    
    Combines momentum (1st moment) with RMSprop (2nd moment).
    Most popular optimizer in deep learning.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate, "Adam")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        # Bias-corrected estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2
        }


class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay
    
    Like Adam, but applies weight decay directly to parameters
    instead of through the gradient (L2 regularization).
    
    Update rule includes: x_{t+1} = x_{t+1} - lr * λ * x_t
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate, "AdamW")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Adam update + weight decay
        x_new = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        x_new = x_new - self.learning_rate * self.weight_decay * x
        
        return x_new
    
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay
        }


def run_optimization(
    loss_surface,
    optimizer: Optimizer,
    initial_position: np.ndarray,
    num_steps: int = 100,
    convergence_threshold: float = 1e-8,
    gradient_clip: Optional[float] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> OptimizationHistory:
    """
    Run optimization on a loss surface.
    
    Args:
        loss_surface: Loss surface object with __call__ and gradient methods
        optimizer: Optimizer instance
        initial_position: Starting point
        num_steps: Maximum number of steps
        convergence_threshold: Stop if loss change < threshold
        gradient_clip: Optional gradient clipping value
        bounds: Optional (x_min, x_max, y_min, y_max) bounds. If provided,
                optimization stops when position exits these bounds.
    
    Returns:
        OptimizationHistory with complete trajectory
    """
    optimizer.reset()
    history = OptimizationHistory(
        optimizer_name=optimizer.name,
        initial_position=initial_position.copy()
    )
    
    x = initial_position.copy()
    prev_loss = float('inf')
    
    def is_within_bounds(pos: np.ndarray) -> bool:
        """Check if position is within the specified bounds."""
        if bounds is None:
            return True
        x_min, x_max, y_min, y_max = bounds
        return (x_min <= pos[0] <= x_max) and (y_min <= pos[1] <= y_max)
    
    for step in range(num_steps):
        # Check if current position is out of bounds
        if not is_within_bounds(x):
            history.diverged = True
            break
        
        # Handle Nesterov look-ahead
        if isinstance(optimizer, NesterovMomentum):
            grad_position = optimizer.get_lookahead_position(x)
            gradient = loss_surface.gradient(grad_position)
        else:
            gradient = loss_surface.gradient(x)
        
        # Optional gradient clipping
        if gradient_clip is not None:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > gradient_clip:
                gradient = gradient * gradient_clip / grad_norm
        
        loss = loss_surface(x)
        
        # Record step
        history.steps.append(OptimizationStep(
            position=x.copy(),
            loss=loss,
            gradient=gradient.copy(),
            step_num=step
        ))
        
        # Check convergence
        if abs(prev_loss - loss) < convergence_threshold:
            break
        
        prev_loss = loss
        
        # Update position
        x = optimizer.step(x, gradient)
    
    # Record final position only if it's within bounds
    if is_within_bounds(x):
        final_loss = loss_surface(x)
        final_gradient = loss_surface.gradient(x)
        history.steps.append(OptimizationStep(
            position=x.copy(),
            loss=final_loss,
            gradient=final_gradient.copy(),
            step_num=len(history.steps)
        ))
    
    return history


def get_all_optimizers(learning_rate: float = 0.01) -> List[Optimizer]:
    """Return instances of all available optimizers."""
    return [
        GradientDescent(learning_rate=learning_rate),
        SGD(learning_rate=learning_rate),
        Momentum(learning_rate=learning_rate, beta=0.9),
        NesterovMomentum(learning_rate=learning_rate, beta=0.9),
        AdaGrad(learning_rate=learning_rate),
        RMSprop(learning_rate=learning_rate, beta=0.9),
        Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999),
    ]


def get_optimizer_by_name(name: str, **kwargs) -> Optimizer:
    """Get an optimizer instance by name."""
    optimizers = {
        "gd": GradientDescent,
        "gradient_descent": GradientDescent,
        "sgd": SGD,
        "momentum": Momentum,
        "nesterov": NesterovMomentum,
        "adagrad": AdaGrad,
        "rmsprop": RMSprop,
        "adam": Adam,
        "adamw": AdamW,
    }
    
    name_lower = name.lower()
    if name_lower not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name_lower](**kwargs)


if __name__ == "__main__":
    from loss_surfaces import QuadraticBowl
    
    print("Testing Optimizers on Quadratic Bowl\n" + "="*50)
    
    surface = QuadraticBowl(a=1.0, b=10.0)
    initial = np.array([4.0, 4.0])
    
    for optimizer in get_all_optimizers(learning_rate=0.1):
        history = run_optimization(
            surface, optimizer, initial, 
            num_steps=100, convergence_threshold=1e-6
        )
        
        print(f"\n{optimizer.name}:")
        print(f"  Steps: {len(history.steps)}")
        print(f"  Final position: [{history.final_position[0]:.6f}, {history.final_position[1]:.6f}]")
        print(f"  Final loss: {history.final_loss:.6f}")

