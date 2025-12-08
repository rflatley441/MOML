"""
Dataset Generators and PCA Utilities

This module provides:
- 2D synthetic datasets for direct visualization
- Higher-dimensional datasets with PCA reduction
- Loss surface fitting from data
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs


class Dataset:
    """Container for dataset with features and labels."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
        self.X = X
        self.y = y
        self.name = name
        self.n_samples, self.n_features = X.shape
    
    def __repr__(self):
        return f"Dataset('{self.name}', samples={self.n_samples}, features={self.n_features})"


# =============================================================================
# 2D Synthetic Datasets
# =============================================================================

def generate_linear_separable(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42
) -> Dataset:
    """
    Generate linearly separable 2D dataset.
    
    Creates a convex loss landscape for logistic regression.
    """
    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_per_class, 2) * noise + np.array([-1, -1])
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_per_class, 2) * noise + np.array([1, 1])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    
    return Dataset(X[idx], y[idx], "Linear Separable")


def generate_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42
) -> Dataset:
    """
    Generate two interleaving half circles (moons).
    
    Creates a non-convex loss landscape - requires non-linear decision boundary.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return Dataset(X, y, "Moons")


def generate_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    factor: float = 0.5,
    random_state: int = 42
) -> Dataset:
    """
    Generate concentric circles dataset.
    
    Creates a challenging loss landscape with potential local minima.
    """
    X, y = make_circles(
        n_samples=n_samples, 
        noise=noise, 
        factor=factor, 
        random_state=random_state
    )
    return Dataset(X, y, "Circles")


def generate_xor(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42
) -> Dataset:
    """
    Generate XOR pattern dataset.
    
    Classic non-linearly separable problem with multiple local minima.
    """
    np.random.seed(random_state)
    
    n_per_quadrant = n_samples // 4
    
    # XOR pattern: (0,0) and (1,1) -> class 0, (0,1) and (1,0) -> class 1
    centers = [
        (np.array([0, 0]), 0),
        (np.array([1, 1]), 0),
        (np.array([0, 1]), 1),
        (np.array([1, 0]), 1),
    ]
    
    X_list = []
    y_list = []
    
    for center, label in centers:
        X_list.append(np.random.randn(n_per_quadrant, 2) * noise + center)
        y_list.extend([label] * n_per_quadrant)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    idx = np.random.permutation(len(y))
    
    return Dataset(X[idx], y[idx], "XOR")


def generate_spiral(
    n_samples: int = 200,
    noise: float = 0.1,
    n_turns: float = 1.5,
    random_state: int = 42
) -> Dataset:
    """
    Generate spiral dataset.
    
    Very challenging - creates highly non-convex loss with many local minima.
    """
    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Generate spiral for class 0
    theta0 = np.linspace(0, n_turns * 2 * np.pi, n_per_class)
    r0 = theta0 / (n_turns * 2 * np.pi)
    X0 = np.column_stack([
        r0 * np.cos(theta0) + np.random.randn(n_per_class) * noise,
        r0 * np.sin(theta0) + np.random.randn(n_per_class) * noise
    ])
    
    # Generate spiral for class 1 (rotated by pi)
    theta1 = theta0 + np.pi
    r1 = theta1 / (n_turns * 2 * np.pi + np.pi)
    X1 = np.column_stack([
        r1 * np.cos(theta1) + np.random.randn(n_per_class) * noise,
        r1 * np.sin(theta1) + np.random.randn(n_per_class) * noise
    ])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    idx = np.random.permutation(n_samples)
    
    return Dataset(X[idx], y[idx], "Spiral")


def generate_clusters(
    n_samples: int = 300,
    n_clusters: int = 3,
    cluster_std: float = 0.5,
    random_state: int = 42
) -> Dataset:
    """
    Generate clustered data with multiple classes.
    
    Creates loss landscape with multiple local minima corresponding to clusters.
    """
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return Dataset(X, y, f"Clusters (k={n_clusters})")


# =============================================================================
# Higher-Dimensional Datasets with PCA
# =============================================================================

def generate_high_dim_classification(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 2,
    random_state: int = 42
) -> Dataset:
    """
    Generate high-dimensional classification dataset.
    
    Use with PCA to visualize in 2D.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=n_classes,
        random_state=random_state
    )
    return Dataset(X, y, f"High-Dim Classification ({n_features}D)")


class PCAProjector:
    """
    Project high-dimensional data and weight vectors to 2D using PCA.
    
    Useful for visualizing optimization trajectories in weight space.
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'PCAProjector':
        """Fit PCA on data."""
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensions."""
        if not self.is_fitted:
            raise ValueError("PCAProjector must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Transform back to original space."""
        X_scaled = self.pca.inverse_transform(X_reduced)
        return self.scaler.inverse_transform(X_scaled)
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        return self.pca.explained_variance_ratio_
    
    @property
    def components(self) -> np.ndarray:
        """Return principal components."""
        return self.pca.components_


# =============================================================================
# Loss Surface from Data (for ML models)
# =============================================================================

class LogisticRegressionLoss:
    """
    Logistic regression loss surface for 2D weight visualization.
    
    For 2D input data, we have 2 weights + 1 bias = 3 parameters.
    We can fix the bias and visualize the 2D weight space.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, regularization: float = 0.0):
        """
        Args:
            X: Input features (n_samples, 2)
            y: Binary labels (n_samples,)
            regularization: L2 regularization strength
        """
        self.X = X
        self.y = y
        self.regularization = regularization
        self.n_samples = len(y)
        
        # Add bias column
        self.X_with_bias = np.column_stack([X, np.ones(self.n_samples)])
        
        self.name = "Logistic Regression Loss"
        self.description = f"Binary cross-entropy on {self.n_samples} samples"
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def __call__(self, w: np.ndarray) -> float:
        """
        Compute loss for weight vector w = [w1, w2, bias].
        
        If w is 2D, assumes bias = 0.
        """
        if len(w) == 2:
            w = np.array([w[0], w[1], 0.0])
        
        z = self.X_with_bias @ w
        p = self._sigmoid(z)
        
        # Binary cross-entropy with numerical stability
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        loss = -np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p))
        
        # L2 regularization (not on bias)
        if self.regularization > 0:
            loss += 0.5 * self.regularization * np.sum(w[:2]**2)
        
        return loss
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """
        Compute gradient for weight vector.
        
        If w is 2D, returns 2D gradient (assumes bias = 0).
        """
        return_2d = len(w) == 2
        if return_2d:
            w = np.array([w[0], w[1], 0.0])
        
        z = self.X_with_bias @ w
        p = self._sigmoid(z)
        
        error = p - self.y
        grad = self.X_with_bias.T @ error / self.n_samples
        
        # L2 regularization gradient (not on bias)
        if self.regularization > 0:
            grad[:2] += self.regularization * w[:2]
        
        if return_2d:
            return grad[:2]
        return grad
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return reasonable bounds for visualization."""
        return (-5, 5, -5, 5)
    
    def get_optimal(self) -> np.ndarray:
        """Return approximate optimal (found by optimization)."""
        from scipy.optimize import minimize
        result = minimize(
            lambda w: self(w),
            x0=np.zeros(3),
            jac=lambda w: self.gradient(w),
            method='L-BFGS-B'
        )
        return result.x[:2]


class NeuralNetworkLoss:
    """
    Simple neural network loss for visualization.
    
    Projects high-dimensional weight space to 2D using two random directions.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        hidden_size: int = 10,
        random_state: int = 42
    ):
        self.X = X
        self.y = y
        self.hidden_size = hidden_size
        self.n_samples, self.n_features = X.shape
        
        np.random.seed(random_state)
        
        # Initialize network weights
        self.W1 = np.random.randn(self.n_features, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # Random directions for 2D projection
        self.dir1 = self._random_direction()
        self.dir2 = self._random_direction()
        
        self.name = "Neural Network Loss"
        self.description = f"2-layer NN on {self.n_samples} samples"
    
    def _random_direction(self) -> dict:
        """Generate random direction in weight space."""
        return {
            'W1': np.random.randn(*self.W1.shape),
            'b1': np.random.randn(*self.b1.shape),
            'W2': np.random.randn(*self.W2.shape),
            'b2': np.random.randn(*self.b2.shape),
        }
    
    def _get_weights_at_point(self, x: np.ndarray) -> Tuple:
        """Get network weights at 2D point x."""
        W1 = self.W1 + x[0] * self.dir1['W1'] + x[1] * self.dir2['W1']
        b1 = self.b1 + x[0] * self.dir1['b1'] + x[1] * self.dir2['b1']
        W2 = self.W2 + x[0] * self.dir1['W2'] + x[1] * self.dir2['W2']
        b2 = self.b2 + x[0] * self.dir1['b2'] + x[1] * self.dir2['b2']
        return W1, b1, W2, b2
    
    def _forward(self, W1, b1, W2, b2) -> np.ndarray:
        """Forward pass."""
        h = np.maximum(0, self.X @ W1 + b1)  # ReLU
        return 1 / (1 + np.exp(-(h @ W2 + b2)))  # Sigmoid
    
    def __call__(self, x: np.ndarray) -> float:
        """Compute loss at 2D point."""
        W1, b1, W2, b2 = self._get_weights_at_point(x)
        p = self._forward(W1, b1, W2, b2).flatten()
        
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        return -np.mean(self.y * np.log(p) + (1 - self.y) * np.log(1 - p))
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Numerical gradient (analytical is complex for projection)."""
        eps = 1e-5
        grad = np.zeros(2)
        
        for i in range(2):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * eps)
        
        return grad
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (-2, 2, -2, 2)
    
    def get_optimal(self) -> np.ndarray:
        return np.array([0.0, 0.0])


# =============================================================================
# Dataset Factory
# =============================================================================

def get_all_2d_datasets() -> List[Dataset]:
    """Return all available 2D datasets."""
    return [
        generate_linear_separable(),
        generate_moons(),
        generate_circles(),
        generate_xor(),
        generate_spiral(),
        generate_clusters(),
    ]


def get_datasets_by_complexity() -> dict:
    """Return datasets organized by optimization complexity."""
    return {
        "convex": [
            generate_linear_separable(noise=0.3),
        ],
        "moderate": [
            generate_moons(noise=0.1),
            generate_circles(noise=0.05),
        ],
        "challenging": [
            generate_xor(noise=0.1),
            generate_spiral(noise=0.05),
        ]
    }


if __name__ == "__main__":
    print("Testing Datasets\n" + "="*50)
    
    for dataset in get_all_2d_datasets():
        print(f"\n{dataset}")
        print(f"  X shape: {dataset.X.shape}")
        print(f"  y unique: {np.unique(dataset.y)}")
        print(f"  X range: [{dataset.X.min():.2f}, {dataset.X.max():.2f}]")
    
    print("\n\nTesting LogisticRegressionLoss\n" + "="*50)
    
    data = generate_linear_separable()
    loss_surface = LogisticRegressionLoss(data.X, data.y)
    
    w = np.array([1.0, 1.0])
    print(f"Loss at w=[1,1]: {loss_surface(w):.4f}")
    print(f"Gradient at w=[1,1]: {loss_surface.gradient(w)}")
    print(f"Optimal weights: {loss_surface.get_optimal()}")
    
    print("\n\nTesting PCA Projector\n" + "="*50)
    
    high_dim_data = generate_high_dim_classification()
    print(f"Original: {high_dim_data}")
    
    projector = PCAProjector(n_components=2)
    X_2d = projector.fit_transform(high_dim_data.X)
    print(f"Projected shape: {X_2d.shape}")
    print(f"Explained variance: {projector.explained_variance_ratio}")

