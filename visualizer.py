"""
Visualization Engine for Optimizer Comparison

This module provides:
- 3D surface plots with optimizer trajectories
- 2D contour plots with paths
- Animation support
- Side-by-side optimizer comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple, Dict
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Color palette for optimizers (colorblind-friendly)
OPTIMIZER_COLORS = {
    'SGD': '#E69F00',        # Orange
    'Momentum': '#56B4E9',   # Sky blue
    'Nesterov': '#009E73',   # Bluish green
    'AdaGrad': '#F0E442',    # Yellow
    'RMSprop': '#0072B2',    # Blue
    'Adam': '#D55E00',       # Vermillion
    'AdamW': '#CC79A7',      # Reddish purple
}

# Fallback colors
DEFAULT_COLORS = plt.cm.tab10.colors


def get_optimizer_color(name: str, idx: int = 0) -> str:
    """Get color for optimizer by name."""
    return OPTIMIZER_COLORS.get(name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


class SurfaceVisualizer:
    """
    Visualize loss surfaces and optimization trajectories.
    """
    
    def __init__(self, loss_surface, resolution: int = 100):
        """
        Args:
            loss_surface: Loss surface object with __call__ and get_bounds methods
            resolution: Grid resolution for surface plotting
        """
        self.loss_surface = loss_surface
        self.resolution = resolution
        
        # Compute surface grid
        bounds = loss_surface.get_bounds()
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        
        self.x_grid = np.linspace(self.x_min, self.x_max, resolution)
        self.y_grid = np.linspace(self.y_min, self.y_max, resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Compute Z values
        self.Z = np.zeros_like(self.X)
        for i in range(resolution):
            for j in range(resolution):
                self.Z[i, j] = loss_surface(np.array([self.X[i, j], self.Y[i, j]]))
    
    def plot_surface_3d(
        self,
        ax: Optional[plt.Axes] = None,
        trajectories: Optional[List] = None,
        title: Optional[str] = None,
        elevation: float = 30,
        azimuth: float = -60,
        alpha: float = 0.7,
        cmap: str = 'viridis'
    ) -> plt.Axes:
        """
        Plot 3D surface with optional optimizer trajectories.
        
        Args:
            ax: Matplotlib 3D axes (created if None)
            trajectories: List of OptimizationHistory objects
            title: Plot title
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle
            alpha: Surface transparency
            cmap: Colormap for surface
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap=cmap,
            alpha=alpha,
            linewidth=0,
            antialiased=True
        )
        
        # Plot trajectories
        if trajectories:
            for idx, history in enumerate(trajectories):
                positions = history.positions
                losses = history.losses
                color = get_optimizer_color(history.optimizer_name, idx)
                
                # Plot trajectory line
                ax.plot(
                    positions[:, 0], positions[:, 1], losses,
                    color=color,
                    linewidth=2,
                    label=history.optimizer_name,
                    zorder=10
                )
                
                # Plot start point
                ax.scatter(
                    [positions[0, 0]], [positions[0, 1]], [losses[0]],
                    color=color, s=100, marker='o', edgecolors='white',
                    linewidths=2, zorder=11
                )
                
                # Plot end point
                ax.scatter(
                    [positions[-1, 0]], [positions[-1, 1]], [losses[-1]],
                    color=color, s=100, marker='*', edgecolors='white',
                    linewidths=2, zorder=11
                )
        
        # Labels and title
        ax.set_xlabel('$w_1$', fontsize=12)
        ax.set_ylabel('$w_2$', fontsize=12)
        ax.set_zlabel('Loss', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(self.loss_surface.name, fontsize=14, fontweight='bold')
        
        ax.view_init(elev=elevation, azim=azimuth)
        
        if trajectories:
            ax.legend(loc='upper left', fontsize=10)
        
        return ax
    
    def plot_contour(
        self,
        ax: Optional[plt.Axes] = None,
        trajectories: Optional[List] = None,
        title: Optional[str] = None,
        levels: int = 30,
        cmap: str = 'viridis',
        show_gradient_field: bool = False,
        log_scale: bool = False
    ) -> plt.Axes:
        """
        Plot 2D contour with optional optimizer trajectories.
        
        Args:
            ax: Matplotlib axes (created if None)
            trajectories: List of OptimizationHistory objects
            title: Plot title
            levels: Number of contour levels
            cmap: Colormap for contours
            show_gradient_field: Whether to show gradient arrows
            log_scale: Use log scale for contours
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare Z for plotting
        Z_plot = self.Z.copy()
        if log_scale:
            Z_plot = np.log1p(Z_plot)
        
        # Plot filled contours
        contour = ax.contourf(
            self.X, self.Y, Z_plot,
            levels=levels,
            cmap=cmap,
            alpha=0.8
        )
        
        # Plot contour lines
        ax.contour(
            self.X, self.Y, Z_plot,
            levels=levels,
            colors='black',
            alpha=0.3,
            linewidths=0.5
        )
        
        # Colorbar
        plt.colorbar(contour, ax=ax, label='Loss' + (' (log)' if log_scale else ''))
        
        # Gradient field
        if show_gradient_field:
            skip = max(1, self.resolution // 15)
            U = np.zeros_like(self.X)
            V = np.zeros_like(self.Y)
            
            for i in range(0, self.resolution, skip):
                for j in range(0, self.resolution, skip):
                    grad = self.loss_surface.gradient(
                        np.array([self.X[i, j], self.Y[i, j]])
                    )
                    # Normalize for visualization
                    norm = np.linalg.norm(grad)
                    if norm > 0:
                        U[i, j] = -grad[0] / norm
                        V[i, j] = -grad[1] / norm
            
            ax.quiver(
                self.X[::skip, ::skip], self.Y[::skip, ::skip],
                U[::skip, ::skip], V[::skip, ::skip],
                color='white', alpha=0.5, scale=25
            )
        
        # Plot trajectories
        if trajectories:
            for idx, history in enumerate(trajectories):
                positions = history.positions
                color = get_optimizer_color(history.optimizer_name, idx)
                
                # Plot trajectory line
                ax.plot(
                    positions[:, 0], positions[:, 1],
                    color=color,
                    linewidth=2.5,
                    label=f"{history.optimizer_name} ({len(history.steps)} steps)",
                    zorder=10
                )
                
                # Plot points along trajectory
                ax.scatter(
                    positions[::5, 0], positions[::5, 1],
                    color=color, s=20, alpha=0.6, zorder=9
                )
                
                # Start point (circle)
                ax.scatter(
                    [positions[0, 0]], [positions[0, 1]],
                    color=color, s=150, marker='o', edgecolors='white',
                    linewidths=2, zorder=11
                )
                
                # End point (star)
                ax.scatter(
                    [positions[-1, 0]], [positions[-1, 1]],
                    color=color, s=200, marker='*', edgecolors='white',
                    linewidths=2, zorder=11
                )
        
        # Plot optimal point if known
        optimal = self.loss_surface.get_optimal()
        ax.scatter(
            [optimal[0]], [optimal[1]],
            color='red', s=200, marker='X', edgecolors='white',
            linewidths=2, zorder=12, label='Optimal'
        )
        
        # Labels and title
        ax.set_xlabel('$w_1$', fontsize=12)
        ax.set_ylabel('$w_2$', fontsize=12)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(self.loss_surface.name, fontsize=14, fontweight='bold')
        
        if trajectories:
            ax.legend(loc='upper right', fontsize=9)
        
        ax.set_aspect('equal')
        
        return ax
    
    def plot_loss_curves(
        self,
        trajectories: List,
        ax: Optional[plt.Axes] = None,
        title: str = "Loss Convergence",
        log_scale: bool = True
    ) -> plt.Axes:
        """
        Plot loss over iterations for multiple optimizers.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, history in enumerate(trajectories):
            color = get_optimizer_color(history.optimizer_name, idx)
            
            losses = history.losses
            steps = np.arange(len(losses))
            
            ax.plot(
                steps, losses,
                color=color,
                linewidth=2,
                label=f"{history.optimizer_name} (final: {losses[-1]:.4f})"
            )
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gradient_norms(
        self,
        trajectories: List,
        ax: Optional[plt.Axes] = None,
        title: str = "Gradient Magnitude",
        log_scale: bool = True
    ) -> plt.Axes:
        """
        Plot gradient norm over iterations.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for idx, history in enumerate(trajectories):
            color = get_optimizer_color(history.optimizer_name, idx)
            
            grad_norms = np.linalg.norm(history.gradients, axis=1)
            steps = np.arange(len(grad_norms))
            
            ax.plot(
                steps, grad_norms,
                color=color,
                linewidth=2,
                label=history.optimizer_name
            )
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('||∇f||', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax


def create_comparison_figure(
    loss_surface,
    trajectories: List,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive comparison figure with multiple views.
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    visualizer = SurfaceVisualizer(loss_surface)
    
    # 3D surface (top-left)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    visualizer.plot_surface_3d(ax=ax1, trajectories=trajectories)
    
    # 2D contour (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    visualizer.plot_contour(ax=ax2, trajectories=trajectories)
    
    # Loss curves (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    visualizer.plot_loss_curves(trajectories, ax=ax3)
    
    # Gradient norms (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    visualizer.plot_gradient_norms(trajectories, ax=ax4)
    
    # Overall title
    fig.suptitle(
        f"Optimizer Comparison on {loss_surface.name}",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    return fig


def create_animation(
    loss_surface,
    trajectories: List,
    fps: int = 10,
    figsize: Tuple[int, int] = (12, 5)
) -> FuncAnimation:
    """
    Create an animated visualization of optimization progress.
    """
    visualizer = SurfaceVisualizer(loss_surface)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Find max steps
    max_steps = max(len(h.steps) for h in trajectories)
    
    # Plot static contour
    visualizer.plot_contour(ax=ax1, title="Optimization Progress")
    
    # Initialize trajectory lines
    lines = []
    points = []
    for idx, history in enumerate(trajectories):
        color = get_optimizer_color(history.optimizer_name, idx)
        line, = ax1.plot([], [], color=color, linewidth=2, label=history.optimizer_name)
        point, = ax1.plot([], [], 'o', color=color, markersize=10)
        lines.append(line)
        points.append(point)
    
    ax1.legend(loc='upper right')
    
    # Loss plot
    loss_lines = []
    for idx, history in enumerate(trajectories):
        color = get_optimizer_color(history.optimizer_name, idx)
        line, = ax2.plot([], [], color=color, linewidth=2, label=history.optimizer_name)
        loss_lines.append(line)
    
    ax2.set_xlim(0, max_steps)
    all_losses = np.concatenate([h.losses for h in trajectories])
    ax2.set_ylim(all_losses.min() * 0.9, all_losses.max() * 1.1)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    def animate(frame):
        for idx, history in enumerate(trajectories):
            positions = history.positions
            losses = history.losses
            
            # Limit to current frame
            n = min(frame + 1, len(positions))
            
            # Update trajectory
            lines[idx].set_data(positions[:n, 0], positions[:n, 1])
            points[idx].set_data([positions[n-1, 0]], [positions[n-1, 1]])
            
            # Update loss curve
            loss_lines[idx].set_data(np.arange(n), losses[:n])
        
        return lines + points + loss_lines
    
    anim = FuncAnimation(
        fig, animate, frames=max_steps,
        interval=1000/fps, blit=True
    )
    
    return anim


def plot_surface_comparison(
    loss_surfaces: List,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot multiple loss surfaces side by side.
    """
    n_surfaces = len(loss_surfaces)
    fig, axes = plt.subplots(1, n_surfaces, figsize=figsize)
    
    if n_surfaces == 1:
        axes = [axes]
    
    for ax, surface in zip(axes, loss_surfaces):
        visualizer = SurfaceVisualizer(surface)
        visualizer.plot_contour(ax=ax, title=surface.name)
    
    fig.suptitle("Loss Surface Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_optimizer_summary_table(trajectories: List) -> str:
    """
    Create a text summary table of optimizer performance.
    """
    header = f"{'Optimizer':<15} {'Steps':<8} {'Final Loss':<12} {'Final Position':<25}"
    separator = "-" * len(header)
    
    rows = [header, separator]
    
    for history in trajectories:
        pos = history.final_position
        pos_str = f"[{pos[0]:.4f}, {pos[1]:.4f}]"
        row = f"{history.optimizer_name:<15} {len(history.steps):<8} {history.final_loss:<12.6f} {pos_str:<25}"
        rows.append(row)
    
    return "\n".join(rows)


if __name__ == "__main__":
    # Demo visualization
    from loss_surfaces import Rosenbrock, QuadraticBowl, Rastrigin
    from optimizers import get_all_optimizers, run_optimization
    
    print("Creating demo visualization...")
    
    # Test on Rosenbrock
    surface = Rosenbrock()
    initial = np.array([-1.5, 1.5])
    
    # Run all optimizers
    trajectories = []
    for opt in get_all_optimizers(learning_rate=0.001):
        history = run_optimization(surface, opt, initial, num_steps=500)
        trajectories.append(history)
    
    # Create comparison figure
    fig = create_comparison_figure(surface, trajectories)
    
    # Print summary
    print("\n" + create_optimizer_summary_table(trajectories))
    
    plt.tight_layout()
    plt.show()

