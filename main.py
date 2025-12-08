#!/usr/bin/env python3
"""
Optimizer Visualizer - Main Application

Interactive tool for visualizing gradient descent optimizers on various loss surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import argparse

from loss_surfaces import (
    QuadraticBowl, EllipticalBowl, Rosenbrock, Rastrigin, 
    Ackley, Beale, SaddlePoint, SixHumpCamel,
    get_loss_surfaces_by_category, get_all_loss_surfaces
)
from optimizers import (
    SGD, Momentum, NesterovMomentum, AdaGrad, RMSprop, Adam, AdamW,
    run_optimization, get_all_optimizers, get_optimizer_by_name
)
from datasets import (
    generate_linear_separable, generate_moons, generate_circles,
    generate_xor, generate_spiral, LogisticRegressionLoss
)
from visualizer import (
    SurfaceVisualizer, create_comparison_figure, 
    create_animation, plot_surface_comparison, create_optimizer_summary_table
)


def demo_convex():
    """
    Demo: Convex loss surface (Quadratic Bowl)
    
    Shows how all optimizers converge to the global minimum.
    Demonstrates the effect of conditioning on convergence speed.
    """
    print("\n" + "="*60)
    print("DEMO 1: Convex Loss Surface (Quadratic Bowl)")
    print("="*60)
    print("\nThis surface has a single global minimum at the origin.")
    print("All optimizers should converge, but at different rates.\n")
    
    # Well-conditioned quadratic
    surface = QuadraticBowl(a=1.0, b=1.0)
    initial = np.array([4.0, 4.0])
    
    print(f"Surface: {surface.name}")
    print(f"Starting point: {initial}")
    print(f"Optimal: {surface.get_optimal()}\n")
    
    # Run optimizers
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.1),
        Momentum(learning_rate=0.1, beta=0.9),
        Adam(learning_rate=0.1),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=50)
        trajectories.append(history)
        print(f"{opt.name}: {len(history.steps)} steps, final loss = {history.final_loss:.6f}")
    
    # Visualize
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_convex.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_convex.png")
    plt.show()
    
    return trajectories


def demo_ill_conditioned():
    """
    Demo: Ill-conditioned quadratic (elongated bowl)
    
    Shows how adaptive methods handle different scales in dimensions.
    """
    print("\n" + "="*60)
    print("DEMO 2: Ill-Conditioned Quadratic")
    print("="*60)
    print("\nThis surface has different curvatures in x and y directions.")
    print("Condition number = 10 (y direction is 10x steeper).\n")
    
    surface = QuadraticBowl(a=1.0, b=10.0)
    initial = np.array([4.0, 4.0])
    
    print(f"Surface: {surface.description}")
    print(f"Starting point: {initial}\n")
    
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.05),
        Momentum(learning_rate=0.05, beta=0.9),
        Adam(learning_rate=0.1),
        RMSprop(learning_rate=0.1),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=100)
        trajectories.append(history)
        print(f"{opt.name}: {len(history.steps)} steps, final loss = {history.final_loss:.6f}")
    
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_ill_conditioned.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_ill_conditioned.png")
    plt.show()
    
    return trajectories


def demo_local_minima():
    """
    Demo: Multiple local minima (Rastrigin function)
    
    Shows how optimizers can get trapped in local minima.
    """
    print("\n" + "="*60)
    print("DEMO 3: Multiple Local Minima (Rastrigin)")
    print("="*60)
    print("\nThis surface has many local minima arranged in a grid.")
    print("Global minimum is at the origin, but many traps exist.\n")
    
    surface = Rastrigin(A=10.0)
    initial = np.array([3.0, 3.0])
    
    print(f"Surface: {surface.name}")
    print(f"Starting point: {initial}")
    print(f"Global optimal: {surface.get_optimal()}\n")
    
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.01),
        Momentum(learning_rate=0.01, beta=0.9),
        Adam(learning_rate=0.05),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=200)
        trajectories.append(history)
        final_pos = history.final_position
        print(f"{opt.name}: final position = [{final_pos[0]:.3f}, {final_pos[1]:.3f}], loss = {history.final_loss:.4f}")
    
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_local_minima.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_local_minima.png")
    plt.show()
    
    return trajectories


def demo_non_convex():
    """
    Demo: Non-convex surface (Rosenbrock)
    
    Shows the challenge of navigating narrow valleys.
    """
    print("\n" + "="*60)
    print("DEMO 4: Non-Convex Surface (Rosenbrock)")
    print("="*60)
    print("\nThe 'banana' function with a narrow curved valley.")
    print("Finding the valley is easy, converging to the minimum is hard.\n")
    
    surface = Rosenbrock(a=1.0, b=100.0)
    initial = np.array([-1.5, 1.5])
    
    print(f"Surface: {surface.name}")
    print(f"Starting point: {initial}")
    print(f"Optimal: {surface.get_optimal()}\n")
    
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.0001),
        Momentum(learning_rate=0.0001, beta=0.9),
        Adam(learning_rate=0.01),
        RMSprop(learning_rate=0.001),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=1000)
        trajectories.append(history)
        final_pos = history.final_position
        dist_to_opt = np.linalg.norm(final_pos - surface.get_optimal())
        print(f"{opt.name}: distance to optimal = {dist_to_opt:.4f}, loss = {history.final_loss:.4f}")
    
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_non_convex.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_non_convex.png")
    plt.show()
    
    return trajectories


def demo_saddle_point():
    """
    Demo: Saddle point behavior
    
    Shows how different optimizers handle saddle points.
    """
    print("\n" + "="*60)
    print("DEMO 5: Saddle Point")
    print("="*60)
    print("\nSaddle point at origin: gradient is zero but not a minimum.")
    print("Momentum helps escape, SGD can get stuck.\n")
    
    surface = SaddlePoint()
    initial = np.array([0.1, 0.1])  # Start near saddle
    
    print(f"Surface: {surface.name}")
    print(f"Starting point: {initial} (near saddle at origin)\n")
    
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.1),
        Momentum(learning_rate=0.1, beta=0.9),
        Adam(learning_rate=0.1),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=50)
        trajectories.append(history)
        final_pos = history.final_position
        print(f"{opt.name}: final position = [{final_pos[0]:.3f}, {final_pos[1]:.3f}]")
    
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_saddle.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_saddle.png")
    plt.show()
    
    return trajectories


def demo_data_driven():
    """
    Demo: Data-driven loss surface (Logistic Regression)
    
    Shows optimization on actual ML loss landscape.
    """
    print("\n" + "="*60)
    print("DEMO 6: Data-Driven Loss (Logistic Regression)")
    print("="*60)
    print("\nOptimizing logistic regression on synthetic classification data.\n")
    
    # Generate dataset
    dataset = generate_moons(n_samples=200, noise=0.1)
    print(f"Dataset: {dataset.name} ({dataset.n_samples} samples)")
    
    # Create loss surface
    surface = LogisticRegressionLoss(dataset.X, dataset.y, regularization=0.01)
    initial = np.array([0.0, 0.0])
    
    print(f"Initial weights: {initial}")
    print(f"Optimal (approx): {surface.get_optimal()}\n")
    
    trajectories = []
    optimizers = [
        SGD(learning_rate=0.5),
        Momentum(learning_rate=0.5, beta=0.9),
        Adam(learning_rate=0.1),
    ]
    
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=100)
        trajectories.append(history)
        print(f"{opt.name}: final loss = {history.final_loss:.4f}")
    
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('demo_data_driven.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_data_driven.png")
    plt.show()
    
    return trajectories


def demo_learning_rate_comparison():
    """
    Demo: Effect of learning rate
    
    Shows how learning rate affects convergence.
    """
    print("\n" + "="*60)
    print("DEMO 7: Learning Rate Comparison")
    print("="*60)
    print("\nComparing different learning rates on SGD.\n")
    
    surface = QuadraticBowl(a=1.0, b=5.0)
    initial = np.array([4.0, 4.0])
    
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        opt = SGD(learning_rate=lr)
        history = run_optimization(surface, opt, initial, num_steps=50)
        
        visualizer = SurfaceVisualizer(surface)
        visualizer.plot_contour(ax=axes[idx], trajectories=[history], 
                               title=f"SGD (lr={lr})")
        
        print(f"lr={lr}: {len(history.steps)} steps, final loss = {history.final_loss:.6f}")
    
    plt.tight_layout()
    plt.savefig('demo_learning_rates.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_learning_rates.png")
    plt.show()


def demo_all_surfaces():
    """
    Demo: Overview of all loss surfaces
    """
    print("\n" + "="*60)
    print("DEMO 8: All Loss Surfaces Overview")
    print("="*60)
    
    surfaces = get_all_loss_surfaces()
    
    n_cols = 3
    n_rows = (len(surfaces) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, surface in enumerate(surfaces):
        visualizer = SurfaceVisualizer(surface, resolution=80)
        visualizer.plot_contour(ax=axes[idx], title=surface.name)
        print(f"{surface.name}: {surface.description}")
    
    # Hide unused axes
    for idx in range(len(surfaces), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('demo_all_surfaces.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_all_surfaces.png")
    plt.show()


def interactive_mode():
    """
    Interactive mode for exploring optimizers.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    # Surface selection
    print("\nAvailable loss surfaces:")
    surfaces = {
        '1': ('Quadratic Bowl', QuadraticBowl()),
        '2': ('Ill-Conditioned Quadratic', QuadraticBowl(a=1.0, b=10.0)),
        '3': ('Rosenbrock', Rosenbrock()),
        '4': ('Rastrigin', Rastrigin()),
        '5': ('Ackley', Ackley()),
        '6': ('Six-Hump Camel', SixHumpCamel()),
        '7': ('Saddle Point', SaddlePoint()),
    }
    
    for key, (name, _) in surfaces.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect surface (1-7): ").strip()
    if choice not in surfaces:
        print("Invalid choice, using Quadratic Bowl")
        choice = '1'
    
    surface_name, surface = surfaces[choice]
    print(f"\nSelected: {surface_name}")
    
    # Starting point
    bounds = surface.get_bounds()
    print(f"\nSurface bounds: x=[{bounds[0]}, {bounds[1]}], y=[{bounds[2]}, {bounds[3]}]")
    
    try:
        x0 = float(input("Enter starting x (or press Enter for default): ").strip() or bounds[1]*0.8)
        y0 = float(input("Enter starting y (or press Enter for default): ").strip() or bounds[3]*0.8)
    except ValueError:
        x0, y0 = bounds[1]*0.8, bounds[3]*0.8
    
    initial = np.array([x0, y0])
    print(f"Starting point: [{x0}, {y0}]")
    
    # Optimizer selection
    print("\nAvailable optimizers:")
    print("  1. SGD")
    print("  2. Momentum")
    print("  3. Nesterov")
    print("  4. AdaGrad")
    print("  5. RMSprop")
    print("  6. Adam")
    print("  7. All of the above")
    
    opt_choice = input("\nSelect optimizer(s) (1-7): ").strip()
    
    # Learning rate
    try:
        lr = float(input("Learning rate (default 0.01): ").strip() or 0.01)
    except ValueError:
        lr = 0.01
    
    # Number of steps
    try:
        num_steps = int(input("Number of steps (default 100): ").strip() or 100)
    except ValueError:
        num_steps = 100
    
    # Create optimizers
    if opt_choice == '7':
        optimizers = get_all_optimizers(learning_rate=lr)
    else:
        opt_map = {
            '1': SGD(learning_rate=lr),
            '2': Momentum(learning_rate=lr),
            '3': NesterovMomentum(learning_rate=lr),
            '4': AdaGrad(learning_rate=lr),
            '5': RMSprop(learning_rate=lr),
            '6': Adam(learning_rate=lr),
        }
        optimizers = [opt_map.get(opt_choice, SGD(learning_rate=lr))]
    
    # Run optimization
    print("\nRunning optimization...")
    trajectories = []
    for opt in optimizers:
        history = run_optimization(surface, opt, initial, num_steps=num_steps)
        trajectories.append(history)
    
    # Display results
    print("\n" + create_optimizer_summary_table(trajectories))
    
    # Visualize
    fig = create_comparison_figure(surface, trajectories)
    plt.savefig('interactive_result.png', dpi=150, bbox_inches='tight')
    print("\nSaved: interactive_result.png")
    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimizer Visualizer for Gradient Descent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo convex      # Run convex surface demo
  python main.py --demo all         # Run all demos
  python main.py --interactive      # Interactive mode
  python main.py                    # Run default demo sequence
        """
    )
    
    parser.add_argument(
        '--demo', 
        choices=['convex', 'ill-conditioned', 'local-minima', 'non-convex', 
                 'saddle', 'data', 'learning-rate', 'surfaces', 'all'],
        help='Run specific demo'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("   OPTIMIZER VISUALIZER FOR GRADIENT DESCENT")
    print("   Math of Machine Learning")
    print("="*60)
    
    if args.interactive:
        interactive_mode()
    elif args.demo:
        demos = {
            'convex': demo_convex,
            'ill-conditioned': demo_ill_conditioned,
            'local-minima': demo_local_minima,
            'non-convex': demo_non_convex,
            'saddle': demo_saddle_point,
            'data': demo_data_driven,
            'learning-rate': demo_learning_rate_comparison,
            'surfaces': demo_all_surfaces,
        }
        
        if args.demo == 'all':
            for name, demo_fn in demos.items():
                demo_fn()
                input("\nPress Enter to continue to next demo...")
        else:
            demos[args.demo]()
    else:
        # Default: run key demos
        print("\nRunning demonstration sequence...")
        print("(Use --demo <name> or --interactive for specific options)\n")
        
        demo_convex()
        input("\nPress Enter to continue...")
        
        demo_non_convex()
        input("\nPress Enter to continue...")
        
        demo_local_minima()
        
        print("\n" + "="*60)
        print("Demo complete! Try --interactive for custom exploration.")
        print("="*60)


if __name__ == "__main__":
    main()

