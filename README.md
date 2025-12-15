# Optimizer Visualizer for Gradient Descent

An interactive visualization tool for understanding how different optimization algorithms navigate loss surfaces during gradient descent.

## What Is This?

When training machine learning models, we use **gradient descent** to find the parameters that minimize a loss function. But vanilla gradient descent isn't the only option—there are many optimization algorithms (SGD, Momentum, Adam, etc.) that behave differently depending on the shape of the loss landscape.

This app lets you **see** how these optimizers work by watching them navigate different loss surfaces in real-time.

## Why Does This Matter?

Understanding optimizer behavior helps you:
1. **Choose the right optimizer** for your problem
2. **Tune hyperparameters** like learning rate more effectively
3. **Debug training issues** (divergence, slow convergence, getting stuck)
4. **Build intuition** for what's happening "under the hood" during training

## How to Use the App

### Running the App

```bash
cd MOML
pip install -r requirements.txt
python app.py
```

Then open your browser to `http://127.0.0.1:8050`

### The Interface

1. **Loss Surface** (left panel): Choose which loss landscape to visualize
2. **Optimizers** (left panel): Select which optimizers to compare (you can run multiple at once)
3. **Parameters** (left panel): Adjust learning rate, number of steps, and starting position
4. **Run Optimization** (button): Click to see the optimizers in action!

### What to Look For

- **Trajectory paths**: Each colored line shows how an optimizer moves through the parameter space
- **Loss curves**: See how quickly each optimizer reduces the loss over time
- **Gradient magnitude**: Observe how the gradient changes as optimizers approach the minimum

## Key Concepts Demonstrated

### 1. Learning Rate Matters
Try the **Quadratic Bowl** with different learning rates:
- Too small (10⁻⁴): Painfully slow convergence
- Just right (10⁻¹): Smooth path to the minimum
- Too large (10⁰): May overshoot and diverge!

### 2. Momentum Helps with Oscillation
On the **Ill-Conditioned Quadratic** (elongated bowl):
- **SGD** oscillates back and forth across the narrow valley
- **Momentum** smooths out the path by building up velocity in consistent directions

### 3. Adaptive Methods Handle Scale
On surfaces with varying curvature:
- **AdaGrad** adapts learning rate per-parameter based on past gradients
- **RMSprop** prevents the learning rate from decaying too aggressively
- **Adam** combines the best of momentum and adaptive learning rates

### 4. Local Minima and Saddle Points
On **Rastrigin** (many local minima) or **Saddle Point**:
- Watch how different optimizers may get stuck or escape
- Momentum-based methods often do better at escaping local minima

## Loss Surfaces Explained

| Surface | Shape | What It Tests |
|---------|-------|---------------|
| **Quadratic Bowl** | Simple parabola | Basic convergence |
| **Ill-Conditioned Quadratic** | Elongated bowl | Handling different curvatures |
| **Elliptical Bowl** | Rotated ellipse | Correlated dimensions |
| **Rastrigin** | Many bumps | Escaping local minima |
| **Ackley** | Flat with deep center | Vanishing gradients |
| **Saddle Point** | Horse saddle shape | Escaping saddle points |

## Optimizers Explained

| Optimizer | Update Rule | Key Insight |
|-----------|-------------|-------------|
| **Gradient Descent** | `x -= lr * gradient` | Simplest possible approach |
| **Momentum** | Adds velocity term | Builds up speed in consistent directions |
| **Nesterov** | Look-ahead gradient | "Smarter" momentum that anticipates |
| **AdaGrad** | Adapts lr per-parameter | Good for sparse gradients |
| **RMSprop** | Moving average of squared gradients | Fixes AdaGrad's aggressive decay |
| **Adam** | Momentum + RMSprop | Most popular in deep learning |

## Suggested Experiments

1. **Compare all optimizers** on the Quadratic Bowl—they should all converge, but at different speeds

2. **Crank up the learning rate** and watch which optimizers diverge first (Adam is usually most stable)

3. **Try the Saddle Point** surface—notice how optimizers behave at the origin where the gradient is zero but it's not a minimum

4. **Use Rastrigin** to see which optimizers escape local minima (hint: momentum helps!)

5. **Start from different positions** to see how initial conditions affect the path

## Project Structure

```
MOML/
├── app.py                # Web application (Dash)
├── loss_surfaces.py      # Loss surface definitions
├── optimizers.py         # Optimizer implementations
├── datasets.py           # Dataset generators
├── visualizer.py         # Visualization utilities
├── main.py               # CLI version
├── optimizer_visualizer.ipynb  # Jupyter notebook version
└── requirements.txt      # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Required packages: numpy, plotly, dash, dash-bootstrap-components

## Credits

Built for Math of Machine Learning course to help visualize and understand gradient descent optimization algorithms.
