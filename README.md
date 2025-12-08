# Optimizer Visualizer for Gradient Descent

An interactive visualization tool for understanding how different optimization algorithms navigate loss surfaces during gradient descent.

## Overview

This project provides:
- **Multiple Loss Surfaces**: Convex, multi-local-minimum, and non-convex surfaces
- **Various Optimizers**: SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
- **Interactive Web App**: Beautiful dark-themed UI with real-time visualization
- **3D Surface Plots**: Interactive 3D visualization with optimizer trajectories
- **PCA Support**: Dimensionality reduction for higher-dimensional datasets

## Installation

```bash
cd MOML
pip install -r requirements.txt
```

## Usage

### 🌐 Web Application (Recommended)

```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

Features:
- Toggle multiple optimizers on/off
- Adjust learning rate and number of steps
- Choose from 9 different loss surfaces
- Interactive 3D surface visualization
- Real-time loss convergence plots
- Detailed results table

### Command Line

```bash
python main.py                    # Run demo sequence
python main.py --demo all         # Run all demos
python main.py --interactive      # Interactive CLI mode
```

### Jupyter Notebook

```bash
jupyter notebook optimizer_visualizer.ipynb
```

## Project Structure

```
MOML/
├── app.py                # 🌐 Web application (Dash)
├── loss_surfaces.py      # Loss surface generators
├── optimizers.py         # Gradient descent optimizers
├── datasets.py           # Dataset generators and PCA utilities
├── visualizer.py         # Matplotlib visualization engine
├── main.py               # Command-line application
├── optimizer_visualizer.ipynb  # Interactive notebook
└── requirements.txt
```

## Loss Surfaces

1. **Convex (Quadratic Bowl)**: Single global minimum, guaranteed convergence
2. **Multi-Local-Min (Rastrigin-like)**: Multiple local minima to test escape ability
3. **Non-Convex (Rosenbrock)**: Challenging banana-shaped valley

## Optimizers Implemented

| Optimizer | Key Feature |
|-----------|-------------|
| SGD | Basic gradient descent |
| Momentum | Accumulates velocity |
| Nesterov | Look-ahead gradient |
| AdaGrad | Adaptive learning rates |
| RMSprop | Exponential moving average of squared gradients |
| Adam | Combines momentum + RMSprop |

## Educational Goals

- Understand how learning rate affects convergence
- Compare optimizer behavior on different surface topologies
- Visualize the effect of momentum and adaptive learning rates
- Explore local minima traps and saddle points

