# Optimizer Visualizer for Gradient Descent

An interactive visualization tool for understanding how different optimization algorithms navigate loss surfaces during gradient descent.

## Overview

This project provides:
- **Multiple Loss Surfaces**: Convex, multi-local-minimum, and non-convex surfaces
- **Various Optimizers**: SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam
- **Interactive Visualization**: 3D surface plots with optimizer trajectories
- **PCA Support**: Dimensionality reduction for higher-dimensional datasets

## Installation

```bash
cd MOML
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python main.py
```

### Jupyter Notebook

```bash
jupyter notebook optimizer_visualizer.ipynb
```

## Project Structure

```
MOML/
├── loss_surfaces.py      # Loss surface generators
├── optimizers.py         # Gradient descent optimizers
├── datasets.py           # Dataset generators and PCA utilities
├── visualizer.py         # Visualization engine
├── main.py               # Main application
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

