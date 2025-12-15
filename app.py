"""
Optimizer Visualizer - Interactive Web Application
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

from loss_surfaces import (
    QuadraticBowl, EllipticalBowl, Rastrigin,
    Ackley, SaddlePoint
)
from optimizers import (
    GradientDescent, SGD, Momentum, NesterovMomentum, AdaGrad, RMSprop, Adam,
    run_optimization
)
from datasets import (
    generate_linear_separable, generate_moons, generate_circles,
    generate_xor, generate_spiral, LogisticRegressionLoss
)

# =============================================================================
# Configuration
# =============================================================================

LOSS_SURFACES = {
    "quadratic": ("Quadratic Bowl (Convex)", QuadraticBowl(a=1.0, b=1.0)),
    "quadratic_ill": ("Ill-Conditioned Quadratic", QuadraticBowl(a=1.0, b=10.0)),
    "elliptical": ("Elliptical Bowl (Rotated)", EllipticalBowl(a=1.0, b=10.0)),
    "rastrigin": ("Rastrigin (Many Local Minima)", Rastrigin(A=10.0)),
    "ackley": ("Ackley (Deep Center)", Ackley()),
    "saddle": ("Saddle Point", SaddlePoint()),
}

OPTIMIZER_CONFIGS = {
    "gd": {"name": "Gradient Descent", "class": GradientDescent, "color": "#CC79A7", "default_lr": 0.1},
    "sgd": {"name": "SGD", "class": SGD, "color": "#E69F00", "default_lr": 0.1},
    "momentum": {"name": "Momentum", "class": Momentum, "color": "#56B4E9", "default_lr": 0.1},
    "nesterov": {"name": "Nesterov", "class": NesterovMomentum, "color": "#009E73", "default_lr": 0.1},
    "adagrad": {"name": "AdaGrad", "class": AdaGrad, "color": "#F0E442", "default_lr": 0.5},
    "rmsprop": {"name": "RMSprop", "class": RMSprop, "color": "#0072B2", "default_lr": 0.1},
    "adam": {"name": "Adam", "class": Adam, "color": "#D55E00", "default_lr": 0.1},
}

DATASETS = {
    "linear": ("Linear Separable", lambda: generate_linear_separable(n_samples=200, noise=0.3)),
    "moons": ("Two Moons", lambda: generate_moons(n_samples=200, noise=0.1)),
    "circles": ("Concentric Circles", lambda: generate_circles(n_samples=200, noise=0.05)),
    "xor": ("XOR Pattern", lambda: generate_xor(n_samples=200, noise=0.1)),
    "spiral": ("Spiral", lambda: generate_spiral(n_samples=200, noise=0.05)),
}

# =============================================================================
# Visualization Functions
# =============================================================================

def create_surface_mesh(surface, resolution=80):
    """Create mesh data for 3D surface plot."""
    bounds = surface.get_bounds()
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = surface(np.array([X[i, j], Y[i, j]]))
    
    # Handle NaN and Inf values
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Clip extreme values to improve visualization
    # Use log-scale friendly clipping for surfaces with large value ranges
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    
    # If the range is very large, clip to a reasonable percentile
    if z_max - z_min > 1000:
        z_clip = np.percentile(Z[np.isfinite(Z)], 95)
        Z = np.clip(Z, z_min, z_clip)
    
    return X, Y, Z


def create_contour_figure(surface, trajectories, show_3d=False):
    """Create interactive contour plot with trajectories."""
    X, Y, Z = create_surface_mesh(surface, resolution=100)
    
    if show_3d:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "xy"}]],
            subplot_titles=("3D Surface", "2D Contour"),
            horizontal_spacing=0.1
        )
        
        # 3D Surface
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                opacity=0.8,
                showscale=False,
                name="Loss Surface"
            ),
            row=1, col=1
        )
        
        # Add trajectories to 3D
        for traj in trajectories:
            positions = traj["positions"]
            losses = traj["losses"]
            fig.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=losses,
                    mode="lines+markers",
                    line=dict(color=traj["color"], width=4),
                    marker=dict(size=3),
                    name=traj["name"]
                ),
                row=1, col=1
            )
        
        # 2D Contour
        fig.add_trace(
            go.Contour(
                x=X[0], y=Y[:, 0], z=Z,
                colorscale="Viridis",
                showscale=True,
                contours=dict(showlabels=False),
                name="Loss"
            ),
            row=1, col=2
        )
        
        # Add trajectories to 2D
        for traj in trajectories:
            positions = traj["positions"]
            fig.add_trace(
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode="lines+markers",
                    line=dict(color=traj["color"], width=3),
                    marker=dict(size=6),
                    name=traj["name"],
                    showlegend=False
                ),
                row=1, col=2
            )
            # Start point
            fig.add_trace(
                go.Scatter(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    mode="markers",
                    marker=dict(size=15, color=traj["color"], symbol="circle",
                               line=dict(color="white", width=2)),
                    showlegend=False
                ),
                row=1, col=2
            )
            # End point
            fig.add_trace(
                go.Scatter(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    mode="markers",
                    marker=dict(size=15, color=traj["color"], symbol="star",
                               line=dict(color="white", width=2)),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add optimal point
        optimal = surface.get_optimal()
        fig.add_trace(
            go.Scatter(
                x=[optimal[0]], y=[optimal[1]],
                mode="markers",
                marker=dict(size=18, color="red", symbol="x",
                           line=dict(color="white", width=2)),
                name="Optimal",
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500)
        
    else:
        # Just 2D contour
        fig = go.Figure()
        
        fig.add_trace(
            go.Contour(
                x=X[0], y=Y[:, 0], z=Z,
                colorscale="Viridis",
                showscale=True,
                contours=dict(showlabels=False),
                name="Loss"
            )
        )
        
        # Add trajectories
        for traj in trajectories:
            positions = traj["positions"]
            fig.add_trace(
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode="lines+markers",
                    line=dict(color=traj["color"], width=3),
                    marker=dict(size=5),
                    name=f"{traj['name']} ({len(positions)} steps)"
                )
            )
            # Start point
            fig.add_trace(
                go.Scatter(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    mode="markers",
                    marker=dict(size=15, color=traj["color"], symbol="circle",
                               line=dict(color="white", width=2)),
                    name=f"{traj['name']} Start",
                    showlegend=False
                )
            )
            # End point
            fig.add_trace(
                go.Scatter(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    mode="markers",
                    marker=dict(size=15, color=traj["color"], symbol="star",
                               line=dict(color="white", width=2)),
                    name=f"{traj['name']} End",
                    showlegend=False
                )
            )
        
        # Add optimal point
        optimal = surface.get_optimal()
        fig.add_trace(
            go.Scatter(
                x=[optimal[0]], y=[optimal[1]],
                mode="markers",
                marker=dict(size=18, color="red", symbol="x",
                           line=dict(color="white", width=2)),
                name="Optimal"
            )
        )
        
        fig.update_layout(height=550)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_loss_curve_figure(trajectories):
    """Create loss convergence plot."""
    fig = go.Figure()
    
    for traj in trajectories:
        losses = traj["losses"]
        steps = np.arange(len(losses))
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode="lines",
                line=dict(color=traj["color"], width=3),
                name=f"{traj['name']} (final: {losses[-1]:.4f})"
            )
        )
    
    fig.update_layout(
        title=dict(text="Loss Convergence", font=dict(size=16)),
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_gradient_figure(trajectories):
    """Create gradient magnitude plot."""
    fig = go.Figure()
    
    for traj in trajectories:
        gradients = traj["gradients"]
        grad_norms = np.linalg.norm(gradients, axis=1)
        steps = np.arange(len(grad_norms))
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=grad_norms,
                mode="lines",
                line=dict(color=traj["color"], width=3),
                name=traj["name"]
            )
        )
    
    fig.update_layout(
        title=dict(text="Gradient Magnitude", font=dict(size=16)),
        xaxis_title="Iteration",
        yaxis_title="||∇f||",
        yaxis_type="log",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_3d_surface_figure(surface, trajectories):
    """Create standalone 3D surface plot."""
    X, Y, Z = create_surface_mesh(surface, resolution=60)
    
    fig = go.Figure()
    
    # Surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale="Viridis",
            opacity=0.85,
            showscale=False,
            name="Loss Surface"
        )
    )
    
    # Trajectories
    for traj in trajectories:
        positions = traj["positions"]
        losses = traj["losses"]
        
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=losses,
                mode="lines+markers",
                line=dict(color=traj["color"], width=6),
                marker=dict(size=3),
                name=traj["name"]
            )
        )
        
        # Start point
        fig.add_trace(
            go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[losses[0]],
                mode="markers",
                marker=dict(size=10, color=traj["color"], symbol="circle"),
                showlegend=False
            )
        )
        
        # End point
        fig.add_trace(
            go.Scatter3d(
                x=[positions[-1, 0]],
                y=[positions[-1, 1]],
                z=[losses[-1]],
                mode="markers",
                marker=dict(size=10, color=traj["color"], symbol="diamond"),
                showlegend=False
            )
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="w₁",
            yaxis_title="w₂",
            zaxis_title="Loss",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace"),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


# =============================================================================
# Dash Application
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Optimizer Visualizer | Math of ML"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-primary: #0a0a0f;
                --bg-secondary: #12121a;
                --bg-card: #1a1a24;
                --accent-primary: #6366f1;
                --accent-secondary: #8b5cf6;
                --accent-tertiary: #06b6d4;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --border-color: #2d2d3a;
            }
            
            body {
                background: linear-gradient(135deg, var(--bg-primary) 0%, #0f0f1a 50%, var(--bg-primary) 100%);
                background-attachment: fixed;
                font-family: 'Space Grotesk', sans-serif;
                color: var(--text-primary);
                min-height: 100vh;
            }
            
            .main-header {
                background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 700;
                font-size: 2.5rem;
                letter-spacing: -0.02em;
            }
            
            .subtitle {
                color: var(--text-secondary);
                font-size: 1.1rem;
                font-weight: 400;
            }
            
            .control-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            
            .control-card h5 {
                color: var(--accent-tertiary);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }
            
            .graph-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            
            .stats-card {
                background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            }
            
            .stats-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--accent-primary);
            }
            
            .stats-label {
                font-size: 0.75rem;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            
            .form-label {
                color: var(--text-secondary);
                font-size: 0.85rem;
                font-weight: 500;
            }
            
            .form-control, .form-select {
                background: var(--bg-secondary) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                border-radius: 8px !important;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .form-control:focus, .form-select:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
                border: none !important;
                font-weight: 600;
                padding: 0.75rem 2rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
            }
            
            .form-check-input:checked {
                background-color: var(--accent-primary);
                border-color: var(--accent-primary);
            }
            
            .optimizer-toggle {
                display: flex;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border-color);
            }
            
            .optimizer-toggle:last-child {
                border-bottom: none;
            }
            
            .optimizer-color {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 0.75rem;
            }
            
            .optimizer-name {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
            }
            
            .rc-slider-track {
                background: var(--accent-primary) !important;
            }
            
            .rc-slider-handle {
                border-color: var(--accent-primary) !important;
            }
            
            .info-badge {
                background: var(--accent-tertiary);
                color: var(--bg-primary);
                font-size: 0.7rem;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--bg-secondary);
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border-color);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--accent-primary);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Optimizer Visualizer", className="main-header mb-1 mt-4"),
            html.P("Explore how gradient descent optimizers navigate loss surfaces", 
                   className="subtitle mb-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        # Left sidebar - Controls
        dbc.Col([
            # Surface Selection
            html.Div([
                html.H5("Loss Surface"),
                dbc.Select(
                    id="surface-select",
                    options=[{"label": v[0], "value": k} for k, v in LOSS_SURFACES.items()],
                    value="quadratic",
                    className="mb-3"
                ),
                html.Div(id="surface-info", className="text-muted small")
            ], className="control-card"),
            
            # Optimizer Selection
            html.Div([
                html.H5("Optimizers"),
                html.Div([
                    html.Div([
                        dbc.Checkbox(
                            id=f"opt-{key}",
                            value=key in ["gd", "momentum", "adam"],
                            className="me-2"
                        ),
                        html.Span("", className="optimizer-color", 
                                 style={"backgroundColor": config["color"]}),
                        html.Span(config["name"], className="optimizer-name"),
                    ], className="optimizer-toggle")
                    for key, config in OPTIMIZER_CONFIGS.items()
                ])
            ], className="control-card"),
            
            # Parameters
            html.Div([
                html.H5("Parameters"),
                dbc.Label("Learning Rate", className="form-label"),
                dcc.Slider(
                    id="learning-rate",
                    min=-4, max=0, step=0.1, value=-1,
                    marks={i: f"10^{i}" for i in range(-4, 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-4"
                ),
                dbc.Label("Number of Steps", className="form-label"),
                dcc.Slider(
                    id="num-steps",
                    min=10, max=500, step=10, value=100,
                    marks={i: str(i) for i in [10, 100, 250, 500]},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-4"
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Start X", className="form-label"),
                        dbc.Input(id="start-x", type="number", value=-1.5, step=0.1)
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Start Y", className="form-label"),
                        dbc.Input(id="start-y", type="number", value=1.5, step=0.1)
                    ], width=6),
                ], className="mb-3"),
            ], className="control-card"),
            
            # Run button
            dbc.Button("Run Optimization", id="run-button", color="primary", 
                      className="w-100 mb-3", size="lg"),
            
            # View options
            html.Div([
                html.H5("View Options"),
                dbc.Checkbox(id="show-3d", label="Show 3D Surface", value=True, className="mb-2"),
            ], className="control-card"),
            
        ], md=3, className="mb-4"),
        
        # Main content - Visualizations
        dbc.Col([
            # Stats row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(id="stat-best-loss", className="stats-value"),
                        html.Div("Best Final Loss", className="stats-label")
                    ], className="stats-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div(id="stat-best-optimizer", className="stats-value"),
                        html.Div("Best Optimizer", className="stats-label")
                    ], className="stats-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div(id="stat-convergence", className="stats-value"),
                        html.Div("Fastest Convergence", className="stats-label")
                    ], className="stats-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div(id="stat-distance", className="stats-value"),
                        html.Div("Closest to Optimal", className="stats-label")
                    ], className="stats-card")
                ], md=3),
            ], className="mb-4"),
            
            # Main visualization
            html.Div([
                dcc.Loading(
                    dcc.Graph(id="main-plot", config={"displayModeBar": True}),
                    type="circle",
                    color="#6366f1"
                )
            ], className="graph-card"),
            
            # Secondary plots
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id="loss-plot", config={"displayModeBar": False}),
                            type="circle",
                            color="#6366f1"
                        )
                    ], className="graph-card")
                ], md=6),
                dbc.Col([
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id="gradient-plot", config={"displayModeBar": False}),
                            type="circle",
                            color="#6366f1"
                        )
                    ], className="graph-card")
                ], md=6),
            ]),
            
            # Results table
            html.Div([
                html.H5("Detailed Results", style={"color": "#06b6d4", "fontFamily": "JetBrains Mono"}),
                html.Div(id="results-table")
            ], className="control-card mt-3"),
            
        ], md=9),
    ]),
    
    # Educational Section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("How to Use This Visualizer", className="mb-4", 
                       style={"color": "#6366f1", "fontWeight": "600"}),
                
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P([
                            "When training machine learning models, we use ", 
                            html.Strong("gradient descent"), 
                            " to find parameters that minimize a loss function. The optimizer starts at some point "
                            "and iteratively moves in the direction that reduces the loss."
                        ]),
                        html.P([
                            "This app lets you ", html.Strong("visualize"), " how different optimization algorithms "
                            "navigate various loss surfaces. Each colored path shows how an optimizer moves through "
                            "the parameter space, trying to find the minimum (marked with a red X)."
                        ]),
                        html.P([
                            html.Strong("The 3D surface"), " shows the loss value (height) at each point in parameter space. ",
                            html.Strong("The loss curve"), " shows how quickly the loss decreases over iterations. ",
                            html.Strong("The gradient magnitude"), " shows the strength of the gradient at each step."
                        ]),
                    ], title="What Am I Looking At?"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H6("Gradient Descent", className="mt-2", style={"color": "#CC79A7"}),
                            html.P("The simplest approach: x ← x - lr × ∇f(x). Moves directly opposite to the gradient.", 
                                  className="small mb-3"),
                            
                            html.H6("Momentum", style={"color": "#56B4E9"}),
                            html.P("Adds a velocity term that accumulates over time. Helps smooth out oscillations and build speed in consistent directions.", 
                                  className="small mb-3"),
                            
                            html.H6("Nesterov", style={"color": "#009E73"}),
                            html.P("\"Smarter\" momentum that computes the gradient at a look-ahead position. Often converges faster than standard momentum.", 
                                  className="small mb-3"),
                            
                            html.H6("AdaGrad", style={"color": "#F0E442"}),
                            html.P("Adapts the learning rate for each parameter based on historical gradients. Good for sparse data, but learning rate can decay too fast.", 
                                  className="small mb-3"),
                            
                            html.H6("RMSprop", style={"color": "#0072B2"}),
                            html.P("Uses exponential moving average of squared gradients. Fixes AdaGrad's aggressive learning rate decay.", 
                                  className="small mb-3"),
                            
                            html.H6("Adam", style={"color": "#D55E00"}),
                            html.P("Combines momentum (1st moment) with RMSprop (2nd moment). The most popular optimizer in deep learning.", 
                                  className="small mb-3"),
                        ])
                    ], title="The Optimizers Explained"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H6("Quadratic Bowl (Convex)", className="mt-2"),
                            html.P("A simple parabola with a single global minimum. All optimizers should converge here.", 
                                  className="small mb-3"),
                            
                            html.H6("Ill-Conditioned Quadratic"),
                            html.P("An elongated bowl where one direction is much steeper than the other. Watch how SGD oscillates while momentum-based methods smooth out the path.", 
                                  className="small mb-3"),
                            
                            html.H6("Rastrigin (Many Local Minima)"),
                            html.P("A bumpy surface with many local minima. Tests whether optimizers can escape local traps. Momentum helps!", 
                                  className="small mb-3"),
                            
                            html.H6("Ackley (Deep Center)"),
                            html.P("Nearly flat outer regions with a deep hole at the center. Tests behavior with vanishing gradients.", 
                                  className="small mb-3"),
                            
                            html.H6("Saddle Point"),
                            html.P("A horse-saddle shape where the gradient is zero at the origin, but it's not a minimum. Watch how optimizers behave at saddle points.", 
                                  className="small mb-3"),
                        ])
                    ], title="Loss Surfaces Explained"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            html.H6("Experiment 1: Learning Rate Effects", className="mt-2"),
                            html.P("Use the Quadratic Bowl. Try learning rates from 10⁻⁴ (very slow) to 10⁰ (may diverge!). Find the sweet spot.", 
                                  className="small mb-3"),
                            
                            html.H6("Experiment 2: Momentum vs Oscillation"),
                            html.P("Use the Ill-Conditioned Quadratic. Compare SGD (oscillates) vs Momentum (smooth path). This shows why momentum matters!", 
                                  className="small mb-3"),
                            
                            html.H6("Experiment 3: Escaping Local Minima"),
                            html.P("Use Rastrigin. Start from (-2, 2). See which optimizers get stuck vs escape. Momentum-based methods usually do better.", 
                                  className="small mb-3"),
                            
                            html.H6("Experiment 4: Saddle Point Behavior"),
                            html.P("Use Saddle Point. Start from (0.1, 0.1) near the saddle. Watch how different optimizers handle this tricky landscape.", 
                                  className="small mb-3"),
                            
                            html.H6("Experiment 5: Adam's Stability"),
                            html.P("Try high learning rates (10⁻¹ or higher) on different surfaces. Notice how Adam tends to be more stable than basic SGD.", 
                                  className="small mb-3"),
                        ])
                    ], title="Try These Experiments"),
                    
                ], start_collapsed=True, className="mb-4"),
                
            ], className="control-card mt-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(style={"borderColor": "#2d2d3a"}),
            html.P("Math of Machine Learning • Optimizer Visualizer", 
                   className="text-center text-muted small mb-4")
        ])
    ])
    
], fluid=True, className="px-4")



@callback(
    Output("surface-info", "children"),
    Input("surface-select", "value")
)
def update_surface_info(surface_key):
    if surface_key:
        name, surface = LOSS_SURFACES[surface_key]
        optimal = surface.get_optimal()
        bounds = surface.get_bounds()
        return html.Div([
            html.Span(surface.description, className="d-block mb-1"),
            html.Span(f"Optimal: ({optimal[0]:.2f}, {optimal[1]:.2f})", className="d-block"),
            html.Span(f"Bounds: [{bounds[0]}, {bounds[1]}] × [{bounds[2]}, {bounds[3]}]", className="d-block"),
        ])
    return ""


@callback(
    [Output("main-plot", "figure"),
     Output("loss-plot", "figure"),
     Output("gradient-plot", "figure"),
     Output("results-table", "children"),
     Output("stat-best-loss", "children"),
     Output("stat-best-optimizer", "children"),
     Output("stat-convergence", "children"),
     Output("stat-distance", "children")],
    Input("run-button", "n_clicks"),
    [State("surface-select", "value"),
     State("learning-rate", "value"),
     State("num-steps", "value"),
     State("start-x", "value"),
     State("start-y", "value"),
     State("show-3d", "value")] +
    [State(f"opt-{key}", "value") for key in OPTIMIZER_CONFIGS.keys()]
)
def run_optimization_callback(n_clicks, surface_key, lr_exp, num_steps, start_x, start_y, 
                              show_3d, *optimizer_states):
    # Get surface
    _, surface = LOSS_SURFACES.get(surface_key, LOSS_SURFACES["quadratic"])
    
    # Learning rate from exponent
    learning_rate = 10 ** lr_exp
    
    # Initial position
    initial = np.array([float(start_x or -1.5), float(start_y or 1.5)])
    
    # Get selected optimizers
    selected_optimizers = []
    for (key, config), is_selected in zip(OPTIMIZER_CONFIGS.items(), optimizer_states):
        if is_selected:
            selected_optimizers.append((key, config))
    
    if not selected_optimizers:
        selected_optimizers = [("adam", OPTIMIZER_CONFIGS["adam"])]
    
    # Run optimizations
    trajectories = []
    bounds = surface.get_bounds()
    for key, config in selected_optimizers:
        opt_class = config["class"]
        optimizer = opt_class(learning_rate=learning_rate)
        history = run_optimization(surface, optimizer, initial, num_steps=int(num_steps), gradient_clip=100.0, bounds=bounds)
        
        trajectories.append({
            "name": config["name"],
            "color": config["color"],
            "positions": history.positions,
            "losses": history.losses,
            "gradients": history.gradients,
            "final_loss": history.final_loss,
            "final_position": history.final_position,
            "steps": len(history.steps)
        })
    
    # Create figures
    if show_3d:
        main_fig = create_3d_surface_figure(surface, trajectories)
    else:
        main_fig = create_contour_figure(surface, trajectories, show_3d=False)
    
    loss_fig = create_loss_curve_figure(trajectories)
    grad_fig = create_gradient_figure(trajectories)
    
    # Create results table
    optimal = surface.get_optimal()
    table_rows = []
    for traj in trajectories:
        dist_to_opt = np.linalg.norm(traj["final_position"] - optimal)
        table_rows.append(
            html.Tr([
                html.Td(html.Span("●", style={"color": traj["color"], "fontSize": "1.5rem"})),
                html.Td(traj["name"], style={"fontFamily": "JetBrains Mono"}),
                html.Td(f"{traj['steps']}", style={"fontFamily": "JetBrains Mono"}),
                html.Td(f"{traj['final_loss']:.6f}", style={"fontFamily": "JetBrains Mono"}),
                html.Td(f"({traj['final_position'][0]:.4f}, {traj['final_position'][1]:.4f})", 
                       style={"fontFamily": "JetBrains Mono"}),
                html.Td(f"{dist_to_opt:.4f}", style={"fontFamily": "JetBrains Mono"}),
            ])
        )
    
    results_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th(""),
            html.Th("Optimizer"),
            html.Th("Steps"),
            html.Th("Final Loss"),
            html.Th("Final Position"),
            html.Th("Dist to Optimal"),
        ])),
        html.Tbody(table_rows)
    ], bordered=False, hover=True, responsive=True, className="mt-2",
       style={"backgroundColor": "transparent"})
    
    # Calculate stats
    best_traj = min(trajectories, key=lambda t: t["final_loss"])
    closest_traj = min(trajectories, key=lambda t: np.linalg.norm(t["final_position"] - optimal))
    
    # Find fastest convergence (first to reach within 10% of best final loss)
    target_loss = best_traj["final_loss"] * 1.1
    fastest = None
    fastest_steps = float('inf')
    for traj in trajectories:
        for i, loss in enumerate(traj["losses"]):
            if loss <= target_loss:
                if i < fastest_steps:
                    fastest_steps = i
                    fastest = traj
                break
    
    return (
        main_fig,
        loss_fig,
        grad_fig,
        results_table,
        f"{best_traj['final_loss']:.4f}",
        best_traj["name"],
        f"{fastest['name'] if fastest else 'N/A'} ({fastest_steps})" if fastest else "N/A",
        f"{np.linalg.norm(closest_traj['final_position'] - optimal):.4f}"
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   OPTIMIZER VISUALIZER - Web Application")
    print("   Math of Machine Learning")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📊 Open http://127.0.0.1:8050 in your browser\n")
    
    app.run(debug=True, host="127.0.0.1", port=8050)

