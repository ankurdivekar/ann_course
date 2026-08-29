"""
demo_utils.py
Shared, reusable functions for the Gradient Descent demos.

Nothing in here imports streamlit -- these are plain, testable functions that the
Streamlit app (01_GradientDescent.py) builds its UI around.
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1-D practice functions:  y = f(x)  and their slopes  dy/dx
# ----------------------------------------------------------------------------
# Each entry gives everything the demo needs to draw and descend a curve.

FUNCTIONS = {
    "y = x^2  (one simple valley)": {
        "f": lambda x: x**2,
        "grad": lambda x: 2 * x,
        "x_range": (-5.0, 5.0),
        "default_start": 4.0,
        "true_min_x": 0.0,
        "note": "The classic bowl. One minimum, at x = 0. Great for a first look.",
    },
    "y = (x - 3)^2 + 5  (shifted valley)": {
        "f": lambda x: (x - 3) ** 2 + 5,
        "grad": lambda x: 2 * (x - 3),
        "x_range": (-4.0, 10.0),
        "default_start": -3.0,
        "true_min_x": 3.0,
        "note": "Same shape, moved. The minimum is at x = 3, y = 5.",
    },
    "y = x^4 - 3x^3 + 2  (two valleys)": {
        "f": lambda x: x**4 - 3 * x**3 + 2,
        "grad": lambda x: 4 * x**3 - 9 * x**2,
        "x_range": (-1.5, 3.3),
        "default_start": -1.0,
        "true_min_x": 2.25,
        "note": (
            "A shallow dip near x = 0 and the true deepest point near x = 2.25. "
            "Where you start decides which one you fall into."
        ),
    },
    "y = x^2 + 3*sin(3x)  (bumpy)": {
        "f": lambda x: x**2 + 3 * np.sin(3 * x),
        "grad": lambda x: 2 * x + 9 * np.cos(3 * x),
        "x_range": (-4.0, 4.0),
        "default_start": 3.5,
        "true_min_x": -0.44,
        "note": "A bowl with ripples. Lots of little traps for gradient descent.",
    },
}


def gradient_descent(grad_fn, start, learning_rate, n_steps):
    """Run gradient descent on a 1-D function.

    Args:
        grad_fn: function returning the slope dy/dx at a point x.
        start: starting x position.
        learning_rate: step-size multiplier.
        n_steps: how many update steps to take.

    Returns:
        1-D numpy array of length ``n_steps + 1`` with every x visited,
        starting with ``start``.
    """
    x = float(start)
    history = [x]
    for _ in range(n_steps):
        x = x - learning_rate * grad_fn(x)
        # Stop recording once the numbers blow up so plots stay drawable.
        if not np.isfinite(x) or abs(x) > 1e6:
            history.append(np.sign(x) * 1e6 if np.isfinite(x) else np.nan)
            break
        history.append(x)
    return np.array(history)


def step_table(f, xs):
    """Build a list of per-step dicts (step, x, y, slope-was-not-computed-here).

    Handy for showing a table of what happened at each iteration.
    """
    rows = []
    for i, x in enumerate(xs):
        rows.append({"step": i, "x": float(x), "y  (height / loss)": float(f(x))})
    return rows


def diverged(xs):
    """True if the descent ran away to infinity / NaN."""
    return not np.all(np.isfinite(xs)) or np.nanmax(np.abs(xs)) > 1e5


# ----------------------------------------------------------------------------
# Fitting a straight line  y = w*x + b  by gradient descent on MSE loss
# ----------------------------------------------------------------------------

def make_line_data(true_w=2.3, true_b=4.0, n=40, noise=3.0, seed=0):
    """Generate noisy (x, y) points that roughly follow a straight line."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = true_w * x + true_b + rng.normal(0, noise, size=n)
    return x, y


def mse_loss(w, b, x, y):
    """Mean squared error of the line y = w*x + b against the data."""
    predictions = w * x + b
    return float(np.mean((predictions - y) ** 2))


def mse_gradients(w, b, x, y):
    """Slopes of the MSE loss with respect to w and b."""
    error = (w * x + b) - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    return float(dw), float(db)


def fit_line_gradient_descent(x, y, learning_rate=0.01, n_steps=400,
                              start_w=0.0, start_b=0.0):
    """Fit a line to data with gradient descent.

    Returns:
        dict with arrays ``w``, ``b``, ``loss`` (each length ``n_steps + 1``).
    """
    w, b = float(start_w), float(start_b)
    ws, bs, losses = [w], [b], [mse_loss(w, b, x, y)]
    for _ in range(n_steps):
        dw, db = mse_gradients(w, b, x, y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ws.append(w)
        bs.append(b)
        losses.append(mse_loss(w, b, x, y))
        if not np.isfinite(losses[-1]) or losses[-1] > 1e12:
            break
    return {"w": np.array(ws), "b": np.array(bs), "loss": np.array(losses)}


# ----------------------------------------------------------------------------
# Plotting helpers (matplotlib, Agg backend -- see the app for backend setup)
# ----------------------------------------------------------------------------

def plot_descent_on_curve(f, x_range, xs, current_step):
    """Figure: the curve y = f(x) with the descent path drawn up to
    ``current_step``, and the current position highlighted."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # Widen the view so the curve is drawn wherever the descent actually goes
    # (e.g. when the student picks a starting point outside the default range).
    finite_xs = xs[np.isfinite(xs)]
    lo = min(x_range[0], finite_xs.min()) if finite_xs.size else x_range[0]
    hi = max(x_range[1], finite_xs.max()) if finite_xs.size else x_range[1]
    margin = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - margin, hi + margin, 500)
    ax.plot(grid, f(grid), color="steelblue", linewidth=2, label="y = f(x)")

    shown = xs[: current_step + 1]
    ax.plot(shown, f(shown), "o-", color="crimson", markersize=5,
            linewidth=1.5, alpha=0.7, label="steps taken")
    ax.scatter([xs[0]], [f(xs[0])], color="green", s=90, zorder=6, label="start")
    ax.scatter([shown[-1]], [f(shown[-1])], color="black", s=110, zorder=7,
               label=f"now (step {current_step})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Walking downhill on the curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Keep the y-axis sane even if a step overshoots hugely.
    # y-limits: show the valley (default range) and the visited points, but
    # don't let one huge overshoot squash everything.
    base_y = f(np.linspace(x_range[0], x_range[1], 200))
    path_y = f(shown[np.isfinite(shown)]) if np.any(np.isfinite(shown)) else base_y
    y_lo = min(base_y.min(), np.min(path_y))
    y_hi = max(base_y.max(), np.percentile(path_y, 95))
    pad = 0.1 * (y_hi - y_lo + 1e-9)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    fig.tight_layout()
    return fig


def plot_loss_curve(values, current_step=None, ylabel="y  (height / loss)"):
    """Figure: value at each step, optionally marking the current step."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.5))
    steps = np.arange(len(values))
    ax.plot(steps, values, "-", color="purple", linewidth=1.5)
    ax.plot(steps, values, "o", color="purple", markersize=3, alpha=0.5)
    if current_step is not None and current_step < len(values):
        ax.scatter([current_step], [values[current_step]], color="black",
                   s=90, zorder=6)
    ax.set_xlabel("step number")
    ax.set_ylabel(ylabel)
    ax.set_title("Loss over time -- we want this to go down")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_line_fit(x, y, w, b, step, loss):
    """Figure: the data scatter with the current fitted line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", label="data", zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, w * xs + b, color="crimson", linewidth=2.5,
            label=f"y = {w:.2f}x + {b:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Step {step}   |   loss = {loss:,.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
