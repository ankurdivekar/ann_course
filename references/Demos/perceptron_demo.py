"""
perceptron_demo.py
Single Perceptron Demonstration with Two Inputs

Educational tool for understanding basic neural network components.
Demonstrates how a perceptron combines weighted inputs and applies
sigmoid activation function.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg", force=True)  # Force non-GUI backend for Streamlit
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode
from perceptron_utils import perceptron_forward, sigmoid


def perceptron_output(x1, x2, w1, w2, b):
    """
    Calculate perceptron output with sigmoid activation.

    Args:
        x1, x2: Input values
        w1, w2: Weight values
        b: Bias term

    Returns:
        Tuple of (weighted_sum, sigmoid_output)
    """
    return perceptron_forward(x1, x2, w1, w2, b)


def sigmoid_curve_data(z_min=-10, z_max=10, num_points=200):
    """
    Generate data points for plotting sigmoid curve.

    Args:
        z_min: Minimum z value
        z_max: Maximum z value
        num_points: Number of points to generate

    Returns:
        Tuple of (z_values, sigmoid_values)
    """
    z_values = np.linspace(z_min, z_max, num_points)
    sigmoid_values = sigmoid(z_values)
    return z_values, sigmoid_values


def plot_sigmoid_with_marker(current_z, current_output):
    """
    Create sigmoid curve plot with current state marked.

    Args:
        current_z: Current weighted sum value
        current_output: Current sigmoid output value

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    z_values, sigmoid_values = sigmoid_curve_data()

    ax.plot(z_values, sigmoid_values, "b-", linewidth=2, label="Sigmoid Function")

    ax.plot(
        current_z,
        current_output,
        "ro",
        markersize=12,
        label=f"Current State: ({current_z:.2f}, {current_output:.3f})",
        zorder=5,
    )

    ax.axvline(x=current_z, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=current_output, color="red", linestyle="--", alpha=0.5, linewidth=1)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Weighted Sum (z)", fontsize=12)
    ax.set_ylabel("Sigmoid Output σ(z)", fontsize=12)
    ax.set_title("Sigmoid Activation Function", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


st.set_page_config(page_title="Perceptron Demo", page_icon="🧠", layout="wide")

st.title("🧠 Single Perceptron Demonstration")
st.markdown("Interactive visualization of a perceptron with two inputs and sigmoid activation")

with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    A **perceptron** is the fundamental building block of neural networks. It:
    1. Takes multiple inputs (x₁, x₂)
    2. Multiplies each input by a weight (w₁, w₂)
    3. Adds a bias term (b)
    4. Applies an activation function (sigmoid) to produce output

    **Formula:** `output = σ(w₁·x₁ + w₂·x₂ + b)`

    **Sigmoid Function:** `σ(z) = 1 / (1 + e⁻ᶻ)`

    Use the sliders to explore how changing inputs, weights, and bias affects the output!
    """)

st.sidebar.header("⚙️ Parameters")

# Initialize session state for slider values if not already set
if "x1" not in st.session_state:
    st.session_state.x1 = 1.0
if "x2" not in st.session_state:
    st.session_state.x2 = 1.0
if "w1" not in st.session_state:
    st.session_state.w1 = 0.5
if "w2" not in st.session_state:
    st.session_state.w2 = 0.5
if "b" not in st.session_state:
    st.session_state.b = 0.0

# Preset buttons section
st.sidebar.subheader("🎯 Quick Presets")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("⚡ AND Gate", use_container_width=True):
        st.session_state.w1 = 2.0
        st.session_state.w2 = 2.0
        st.session_state.b = -3.0
        st.session_state.x1 = 0.0
        st.session_state.x2 = 0.0

with col2:
    if st.button("⚡ OR Gate", use_container_width=True):
        st.session_state.w1 = 2.0
        st.session_state.w2 = 2.0
        st.session_state.b = -1.0
        st.session_state.x1 = 0.0
        st.session_state.x2 = 0.0

if st.sidebar.button("🔄 Reset to Default", use_container_width=True):
    st.session_state.x1 = 1.0
    st.session_state.x2 = 1.0
    st.session_state.w1 = 0.5
    st.session_state.w2 = 0.5
    st.session_state.b = 0.0

st.sidebar.markdown("---")

# Sliders with session state
st.sidebar.subheader("Input Values")
x1 = st.sidebar.slider("x₁ (Input 1)", min_value=-5.0, max_value=5.0, value=st.session_state.x1, step=0.1, key="x1")
x2 = st.sidebar.slider("x₂ (Input 2)", min_value=-5.0, max_value=5.0, value=st.session_state.x2, step=0.1, key="x2")

st.sidebar.subheader("Weights")
w1 = st.sidebar.slider("w₁ (Weight 1)", min_value=-3.0, max_value=3.0, value=st.session_state.w1, step=0.1, key="w1")
w2 = st.sidebar.slider("w₂ (Weight 2)", min_value=-3.0, max_value=3.0, value=st.session_state.w2, step=0.1, key="w2")

st.sidebar.subheader("Bias")
b = st.sidebar.slider("b (Bias)", min_value=-5.0, max_value=5.0, value=st.session_state.b, step=0.1, key="b")

z, output = perceptron_output(x1, x2, w1, w2, b)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Weighted Sum (z)", value=f"{z:.3f}", help="Sum of weighted inputs plus bias: z = w₁·x₁ + w₂·x₂ + b"
    )

with col2:
    output_color = "green" if output > 0.5 else "red"
    st.metric(
        label="Sigmoid Output σ(z)", value=f"{output:.3f}", help="Result after applying sigmoid activation function"
    )
    st.markdown(
        f"<span style='color:{output_color}; font-size:14px;'>{'✓ Activated (>0.5)' if output > 0.5 else '✗ Not Activated (<0.5)'}</span>",
        unsafe_allow_html=True,
    )

st.subheader("📊 Step-by-Step Calculation")

st.markdown(f"""
**Step 1: Calculate Weighted Sum**

$$z = w_1 \\cdot x_1 + w_2 \\cdot x_2 + b$$

$$z = ({w1:.1f}) \\cdot ({x1:.1f}) + ({w2:.1f}) \\cdot ({x2:.1f}) + ({b:.1f})$$

$$z = {w1 * x1:.2f} + {w2 * x2:.2f} + {b:.1f} = {z:.3f}$$

**Step 2: Apply Sigmoid Activation**

$$\\sigma(z) = \\frac{{1}}{{1 + e^{{-z}}}}$$

$$\\sigma({z:.3f}) = \\frac{{1}}{{1 + e^{{-{z:.3f}}}}} = {output:.3f}$$
""")

st.subheader("📈 Sigmoid Function Visualization")

fig = plot_sigmoid_with_marker(z, output)
st.pyplot(fig, clear_figure=True)

with st.expander("💡 Interpretation Guide"):
    if output > 0.9:
        interpretation = "**Strong activation**: The perceptron is highly confident (output close to 1). This typically represents a strong 'yes' or positive classification."
    elif output > 0.7:
        interpretation = (
            "**Moderate-high activation**: The perceptron leans toward activation but with some uncertainty."
        )
    elif output > 0.5:
        interpretation = "**Weak activation**: The perceptron is slightly above the threshold. Small changes could flip the decision."
    elif output > 0.3:
        interpretation = (
            "**Weak deactivation**: The perceptron is slightly below the threshold. Close to the decision boundary."
        )
    elif output > 0.1:
        interpretation = "**Moderate-low activation**: The perceptron leans away from activation."
    else:
        interpretation = (
            "**Strong deactivation**: The perceptron is highly confident in the negative direction (output close to 0)."
        )

    st.markdown(f"""
    **Current Output: {output:.3f}**

    {interpretation}

    **Key Insights:**
    - The sigmoid function smoothly maps any input to a value between 0 and 1
    - Outputs near 0.5 indicate uncertainty (the decision boundary)
    - The steepest part of the sigmoid curve is around z=0
    - As z approaches ±∞, the output approaches 1 or 0 (but never quite reaches them)
    - Increasing positive weights strengthens the influence of positive inputs
    - The bias shifts the activation threshold left (negative bias) or right (positive bias)
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Try These Scenarios:**")
st.sidebar.markdown("""
- **Zero everything**: All sliders to 0 → output = 0.5
- **AND gate**: w₁=2, w₂=2, b=-3, test (0,0), (0,1), (1,0), (1,1)
- **OR gate**: w₁=2, w₂=2, b=-1
- **Negative weights**: Set w₁ or w₂ negative to invert input influence
""")
