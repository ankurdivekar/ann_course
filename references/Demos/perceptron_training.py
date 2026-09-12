"""
perceptron_training.py
Backpropagation Training Demonstration for Logic Gates

Educational tool for understanding how backpropagation trains a perceptron
to learn AND and OR gate behavior. Shows weight evolution, loss curves,
gradients, and predictions in real-time.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)  # Force non-GUI backend for Streamlit
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pandas as pd
from perceptron_utils import sigmoid, sigmoid_derivative, perceptron_forward, mean_squared_error


class PerceptronTrainer:
    """Handles training logic for a single perceptron."""

    def __init__(self, w1, w2, b, learning_rate=0.1):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.learning_rate = learning_rate

        # Training history
        self.loss_history = []
        self.w1_history = [w1]
        self.w2_history = [w2]
        self.b_history = [b]
        self.epoch = 0

        # Gradient tracking
        self.grad_w1 = 0.0
        self.grad_w2 = 0.0
        self.grad_b = 0.0

    def forward(self, x1, x2):
        """Forward pass."""
        z = self.w1 * x1 + self.w2 * x2 + self.b
        output = sigmoid(z)
        return z, output

    def train_step(self, X, y):
        """
        Perform one training step (one epoch through all data).

        Args:
            X: Input data array, shape (n_samples, 2)
            y: Target values array, shape (n_samples,)

        Returns:
            Loss value for this epoch
        """
        n_samples = len(X)
        total_loss = 0.0

        # Accumulate gradients
        grad_w1_acc = 0.0
        grad_w2_acc = 0.0
        grad_b_acc = 0.0

        predictions = []

        for i in range(n_samples):
            x1, x2 = X[i]
            target = y[i]

            # Forward pass
            z, output = self.forward(x1, x2)
            predictions.append(output)

            # Calculate loss (MSE)
            loss = (output - target) ** 2
            total_loss += loss

            # Backward pass (compute gradients)
            # d_loss/d_output = 2 * (output - target)
            d_loss_d_output = 2 * (output - target)

            # d_output/d_z = sigmoid_derivative(z)
            d_output_d_z = sigmoid_derivative(z)

            # Chain rule: d_loss/d_z
            d_loss_d_z = d_loss_d_output * d_output_d_z

            # Gradients for weights and bias
            grad_w1_acc += d_loss_d_z * x1
            grad_w2_acc += d_loss_d_z * x2
            grad_b_acc += d_loss_d_z

        # Average gradients
        self.grad_w1 = grad_w1_acc / n_samples
        self.grad_w2 = grad_w2_acc / n_samples
        self.grad_b = grad_b_acc / n_samples

        # Update weights using gradient descent
        self.w1 -= self.learning_rate * self.grad_w1
        self.w2 -= self.learning_rate * self.grad_w2
        self.b -= self.learning_rate * self.grad_b

        # Calculate average loss
        avg_loss = total_loss / n_samples

        # Store history
        self.loss_history.append(avg_loss)
        self.w1_history.append(self.w1)
        self.w2_history.append(self.w2)
        self.b_history.append(self.b)
        self.epoch += 1

        return avg_loss

    def predict(self, X):
        """Make predictions for input data."""
        predictions = []
        for x1, x2 in X:
            _, output = self.forward(x1, x2)
            predictions.append(output)
        return np.array(predictions)

    def reset(self, w1, w2, b):
        """Reset trainer to initial state."""
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.loss_history = []
        self.w1_history = [w1]
        self.w2_history = [w2]
        self.b_history = [b]
        self.epoch = 0
        self.grad_w1 = 0.0
        self.grad_w2 = 0.0
        self.grad_b = 0.0


def plot_loss_curve(trainer, gate_name, color='blue'):
    """Plot loss curve over epochs."""
    fig, ax = plt.subplots(figsize=(5, 3))

    try:
        if len(trainer.loss_history) > 0:
            epochs = range(len(trainer.loss_history))
            ax.plot(epochs, trainer.loss_history, color=color, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss (MSE)', fontsize=10)
            ax.set_title(f'{gate_name} Gate - Loss Curve', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'No training data yet',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss (MSE)', fontsize=10)
            ax.set_title(f'{gate_name} Gate - Loss Curve', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close(fig)
        raise e


def plot_weight_evolution(trainer, gate_name, color='blue'):
    """Plot weight evolution over epochs."""
    fig, ax = plt.subplots(figsize=(5, 3))

    try:
        if len(trainer.w1_history) > 1:
            epochs = range(len(trainer.w1_history))
            ax.plot(epochs, trainer.w1_history, label='w₁', linewidth=2, alpha=0.8)
            ax.plot(epochs, trainer.w2_history, label='w₂', linewidth=2, alpha=0.8)
            ax.plot(epochs, trainer.b_history, label='b', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Parameter Value', fontsize=10)
            ax.set_title(f'{gate_name} Gate - Weight Evolution', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
        else:
            ax.text(0.5, 0.5, 'No training data yet',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Parameter Value', fontsize=10)
            ax.set_title(f'{gate_name} Gate - Weight Evolution', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig
    except Exception as e:
        plt.close(fig)
        raise e


# Page configuration
st.set_page_config(
    page_title="Perceptron Training",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Perceptron Training with Backpropagation")
st.markdown("Watch how gradient descent trains perceptrons to learn AND and OR gate behavior")

with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    This demo shows **backpropagation** in action for training a single perceptron.

    **Training Process:**
    1. **Forward Pass**: Calculate predictions using current weights
    2. **Loss Calculation**: Compare predictions to targets (MSE loss)
    3. **Backward Pass**: Compute gradients using chain rule
    4. **Weight Update**: Adjust weights using gradient descent: `w = w - α·∇w`

    **Key Concepts:**
    - **Gradient**: Direction and magnitude of steepest increase in loss
    - **Learning Rate (α)**: Step size for weight updates
    - **Epoch**: One complete pass through all training examples

    Watch how the perceptron discovers the right weights to implement logic gates!
    """)

# Sidebar controls
st.sidebar.header("⚙️ Training Controls")

# Learning rate
learning_rate = st.sidebar.slider(
    "Learning Rate (α)",
    min_value=0.01,
    max_value=2.0,
    value=0.5,
    step=0.01,
    help="Step size for weight updates. Higher = faster but may overshoot."
)

# Random seed
random_seed = st.sidebar.slider(
    "Random Seed",
    min_value=0,
    max_value=100,
    value=42,
    step=1,
    help="Seed for random weight initialization. Same seed = same starting weights."
)

# Max epochs
max_epochs = st.sidebar.slider(
    "Max Epochs",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Maximum number of training iterations."
)

# Animation speed (for continuous training)
if 'is_training' in st.session_state and st.session_state.is_training:
    animation_speed = st.sidebar.slider(
        "Animation Speed (epochs/update)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many epochs to run per screen update during continuous training."
    )
else:
    animation_speed = 5

st.sidebar.markdown("---")

# Initialize session state
if 'trainers_initialized' not in st.session_state:
    np.random.seed(random_seed)
    init_w1 = np.random.randn() * 0.5
    init_w2 = np.random.randn() * 0.5
    init_b = np.random.randn() * 0.5

    st.session_state.and_trainer = PerceptronTrainer(init_w1, init_w2, init_b, learning_rate)
    st.session_state.or_trainer = PerceptronTrainer(init_w1, init_w2, init_b, learning_rate)
    st.session_state.trainers_initialized = True
    st.session_state.is_training = False
    st.session_state.last_seed = random_seed

# Check if seed changed - reset if so
if st.session_state.last_seed != random_seed:
    np.random.seed(random_seed)
    init_w1 = np.random.randn() * 0.5
    init_w2 = np.random.randn() * 0.5
    init_b = np.random.randn() * 0.5

    st.session_state.and_trainer.reset(init_w1, init_w2, init_b)
    st.session_state.or_trainer.reset(init_w1, init_w2, init_b)
    st.session_state.last_seed = random_seed

# Update learning rate if changed
st.session_state.and_trainer.learning_rate = learning_rate
st.session_state.or_trainer.learning_rate = learning_rate

# Training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Control buttons
st.sidebar.subheader("Training Actions")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Play", use_container_width=True, disabled=st.session_state.is_training):
        st.session_state.is_training = True
        st.rerun()

    if st.button("⏭️ Step", use_container_width=True, disabled=st.session_state.is_training):
        if st.session_state.and_trainer.epoch < max_epochs:
            st.session_state.and_trainer.train_step(X_train, y_and)
            st.session_state.or_trainer.train_step(X_train, y_or)
            st.rerun()

with col2:
    if st.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.is_training):
        st.session_state.is_training = False
        st.rerun()

    if st.button("🔄 Reset", use_container_width=True):
        np.random.seed(random_seed)
        init_w1 = np.random.randn() * 0.5
        init_w2 = np.random.randn() * 0.5
        init_b = np.random.randn() * 0.5

        st.session_state.and_trainer.reset(init_w1, init_w2, init_b)
        st.session_state.or_trainer.reset(init_w1, init_w2, init_b)
        st.session_state.is_training = False
        st.rerun()

# Continuous training loop
if st.session_state.is_training:
    if st.session_state.and_trainer.epoch < max_epochs:
        for _ in range(animation_speed):
            if st.session_state.and_trainer.epoch < max_epochs:
                st.session_state.and_trainer.train_step(X_train, y_and)
                st.session_state.or_trainer.train_step(X_train, y_or)
        st.rerun()
    else:
        st.session_state.is_training = False

# Display training status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Training Status")
st.sidebar.metric("Current Epoch", st.session_state.and_trainer.epoch)
if len(st.session_state.and_trainer.loss_history) > 0:
    st.sidebar.metric("AND Gate Loss", f"{st.session_state.and_trainer.loss_history[-1]:.4f}")
    st.sidebar.metric("OR Gate Loss", f"{st.session_state.or_trainer.loss_history[-1]:.4f}")

# Main display - side by side comparison
left_col, right_col = st.columns(2)

# AND Gate (Left)
with left_col:
    st.subheader("🔵 AND Gate Training")

    and_trainer = st.session_state.and_trainer

    # Current weights
    st.markdown("**Current Weights:**")
    weight_col1, weight_col2, weight_col3 = st.columns(3)
    with weight_col1:
        st.metric("w₁", f"{and_trainer.w1:.3f}")
    with weight_col2:
        st.metric("w₂", f"{and_trainer.w2:.3f}")
    with weight_col3:
        st.metric("b", f"{and_trainer.b:.3f}")

    # Current gradients
    if and_trainer.epoch > 0:
        st.markdown("**Current Gradients:**")
        grad_col1, grad_col2, grad_col3 = st.columns(3)
        with grad_col1:
            st.metric("∂L/∂w₁", f"{and_trainer.grad_w1:.3f}")
        with grad_col2:
            st.metric("∂L/∂w₂", f"{and_trainer.grad_w2:.3f}")
        with grad_col3:
            st.metric("∂L/∂b", f"{and_trainer.grad_b:.3f}")

    # Predictions vs Targets
    st.markdown("**Predictions vs Targets:**")
    and_predictions = and_trainer.predict(X_train)
    pred_df = pd.DataFrame({
        'x₁': X_train[:, 0],
        'x₂': X_train[:, 1],
        'Target': y_and,
        'Prediction': [f"{p:.3f}" for p in and_predictions],
        'Error': [f"{abs(p - t):.3f}" for p, t in zip(and_predictions, y_and)]
    })
    st.dataframe(pred_df, hide_index=True, use_container_width=True)

    # Loss curve
    st.pyplot(plot_loss_curve(and_trainer, "AND", color='blue'), clear_figure=True)

    # Weight evolution
    st.pyplot(plot_weight_evolution(and_trainer, "AND", color='blue'), clear_figure=True)

# OR Gate (Right)
with right_col:
    st.subheader("🟢 OR Gate Training")

    or_trainer = st.session_state.or_trainer

    # Current weights
    st.markdown("**Current Weights:**")
    weight_col1, weight_col2, weight_col3 = st.columns(3)
    with weight_col1:
        st.metric("w₁", f"{or_trainer.w1:.3f}")
    with weight_col2:
        st.metric("w₂", f"{or_trainer.w2:.3f}")
    with weight_col3:
        st.metric("b", f"{or_trainer.b:.3f}")

    # Current gradients
    if or_trainer.epoch > 0:
        st.markdown("**Current Gradients:**")
        grad_col1, grad_col2, grad_col3 = st.columns(3)
        with grad_col1:
            st.metric("∂L/∂w₁", f"{or_trainer.grad_w1:.3f}")
        with grad_col2:
            st.metric("∂L/∂w₂", f"{or_trainer.grad_w2:.3f}")
        with grad_col3:
            st.metric("∂L/∂b", f"{or_trainer.grad_b:.3f}")

    # Predictions vs Targets
    st.markdown("**Predictions vs Targets:**")
    or_predictions = or_trainer.predict(X_train)
    pred_df = pd.DataFrame({
        'x₁': X_train[:, 0],
        'x₂': X_train[:, 1],
        'Target': y_or,
        'Prediction': [f"{p:.3f}" for p in or_predictions],
        'Error': [f"{abs(p - t):.3f}" for p, t in zip(or_predictions, y_or)]
    })
    st.dataframe(pred_df, hide_index=True, use_container_width=True)

    # Loss curve
    st.pyplot(plot_loss_curve(or_trainer, "OR", color='green'), clear_figure=True)

    # Weight evolution
    st.pyplot(plot_weight_evolution(or_trainer, "OR", color='green'), clear_figure=True)

# Educational notes
with st.expander("📚 Understanding the Training Process"):
    st.markdown("""
    ### What You're Seeing

    **Predictions vs Targets Table:**
    - Shows current output for each input combination
    - Error shows how far predictions are from target (0 or 1)
    - Watch errors shrink as training progresses

    **Loss Curve:**
    - Mean Squared Error (MSE) = average of squared errors
    - Should decrease over time as model learns
    - Flat line = convergence (model has learned)

    **Weight Evolution:**
    - Shows how w₁, w₂, and b change during training
    - Weights move in direction that reduces loss
    - Final values determine the learned decision boundary

    **Gradients:**
    - ∂L/∂w = how much loss changes when weight changes
    - Positive gradient → weight should decrease
    - Negative gradient → weight should increase
    - Magnitude = how strongly the weight should change

    ### Try This
    - Adjust learning rate: higher = faster but may oscillate
    - Change random seed: different starting points, same final weights
    - Use Step button to see each iteration in detail
    - Watch how AND and OR converge differently despite same starting weights
    """)
