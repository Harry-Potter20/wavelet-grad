import numpy as np

# ==============================================
# META NETWORK - maps training state to wavelet scale weights
# Architecture:
#   - Input (4) -> Linear -> Tanh -> Linear -> Sigmoid -> weights (J)
#
# All operation implemented by hand - no frameworks
# This forces you to understand exactly what backprop does
# Store params as flat arrays and index into them.
# ==============================================

def tanh(x):
    """
    Compute the hyperbolic tangent of x.
    """
    return np.tanh(x)

def tanh_grad(x):
    """
    Compute the gradient of the hyperbolic tangent function.
    """
    return 1.0 - np.tanh(x)**2

def sigmoid(x):
    """
    Compute the sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    """
    Compute the gradient of the sigmoid function.
    """
    s = sigmoid(x)
    return s * (1 - s)

class MetaNet:
    """
    Small neural network: state → scale weights.

    Layers:
        fc1: (input_dim, hidden_dim)   with bias
        fc2: (hidden_dim, output_dim)  with bias

    Forward pass:
        h_pre  = fc1 @ x + b1          (pre-activation, hidden)
        h      = tanh(h_pre)           (hidden activation)
        out_pre = fc2 @ h + b2         (pre-activation, output)
        weights = sigmoid(out_pre)     (scale weights in (0,1))

    We store all parameters in a dict for clean access.
    Gradients are stored in a parallel dict with same keys.
    """

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3, seed=42):
        np.random.seed(seed)
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization — keeps activations well-scaled at the start
        # Scale = sqrt(1 / fan_in)
        scale1 = np.sqrt(1.0 / input_dim)
        scale2 = np.sqrt(1.0 / hidden_dim)

        self.params = {
            'W1': np.random.randn(hidden_dim, input_dim)  * scale1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(output_dim, hidden_dim) * scale2,
            'b2': np.zeros(output_dim),
        }

        # Gradient dict — same structure as params
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        # Cache for backward pass — stores intermediate values
        self._cache = {}

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input vector, shape (input_dim,)

        Returns:
            weights: shape (output_dim,), values in (0, 1)
        """
        # Layer 1
        h_pre = self.params['W1'] @ x + self.params['b1']
        h     = tanh(h_pre)

        # Layer 2
        out_pre = self.params['W2'] @ h + self.params['b2']
        weights = sigmoid(out_pre)

        # Cache everything needed for backward
        self._cache = {
            'x':       x,
            'h_pre':   h_pre,
            'h':       h,
            'out_pre': out_pre,
            'weights': weights,
        }

        return weights

    def backward(self, dL_dweights):
        """
        Backward pass — chain rule through the network.

        Args:
            dL_dweights: gradient of loss w.r.t. output weights,
                         shape (output_dim,)

        Returns:
            dL_dx: gradient w.r.t. input, shape (input_dim,)
                   (needed if we want to backprop further)

        Also populates self.grads with parameter gradients.
        """
        x       = self._cache['x']
        h_pre   = self._cache['h_pre']
        h       = self._cache['h']
        out_pre = self._cache['out_pre']

        # --- Layer 2 backward ---
        # dL/d(out_pre) = dL/d(weights) * d(weights)/d(out_pre)
        #               = dL_dweights * sigmoid_grad(out_pre)
        dL_dout_pre = dL_dweights * sigmoid_grad(out_pre)

        # dL/dW2 = dL/d(out_pre) ⊗ h   (outer product)
        self.grads['W2'] = np.outer(dL_dout_pre, h)
        self.grads['b2'] = dL_dout_pre

        # dL/dh = W2.T @ dL/d(out_pre)
        dL_dh = self.params['W2'].T @ dL_dout_pre

        # --- Layer 1 backward ---
        # dL/d(h_pre) = dL/dh * tanh_grad(h_pre)
        dL_dh_pre = dL_dh * tanh_grad(h_pre)

        self.grads['W1'] = np.outer(dL_dh_pre, x)
        self.grads['b1'] = dL_dh_pre

        # dL/dx = W1.T @ dL/d(h_pre)
        dL_dx = self.params['W1'].T @ dL_dh_pre

        return dL_dx

    def update(self, lr, clip=1.0):
        """SGD update on all parameters."""
        for k in self.params:
            grad = self.grads[k]
            # Clip gradient norm to prevent large updates
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip:
                grad = grad * (clip / grad_norm)
            self.params[k] -= lr * grad

    def zero_grad(self):
        """Reset all gradients to zero."""
        for k in self.grads:
            self.grads[k][:] = 0.0
            
if __name__ == "__main__":
    net = MetaNet(input_dim=4, hidden_dim=8, output_dim=3)

    # Random input
    x = np.array([0.1, -2.3, 0.8, 1.2])
    weights = net.forward(x)

    print(f"Input:   {x}")
    print(f"Weights: {np.round(weights, 4)}")
    print(f"All in (0,1): {np.all(weights > 0) and np.all(weights < 1)}")

    # Gradient check
    dL_dw = np.ones(3)   # dummy upstream gradient
    dL_dx = net.backward(dL_dw)
    print(f"dL/dx:   {np.round(dL_dx, 4)}")
    print(f"dL/dW1 shape: {net.grads['W1'].shape}")
    print(f"dL/dW2 shape: {net.grads['W2'].shape}")
    
    
# Gradient check — numerical vs analytical
    print("\n--- Gradient check ---")

    def net_loss(params_flat, x, target):
        """Compute MSE loss for a given flat parameter vector."""
        # Unpack flat params back into net
        net2 = MetaNet(input_dim=4, hidden_dim=8, output_dim=3)
        idx = 0
        for k in ['W1', 'b1', 'W2', 'b2']:
            size = net2.params[k].size
            net2.params[k] = params_flat[idx:idx+size].reshape(net2.params[k].shape)
            idx += size
        weights = net2.forward(x)
        return np.sum((weights - target)**2)

    # Flatten current params
    flat = np.concatenate([net.params[k].flatten()
                           for k in ['W1', 'b1', 'W2', 'b2']])
    target = np.array([0.8, 0.5, 0.2])   # dummy target weights

    # Numerical gradient
    eps = 1e-5
    num_grad = np.zeros_like(flat)
    for i in range(len(flat)):
        f_plus  = net_loss(flat.copy() + np.eye(len(flat))[i]*eps, x, target)
        f_minus = net_loss(flat.copy() - np.eye(len(flat))[i]*eps, x, target)
        num_grad[i] = (f_plus - f_minus) / (2*eps)

    # Analytical gradient
    net.forward(x)
    weights = net._cache['weights']
    dL_dw = 2 * (weights - target)
    net.backward(dL_dw)
    anal_grad = np.concatenate([net.grads[k].flatten()
                                for k in ['W1', 'b1', 'W2', 'b2']])

    max_diff = np.max(np.abs(num_grad - anal_grad))
    print(f"Max gradient difference: {max_diff:.2e}")
    print(f"Gradient check passed: {max_diff < 1e-4}")