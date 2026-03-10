import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.meta_net import MetaNet, sigmoid, sigmoid_grad


def test_forward_output_shape():
    """MetaNet output shape matches output_dim."""
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    x = np.random.randn(5)
    out = net.forward(x)
    assert out.shape == (3,), f"Expected (3,), got {out.shape}"
    print("test_forward_output_shape passed")


def test_output_in_range():
    """Output is sigmoid — all values in (0, 1)."""
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    for _ in range(10):
        x = np.random.randn(5) * 3
        out = net.forward(x)
        assert np.all(out > 0) and np.all(out < 1), \
            f"Output out of (0,1): {out}"
    print("test_output_in_range passed")


def test_backward_shapes():
    """Gradients have same shape as parameters."""
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    x = np.random.randn(5)
    out = net.forward(x)
    net.zero_grad()
    net.backward(np.ones(3))
    for k in net.params:
        assert net.grads[k].shape == net.params[k].shape, \
            f"Grad shape mismatch for {k}"
    print("test_backward_shapes passed")


def test_gradient_check():
    """Analytical gradients match numerical gradients."""
    np.random.seed(42)
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    x = np.random.randn(5)
    dL_dout = np.random.randn(3)

    # Analytical
    net.forward(x)
    net.zero_grad()
    net.backward(dL_dout)

    eps = 1e-5
    for k in ['W1', 'b1', 'W2', 'b2']:
        anal = net.grads[k].flatten()
        num  = np.zeros_like(anal)
        p    = net.params[k].flatten()

        for i in range(len(p)):
            p[i] += eps
            net.params[k] = p.reshape(net.params[k].shape)
            out_plus = net.forward(x)

            p[i] -= 2*eps
            net.params[k] = p.reshape(net.params[k].shape)
            out_minus = net.forward(x)

            num[i] = np.dot(dL_dout, (out_plus - out_minus)) / (2*eps)
            p[i] += eps
            net.params[k] = p.reshape(net.params[k].shape)

        max_diff = np.max(np.abs(anal - num))
        assert max_diff < 1e-5, \
            f"Gradient check failed for {k}: max_diff={max_diff:.2e}"

    print("test_gradient_check passed")


def test_zero_grad_clears():
    """zero_grad resets all gradients to zero."""
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    x = np.random.randn(5)
    net.forward(x)
    net.backward(np.ones(3))
    net.zero_grad()
    for k in net.grads:
        assert np.all(net.grads[k] == 0), f"Grad not zeroed for {k}"
    print("test_zero_grad_clears passed")


def test_xavier_init_scale():
    """Weights initialized with reasonable scale — not too large."""
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    assert np.max(np.abs(net.params['W1'])) < 2.0, "W1 too large"
    assert np.max(np.abs(net.params['W2'])) < 2.0, "W2 too large"
    print("test_xavier_init_scale passed")


if __name__ == "__main__":
    test_forward_output_shape()
    test_output_in_range()
    test_backward_shapes()
    test_gradient_check()
    test_zero_grad_clears()
    test_xavier_init_scale()
    print("\nAll meta_net tests passed.")