import numpy as np

# Implementing the Rosenbrock standard test function
def rosenbrock(x, y):
    """
    Computes the Rosenbrock function and its gradient at (x, y).
    Returns:
    loss: scalar f(x, y)
    gx: gradient of f with respect to x
    gy: gradient of f with respect to y
    """
    loss = (1 - x)**2 + 100 * (y - x**2)**2
    gx = -2 * (1 - x) - 400 * x * (y - x**2)
    gy = 200 * (y - x**2)
    return loss, gx, gy

if __name__ == "__main__":
    # Sanity checks
    loss, gx, gy = rosenbrock(1.0, 1.0)
    print(f"At minimum (1, 1): loss={loss: .4f}, gx={gx: .4f}, gy={gy: .4f}")

    loss, gx, gy = rosenbrock(-1.0, 1.0)
    print(f"At point (-1, 1): loss={loss: .4f}, gx={gx: .4f}, gy={gy: .4f}")


# STEP 2: SGD

def run_sgd(lr, n_steps, start=(-1.0, 1.0)):
    x, y = start
    history = []
    for i in range(n_steps):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))
        
        x -= lr * gx
        y -= lr * gy
        
    return history

history_more = run_sgd(lr=0.0001, n_steps=5000)
last = history_more[-1]
print(f"\nSGD  lr=0.001  50000 steps")
print(f"  End: x={last[0]:.4f}  y={last[1]:.4f}  loss={last[2]:.6f}")


# STEP 3: ADAM

def run_adam(lr, n_steps, start=(-1.0, 1.0),
             beta1=0.9, beta2=0.999, eps=1e-8):
    x, y = start
    history = []

    # Moment estimates — initialized to zero
    mx, my = 0.0, 0.0   # 1st moments
    vx, vy = 0.0, 0.0   # 2nd moments

    for t in range(1, n_steps + 1):   # t starts at 1 (needed for bias correction)
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        # --- 1st moment update ---
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy

        # --- 2nd moment update ---
        vx = beta2 * vx + (1 - beta2) * gx**2
        vy = beta2 * vy + (1 - beta2) * gy**2

        # --- Bias correction ---
        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        # --- Update ---
        x = x - lr * mx_hat / (vx_hat**0.5 + eps)
        y = y - lr * my_hat / (vy_hat**0.5 + eps)

    return history


history = run_adam(lr=0.01, n_steps=5000)

first = history[0]
last  = history[-1]
print(f"\nAdam  lr=0.01  5000 steps")
print(f"  Start: x={first[0]:.4f}  y={first[1]:.4f}  loss={first[2]:.4f}")
print(f"  End:   x={last[0]:.4f}   y={last[1]:.4f}   loss={last[2]:.6f}")
print(f"  Reached (1,1)? {abs(last[0]-1) < 1e-4 and abs(last[1]-1) < 1e-4}")

# How many steps did it take to converge?
for n in [500, 1000, 2000, 5000]:
    h = run_adam(lr=0.01, n_steps=n)
    last = h[-1]
    reached = abs(last[0]-1) < 1e-4 and abs(last[1]-1) < 1e-4
    print(f"  Steps: {n}  End: x={last[0]:.4f}  y={last[1]:.4f}  loss={last[2]:.6f}")
    print(f"  Reached (1,1)? {reached}")
    

# =============================================================
# STEP 4: CIRCULAR GRADIENT BUFFER
# =============================================================
#
# Stores the last `capacity` gradients in a fixed-size array.
# Overwrites oldest entry when full.
#
# Key invariants:
#   - capacity must be a power of 2
#   - read() always returns entries in chronological order
#   - write() is O(1) always
# =============================================================

class CircularGradientBuffer:

    def __init__(self, capacity, grad_shape):
        # Validate: capacity must be a power of 2
        # Trick: a power of 2 in binary is 1000...0
        # Subtracting 1 gives 0111...1
        # AND of the two is always 0 for powers of 2
        # e.g. 8 = 1000, 7 = 0111, 8 & 7 = 0000 ✓
        # e.g. 6 = 0110, 5 = 0101, 6 & 5 = 0100 ✗
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError(f"capacity must be a power of 2, got {capacity}")

        self.capacity = capacity
        self.grad_shape = grad_shape
        self._head = 0       # next write position
        self._n_writes = 0   # total writes ever

        # Pre-allocate — shape is (capacity, *grad_shape)
        # For 1D gradients of size 2: shape = (8, 2)
        self._buffer = np.zeros((capacity, *grad_shape))

    def write(self, grad):
        self._buffer[self._head] = grad
        self._head = (self._head + 1) % self.capacity
        self._n_writes += 1

    def read(self):
        if not self.is_full():
            return self._buffer[:self._n_writes].copy()

        # Unroll the circular structure into chronological order
        # Oldest entry sits at self._head, newest at self._head - 1
        return np.concatenate([
            self._buffer[self._head:],
            self._buffer[:self._head]
        ], axis=0)

    def is_full(self):
        return self._n_writes >= self.capacity

    @property
    def n_entries(self):
        return min(self._n_writes, self.capacity)

# --- Tests ---
print("\n--- CircularGradientBuffer tests ---")

buf = CircularGradientBuffer(capacity=4, grad_shape=(2,))

grads = [
    np.array([1.0, 0.1]),
    np.array([0.8, 0.2]),
    np.array([0.6, 0.3]),
    np.array([0.4, 0.4]),
    np.array([0.2, 0.5]),   # overwrites slot 0
    np.array([0.1, 0.6]),   # overwrites slot 1
]

for i, g in enumerate(grads):
    buf.write(g)
    contents = buf.read()
    print(f"write {i+1}: full={buf.is_full()}  "
          f"entries={buf.n_entries}  "
          f"oldest={contents[0]}  newest={contents[-1]}")

# Power of 2 validation
print("\n--- Validation test ---")
try:
    bad = CircularGradientBuffer(capacity=6, grad_shape=(2,))
except ValueError as e:
    print(f"Caught: {e}")