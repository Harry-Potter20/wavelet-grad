import numpy as np

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

if __name__ == "__main__":
    # --- Tests ---
    print("\n--- CircularGradientBuffer tests ---")

    buf = CircularGradientBuffer(capacity=4, grad_shape=(2,))

    grads = [
        np.array([1.0, 0.1]),
        np.array([0.8, 0.2]),
        np.array([0.6, 0.3]),
        np.array([0.4, 0.4]),
        np.array([0.2, 0.5]),
        np.array([0.1, 0.6]),
    ]

    for i, g in enumerate(grads):
        buf.write(g)
        contents = buf.read()
        print(f"write {i+1}: full={buf.is_full()}  "
              f"entries={buf.n_entries}  "
              f"oldest={contents[0]}  newest={contents[-1]}")

    print("\n--- Validation test ---")
    try:
        bad = CircularGradientBuffer(capacity=6, grad_shape=(2,))
    except ValueError as e:
        print(f"Caught: {e}")