import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.buffers import CircularGradientBuffer

# =============================================================
# Unit tests for CircularGradientBuffer
#
# Each test function checks one invariant.
# A test passes if it runs without raising AssertionError.
# We use assert statements — the simplest possible test framework.
# Later we'll migrate to pytest, but the logic stays identical.
# =============================================================

def test_capacity_must_be_power_of_2():
    """Invalid capacities should raise ValueError."""
    for bad in [0, 3, 5, 6, 7, 9, 100]:
        try:
            CircularGradientBuffer(capacity=bad, grad_shape=(2,))
            assert False, f"Should have raised ValueError for capacity={bad}"
        except ValueError:
            pass  # expected

    for good in [1, 2, 4, 8, 16, 32]:
        CircularGradientBuffer(capacity=good, grad_shape=(2,))  # should not raise

    print("test_capacity_must_be_power_of_2 passed")


def test_is_full_triggers_at_correct_step():
    """Buffer should become full after exactly `capacity` writes."""
    buf = CircularGradientBuffer(capacity=4, grad_shape=(1,))

    for i in range(3):
        assert not buf.is_full(), f"Should not be full after {i+1} writes"
        buf.write(np.array([float(i)]))

    buf.write(np.array([3.0]))
    assert buf.is_full(), "Should be full after 4 writes"

    print("test_is_full_triggers_at_correct_step passed")


def test_read_returns_chronological_order():
    """
    read() must always return oldest → newest regardless of
    how many times the buffer has wrapped around.
    """
    buf = CircularGradientBuffer(capacity=4, grad_shape=(1,))

    # Write 6 items — buffer wraps once
    for i in range(6):
        buf.write(np.array([float(i)]))

    contents = buf.read().flatten()

    # Should contain [2, 3, 4, 5] in that order
    expected = np.array([2.0, 3.0, 4.0, 5.0])
    assert np.allclose(contents, expected), \
        f"Expected {expected}, got {contents}"

    print("test_read_returns_chronological_order passed")


def test_newest_entry_is_last():
    """The last row of read() should always be the most recent write."""
    buf = CircularGradientBuffer(capacity=4, grad_shape=(2,))

    for i in range(10):
        g = np.array([float(i), float(i)*2])
        buf.write(g)
        if buf.is_full():
            last = buf.read()[-1]
            assert np.allclose(last, g), \
                f"Last entry should be {g}, got {last}"

    print("test_newest_entry_is_last passed")


def test_write_wrong_shape_raises():
    """Writing a gradient with wrong shape should raise ValueError."""
    buf = CircularGradientBuffer(capacity=4, grad_shape=(3,))
    try:
        buf.write(np.array([1.0, 2.0]))  # shape (2,) not (3,)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("test_write_wrong_shape_raises passed")


def test_n_entries_before_full():
    """n_entries should count correctly before buffer fills."""
    buf = CircularGradientBuffer(capacity=8, grad_shape=(1,))
    for i in range(5):
        assert buf.n_entries == i
        buf.write(np.array([0.0]))
    assert buf.n_entries == 5

    print("test_n_entries_before_full passed")


if __name__ == "__main__":
    test_capacity_must_be_power_of_2()
    test_is_full_triggers_at_correct_step()
    test_read_returns_chronological_order()
    test_newest_entry_is_last()
    test_write_wrong_shape_raises()
    test_n_entries_before_full()
    print("\nAll buffer tests passed.")