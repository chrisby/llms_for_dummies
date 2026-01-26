"""
Tests for gpu_sim.py

Run with:
  python tests/test_gpu_sim.py          # Simple runner (no dependencies)
  python -m pytest tests/test_gpu_sim.py -v  # With pytest
"""

from llms_for_dummies.gpu_sim import GPUSpec, Profiler, Tensor
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gpu():
    """A small GPU for testing (easy to reason about sizes)."""
    return GPUSpec(
        name="Test GPU",
        bytes_per_float=2,
        sram_size=1000,  # 1000 floats - easy to calculate
        hbm_bandwidth=1.0,
        sram_bandwidth=10.0,
        flop_rate=100.0,
    )


@pytest.fixture
def prof(gpu):
    """Fresh profiler for each test."""
    return Profiler(gpu, "Test")


# =============================================================================
# Profiler Tests
# =============================================================================


class TestProfiler:
    def test_initial_state(self, prof):
        """Profiler starts with zero usage."""
        assert prof.current_hbm_usage == 0
        assert prof.current_sram_usage == 0
        assert prof.peak_hbm_usage == 0
        assert prof.peak_sram_usage == 0
        assert prof.sram_overflowed == False

    def test_hbm_allocation(self, prof):
        """HBM allocation tracking."""
        prof.allocate_hbm(100, "A")
        assert prof.current_hbm_usage == 100
        assert prof.peak_hbm_usage == 100

        prof.allocate_hbm(200, "B")
        assert prof.current_hbm_usage == 300
        assert prof.peak_hbm_usage == 300

        prof.free_hbm(100, "A")
        assert prof.current_hbm_usage == 200
        assert prof.peak_hbm_usage == 300  # Peak unchanged

    def test_sram_push_pop(self, prof):
        """SRAM push/pop tracking."""
        prof.sram_push(100, "A")
        assert prof.current_sram_usage == 100
        assert prof.peak_sram_usage == 100
        assert prof.sram_overflowed == False

        prof.sram_push(200, "B")
        assert prof.current_sram_usage == 300
        assert prof.peak_sram_usage == 300

        prof.sram_pop(100, "A")
        assert prof.current_sram_usage == 200
        assert prof.peak_sram_usage == 300  # Peak unchanged

    def test_sram_overflow_flag(self, gpu):
        """SRAM overflow reflects current state."""
        prof = Profiler(gpu, "Test")  # sram_size = 1000

        prof.sram_push(500, "A")
        assert prof.sram_overflowed == False

        prof.sram_push(600, "B")  # Total 1100 > 1000
        assert prof.sram_overflowed == True
        assert prof.current_sram_usage == 1100

        # Flag becomes False after freeing (current state)
        prof.sram_pop(600, "B")
        assert prof.sram_overflowed == False
        assert prof.current_sram_usage == 500

    def test_hbm_traffic(self, prof):
        """HBM read/write tracking."""
        prof.load_from_hbm(100, "A")
        assert prof.total_hbm_reads == 100
        assert prof.total_hbm_writes == 0

        prof.store_to_hbm(50, "B")
        assert prof.total_hbm_reads == 100
        assert prof.total_hbm_writes == 50

    def test_cycles(self, prof):
        """Cycle tracking for compute and memory."""
        # HBM transfer: 100 floats / 1.0 bandwidth = 100 cycles
        prof.load_from_hbm(100, "A")
        assert prof.cycles_hbm == 100

        # Matmul: 2*M*N*K / flop_rate = 2*10*10*10 / 100 = 20 cycles
        prof.matmul(10, 10, 10, "test")
        assert prof.cycles_compute == 20


# =============================================================================
# Tensor Tests - Memory Movement
# =============================================================================


class TestTensorMemory:
    def test_tensor_allocates_hbm(self, prof):
        """Creating a tensor allocates HBM."""
        t = Tensor((10, 10), "A", prof)
        assert prof.current_hbm_usage == 100
        assert prof.current_sram_usage == 0

    def test_load_moves_to_sram(self, prof):
        """Loading a tensor moves it to SRAM."""
        t = Tensor((10, 10), "A", prof)
        assert prof.current_sram_usage == 0

        t_sram = t.load()
        assert prof.current_sram_usage == 100
        assert prof.total_hbm_reads == 100
        assert t_sram._in_sram == True

    def test_load_slice(self, prof):
        """Loading a slice only loads that portion."""
        t = Tensor((100, 10), "A", prof)  # 1000 floats in HBM

        # Load rows 0-50 (500 floats)
        t_slice = t.load(rows=(0, 50))
        assert prof.current_sram_usage == 500
        assert prof.total_hbm_reads == 500
        assert t_slice.shape == (50, 10)

    def test_load_slice_cols(self, prof):
        """Loading column slice."""
        t = Tensor((10, 100), "A", prof)

        t_slice = t.load(cols=(0, 50))
        assert prof.current_sram_usage == 500
        assert t_slice.shape == (10, 50)

    def test_load_slice_both(self, prof):
        """Loading both row and column slice."""
        t = Tensor((100, 100), "A", prof)

        t_slice = t.load(rows=(0, 10), cols=(0, 20))
        assert prof.current_sram_usage == 200
        assert t_slice.shape == (10, 20)

    def test_write_hbm(self, prof):
        """Writing to HBM records the traffic."""
        t = Tensor((10, 10), "A", prof)
        t_sram = t.load()
        assert prof.total_hbm_writes == 0

        t_sram.write_hbm()
        assert prof.total_hbm_writes == 100

    def test_free_sram(self, prof):
        """Freeing releases SRAM."""
        t = Tensor((10, 10), "A", prof)
        t_sram = t.load()
        assert prof.current_sram_usage == 100

        t_sram.free()
        assert prof.current_sram_usage == 0

    def test_free_hbm(self, prof):
        """Freeing HBM tensor."""
        t = Tensor((10, 10), "A", prof)
        assert prof.current_hbm_usage == 100

        t.free_hbm()
        assert prof.current_hbm_usage == 0


# =============================================================================
# Tensor Tests - Transpose
# =============================================================================


class TestTensorTranspose:
    def test_transpose_shape(self, prof):
        """Transpose swaps dimensions."""
        t = Tensor((10, 20), "A", prof)
        t_sram = t.load()

        t_T = t_sram.T
        assert t_T.shape == (20, 10)

    def test_transpose_no_memory_cost(self, prof):
        """Transpose doesn't allocate memory."""
        t = Tensor((10, 20), "A", prof)
        t_sram = t.load()
        sram_before = prof.current_sram_usage

        t_T = t_sram.T
        assert prof.current_sram_usage == sram_before

    def test_transpose_in_matmul(self, prof):
        """Transpose works in matmul."""
        a = Tensor((10, 20), "A", prof)
        b = Tensor((30, 20), "B", prof)  # Will be transposed to (20, 30)

        a_sram = a.load()
        b_sram = b.load()

        # (10, 20) @ (20, 30) -> (10, 30)
        c = a_sram @ b_sram.T
        assert c.shape == (10, 30)


# =============================================================================
# Tensor Tests - Matmul
# =============================================================================


class TestTensorMatmul:
    def test_matmul_shape(self, prof):
        """Matmul produces correct shape."""
        a = Tensor((10, 20), "A", prof)
        b = Tensor((20, 30), "B", prof)

        a_sram = a.load()
        b_sram = b.load()

        c = a_sram @ b_sram
        assert c.shape == (10, 30)
        assert c._in_sram == True

    def test_matmul_allocates_result(self, prof):
        """Matmul allocates result in SRAM."""
        a = Tensor((10, 20), "A", prof)
        b = Tensor((20, 30), "B", prof)

        a_sram = a.load()
        b_sram = b.load()
        sram_before = prof.current_sram_usage  # 200 + 600 = 800

        c = a_sram @ b_sram
        # Result is 10*30 = 300 floats
        assert prof.current_sram_usage == sram_before + 300

    def test_matmul_requires_sram(self, prof):
        """Matmul fails if operands not in SRAM."""
        a = Tensor((10, 20), "A", prof)
        b = Tensor((20, 30), "B", prof)

        with pytest.raises(ValueError, match="must be in SRAM"):
            a @ b

    def test_matmul_shape_mismatch(self, prof):
        """Matmul fails on shape mismatch."""
        a = Tensor((10, 20), "A", prof)
        b = Tensor((30, 40), "B", prof)  # Wrong inner dimension

        a_sram = a.load()
        b_sram = b.load()

        with pytest.raises(ValueError, match="Shape mismatch"):
            a_sram @ b_sram


# =============================================================================
# Tensor Tests - Softmax
# =============================================================================


class TestTensorSoftmax:
    def test_softmax_shape(self, prof):
        """Softmax preserves shape."""
        t = Tensor((10, 20), "A", prof)
        t_sram = t.load()

        result = t_sram.softmax()
        assert result.shape == (10, 20)
        assert result._in_sram == True

    def test_softmax_allocates_result(self, prof):
        """Softmax allocates new tensor in SRAM."""
        t = Tensor((10, 20), "A", prof)
        t_sram = t.load()
        sram_before = prof.current_sram_usage

        result = t_sram.softmax()
        assert prof.current_sram_usage == sram_before + 200

    def test_softmax_requires_sram(self, prof):
        """Softmax fails if tensor not in SRAM."""
        t = Tensor((10, 20), "A", prof)

        with pytest.raises(ValueError, match="must be in SRAM"):
            t.softmax()


# =============================================================================
# Tensor Tests - Zeros and Add
# =============================================================================


class TestTensorZerosAndAdd:
    def test_zeros_in_sram(self, prof):
        """Zeros creates tensor in SRAM."""
        t = Tensor.zeros((10, 20), "acc", prof)
        assert t._in_sram == True
        assert t.shape == (10, 20)
        assert prof.current_sram_usage == 200

    def test_zeros_no_hbm(self, prof):
        """Zeros doesn't allocate HBM."""
        t = Tensor.zeros((10, 20), "acc", prof)
        assert prof.current_hbm_usage == 0

    def test_add_in_place(self, prof):
        """Add_ modifies tensor in place."""
        a = Tensor.zeros((10, 20), "acc", prof)
        b = Tensor((10, 20), "B", prof)
        b_sram = b.load()

        sram_before = prof.current_sram_usage
        result = a.add_(b_sram)

        assert result is a  # Same object
        assert prof.current_sram_usage == sram_before  # No new allocation

    def test_add_shape_mismatch(self, prof):
        """Add_ fails on shape mismatch."""
        a = Tensor.zeros((10, 20), "acc", prof)
        b = Tensor((10, 30), "B", prof)
        b_sram = b.load()

        with pytest.raises(ValueError, match="Shape mismatch"):
            a.add_(b_sram)

    def test_add_requires_sram(self, prof):
        """Add_ requires both tensors in SRAM."""
        a = Tensor.zeros((10, 20), "acc", prof)
        b = Tensor((10, 20), "B", prof)  # In HBM

        with pytest.raises(ValueError, match="must be in SRAM"):
            a.add_(b)


# =============================================================================
# Run tests
# =============================================================================

def run_tests_simple():
    """Run tests without pytest."""
    gpu = GPUSpec(
        name="Test GPU",
        bytes_per_float=2,
        sram_size=1000,
        hbm_bandwidth=1.0,
        sram_bandwidth=10.0,
        flop_rate=100.0,
    )

    tests_passed = 0
    tests_failed = 0

    def run_test(name, test_fn):
        nonlocal tests_passed, tests_failed
        try:
            test_fn()
            print(f"✓ {name}")
            tests_passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")
            tests_failed += 1

    def make_prof():
        return Profiler(gpu, "Test")

    # Profiler tests
    def test_initial_state():
        prof = make_prof()
        assert prof.current_hbm_usage == 0
        assert prof.current_sram_usage == 0
        assert prof.sram_overflowed == False

    def test_sram_overflow():
        prof = make_prof()
        prof.sram_push(500, "A")
        assert prof.sram_overflowed == False
        prof.sram_push(600, "B")  # 1100 > 1000
        assert prof.sram_overflowed == True
        prof.sram_pop(600, "B")  # Back to 500
        assert prof.sram_overflowed == False

    # Tensor tests
    def test_tensor_hbm():
        prof = make_prof()
        t = Tensor((10, 10), "A", prof)
        assert prof.current_hbm_usage == 100

    def test_tensor_load():
        prof = make_prof()
        t = Tensor((10, 10), "A", prof)
        t_sram = t.load()
        assert prof.current_sram_usage == 100
        assert t_sram._in_sram == True

    def test_tensor_load_slice():
        prof = make_prof()
        t = Tensor((100, 10), "A", prof)
        t_slice = t.load(rows=(0, 50))
        assert prof.current_sram_usage == 500
        assert t_slice.shape == (50, 10)

    def test_tensor_free():
        prof = make_prof()
        t = Tensor((10, 10), "A", prof)
        t_sram = t.load()
        t_sram.free()
        assert prof.current_sram_usage == 0

    def test_transpose():
        prof = make_prof()
        t = Tensor((10, 20), "A", prof)
        t_sram = t.load()
        assert t_sram.T.shape == (20, 10)

    def test_matmul():
        prof = make_prof()
        a = Tensor((10, 20), "A", prof)
        b = Tensor((20, 30), "B", prof)
        a_sram = a.load()
        b_sram = b.load()
        c = a_sram @ b_sram
        assert c.shape == (10, 30)

    def test_matmul_transpose():
        prof = make_prof()
        a = Tensor((10, 20), "A", prof)
        b = Tensor((30, 20), "B", prof)
        a_sram = a.load()
        b_sram = b.load()
        c = a_sram @ b_sram.T
        assert c.shape == (10, 30)

    # Run all tests
    run_test("test_initial_state", test_initial_state)
    run_test("test_sram_overflow", test_sram_overflow)
    run_test("test_tensor_hbm", test_tensor_hbm)
    run_test("test_tensor_load", test_tensor_load)
    run_test("test_tensor_load_slice", test_tensor_load_slice)
    run_test("test_tensor_free", test_tensor_free)
    run_test("test_transpose", test_transpose)
    run_test("test_matmul", test_matmul)
    run_test("test_matmul_transpose", test_matmul_transpose)

    print()
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed, running simple test runner\n")
        success = run_tests_simple()
        exit(0 if success else 1)
