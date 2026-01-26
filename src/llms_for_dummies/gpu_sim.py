"""
gpu_sim.py
A cycle-accurate simulator for GPU memory hierarchies.

Key concepts:
- Cycle: One "tick" of the GPU clock. Modern GPUs run at ~1-2 GHz (1-2 billion
  cycles/second). We use cycles as our unit of time to reason about ratios
  without worrying about actual clock speeds.
- HBM (High Bandwidth Memory): Large but slow. Data here costs cycles to access.
- SRAM (On-chip cache): Small but fast. Data here is "free" to access.
- The goal is to minimize HBM traffic by keeping working data in SRAM.
"""
from dataclasses import dataclass


@dataclass
class GPUSpec:
    name: str
    bytes_per_float: int  # e.g., 2 for FP16, 4 for FP32
    sram_size: int        # capacity in floats
    hbm_bandwidth: float  # floats per cycle
    sram_bandwidth: float # floats per cycle
    flop_rate: float      # floating point ops per cycle

    @staticmethod
    def sim_v100():
        """
        Simplified V100 (2017):
        - Real: ~900 GB/s HBM2, ~125 TFLOPS FP16 → ~140 FLOPs/byte
        - Oldest of the three, least compute-bound
        """
        return GPUSpec(
            name="Simulated V100",
            bytes_per_float=2,    # FP16
            sram_size=64 * 1024,  # 64k floats * 2 (bytes_per_float) = 128KB
            hbm_bandwidth=1.0,    # Baseline unit
            sram_bandwidth=10.0,  # ~10x faster than HBM
            flop_rate=70.0        # ~70x faster than HBM (less compute-bound)
        )

    @staticmethod
    def sim_a100():
        """
        Simplified A100 (2020):
        - Real: ~2 TB/s HBM2e, ~312 TFLOPS FP16 → ~156 FLOPs/byte
        - Middle ground, significantly more compute-bound than V100
        """
        return GPUSpec(
            name="Simulated A100",
            bytes_per_float=2,    # FP16
            sram_size=128 * 1024, # 128k floats * 2 (bytes_per_float) = 256KB
            hbm_bandwidth=1.0,    # Baseline unit
            sram_bandwidth=10.0,  # ~10x faster than HBM
            flop_rate=150.0       # ~150x faster than HBM
        )

    @staticmethod
    def sim_h100():
        """
        Simplified H100 (2022):
        - Real: ~3.35 TB/s HBM3, ~990 TFLOPS FP16 → ~295 FLOPs/byte
        - Most compute-bound, memory optimization is critical
        """
        return GPUSpec(
            name="Simulated H100",
            bytes_per_float=2,    # FP16
            sram_size=192 * 1024, # 192k floats * 2 (bytes_per_float) = 384KB
            hbm_bandwidth=1.0,    # Baseline unit
            sram_bandwidth=10.0,  # ~10x faster than HBM
            flop_rate=300.0       # ~300x faster than HBM (most compute-bound)
        )

    def optimal_tile_size_2d(self, d: int, usage: float = 1.0) -> int:
        """
        Compute optimal square tile size for full 2D tiled attention.

        The constraining step is O = P @ V which needs:
          - P tile: Br × Bc = Br² (square tiles)
          - V tile: Bc × d = Br × d
          - partial result: Br × d
          - accumulator: Br × d

        Total SRAM: Br² + 3*Br*d ≤ sram_size * usage

        Solving: Br = (-3d + sqrt(9d² + 4*sram_size*usage)) / 2

        Args:
            d: Head dimension
            usage: Fraction of SRAM to use (default 1.0 = 100%)

        Returns:
            Optimal tile size Br (rounded down to be safe)
        """
        import math
        available = self.sram_size * usage
        # Quadratic formula: Br² + 3*d*Br - available = 0
        Br = (-3 * d + math.sqrt(9 * d * d + 4 * available)) / 2
        return int(Br)

    def optimal_tile_size_flash(self, d: int, usage: float = 1.0) -> int:
        """
        Compute optimal tile size for FlashAttention.

        FlashAttention has a larger working set than naive 2D tiling because
        of online softmax intermediate tensors. Two peak SRAM moments:

        Peak 1 (during s_shifted = s_tile.sub_rowvec):
          q_tile(Bd) + o(Bd) + v_tile(Bd) + s_tile(B²) + s_shifted(B²) + small vectors
          ≈ 3Bd + 2B²

        Peak 2 (during o_local = p_tile @ v_tile):
          q_tile(Bd) + o(Bd) + v_tile(Bd) + p_tile(B²) + o_local(Bd) + small vectors
          ≈ 4Bd + B²

        Combined constraint: 2*B² + 4*B*d ≤ sram_size * usage

        Args:
            d: Head dimension
            usage: Fraction of SRAM to use (default 1.0 = 100%)

        Returns:
            Optimal tile size for FlashAttention
        """
        import math
        available = self.sram_size * usage
        # Quadratic formula: 2*Br² + 4*d*Br - available = 0
        # Br = (-4d + sqrt(16d² + 8*available)) / 4 = -d + sqrt(d² + available/2)
        Br = -d + math.sqrt(d * d + available / 2)
        return int(Br)

    def optimal_tile_size_1d(self, N: int, d: int, usage: float = 1.0) -> int | None:
        """
        Compute optimal tile size for 1D tiled attention.

        For full 1D tiled attention:
          - K (full): N × d  (must fit entirely)
          - V (full): N × d  (must fit entirely)
          - Q tile: Br × d
          - S tile: Br × N
          - Softmax peak: 3 × Br × N  (input + output + scratch)
          - O tile: Br × d

        Peak SRAM (during softmax): 2*N*d + Br*d + 3*Br*N ≤ sram_size * usage

        Solving: Br = (sram_size * usage - 2*N*d) / (d + 3*N)

        Args:
            N: Sequence length
            d: Head dimension
            usage: Fraction of SRAM to use (default 1.0 = 100%)

        Returns:
            Optimal tile size Br, or None if K+V don't fit
        """
        available = int(self.sram_size * usage)
        kv_size = 2 * N * d

        if kv_size >= available:
            print(f"1D tiling not possible:")
            print(f"  K + V = 2 × {N} × {d} = {kv_size:,} floats")
            print(f"  SRAM  = {available:,} floats")
            print(f"  K + V alone fill/exceed SRAM — need 2D tiling")
            return None

        Br = (available - kv_size) / (d + 3 * N)
        if Br < 1:
            remaining = available - kv_size
            print(f"1D tiling not possible:")
            print(f"  K + V = {kv_size:,} floats")
            print(f"  Remaining = {remaining:,} floats")
            print(f"  Not enough room for even 1 row of Q + S tiles")
            return None

        return int(Br)


class Profiler:
    """
    Tracks cycles spent on compute vs memory, and memory usage patterns.

    Key insight: The cost of an operation depends on WHERE the data lives.
    - Data in SRAM: compute-bound (fast)
    - Data in HBM: memory-bound (slow, pay HBM bandwidth cost)
    """

    def __init__(self, spec: GPUSpec, name: str):
        self.spec = spec
        self.name = name

        # Cycle tracking
        self.cycles_compute = 0
        self.cycles_hbm = 0

        # HBM allocation tracking
        self.current_hbm_usage = 0
        self.peak_hbm_usage = 0

        # SRAM working set tracking
        self.current_sram_usage = 0
        self.peak_sram_usage = 0

        # Traffic tracking
        self.total_hbm_reads = 0
        self.total_hbm_writes = 0

        # Detailed log for visualization
        self.log = []

    # --- HBM ALLOCATION ---
    # Use these to track what matrices exist in HBM

    def allocate_hbm(self, num_floats: int, name: str):
        """Allocate a matrix in HBM. No cycle cost (just tracking)."""
        self.current_hbm_usage += num_floats
        self.peak_hbm_usage = max(self.peak_hbm_usage, self.current_hbm_usage)
        self.log.append(("hbm_alloc", name, num_floats))

    def free_hbm(self, num_floats: int, name: str):
        """Free a matrix from HBM."""
        self.current_hbm_usage -= num_floats
        self.log.append(("hbm_free", name, num_floats))

    # --- SRAM WORKING SET ---
    # Use these to track what must fit in SRAM simultaneously

    @property
    def sram_overflowed(self) -> bool:
        """True if current SRAM usage exceeds capacity."""
        return self.current_sram_usage > self.spec.sram_size

    def sram_push(self, num_floats: int, name: str):
        """Add to SRAM working set. Call before an operation that needs this data."""
        self.current_sram_usage += num_floats
        self.peak_sram_usage = max(self.peak_sram_usage, self.current_sram_usage)
        self.log.append(("sram_push", name, num_floats))

        # Log overflow if it occurs
        if self.current_sram_usage > self.spec.sram_size:
            overflow = self.current_sram_usage - self.spec.sram_size
            self.log.append(("sram_overflow", name, overflow))

    def sram_pop(self, num_floats: int, name: str):
        """Remove from SRAM working set. Call when done with this data."""
        self.current_sram_usage -= num_floats
        self.log.append(("sram_pop", name, num_floats))

    # --- HBM TRANSFERS ---
    # These cost cycles (HBM bandwidth is the bottleneck)

    def load_from_hbm(self, num_floats: int, name: str):
        """Load data from HBM into SRAM. Costs HBM bandwidth cycles."""
        cycles = num_floats / self.spec.hbm_bandwidth
        self.cycles_hbm += cycles
        self.total_hbm_reads += num_floats
        self.log.append(("hbm_read", name, num_floats))

    def store_to_hbm(self, num_floats: int, name: str):
        """Store data from SRAM to HBM. Costs HBM bandwidth cycles."""
        cycles = num_floats / self.spec.hbm_bandwidth
        self.cycles_hbm += cycles
        self.total_hbm_writes += num_floats
        self.log.append(("hbm_write", name, num_floats))

    # --- COMPUTE ---
    # These cost compute cycles (but are usually much faster than memory)

    def matmul(self, M: int, N: int, K: int, name: str = ""):
        """
        Matrix multiply C = A @ B where A is (M, K) and B is (K, N).
        Cost: 2 * M * N * K FLOPs.
        """
        flops = 2 * M * N * K
        cycles = flops / self.spec.flop_rate
        self.cycles_compute += cycles
        self.log.append(("compute_matmul", name, flops))

    def elementwise(self, num_floats: int, ops_per_element: int = 1, name: str = ""):
        """
        Elementwise operations (softmax, scaling, etc).
        Cost: num_floats * ops_per_element FLOPs.
        """
        flops = num_floats * ops_per_element
        cycles = flops / self.spec.flop_rate
        self.cycles_compute += cycles
        self.log.append(("compute_elementwise", name, flops))

    # --- REPORTING ---

    def report(self):
        """Print a summary of the simulation."""
        total_cycles = self.cycles_compute + self.cycles_hbm

        print(f"{'='*60}")
        print(f"{self.name}")
        print(f"{'='*60}")

        # Memory usage with visual bars
        print(f"\nMemory Usage:")
        hbm_bytes = self.peak_hbm_usage * self.spec.bytes_per_float
        if hbm_bytes >= 1024 * 1024:
            hbm_str = f"{hbm_bytes / 1024 / 1024:.1f} MB"
        else:
            hbm_str = f"{hbm_bytes / 1024:.0f} KB"
        print(f"  Peak HBM:   {hbm_str:>8}")

        sram_pct = self.peak_sram_usage / self.spec.sram_size * 100
        bar_width = 15
        filled = min(int(sram_pct / 100 * bar_width), bar_width)
        sram_bar = "█" * filled + "·" * (bar_width - filled)
        overflow = "" if sram_pct <= 100 else f" ⚠️ OVERFLOW"
        print(f"  Peak SRAM:  │{sram_bar}│{sram_pct:>4.0f}%{overflow}")

        # HBM traffic
        print(f"\nHBM Traffic:")
        print(f"  Reads:  {self.total_hbm_reads:>15,} floats "
              f"({self.total_hbm_reads * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")
        print(f"  Writes: {self.total_hbm_writes:>15,} floats "
              f"({self.total_hbm_writes * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")
        print(f"  Total:  {self.total_hbm_reads + self.total_hbm_writes:>15,} floats "
              f"({(self.total_hbm_reads + self.total_hbm_writes) * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")

        # Time breakdown with ASCII bar
        print(f"\nTime Breakdown:")
        bar_width = 50
        compute_pct = self.cycles_compute / total_cycles
        hbm_pct = self.cycles_hbm / total_cycles
        compute_bars = int(compute_pct * bar_width)
        hbm_bars = bar_width - compute_bars

        print(f"  ┌{'─' * bar_width}┐")
        print(f"  │{'█' * compute_bars}{'░' * hbm_bars}│")
        print(f"  └{'─' * bar_width}┘")
        print(f"   Computing ({compute_pct:.0%}){'':>{bar_width - 32}}Waiting for HBM ({hbm_pct:.0%})")

        # Bottleneck
        if self.cycles_hbm > self.cycles_compute:
            print(f"\n→ Memory-bound (GPU is waiting on HBM)")
        else:
            print(f"\n→ Compute-bound (GPU is busy computing)")

    def get_hbm_traffic_log(self):
        """Return just the HBM read/write operations for visualization."""
        return [(op, name, size) for op, name, size in self.log
                if op in ("hbm_read", "hbm_write")]


class Tensor:
    """
    A simulated tensor for the GPU memory simulator.

    This class wraps profiler calls to provide PyTorch-like syntax.
    There is no actual data - just shape tracking and profiler calls.

    Tensors live in HBM by default. Use .load() to bring data into SRAM
    for computation, then .write_hbm() to store results.

    Example:
        Q = Tensor((N, d), "Q", prof)       # Allocate in HBM
        K = Tensor((N, d), "K", prof)       # Allocate in HBM

        # Explicit tiling - you control what's in SRAM
        for i in range(0, N, Br):
            q_tile = Q.load(rows=(i, i+Br)) # Load slice to SRAM
            k = K.load()                     # Load full K to SRAM
            result = q_tile @ k.T            # Compute in SRAM
            result.write_hbm()               # Write result to HBM
            result.free()                    # Free SRAM
            k.free()
            q_tile.free()
    """

    verbose: bool = False

    def __init__(
        self,
        shape: tuple,
        name: str,
        profiler: "Profiler",
        *,
        _in_sram: bool = False,
        _transposed: bool = False,
    ):
        if len(shape) != 2:
            raise ValueError("Only 2D tensors are supported")

        self.shape = shape
        self.name = name
        self._profiler = profiler
        self._in_sram = _in_sram
        self._transposed = _transposed
        self._freed = False

        # Allocate in HBM (unless already in SRAM or transposed view)
        if not _in_sram and not _transposed:
            self._profiler.allocate_hbm(self.size, self.name)
            if Tensor.verbose:
                self._print_hbm_status(f"[+HBM] Allocate {self.name} {self.shape}")

    @property
    def size(self) -> int:
        """Total number of floats in the tensor."""
        return self.shape[0] * self.shape[1]

    @classmethod
    def zeros(cls, shape: tuple, name: str, profiler: "Profiler") -> "Tensor":
        """Create a zero-initialized tensor in SRAM (for accumulators)."""
        t = cls(shape, name, profiler, _in_sram=True)
        profiler.sram_push(t.size, name)
        if cls.verbose:
            t._print_sram_status(f"[SRAM] Allocate {name} (zeros)")
        return t

    def load(self, rows: tuple = None, cols: tuple = None) -> "Tensor":
        """
        Load this tensor (or a slice) from HBM to SRAM.

        Args:
            rows: Optional (start, end) tuple to load only certain rows
            cols: Optional (start, end) tuple to load only certain columns

        Returns a new Tensor that lives in SRAM.
        """
        if self._in_sram:
            raise ValueError(f"{self.name} is already in SRAM")

        # Determine slice bounds
        r_start = rows[0] if rows else 0
        r_end = rows[1] if rows else self.shape[0]
        c_start = cols[0] if cols else 0
        c_end = cols[1] if cols else self.shape[1]

        # Compute slice shape and name
        slice_shape = (r_end - r_start, c_end - c_start)
        if rows or cols:
            if cols and not rows:
                slice_name = f"{self.name}[:, {c_start}:{c_end}]"
            elif rows and not cols:
                slice_name = f"{self.name}[{r_start}:{r_end}]"
            else:
                slice_name = f"{self.name}[{r_start}:{r_end}, {c_start}:{c_end}]"
        else:
            slice_name = self.name

        slice_size = slice_shape[0] * slice_shape[1]

        self._profiler.load_from_hbm(slice_size, slice_name)
        self._profiler.sram_push(slice_size, slice_name)

        if Tensor.verbose:
            self._print_sram_status(f"[HBM → SRAM] Load {slice_name}")

        return Tensor(slice_shape, slice_name, self._profiler, _in_sram=True)

    @property
    def T(self) -> "Tensor":
        """Transpose view - swaps dimensions, no memory cost."""
        return Tensor(
            (self.shape[1], self.shape[0]),
            f"{self.name}.T",
            self._profiler,
            _in_sram=self._in_sram,
            _transposed=True,
        )

    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        """Format bytes as KB or MB depending on size."""
        if num_bytes >= 1024 * 1024:
            return f"{num_bytes / 1024 / 1024:>5.1f} MB"
        else:
            return f"{num_bytes / 1024:>5.0f} KB"

    @staticmethod
    def _make_bar(pct: float, width: int, fill_char: str, empty_char: str) -> str:
        """Create a progress bar string."""
        filled = min(int(pct / 100 * width), width)
        return fill_char * filled + empty_char * (width - filled)

    def _print_hbm_status(self, message: str) -> None:
        """Print HBM allocation status."""
        prof = self._profiler
        hbm_bytes = prof.current_hbm_usage * prof.spec.bytes_per_float
        hbm_str = self._format_bytes(hbm_bytes)
        # Truncate message if too long, align with SRAM output
        msg = message[:42] if len(message) > 42 else message
        print(f"{msg:<42}  HBM  {hbm_str}")

    def _print_sram_status(self, message: str) -> None:
        """Print SRAM operation status with visual bar."""
        prof = self._profiler
        sram_pct = prof.current_sram_usage / prof.spec.sram_size * 100

        # Build SRAM bar (15 chars wide)
        bar_width = 15
        if sram_pct <= 100:
            sram_bar = self._make_bar(sram_pct, bar_width, "█", "·")
            overflow = ""
        else:
            sram_bar = "█" * bar_width
            overflow = f" ⚠️ +{sram_pct - 100:.0f}%"

        # Truncate message if too long, fixed-width ensures bars align
        msg = message[:42] if len(message) > 42 else message
        print(f"{msg:<42}  SRAM │{sram_bar}│{sram_pct:>4.0f}%{overflow}")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication: C = self @ other

        Both operands must be in SRAM. Result is allocated in SRAM.
        Use .write_hbm() to store result, .free() to release SRAM.
        """
        # Both must be in SRAM
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for matmul (call .load() first)")
        if not other._in_sram:
            raise ValueError(f"{other.name} must be in SRAM for matmul (call .load() first)")

        # Shape validation: (M, K) @ (K, N) -> (M, N)
        M, K = self.shape
        K2, N = other.shape
        if K != K2:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")

        # Allocate result in SRAM
        result_name = f"({self.name}@{other.name})"
        result = Tensor((M, N), result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(result.size, result_name)

        # Compute
        self._profiler.matmul(M, N, K, result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} @ {other.name} → {result.shape}")

        return result

    def scale(self, factor: float, name_suffix: str = "") -> "Tensor":
        """Scale tensor by a constant in-place. Must be in SRAM. Returns self."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for scale")

        old_name = self.name
        self.name = f"{self.name}{name_suffix}" if name_suffix else f"scaled({self.name})"
        self._profiler.elementwise(self.size, ops_per_element=1, name=self.name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] scale({old_name}) in-place")

        return self

    def add_(self, other: "Tensor") -> "Tensor":
        """Add another tensor in-place. Both must be in SRAM with same shape. Returns self."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for add_")
        if not other._in_sram:
            raise ValueError(f"{other.name} must be in SRAM for add_")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        self._profiler.elementwise(self.size, ops_per_element=1, name=f"{self.name}+=")

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} += {other.name}")

        return self

    def mul_(self, other: "Tensor") -> "Tensor":
        """Multiply by another tensor in-place. Supports broadcasting (rows,1) to (rows,cols)."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for mul_")
        if not other._in_sram:
            raise ValueError(f"{other.name} must be in SRAM for mul_")

        # Check shapes: either same shape or other is (rows, 1) for broadcasting
        if other.shape == self.shape:
            pass  # Same shape, element-wise
        elif other.shape == (self.shape[0], 1):
            pass  # Broadcasting column vector
        else:
            raise ValueError(f"Shape mismatch for mul_: {self.shape} vs {other.shape}")

        self._profiler.elementwise(self.size, ops_per_element=1, name=f"{self.name}*=")

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} *= {other.name}")

        return self

    def rowmax(self) -> "Tensor":
        """Compute max of each row. Returns (rows, 1) tensor in SRAM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for rowmax")

        rows = self.shape[0]
        result_name = f"rowmax({self.name})"
        result = Tensor((rows, 1), result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(rows, result_name)

        # Cost: read each element once, compare
        self._profiler.elementwise(self.size, ops_per_element=1, name=result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] rowmax({self.name}) → ({rows}, 1)")

        return result

    def rowsum(self) -> "Tensor":
        """Compute sum of each row. Returns (rows, 1) tensor in SRAM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for rowsum")

        rows = self.shape[0]
        result_name = f"rowsum({self.name})"
        result = Tensor((rows, 1), result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(rows, result_name)

        # Cost: read each element once, add
        self._profiler.elementwise(self.size, ops_per_element=1, name=result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] rowsum({self.name}) → ({rows}, 1)")

        return result

    def sub_rowvec(self, vec: "Tensor") -> "Tensor":
        """Subtract column vector from each row. Returns new tensor in SRAM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for sub_rowvec")
        if not vec._in_sram:
            raise ValueError(f"{vec.name} must be in SRAM for sub_rowvec")
        if vec.shape != (self.shape[0], 1):
            raise ValueError(f"Expected ({self.shape[0]}, 1), got {vec.shape}")

        result_name = f"({self.name}-{vec.name})"
        result = Tensor(self.shape, result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(result.size, result_name)

        self._profiler.elementwise(self.size, ops_per_element=1, name=result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} - {vec.name} (broadcast)")

        return result

    def div_rowvec(self, vec: "Tensor") -> "Tensor":
        """Divide each row by column vector. Returns new tensor in SRAM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for div_rowvec")
        if not vec._in_sram:
            raise ValueError(f"{vec.name} must be in SRAM for div_rowvec")
        if vec.shape != (self.shape[0], 1):
            raise ValueError(f"Expected ({self.shape[0]}, 1), got {vec.shape}")

        result_name = f"({self.name}/{vec.name})"
        result = Tensor(self.shape, result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(result.size, result_name)

        # Division is SFU op, ~8x slower
        self._profiler.elementwise(self.size, ops_per_element=8, name=result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} / {vec.name} (broadcast)")

        return result

    def exp(self) -> "Tensor":
        """Elementwise exp. Returns new tensor in SRAM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for exp")

        result_name = f"exp({self.name})"
        result = Tensor(self.shape, result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(result.size, result_name)

        # Exp is SFU op, ~8x slower than ALU
        self._profiler.elementwise(self.size, ops_per_element=8, name=result_name)

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] exp({self.name}) → {result.shape}")

        return result

    def max_(self, other: "Tensor") -> "Tensor":
        """Element-wise max in-place. Both must have same shape. Returns self."""
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for max_")
        if not other._in_sram:
            raise ValueError(f"{other.name} must be in SRAM for max_")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        self._profiler.elementwise(self.size, ops_per_element=1, name=f"max({self.name},{other.name})")

        if Tensor.verbose:
            self._print_sram_status(f"[Compute] {self.name} = max({self.name}, {other.name})")

        return self

    def softmax(self) -> "Tensor":
        """
        Apply softmax row-wise. Must be in SRAM. Returns new tensor in SRAM.
        
        Simulates "Standard Safe Softmax" (3-pass):
          Pass 1: Max (read x)
          Pass 2: Exp (read x, write temp_exp)
          Pass 3: Div (read temp_exp, write out)
        """
        if not self._in_sram:
            raise ValueError(f"{self.name} must be in SRAM for softmax")

        # 1. COST REFINEMENT: Weight SFU ops higher
        # SFU (Special Function Unit) ops like exp/div are ~8-16x slower than ALU.
        # Approx cost: 1 cmp + 1 sub + 1 add + 1 exp(SFU) + 1 div(SFU)
        # Weighted: 3 ALU + 2 SFU (~16 ALU) = ~19 ops
        OPS_WEIGHTED = 19

        result_name = f"softmax({self.name})"
        result = Tensor(self.shape, result_name, self._profiler, _in_sram=True)
        self._profiler.sram_push(result.size, result_name)

        # 2. MEMORY REFINEMENT: Account for intermediate storage
        # Standard softmax creates a temporary tensor of exponentials: y = exp(x-m).
        # We must store 'y' before we can divide by sum(y).
        # This "scratch" memory is what kills performance/capacity in standard attention.
        scratch_name = f"temp_exp({self.name})"
        self._profiler.sram_push(self.size, scratch_name)

        # Show peak memory (with scratch allocated)
        if Tensor.verbose:
            self._print_sram_status(f"[Compute] softmax({self.name}) → {result.shape}")

        # Charge the compute cost
        self._profiler.elementwise(self.size, ops_per_element=OPS_WEIGHTED, name=result_name)

        # Free the scratchpad (Pass 3 consumes it to produce result)
        self._profiler.sram_pop(self.size, scratch_name)

        return result

    def write_hbm(self) -> None:
        """Write this SRAM tensor to HBM."""
        if not self._in_sram:
            raise ValueError(f"{self.name} is not in SRAM")

        self._profiler.store_to_hbm(self.size, self.name)

        if Tensor.verbose:
            self._print_sram_status(f"[SRAM → HBM] Store {self.name}")

    def free(self) -> None:
        """Free this tensor from SRAM."""
        if self._freed:
            return
        if not self._in_sram:
            raise ValueError(f"{self.name} is not in SRAM (use .free_hbm() for HBM tensors)")

        self._profiler.sram_pop(self.size, self.name)
        self._freed = True

        if Tensor.verbose:
            self._print_sram_status(f"[SRAM] Free {self.name}")

    def free_hbm(self) -> None:
        """Free this tensor from HBM."""
        if self._freed:
            return
        if self._in_sram:
            raise ValueError(f"{self.name} is in SRAM, not HBM")
        if self._transposed:
            return  # Transposed views don't own memory

        self._profiler.free_hbm(self.size, self.name)
        self._freed = True

        if Tensor.verbose:
            self._print_hbm_status(f"[-HBM] Free {self.name}")

    def __repr__(self) -> str:
        loc = "SRAM" if self._in_sram else "HBM"
        return f"Tensor({self.shape}, '{self.name}', {loc})"
