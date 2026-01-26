"""
gpu_sim.py
A cycle-accurate simulator for GPU memory hierarchies.

Key concepts:
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

    def sram_push(self, num_floats: int, name: str):
        """Add to SRAM working set. Call before an operation that needs this data."""
        self.current_sram_usage += num_floats
        self.peak_sram_usage = max(self.peak_sram_usage, self.current_sram_usage)
        self.log.append(("sram_push", name, num_floats))

        # Check if we exceed SRAM - this is informational, not a hard error
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

        # Memory usage
        print(f"\nMemory Usage:")
        print(f"  Peak HBM:  {self.peak_hbm_usage:>15,} floats "
              f"({self.peak_hbm_usage * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")

        sram_pct = self.peak_sram_usage / self.spec.sram_size * 100
        sram_note = "" if sram_pct <= 100 else " ⚠️  OVERFLOW"
        print(f"  Peak SRAM: {self.peak_sram_usage:>15,} floats "
              f"({sram_pct:.0f}% of {self.spec.sram_size:,}){sram_note}")

        # HBM traffic
        print(f"\nHBM Traffic:")
        print(f"  Reads:  {self.total_hbm_reads:>15,} floats "
              f"({self.total_hbm_reads * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")
        print(f"  Writes: {self.total_hbm_writes:>15,} floats "
              f"({self.total_hbm_writes * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")
        print(f"  Total:  {self.total_hbm_reads + self.total_hbm_writes:>15,} floats "
              f"({(self.total_hbm_reads + self.total_hbm_writes) * self.spec.bytes_per_float / 1024 / 1024:.1f} MB)")

        # Time breakdown
        print(f"\nCycles:")
        print(f"  Compute: {self.cycles_compute:>15,.0f} ({self.cycles_compute/total_cycles:>5.1%})")
        print(f"  HBM:     {self.cycles_hbm:>15,.0f} ({self.cycles_hbm/total_cycles:>5.1%})")
        print(f"  Total:   {total_cycles:>15,.0f}")

        # Bottleneck
        print(f"\n→ {'Memory-bound' if self.cycles_hbm > self.cycles_compute else 'Compute-bound'} "
              f"({'HBM' if self.cycles_hbm > self.cycles_compute else 'compute'} is the bottleneck)")

    def get_hbm_traffic_log(self):
        """Return just the HBM read/write operations for visualization."""
        return [(op, name, size) for op, name, size in self.log
                if op in ("hbm_read", "hbm_write")]


class Tensor:
    """
    A simulated tensor for the GPU memory simulator.

    This class wraps profiler calls to provide PyTorch-like syntax.
    There is no actual data - just shape tracking and profiler calls.

    Example:
        Tensor.verbose = True  # Enable operation logging
        Q = Tensor((N, d), name="Q", profiler=prof)
        K = Tensor((N, d), name="K", profiler=prof)
        S = Q @ K.T           # Automatic HBM load/store
        P = S.softmax()       # Automatic HBM load/store
        S.free()              # Explicit memory management
    """

    # Class-level flag to enable verbose output
    verbose: bool = False

    def __init__(
        self,
        shape: tuple,
        name: str,
        profiler: "Profiler",
        *,
        _is_view: bool = False,
        _parent: "Tensor | None" = None,
    ):
        """
        Create a tensor and allocate it in HBM.

        Args:
            shape: Tuple of dimensions, e.g., (N, d) for a matrix
            name: Human-readable name for profiler logs
            profiler: The Profiler instance to track operations
            _is_view: Internal - True if this is a view (like .T)
            _parent: Internal - Parent tensor for views
        """
        if len(shape) != 2:
            raise ValueError("Only 2D tensors are supported")

        self.shape = shape
        self.name = name
        self._profiler = profiler
        self._is_view = _is_view
        self._parent = _parent
        self._freed = False

        # Allocate in HBM (unless this is a view)
        if not _is_view:
            self._profiler.allocate_hbm(self.size, self.name)
            if Tensor.verbose:
                self._print_status(f"Allocate {self.name} {self.shape}")

    @property
    def size(self) -> int:
        """Total number of floats in the tensor."""
        return self.shape[0] * self.shape[1]

    @property
    def T(self) -> "Tensor":
        """
        Transpose view - swaps dimensions, no memory cost.

        Returns a new Tensor that references the same underlying allocation
        but with swapped dimensions for matmul shape checking.
        """
        base = self._parent if self._is_view else self
        return Tensor(
            shape=(self.shape[1], self.shape[0]),
            name=f"{base.name}.T",
            profiler=self._profiler,
            _is_view=True,
            _parent=base,
        )

    def _base_tensor(self) -> "Tensor":
        """Get the underlying tensor (self or parent if view)."""
        return self._parent if self._is_view else self

    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        """Format bytes as KB or MB depending on size."""
        if num_bytes >= 1024 * 1024:  # >= 1 MB
            return f"{num_bytes / 1024 / 1024:>6.1f} MB"
        else:
            return f"{num_bytes / 1024:>6.1f} KB"

    def _print_status(self, message: str, indent: int = 0) -> None:
        """Print a message with current HBM/SRAM usage."""
        prof = self._profiler
        hbm_bytes = prof.current_hbm_usage * prof.spec.bytes_per_float
        sram_bytes = prof.current_sram_usage * prof.spec.bytes_per_float
        sram_pct = prof.current_sram_usage / prof.spec.sram_size * 100

        prefix = "    " * indent
        overflow = " ⚠️  OVERFLOW!" if sram_pct > 100 else ""
        hbm_str = self._format_bytes(hbm_bytes)
        sram_str = self._format_bytes(sram_bytes)
        print(f"{prefix}{message:<45} | HBM: {hbm_str} | SRAM: {sram_str} ({sram_pct:>5.1f}%){overflow}")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication: C = self @ other

        Handles the full memory lifecycle with SRAM-aware 2D tiling.
        If operands don't fit in SRAM, tiles both matrices and pays
        extra HBM traffic for reloading data.
        """
        # Shape validation: (M, K) @ (K, N) -> (M, N)
        M, K = self.shape
        K2, N = other.shape
        if K != K2:
            raise ValueError(
                f"Shape mismatch for matmul: {self.shape} @ {other.shape}"
            )

        # Get base tensors for memory operations
        self_base = self._base_tensor()
        other_base = other._base_tensor()
        prof = self._profiler
        sram_cap = prof.spec.sram_size

        if Tensor.verbose:
            print(f"\n>>> {self.name} @ {other.name}")

        # Create result tensor (allocates in HBM)
        result_name = f"{self.name}@{other.name}"
        result = Tensor((M, N), name=result_name, profiler=prof)

        # Check if we need to tile: do both operands fit in SRAM?
        working_set = self_base.size + other_base.size

        if working_set <= sram_cap:
            # Single pass - everything fits
            prof.load_from_hbm(self_base.size, self_base.name)
            prof.sram_push(self_base.size, self_base.name)
            if Tensor.verbose:
                self._print_status(f"[HBM → SRAM] Load {self_base.name}", indent=1)

            prof.load_from_hbm(other_base.size, other_base.name)
            prof.sram_push(other_base.size, other_base.name)
            if Tensor.verbose:
                self._print_status(f"[HBM → SRAM] Load {other_base.name}", indent=1)

            prof.matmul(M, N, K, f"{result.name} = {self.name} @ {other.name}")
            if Tensor.verbose:
                self._print_status(f"[Compute] matmul → {result.shape}", indent=1)

            prof.store_to_hbm(result.size, result.name)
            if Tensor.verbose:
                self._print_status(f"[SRAM → HBM] Store {result.name}", indent=1)

            prof.sram_pop(other_base.size, other_base.name)
            prof.sram_pop(self_base.size, self_base.name)
            if Tensor.verbose:
                self._print_status(f"[SRAM] Clear working set", indent=1)
        else:
            # Need 2D tiling!
            # For A (M, K) @ B (K, N) -> C (M, N)
            # Tile A by rows (M_tile), tile B by columns (N_tile)
            # Working set per tile: M_tile * K + K * N_tile = K * (M_tile + N_tile)
            # Constraint: K * (M_tile + N_tile) <= SRAM
            # So: M_tile + N_tile <= SRAM / K

            max_sum = sram_cap // K
            # Split evenly between M and N tiles, but cap at actual dimensions
            M_tile = min(M, max(1, max_sum // 2))
            N_tile = min(N, max(1, max_sum - M_tile))

            # Recalculate in case N was smaller
            M_tile = min(M, max(1, max_sum - N_tile))

            num_M_chunks = (M + M_tile - 1) // M_tile
            num_N_chunks = (N + N_tile - 1) // N_tile
            total_chunks = num_M_chunks * num_N_chunks

            if Tensor.verbose:
                self._print_status(
                    f"[2D tiling] {num_M_chunks}×{num_N_chunks} = {total_chunks} tiles "
                    f"({M_tile}×{N_tile} each)", indent=1)

            max_verbose_chunks = 3
            chunk_count = 0
            total_a_loads = 0
            total_b_loads = 0

            for i in range(num_M_chunks):
                m_start = i * M_tile
                m_end = min(m_start + M_tile, M)
                m_size = m_end - m_start
                a_tile_size = m_size * K

                for j in range(num_N_chunks):
                    n_start = j * N_tile
                    n_end = min(n_start + N_tile, N)
                    n_size = n_end - n_start
                    b_tile_size = K * n_size
                    c_tile_size = m_size * n_size

                    # Load tiles
                    prof.load_from_hbm(a_tile_size, f"{self_base.name}[{m_start}:{m_end},:]")
                    prof.sram_push(a_tile_size, f"{self_base.name}[{m_start}:{m_end},:]")
                    total_a_loads += a_tile_size

                    prof.load_from_hbm(b_tile_size, f"{other_base.name}[:,{n_start}:{n_end}]")
                    prof.sram_push(b_tile_size, f"{other_base.name}[:,{n_start}:{n_end}]")
                    total_b_loads += b_tile_size

                    show_tile = (chunk_count < max_verbose_chunks or
                                 chunk_count == total_chunks - 1 or
                                 total_chunks <= max_verbose_chunks + 1)

                    if Tensor.verbose and show_tile:
                        self._print_status(
                            f"[Tile {chunk_count+1}/{total_chunks}] "
                            f"A[{m_start}:{m_end},:] @ B[:,{n_start}:{n_end}]", indent=1)
                    elif Tensor.verbose and chunk_count == max_verbose_chunks:
                        print(f"    ... ({total_chunks - max_verbose_chunks - 1} more tiles) ...")

                    # Compute tile
                    prof.matmul(m_size, n_size, K,
                               f"{result.name}[{m_start}:{m_end},{n_start}:{n_end}]")

                    # Store result tile
                    prof.store_to_hbm(c_tile_size,
                                     f"{result.name}[{m_start}:{m_end},{n_start}:{n_end}]")

                    # Clear SRAM
                    prof.sram_pop(b_tile_size, f"{other_base.name}[:,{n_start}:{n_end}]")
                    prof.sram_pop(a_tile_size, f"{self_base.name}[{m_start}:{m_end},:]")

                    chunk_count += 1

            if Tensor.verbose:
                # Show the overhead compared to loading each matrix once
                naive_loads = self_base.size + other_base.size
                actual_loads = total_a_loads + total_b_loads
                overhead = actual_loads - naive_loads
                overhead_bytes = overhead * prof.spec.bytes_per_float
                if overhead > 0:
                    self._print_status(
                        f"[Done] {total_chunks} tiles → +{self._format_bytes(overhead_bytes)} extra traffic",
                        indent=1)
                else:
                    self._print_status(f"[Done] {total_chunks} tiles, no extra traffic", indent=1)

        return result

    def softmax(self, name: str | None = None) -> "Tensor":
        """
        Apply softmax row-wise.

        Handles the full memory lifecycle with SRAM-aware chunking.
        Softmax is computed row by row (each row is independent).
        If the matrix doesn't fit in SRAM, processes in row chunks.
        """
        if Tensor.verbose:
            print(f"\n>>> softmax({self.name})")

        result_name = name or f"softmax({self.name})"
        result = Tensor(self.shape, name=result_name, profiler=self._profiler)

        rows, cols = self.shape
        prof = self._profiler
        sram_cap = prof.spec.sram_size

        # For softmax, we need input rows + output rows in SRAM
        # Minimum: 2 * cols (one input row, one output row)
        floats_per_row = cols
        rows_per_chunk = max(1, sram_cap // (2 * floats_per_row))
        num_chunks = (rows + rows_per_chunk - 1) // rows_per_chunk  # ceil division

        if num_chunks == 1:
            # Single pass - everything fits
            prof.load_from_hbm(self.size, self.name)
            prof.sram_push(self.size, self.name)
            if Tensor.verbose:
                self._print_status(f"[HBM → SRAM] Load {self.name}", indent=1)

            prof.elementwise(self.size, ops_per_element=5, name=result_name)
            if Tensor.verbose:
                self._print_status(f"[Compute] softmax", indent=1)

            prof.store_to_hbm(result.size, result.name)
            if Tensor.verbose:
                self._print_status(f"[SRAM → HBM] Store {result.name}", indent=1)

            prof.sram_pop(self.size, self.name)
            if Tensor.verbose:
                self._print_status(f"[SRAM] Clear working set", indent=1)
        else:
            # Need to chunk by rows
            if Tensor.verbose:
                self._print_status(f"[SRAM overflow] Need {num_chunks} chunks ({rows_per_chunk} rows each)", indent=1)

            max_verbose_chunks = 3

            for chunk_idx in range(num_chunks):
                start_row = chunk_idx * rows_per_chunk
                end_row = min(start_row + rows_per_chunk, rows)
                chunk_rows = end_row - start_row
                chunk_size = chunk_rows * cols

                # Load chunk of input
                prof.load_from_hbm(chunk_size, f"{self.name}[{start_row}:{end_row}]")
                prof.sram_push(chunk_size, f"{self.name}[{start_row}:{end_row}]")

                show_chunk = (chunk_idx < max_verbose_chunks or
                              chunk_idx == num_chunks - 1 or
                              num_chunks <= max_verbose_chunks + 1)

                if Tensor.verbose and show_chunk:
                    self._print_status(f"[Chunk {chunk_idx+1}/{num_chunks}] Load rows {start_row}:{end_row}", indent=1)
                elif Tensor.verbose and chunk_idx == max_verbose_chunks:
                    print(f"    ... ({num_chunks - max_verbose_chunks - 1} more chunks) ...")

                # Compute softmax on this chunk
                prof.elementwise(chunk_size, ops_per_element=5, name=f"{result_name}[{start_row}:{end_row}]")

                # Store chunk of output
                prof.store_to_hbm(chunk_size, f"{result.name}[{start_row}:{end_row}]")

                if Tensor.verbose and show_chunk:
                    self._print_status(f"[Chunk {chunk_idx+1}/{num_chunks}] Compute & store rows {start_row}:{end_row}", indent=1)

                # Clear SRAM for next chunk
                prof.sram_pop(chunk_size, f"{self.name}[{start_row}:{end_row}]")

            if Tensor.verbose:
                self._print_status(f"[Done] Processed {rows} rows in {num_chunks} chunks", indent=1)

        return result

    def free(self) -> None:
        """
        Free this tensor's HBM allocation.

        Views cannot be freed - they reference the parent tensor's memory.
        """
        if self._freed:
            return
        if self._is_view:
            return  # Views don't own memory
        self._profiler.free_hbm(self.size, self.name)
        self._freed = True
        if Tensor.verbose:
            print()
            self._print_status(f"Free {self.name}")

    def __repr__(self) -> str:
        return f"Tensor({self.shape}, name='{self.name}')"
