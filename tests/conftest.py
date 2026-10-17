"""Shared test setup.

The tests use small models, so the CPU runs them fast. The CPU also gives the
same results on each machine.
"""

import os
import pathlib

# JAX reads this value at import time, so set it before the first import.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# Most of the test time goes to the compilation of small programs. A disk cache
# keeps the compiled programs, so the next test run is much faster.
cache = pathlib.Path(__file__).parent.parent / ".pytest_cache" / "jax"
jax.config.update("jax_compilation_cache_dir", str(cache))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
