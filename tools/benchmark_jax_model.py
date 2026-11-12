"""Compare warmed JAX and NumPy forward-model evaluation times."""

from pathlib import Path
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.models import evaluate_voigt_lya_model
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


def _average_seconds(function, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        result = function()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
    return (time.perf_counter() - start) / iterations


def main(iterations=200):
    reference = evaluate_gj411_legacy_case()
    wave = np.asarray(reference["wave"], dtype=np.float64)
    kernel = np.asarray(reference["lsf_kernel"], dtype=np.float64)
    parameters = np.asarray(PARAMETERS, dtype=np.float64)

    jax_wave = jnp.asarray(wave)
    jax_kernel = jnp.asarray(kernel)
    jax_parameters = jnp.asarray(parameters)
    compiled_model = jax.jit(evaluate_voigt_lya_model_jax)

    compile_start = time.perf_counter()
    compiled_model(
        jax_wave, jax_parameters, jax_kernel
    ).convolved.observed.block_until_ready()
    compile_seconds = time.perf_counter() - compile_start

    numpy_seconds = _average_seconds(
        lambda: evaluate_voigt_lya_model(wave, parameters, kernel).convolved.observed,
        iterations,
    )
    jax_seconds = _average_seconds(
        lambda: compiled_model(
            jax_wave, jax_parameters, jax_kernel
        ).convolved.observed,
        iterations,
    )

    print(f"JAX compile + first evaluation: {compile_seconds:.4f} s")
    print(f"NumPy average evaluation:       {numpy_seconds:.8f} s")
    print(f"JAX average evaluation:         {jax_seconds:.8f} s")
    print(f"Warmed JAX speedup:             {numpy_seconds / jax_seconds:.2f}x")


if __name__ == "__main__":
    main()
