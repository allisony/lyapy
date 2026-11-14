import subprocess
import sys
import textwrap


def _run_isolated_import(blocked, statement):
    code = textwrap.dedent(
        f"""
        import sys

        class BlockedImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.', 1)[0] in {set(blocked)!r}:
                    raise ModuleNotFoundError(
                        f"blocked optional dependency: {{fullname}}"
                    )
                return None

        sys.meta_path.insert(0, BlockedImports())
        {statement}
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )


def test_base_package_import_does_not_require_jax_or_inference_backends():
    completed = _run_isolated_import(
        {"jax", "numpyro", "dynesty", "arviz"},
        "import lyapy; assert callable(lyapy.load_stis_lsf)",
    )

    assert completed.returncode == 0, completed.stderr


def test_inference_import_does_not_require_sampler_or_comparison_backends():
    completed = _run_isolated_import(
        {"numpyro", "dynesty", "arviz"},
        (
            "import lyapy.inference as inference; "
            "assert callable(inference.run_inference)"
        ),
    )

    assert completed.returncode == 0, completed.stderr
