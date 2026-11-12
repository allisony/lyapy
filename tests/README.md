# lyapy tests

The test suite has two complementary roles:

- Unit tests check mathematical and physical invariants such as normalized
  convolution kernels, nonnegative optical depth, and analytic Gaussian flux.
- Regression tests compare the GJ 411 legacy model against a reviewed reference
  artifact in `data/reference/`.

Install the test runner with `python -m pip install -r requirements-test.txt`,
then run the suite from the repository root with:

```bash
python -m pytest
```

Large chains, plots, and temporary fit products do not belong in this
directory. Tests that create files should use pytest's `tmp_path` fixture.
