from pathlib import Path

import numpy as np
import pytest


REFERENCE_FILE = (
    Path(__file__).parent / "data" / "reference" / "gj411_legacy_bfb27ca.npz"
)


@pytest.fixture(scope="session")
def legacy_reference():
    """Load the reviewed legacy GJ 411 reference arrays."""
    with np.load(REFERENCE_FILE, allow_pickle=False) as reference:
        return {name: reference[name] for name in reference.files}
