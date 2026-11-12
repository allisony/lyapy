"""Regenerate the reviewed legacy GJ 411 reference artifact.

Do not run this script merely to make a failing regression test pass. Changes
to the reference artifact should accompany an explicit scientific review.
"""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests._legacy_reference import evaluate_gj411_legacy_case


OUTPUT_FILE = (
    REPOSITORY_ROOT
    / "tests"
    / "data"
    / "reference"
    / "gj411_legacy_bfb27ca.npz"
)


def main():
    reference = evaluate_gj411_legacy_case()
    reference["source_commit"] = np.array(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
    )
    reference["generated_utc"] = np.array(
        datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, **reference)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
