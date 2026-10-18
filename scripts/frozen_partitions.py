"""One registry of the frozen test partitions, shared by every script that needs to know.

Why this exists: the decision and evaluation records of this project carried the pair
``"validation_only": true`` and ``"test_sets_touched": false`` as literals written into the
output.  They were written that way by the code that produced the records, not derived from
what the code had read, so a run on a frozen test would have filed itself as a validation
run, and no downstream check could have told the difference.  The frozen-test records on disk
from before this module exists do carry that pair, which is why the paper's frozen-test
discipline is stated in Sections 3.1 and 3.5 from the directories and the one-shot runners
rather than from those two fields.

This module makes the claim computable: give it the directories a run read, and it says
which frozen partition, if any, they fall inside.  Any script that adds a frozen partition
adds it here once.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT / "strawberry_sci"
DATA = PROJECT_ROOT / "data"

# Every partition that this project treats as frozen against tuning, keyed by the name used
# in the paper.  ``Yurdakul full released set`` ships without a split, so the whole archive
# is the frozen view.
FROZEN_PARTITIONS: dict[str, Path] = {
    "merchant test": DATA / "merchant_clean" / "test",
    "external test": DATA / "external_protocol_v2_frozen" / "test",
    "SDS test": WORKSPACE_ROOT / "strawberry-DS" / "test",
    "Yurdakul full released set": DATA / "yurdakul_turkey_frozen" / "all",
}


def frozen_partition_evaluated(*paths) -> str | None:
    """Name the frozen partition an input directory falls inside; None for development data.

    Paths that do not exist are still classified: the check is about where a run was pointed,
    and a missing directory should not read as "no test set touched".
    """
    for name, directory in FROZEN_PARTITIONS.items():
        target = Path(directory).resolve()
        for path in paths:
            if not path:
                continue
            try:
                Path(path).resolve().relative_to(target)
            except ValueError:
                continue
            return name
    return None


def partitions_read(paths) -> list[str]:
    """Every frozen partition a collection of input directories touches, sorted by name.

    Pass the pairs a run actually evaluated; a decision script that reads only validation
    partitions gets an empty list, and ``test_sets_touched`` follows from that.
    """
    found = set()
    for path in paths:
        if isinstance(path, (tuple, list)):
            name = frozen_partition_evaluated(*path)
        else:
            name = frozen_partition_evaluated(path)
        if name:
            found.add(name)
    return sorted(found)
