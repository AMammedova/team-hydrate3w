"""Member 1, W1.2 — fetch 3W from Figshare/GitHub, verify checksums."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/petrobras/3W.git"


def download(target_dir: str) -> None:
    """
    Thin Python wrapper around data/download_data.sh's git-clone approach --
    reimplemented here (rather than shelling out to the .sh file) so it also
    runs on Windows without requiring bash/WSL.

    Skips the clone if `target_dir` already exists, same as the shell script.
    """
    target = Path(target_dir)
    if target.exists():
        logger.info("%s already exists, skipping clone. Delete it to re-download.", target)
        return

    logger.info("Cloning %s into %s (multi-GB, this may take a while)...", REPO_URL, target)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(target)],
        check=True,
    )
