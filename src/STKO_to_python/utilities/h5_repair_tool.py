from __future__ import annotations

from pathlib import Path
from typing import Literal
import shutil
import subprocess
import logging
import h5py
import sys
import argparse

log = logging.getLogger(__name__)

WRITE_OPEN_MARKERS = (
    "file is already open for write",   # common h5py/HDF5 message
    "file is open for write",
    "unable to lock file",              # seen on some HDF5 builds
    "H5Fopen failed",                   # generic open failure (fall back to ERROR)
)

class H5RepairTool:
    """
    Scan .mpco (HDF5) partitions and clear HDF5's "file is already open for write"
    flag using `h5clear -s -i <file>` when needed.

    Notes
    -----
    - Requires the HDF5 command-line tools. If `h5clear` is not on PATH, pass its
      absolute path via `h5clear_cmd=...` or load the proper module on HPC.
    - We *only* call `h5clear` for files detected as FLAGGED.
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = "results.part-*.mpco",
        *,
        h5clear_cmd: str = None, #r"C:\Program Files\HDF_Group\HDF5\1.14.6\bin\h5clear.exe"
    ) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self.files: list[Path] = sorted(self.directory.glob(pattern))
        self.status: dict[Path, str] = {}
        self._h5clear = h5clear_cmd or shutil.which("h5clear")

    # ───────────────────────────────────────────────────────────────────── #
    def scan(self, verbose: bool = False) -> dict[Path, str]:
        """
        Scan all files and store their status.

        Returns
        -------
        dict[Path, str] with statuses: "OK", "FLAGGED", or "ERROR: <msg>"
        """
        self.status.clear()
        for f in self.files:
            try:
                # Fast reject for non-HDF5 files (e.g., partial temp files)
                if not h5py.is_hdf5(str(f)):
                    self.status[f] = "ERROR: not an HDF5 file"
                    if verbose:
                        log.info(f"{f.name:<30} →  {self.status[f]}")
                    continue

                # Attempt read-only open. If the file has an “open for write” flag,
                # some HDF5 builds raise an OSError with a recognizable message.
                with h5py.File(f, "r"):
                    self.status[f] = "OK"

            except OSError as e:
                msg = str(e).lower()
                if any(marker in msg for marker in WRITE_OPEN_MARKERS):
                    self.status[f] = "FLAGGED"
                else:
                    self.status[f] = f"ERROR: {e}"

            if verbose:
                log.info(f"{f.name:<30} →  {self.status[f]}")
        return self.status

    # ───────────────────────────────────────────────────────────────────── #
    def print_report(self) -> None:
        """Print a summary table of file statuses."""
        print("File Status Report:")
        for f, stat in self.status.items():
            print(f"{f.name:<30} →  {stat}")

        print("\nSummary:")
        counts = {"OK": 0, "FLAGGED": 0, "ERROR": 0}
        for s in self.status.values():
            if s == "OK":
                counts["OK"] += 1
            elif s == "FLAGGED":
                counts["FLAGGED"] += 1
            else:
                counts["ERROR"] += 1
        for k, v in counts.items():
            print(f"{k:<8}: {v}")

    # ───────────────────────────────────────────────────────────────────── #
    def fix_flagged(self, *, dry_run: bool = False) -> None:
        """
        Attempt to clear the write-in-progress flag from all flagged files.

        Parameters
        ----------
        dry_run : bool
            If True, only log what would be done.
        """
        if not self._h5clear:
            log.warning(
                "h5clear not found on PATH, and no explicit path provided. "
                "Skipping fix. Load your HDF5 tools module or set h5clear_cmd=..."
            )
            return

        for f, stat in self.status.items():
            if stat != "FLAGGED":
                continue

            log.info(f"Fixing: {f.name}")
            if dry_run:
                log.info("  → dry-run: would run `h5clear -s -i %s`", f)
                continue

            # h5clear exits 0 on success
            result = subprocess.run(
                [self._h5clear, "-s", "-i", str(f)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                log.info(f"  → Cleared flag on {f.name}")
                # Re-scan that single file to confirm
                try:
                    with h5py.File(f, "r"):
                        self.status[f] = "OK"
                except Exception as e:
                    self.status[f] = f"ERROR: {e}"
            else:
                log.error(f"  ✗ Failed to clear {f.name}")
                if result.stderr:
                    log.error(result.stderr.strip())

    # ───────────────────────────────────────────────────────────────────── #
    def run_full_check_and_fix(
        self,
        *,
        verbose: bool = True,
        dry_run: bool = False,
    ) -> None:
        """Run scan, print report, and attempt to fix any flagged files."""
        self.scan(verbose=verbose)
        self.print_report()
        self.fix_flagged(dry_run=dry_run)


# ─────────────────────────────── CLI ──────────────────────────────────── #
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="h5_repair",
        description="Scan and fix HDF5 write-in-progress flags on .mpco partitions.",
    )
    p.add_argument("directory", type=Path, help="Directory with *.mpco partitions")
    p.add_argument(
        "--pattern",
        default="results.part-*.mpco",
        help="Glob pattern for partitions (default: results.part-*.mpco)",
    )
    p.add_argument(
        "--h5clear",
        dest="h5clear_cmd",
        default=None,
        help="Path to h5clear executable if not on PATH",
    )
    p.add_argument("--no-verbose", action="store_true", help="Silence per-file log lines")
    p.add_argument("--dry-run", action="store_true", help="Only report actions, don't modify files")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    tool = H5RepairTool(
        directory=args.directory,
        pattern=args.pattern,
        h5clear_cmd=args.h5clear_cmd,
    )
    tool.run_full_check_and_fix(verbose=not args.no_verbose, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
