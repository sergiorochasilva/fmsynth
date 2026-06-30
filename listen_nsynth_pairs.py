"""Play NSynth original/resynthesized pairs in sequence for auditory checks.

This helper reads a simple pairs file and reproduces each original file
followed by its resynthesized counterpart. It is meant for quick manual
validation of the transfer pipeline.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_PAIRS_FILE = Path("listen_big18_examples.txt")
ORIGINAL_RE = re.compile(r"original:\s*(.+)$", re.IGNORECASE)
RESYNTH_RE = re.compile(r"resynth:\s*(.+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play NSynth original/resynth pairs in sequence.")
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=DEFAULT_PAIRS_FILE,
        help="Text file with `original:` and `resynth:` entries.",
    )
    parser.add_argument(
        "--player",
        choices=["auto", "ffplay", "aplay"],
        default="auto",
        help="Audio player to use. `auto` prefers ffplay and falls back to aplay.",
    )
    return parser.parse_args()


def load_pairs(path: Path) -> list[tuple[Path, Path]]:
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    pairs: list[tuple[Path, Path]] = []
    current_original: Path | None = None
    current_resynth: Path | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        original_match = ORIGINAL_RE.search(line)
        if original_match:
            current_original = Path(original_match.group(1).strip())
            continue

        resynth_match = RESYNTH_RE.search(line)
        if resynth_match:
            current_resynth = Path(resynth_match.group(1).strip())
            if current_original is None:
                raise ValueError(f"Found resynth path before original in {path}: {raw_line}")
            pairs.append((current_original, current_resynth))
            current_original = None
            current_resynth = None

    if current_original is not None or current_resynth is not None:
        raise ValueError(f"Unfinished pair at end of {path}")

    if not pairs:
        raise ValueError(f"No pairs found in {path}")

    return pairs


def resolve_player(choice: str) -> tuple[str, list[str]]:
    if choice == "auto":
        if shutil.which("ffplay"):
            choice = "ffplay"
        elif shutil.which("aplay"):
            choice = "aplay"
        else:
            raise FileNotFoundError("No supported player found. Install ffplay or aplay.")

    if choice == "ffplay":
        return choice, ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    if choice == "aplay":
        return choice, ["aplay"]
    raise ValueError(f"Unsupported player: {choice}")


def play(path: Path, command: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    subprocess.run(command + [str(path)], check=True)


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.pairs_file)
    player_name, base_command = resolve_player(args.player)

    print(f"Using player: {player_name}")
    print(f"Pairs file: {args.pairs_file}")
    for idx, (original, resynth) in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] original: {original}")
        play(original, base_command)
        print(f"[{idx}/{len(pairs)}] resynth:  {resynth}")
        play(resynth, base_command)


if __name__ == "__main__":
    main()
