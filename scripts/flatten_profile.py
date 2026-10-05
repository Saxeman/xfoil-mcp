"""Resolve a Bambu Studio preset into one complete settings file.

Bambu Studio's built-in presets are layered: each file lists only what
differs from the preset it "inherits" from, and some also pull in
"include" templates (start G-code, filament templates). The GUI resolves
the whole chain. The command line does not: it reads only the top file and
fills every other setting with generic defaults, which is how a slice for
an X1 Carbon came out at 60 mm/s with "PLA" as the material.

This script does the resolving, so the slicer gets every setting explicitly:

    python scripts/flatten_profile.py "$P/machine"  "Bambu Lab X1 Carbon 0.4 nozzle" slicer/machine.json
    python scripts/flatten_profile.py "$P/process"  "0.20mm Standard @BBL X1C" slicer/process.json \
        --set "curr_bed_type=Textured PEI Plate"
    python scripts/flatten_profile.py "$P/filament" "Bambu ABS @BBL X1C" slicer/filament.json

where P is Bambu Studio's profiles/BBL folder. --set overrides one setting
after resolving; repeat it for more. The output files are meant to be
committed, so the exact settings behind every print are in version control.
"""

import argparse
import json
import sys
from pathlib import Path


def index(folder: Path) -> dict[str, Path]:
    """Every preset in a folder, by the name other presets use to refer to it."""
    names: dict[str, Path] = {}
    for path in sorted(folder.rglob("*.json")):
        try:
            name = json.loads(path.read_text()).get("name")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        names[name or path.stem] = path
        names.setdefault(path.stem, path)
    return names


def resolve(name: str, names: dict[str, Path], chain: tuple[str, ...] = ()) -> dict:
    """The preset's settings with its whole inheritance chain applied.

    Order: the parent's resolved settings, then each included template, then
    the preset's own settings, so the most specific value always wins.
    """
    if name in chain:
        sys.exit(f"inheritance loop: {' -> '.join(chain + (name,))}")
    if name not in names:
        sys.exit(f"no preset named {name!r} (needed by {chain[-1] if chain else 'the command line'})")
    own = json.loads(names[name].read_text())
    chain = chain + (name,)
    merged = resolve(own["inherits"], names, chain) if own.get("inherits") else {}
    for template in own.get("include", []):
        merged.update(resolve(template, names, chain))
    merged.update(own)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="profiles/BBL/machine, process or filament")
    parser.add_argument("name", help="the preset's name, as shown in Bambu Studio")
    parser.add_argument("out", type=Path, help="where to write the complete settings file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one setting after resolving (repeatable)")
    args = parser.parse_args()

    settings = resolve(args.name, index(args.folder))
    # Resolved: nothing left for the slicer to look up.
    settings.pop("inherits", None)
    settings.pop("include", None)

    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--set needs KEY=VALUE, got {item!r}")
        if key not in settings:
            # Some settings (the plate type) live outside every preset chain.
            # Allowed, but said out loud, so a misspelled key is visible.
            print(f"note: {key} is not in {args.name!r}'s chain; adding it")
            settings[key] = value
            continue
        # Per-filament and per-extruder settings are lists; keep the shape.
        settings[key] = [value] * len(settings[key]) if isinstance(settings[key], list) else value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(settings, indent=4) + "\n")
    print(f"{args.name}: {len(settings)} settings -> {args.out}")


if __name__ == "__main__":
    main()
