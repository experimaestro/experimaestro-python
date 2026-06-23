#!/usr/bin/env python3
"""Generate the web-app icon from the master logo SVG.

The master logo lives in the documentation (``docs/source/img/icon.svg``) and
carries the full artwork: the network graph, the conductor's hand and the
``Experimaestro`` word-mark. The web app shows it on a **dark** navbar and does
not want the word-mark, so this script derives a compact dark-background variant
with three generic, flag-driven steps:

1. **Drop** the Inkscape-labelled group(s) you don't want in the icon
   (default: the layer/group named ``label`` — the word-mark).
2. **Recolour for a dark background**: every painted ``fill``/``stroke`` keeps
   its *hue* but its lightness is lifted (dark inks become light) so it reads on
   black. Saturation is preserved/boosted to keep the colours vivid rather than
   washed out. ``none`` and ``<defs>``/``<mask>`` internals are left untouched
   so masks keep working.
3. **Crop** the canvas to the remaining drawing and emit a clean plain SVG
   (unused ``<defs>`` are vacuumed away).

Nothing about the artwork is hard-coded: groups are matched by their
``inkscape:label`` and colours are remapped by hue, so the script keeps working
if the logo is re-drawn.

The crop/clean step shells out to Inkscape (must be on ``PATH``).

Usage::

    uv run python scripts/make-app-icon.py
    uv run python scripts/make-app-icon.py --saturation 1.2 --min-lightness 0.7
    uv run python scripts/make-app-icon.py --drop label --drop maestro
    uv run python scripts/make-app-icon.py src.svg out.svg --no-crop --show

With ``--adaptive`` it instead emits a single self-contained SVG (default
``docs/source/img/icon-adaptive.svg``) that keeps the full artwork and its
original light colours, plus a ``prefers-color-scheme: dark`` ``<style>`` block
that lightens every ink for a dark background (used by the README)::

    uv run python scripts/make-app-icon.py --adaptive

Re-run it whenever the master ``icon.svg`` changes.
"""

from __future__ import annotations

import argparse
import colorsys
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Default source (documentation master) / targets.
DEFAULT_SRC = Path("docs/source/img/icon.svg")
DEFAULT_DST = Path("app/public/icon.svg")
# Adaptive variant: full artwork that recolours itself for a dark background via
# a ``prefers-color-scheme`` media query (used by the README, which is shown on
# both light and dark GitHub themes).
DEFAULT_ADAPTIVE_DST = Path("docs/source/img/icon-adaptive.svg")

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
_NAMESPACES = {
    "": SVG_NS,
    "svg": SVG_NS,
    "inkscape": INK_NS,
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "xlink": "http://www.w3.org/1999/xlink",
}

# Properties that paint a shape and should therefore be recoloured.
_PAINT = ("fill", "stroke")

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RGB_PCT_RE = re.compile(
    r"^rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)$", re.IGNORECASE
)


def _parse_rgb(value: str) -> tuple[float, float, float] | None:
    """Return ``(r, g, b)`` in 0..1 from a hex or ``rgb(%)`` colour, else ``None``."""
    value = value.strip()
    if m := _HEX_RE.match(value):
        body = m.group(1)
        if len(body) == 3:
            body = "".join(c * 2 for c in body)
        return tuple(int(body[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    if m := _RGB_PCT_RE.match(value):
        return tuple(float(m.group(i)) / 100 for i in (1, 2, 3))  # type: ignore[return-value]
    return None


class DarkRecolour:
    """Hue-preserving recolour that makes dark inks read on a dark background."""

    def __init__(self, *, saturation: float, min_lightness: float) -> None:
        self.saturation = saturation
        self.min_lightness = min_lightness
        self.mapping: dict[str, str] = {}

    def __call__(self, value: str | None) -> str | None:
        if value is None or value.strip().lower() == "none":
            return value
        rgb = _parse_rgb(value)
        if rgb is None:  # url(#…), currentColor, named colours: leave as-is
            return value

        r, g, b = rgb
        h, lightness, s = colorsys.rgb_to_hls(r, g, b)
        # Lift lightness (dark -> light) so the ink shows on black; keep colours
        # vivid by scaling saturation up rather than down.
        new_l = max(self.min_lightness, 1.0 - lightness)
        new_s = min(1.0, s * self.saturation)
        nr, ng, nb = colorsys.hls_to_rgb(h, new_l, new_s)
        out = "#" + "".join(f"{round(c * 255):02x}" for c in (nr, ng, nb))
        self.mapping[value.strip().lower()] = out
        return out

    def style(self, style: str) -> str:
        """Rewrite ``fill``/``stroke`` declarations inside a ``style`` attribute."""
        parts = []
        for decl in style.split(";"):
            key, sep, val = decl.partition(":")
            if sep and key.strip() in _PAINT:
                val = self(val)
            parts.append(f"{key}{sep}{val}" if sep else decl)
        return ";".join(parts)


def transform(
    tree: ET.ElementTree, *, drop: set[str], recolour: DarkRecolour | None
) -> None:
    """Apply the drop + recolour steps to ``tree`` in place."""
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    # Elements living inside <defs> are left untouched: mask luminance relies on
    # their black/white fills, and recolouring them would break the artwork.
    protected = {
        node for defs in root.iter(f"{{{SVG_NS}}}defs") for node in defs.iter()
    }

    # 1. Drop unwanted labelled groups (and everything they contain).
    for element in list(root.iter()):
        if element.get(f"{{{INK_NS}}}label") in drop:
            parents[element].remove(element)

    # 2. Recolour every remaining painted colour for a dark background.
    if recolour is None:  # keep the original (light-background) colours
        return
    for element in root.iter():
        if element in protected:
            continue
        for prop in _PAINT:
            if (value := element.get(prop)) is not None:
                element.set(prop, recolour(value))
        if style := element.get("style"):
            element.set("style", recolour.style(style))


def crop_and_clean(src: Path, dst: Path) -> None:
    """Crop the canvas to the drawing and write a vacuumed plain SVG via Inkscape."""
    actions = (
        "page-fit-to-selection;"  # no selection -> fit page to whole drawing
        "export-plain-svg;"
        f"export-filename:{dst};"
        "export-do"
    )
    subprocess.run(
        ["inkscape", str(src), "--vacuum-defs", f"--actions={actions}"],
        check=True,
        capture_output=True,
        text=True,
    )


def collect_paint(
    tree: ET.ElementTree, recolour: DarkRecolour
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(fill_map, stroke_map)`` of original -> dark colour for the drawing.

    Colours living inside ``<defs>`` (mask luminance) are ignored, and colours
    that the recolour leaves unchanged are dropped, so the resulting maps only
    contain the inks that actually need a dark-mode override.
    """
    root = tree.getroot()
    protected = {
        node for defs in root.iter(f"{{{SVG_NS}}}defs") for node in defs.iter()
    }
    fill_map: dict[str, str] = {}
    stroke_map: dict[str, str] = {}

    def record(prop: str, value: str | None) -> None:
        if value is None or value.strip().lower() == "none":
            return
        original = value.strip()
        dark = recolour(original)
        if dark is None or dark.lower() == original.lower():
            return  # non-colour (url(#…), currentColor) or already light enough
        (fill_map if prop == "fill" else stroke_map)[original] = dark

    for element in root.iter():
        if element in protected:
            continue
        for prop in _PAINT:
            record(prop, element.get(prop))
        if style := element.get("style"):
            for decl in style.split(";"):
                key, sep, val = decl.partition(":")
                if sep and key.strip() in _PAINT:
                    record(key.strip(), val)
    return fill_map, stroke_map


def build_dark_css(fill_map: dict[str, str], stroke_map: dict[str, str]) -> str:
    """Build a ``prefers-color-scheme: dark`` stylesheet from the colour maps.

    Each original ink is matched both as a presentation attribute (``fill="…"``)
    and inside an inline ``style`` declaration, and overridden with ``!important``
    so it wins over the inline value when the dark scheme is active.
    """
    lines = ["@media (prefers-color-scheme: dark) {"]
    for prop, mapping in (("fill", fill_map), ("stroke", stroke_map)):
        for original, dark in sorted(mapping.items()):
            lines.append(
                f'    [{prop}="{original}"], [style*="{prop}:{original}"]'
                f" {{ {prop}: {dark} !important; }}"
            )
    lines.append("  }")
    return "\n".join(lines)


def write_adaptive(src: Path, dst: Path, recolour: DarkRecolour, *, crop: bool) -> int:
    """Emit a self-contained SVG that recolours itself on a dark background.

    The artwork keeps its original (light) colours inline; a ``<style>`` element
    holding a ``prefers-color-scheme: dark`` media query flips each ink to its
    lightened counterpart. The colour map is built from the *cropped/cleaned*
    output so the selectors match the exact strings Inkscape serialises.
    """
    if crop:
        try:
            crop_and_clean(src, dst)
        except FileNotFoundError:
            sys.stderr.write("inkscape not found on PATH (needed for --crop)\n")
            return 1
        tree = ET.parse(dst)
    else:
        tree = ET.parse(src)

    fill_map, stroke_map = collect_paint(tree, recolour)
    css = build_dark_css(fill_map, stroke_map)

    root = tree.getroot()
    style_el = ET.Element(f"{{{SVG_NS}}}style")
    style_el.text = "\n    " + css + "\n  "
    root.insert(0, style_el)
    # Serialise SVG elements in the default namespace (<svg>, not <svg:svg>) to
    # match the Inkscape-written icons; _NAMESPACES registers both "" and "svg"
    # for the same URI, so the last one (svg) would otherwise win.
    ET.register_namespace("", SVG_NS)
    tree.write(dst, xml_declaration=True, encoding="UTF-8")

    print(
        f"Wrote {dst} (adaptive, "
        f"{len(fill_map) + len(stroke_map)} colour overrides for dark mode)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "src", nargs="?", type=Path, default=DEFAULT_SRC, help="master SVG"
    )
    parser.add_argument(
        "dst",
        nargs="?",
        type=Path,
        default=None,
        help="output (default: app/public/icon.svg, or the adaptive path with --adaptive)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="emit one self-contained SVG that recolours for dark mode via a "
        "prefers-color-scheme media query (keeps the word-mark by default)",
    )
    parser.add_argument(
        "--drop",
        action="append",
        metavar="LABEL",
        help="inkscape:label of a group to remove (repeatable; "
        "default: 'label', or nothing with --adaptive)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="saturation multiplier (>1 = more vivid; default 1.0)",
    )
    parser.add_argument(
        "--min-lightness",
        type=float,
        default=0.7,
        help="floor on output lightness 0..1 (higher = lighter; default 0.7)",
    )
    parser.add_argument(
        "--no-recolor",
        "--no-recolour",
        dest="recolour",
        action="store_false",
        help="keep the original (light-background) colours instead of lightening",
    )
    parser.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="keep the original canvas instead of cropping to the drawing",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the colour mapping that was applied",
    )
    args = parser.parse_args()

    if not args.src.exists():
        parser.error(f"source not found: {args.src}")

    dst = args.dst or (DEFAULT_ADAPTIVE_DST if args.adaptive else DEFAULT_DST)
    if args.drop is not None:
        drop = set(args.drop)
    else:
        drop = set() if args.adaptive else {"label"}

    for prefix, uri in _NAMESPACES.items():
        ET.register_namespace(prefix, uri)

    if args.adaptive:
        # Keep the original (light) colours inline and add a dark-mode <style>;
        # the recolour object computes the lightened counterparts.
        recolour = DarkRecolour(
            saturation=args.saturation, min_lightness=args.min_lightness
        )
        tree = ET.parse(args.src)
        transform(tree, drop=drop, recolour=None)  # only drop unwanted groups
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as tmp:
            tree.write(tmp.name, xml_declaration=True, encoding="UTF-8")
            tmp_path = Path(tmp.name)
        try:
            return write_adaptive(tmp_path, dst, recolour, crop=args.crop)
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr or "")
            parser.error("inkscape failed while cropping the icon")
        finally:
            tmp_path.unlink(missing_ok=True)

    recolour = (
        DarkRecolour(saturation=args.saturation, min_lightness=args.min_lightness)
        if args.recolour
        else None
    )

    tree = ET.parse(args.src)
    transform(tree, drop=drop, recolour=recolour)

    if args.crop:
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as tmp:
            tree.write(tmp.name, xml_declaration=True, encoding="UTF-8")
            tmp_path = Path(tmp.name)
        try:
            crop_and_clean(tmp_path, dst)
        except FileNotFoundError:
            parser.error("inkscape not found on PATH (needed for --crop)")
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr or "")
            parser.error("inkscape failed while cropping the icon")
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        tree.write(dst, xml_declaration=True, encoding="UTF-8")

    if recolour is None:
        print(f"Wrote {dst} (dropped {sorted(drop)}, original colours kept)")
    else:
        print(
            f"Wrote {dst} (dropped {sorted(drop)}, "
            f"{len(recolour.mapping)} colours remapped for a dark background)"
        )
        if args.show:
            for src_colour, dst_colour in sorted(recolour.mapping.items()):
                print(f"  {src_colour} -> {dst_colour}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
