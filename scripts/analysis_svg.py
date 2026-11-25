"""Dependency-free SVG primitives for the offline analysis figures."""

from __future__ import annotations

import html


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    anchor: str = "start",
    weight: int = 400,
    color: str = "#0f172a",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"{html.escape(value)}</text>"
    )


def _polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    values = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def _legend(x: float, y: float, items: tuple[tuple[str, str], ...]) -> str:
    parts = []
    cursor = x
    for label, color in items:
        parts.append(
            f'<line x1="{cursor}" y1="{y}" x2="{cursor + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(_text(cursor + 36, y + 5, label, 13, color="#334155"))
        cursor += 180
    return "".join(parts)
