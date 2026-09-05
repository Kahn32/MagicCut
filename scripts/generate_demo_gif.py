#!/usr/bin/env python3
"""Render a deterministic interaction GIF from the completed demo backend."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from magiccut.demo import MagicCutDemo
from magiccut.io import read_json, sha256_file, write_json
from generate_figures import COLORS, medium_renders


ROOT = Path(__file__).resolve().parents[1]


def choose_case(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["candidate_count"] <= 80 and len(row["active_history"]) >= 2]
    if not eligible:
        eligible = [row for row in rows if len(row["active_history"]) >= 2]
    return max(
        eligible,
        key=lambda row: (
            row["active_history"][-1]["f1"] - row["methods"]["magiccut"]["f1"],
            -row["candidate_count"],
            row["uid"],
            row["query"],
        ),
    )


def shown_parts(row: dict, states: list[dict]) -> list[int]:
    distance = {item["part_id"]: item["distance"] for item in row["candidate_records"]}
    priority = {row["query"]} | set(row["target_ids"])
    for state in states:
        priority.update(state["selected"])
        if state["recommended_part"] is not None:
            priority.add(state["recommended_part"])
    ordered = sorted(priority, key=lambda part: (distance.get(part, float("inf")), part))[:20]
    for part in sorted(distance, key=lambda part: (distance[part], part)):
        if len(ordered) >= 20:
            break
        if part not in ordered:
            ordered.append(part)
    return ordered


def frame(uid: str, parts: list[int], images: dict[int, Image.Image], state: dict, index: int) -> Image.Image:
    width, height = 1050, 780
    canvas = Image.new("RGB", (width, height), "#F7F5EF")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((40, 28), "MagicCut", fill=COLORS["ink"], font=font)
    draw.text((40, 52), f"mesh {uid[:8]}  |  click {state['click_count']}  |  selected {len(state['selected'])}  |  uncertain {len(state['uncertain'])}", fill="#66727D", font=font)
    if state["recommended_part"] is not None:
        draw.text((40, 74), f"recommended clarification: part {state['recommended_part']}", fill="#246BCE", font=font)
    cell, image_size, left, top = 190, 148, 42, 112
    selected, uncertain = set(state["selected"]), set(state["uncertain"])
    for position, part in enumerate(parts):
        row, column = divmod(position, 5)
        x, y = left + column * cell, top + row * 160
        if part == state["query_part"]:
            color = "#D94B4B"
        elif part in uncertain:
            color = "#F2CF55"
        elif part in selected:
            color = "#EE8B2D"
        else:
            color = "#D9DEE3"
        border = 7 if part == state["recommended_part"] else 3
        outline = "#246BCE" if part == state["recommended_part"] else color
        draw.rounded_rectangle((x, y, x + 164, y + 150), radius=10, fill=color, outline=outline, width=border)
        image = images.get(part)
        if image is not None:
            copy = image.copy()
            copy.thumbnail((image_size, 116))
            canvas.paste(copy, (x + (164 - copy.width) // 2, y + 8))
        draw.text((x + 8, y + 132), f"part {part}", fill=COLORS["ink"], font=font)
    draw.text((40, 754), "red query  |  orange selected  |  yellow uncertain  |  blue outline next question", fill="#66727D", font=font)
    return canvas


def main() -> None:
    raw = read_json(ROOT / "outputs/locked_evaluation/raw_results.json")
    report35 = read_json(ROOT / "outputs/locked_evaluation/report.json")
    if sha256_file(ROOT / "outputs/locked_evaluation/raw_results.json") != report35["raw_results_sha256"]:
        raise ValueError("locked evaluation digest mismatch")
    row = choose_case(raw["rows"])
    engine = MagicCutDemo(ROOT)
    states = [engine.start(row["uid"], row["query"])]
    target = set(row["target_ids"])
    for _ in range(3):
        recommended = states[-1]["recommended_part"]
        if recommended is None:
            break
        states.append(engine.feedback(states[-1]["session_id"], recommended, recommended in target))
    parts = shown_parts(row, states)
    manifest33 = read_json(ROOT / "outputs/benchmark_preparation/manifest.json")
    images = medium_renders(row["uid"], set(parts), manifest33)
    frames = [frame(row["uid"], parts, images, state, index) for index, state in enumerate(states)]
    output = ROOT / "outputs/presentation"
    output.mkdir(parents=True, exist_ok=True)
    gif = output / "demo.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1300, loop=0, optimize=True)
    report = {
        "source_raw_results_sha256": report35["raw_results_sha256"],
        "uid": row["uid"],
        "query": row["query"],
        "frame_count": len(frames),
        "oracle_answers": [state["recommended_part"] in target for state in states[:-1] if state["recommended_part"] is not None],
        "demo_gif": str(gif.relative_to(ROOT)),
        "demo_gif_sha256": sha256_file(gif),
    }
    write_json(output / "demo_manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
