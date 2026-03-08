#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path


TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_impression_token(token: str) -> tuple[str, str | None]:
    if "-" not in token:
        return token, None
    news_id, label = token.rsplit("-", 1)
    if not news_id:
        return token, None
    return news_id, label


def parse_time_or_none(time_text: str) -> datetime | None:
    try:
        return datetime.strptime(time_text, TIME_FORMAT)
    except ValueError:
        return None


def load_user_impressions(
    behaviors_path: Path,
) -> dict[str, list[tuple[int, datetime | None, dict[str, str]]]]:
    # user_id -> [(row_index, parsed_time, {news_id: label})]
    user_records: dict[str, list[tuple[int, datetime | None, dict[str, str]]]] = defaultdict(list)

    with behaviors_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue

            user_id = cols[1]
            parsed_time = parse_time_or_none(cols[2])
            impressions = cols[4].split()

            labels_by_news: dict[str, str] = {}
            for token in impressions:
                news_id, label = parse_impression_token(token)
                if label not in {"0", "1"}:
                    continue
                if news_id not in labels_by_news or label == "1":
                    labels_by_news[news_id] = label

            user_records[user_id].append((row_index, parsed_time, labels_by_news))

    for records in user_records.values():
        records.sort(
            key=lambda x: (
                x[1] is None,
                x[1] if x[1] is not None else datetime.min,
                x[0],
            )
        )

    return user_records


def build_user_news_appearances(
    user_records: dict[str, list[tuple[int, datetime | None, dict[str, str]]]],
) -> dict[str, dict[str, tuple[list[int], list[str]]]]:
    # user_id -> news_id -> (appearance_indices, labels)
    out: dict[str, dict[str, tuple[list[int], list[str]]]] = {}
    for user_id, records in user_records.items():
        news_to_idx: dict[str, list[int]] = defaultdict(list)
        news_to_label: dict[str, list[str]] = defaultdict(list)
        for imp_index, (_, _, labels_by_news) in enumerate(records):
            for news_id, label in labels_by_news.items():
                news_to_idx[news_id].append(imp_index)
                news_to_label[news_id].append(label)
        out[user_id] = {
            news_id: (news_to_idx[news_id], news_to_label[news_id])
            for news_id in news_to_idx
        }
    return out


def classify_next_state(
    news_appearances: dict[str, tuple[list[int], list[str]]],
    news_id: str,
    run_end_imp_index: int,
) -> tuple[str, int | None, str | None]:
    if news_id not in news_appearances:
        return "no_future_appearance", None, None

    imp_indices, labels = news_appearances[news_id]
    pos = bisect_right(imp_indices, run_end_imp_index)
    if pos >= len(imp_indices):
        return "no_future_appearance", None, None

    next_imp = imp_indices[pos]
    next_label = labels[pos]
    if next_label == "1":
        return "next_is_1", next_imp, next_label
    if next_label == "0":
        return "next_is_0_after_gap", next_imp, next_label
    return "other", next_imp, next_label


def collect_run2_followup_patterns(
    user_records: dict[str, list[tuple[int, datetime | None, dict[str, str]]]],
    user_news_appearances: dict[str, dict[str, tuple[list[int], list[str]]]],
) -> tuple[dict[str, int], list[tuple[str, str, int, int, str, int | None, str | None]]]:
    counts = {
        "next_is_1": 0,               # 0->0->1
        "no_future_appearance": 0,    # 0->0->(no more appearance)
        "next_is_0_after_gap": 0,     # 0->0->...->0
        "other": 0,
    }
    events: list[tuple[str, str, int, int, str, int | None, str | None]] = []
    # event tuple:
    # (user_id, news_id, run_start_imp_idx, run_end_imp_idx, class_name, next_imp_idx, next_label)

    for user_id, records in user_records.items():
        active_runs: dict[str, tuple[int, int, int]] = {}
        appearances = user_news_appearances.get(user_id, {})

        for imp_index, (_, _, labels_by_news) in enumerate(records):
            current_zero_news = {
                news_id for news_id, label in labels_by_news.items() if label == "0"
            }
            next_active_runs: dict[str, tuple[int, int, int]] = {}

            for news_id in current_zero_news:
                if news_id in active_runs:
                    prev_len, start_imp, _ = active_runs[news_id]
                    next_active_runs[news_id] = (prev_len + 1, start_imp, imp_index)
                else:
                    next_active_runs[news_id] = (1, imp_index, imp_index)

            for news_id, (run_len, start_imp, end_imp) in active_runs.items():
                if news_id in next_active_runs:
                    continue
                if run_len == 2:
                    cls, next_imp, next_label = classify_next_state(
                        appearances,
                        news_id,
                        end_imp,
                    )
                    counts[cls] = counts.get(cls, 0) + 1
                    events.append((user_id, news_id, start_imp, end_imp, cls, next_imp, next_label))

            active_runs = next_active_runs

        # run that reaches user timeline end
        for news_id, (run_len, start_imp, end_imp) in active_runs.items():
            if run_len != 2:
                continue
            cls, next_imp, next_label = classify_next_state(
                appearances,
                news_id,
                end_imp,
            )
            counts[cls] = counts.get(cls, 0) + 1
            events.append((user_id, news_id, start_imp, end_imp, cls, next_imp, next_label))

    return counts, events


def save_svg_bar_chart(
    labels: list[str],
    values: list[int],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    total: int,
) -> None:
    width, height = 1200, 680
    margin_left, margin_right, margin_top, margin_bottom = 110, 30, 70, 110
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_val = max(values) if values else 1
    bar_w = plot_w / max(1, len(values))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="black"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>'
    )

    for i, val in enumerate(values):
        x = margin_left + i * bar_w
        bar_h = 0 if max_val == 0 else (val / max_val) * plot_h
        y = margin_top + (plot_h - bar_h)
        color = colors[i % len(colors)]
        parts.append(
            f'<rect x="{x+10:.2f}" y="{y:.2f}" width="{max(bar_w-20, 1):.2f}" height="{bar_h:.2f}" fill="{color}"/>'
        )
        ratio = 0.0 if total == 0 else (val / total) * 100
        parts.append(
            f'<text x="{x + bar_w/2:.2f}" y="{y-8:.2f}" text-anchor="middle" font-size="12" font-family="Arial">{val} ({ratio:.2f}%)</text>'
        )
        parts.append(
            f'<text x="{x + bar_w/2:.2f}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="12" font-family="Arial">{labels[i]}</text>'
        )

    for i in range(6):
        frac = i / 5
        y = margin_top + plot_h - frac * plot_h
        tick_val = int(round(frac * max_val))
        parts.append(
            f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{tick_val}</text>'
        )

    parts.append(
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-size="16" font-family="Arial">{xlabel}</text>'
    )
    parts.append(
        f'<text x="24" y="{height/2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {height/2})">{ylabel}</text>'
    )
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For exact 0->0 runs (run length == 2), measure follow-up patterns: "
            "0->0->1 vs 0->0->(no future appearance), and save bar figure."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("../MIND-small"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_zero_run2_followup_pattern.svg"),
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=None,
        help="Optional TSV path to save each run2 event and its follow-up class",
    )
    args = parser.parse_args()

    behaviors_path = args.dataset_root / args.split / "behaviors.tsv"
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Missing file: {behaviors_path}")

    user_records = load_user_impressions(behaviors_path)
    user_news_appearances = build_user_news_appearances(user_records)
    counts, events = collect_run2_followup_patterns(user_records, user_news_appearances)

    total_run2 = len(events)
    if total_run2 == 0:
        raise ValueError("No exact run length 2 (0->0) events were found.")

    labels = [
        "0->0->1",
        "0->0->(not shown again)",
        "0->0->0 (after gap)",
    ]
    values = [
        counts.get("next_is_1", 0),
        counts.get("no_future_appearance", 0),
        counts.get("next_is_0_after_gap", 0),
    ]
    if counts.get("other", 0) > 0:
        labels.append("other")
        values.append(counts["other"])

    save_svg_bar_chart(
        labels=labels,
        values=values,
        output_path=args.output,
        title=f"{args.split}: Follow-up Patterns After Exact 0->0 Run (length=2)",
        xlabel="follow-up pattern",
        ylabel="count",
        total=total_run2,
    )

    if args.events_output is not None:
        with args.events_output.open("w", encoding="utf-8") as f:
            f.write(
                "user_id\tnews_id\trun_start_imp_idx\trun_end_imp_idx\tfollowup_class\tnext_imp_idx\tnext_label\n"
            )
            for user_id, news_id, start_imp, end_imp, cls, next_imp, next_label in events:
                f.write(
                    f"{user_id}\t{news_id}\t{start_imp}\t{end_imp}\t{cls}\t"
                    f"{'' if next_imp is None else next_imp}\t{'' if next_label is None else next_label}\n"
                )

    print(f"behaviors_path: {behaviors_path}")
    print(f"users: {len(user_records)}")
    print(f"run2_total: {total_run2}")
    print(f"run2_then_1: {counts.get('next_is_1', 0)}")
    print(f"run2_then_not_shown_again: {counts.get('no_future_appearance', 0)}")
    print(f"run2_then_0_after_gap: {counts.get('next_is_0_after_gap', 0)}")
    print(f"run2_other: {counts.get('other', 0)}")
    print(f"output_figure: {args.output}")
    if args.events_output is not None:
        print(f"output_events: {args.events_output}")


if __name__ == "__main__":
    main()
