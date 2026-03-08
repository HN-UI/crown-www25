#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
) -> tuple[dict[str, list[tuple[int, datetime | None, dict[str, str]]]], int]:
    # user_id -> [(row_index, parsed_time, {news_id: label})]
    user_records: dict[str, list[tuple[int, datetime | None, dict[str, str]]]] = defaultdict(list)
    row_count = 0

    with behaviors_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            row_count += 1

            user_id = cols[1]
            parsed_time = parse_time_or_none(cols[2])
            impressions = cols[4].split()

            labels_by_news: dict[str, str] = {}
            for token in impressions:
                news_id, label = parse_impression_token(token)
                if label not in {"0", "1"}:
                    continue
                # 중복 토큰이 있을 경우 1을 우선 (데이터 이상치 방어)
                if news_id not in labels_by_news or label == "1":
                    labels_by_news[news_id] = label

            user_records[user_id].append((row_index, parsed_time, labels_by_news))

    for records in user_records.values():
        # time 파싱 실패는 뒤로 보내고, 동시간대는 원본 row_index 순서 유지
        records.sort(
            key=lambda x: (
                x[1] is None,
                x[1] if x[1] is not None else datetime.min,
                x[0],
            )
        )

    return user_records, row_count


def finalize_run(
    run_length: int,
    user_id: str,
    news_id: str,
    start_imp_index: int,
    end_imp_index: int,
    min_run_length: int,
    run_lengths: list[int],
    longest_runs: list[tuple[str, str, int, int, int]],
    max_run_length: int,
) -> int:
    if run_length < min_run_length:
        return max_run_length

    run_lengths.append(run_length)
    if run_length > max_run_length:
        longest_runs.clear()
        longest_runs.append((user_id, news_id, run_length, start_imp_index, end_imp_index))
        return run_length
    if run_length == max_run_length and len(longest_runs) < 20:
        longest_runs.append((user_id, news_id, run_length, start_imp_index, end_imp_index))
    return max_run_length


def collect_zero_run_lengths(
    user_records: dict[str, list[tuple[int, datetime | None, dict[str, str]]]],
    min_run_length: int,
) -> tuple[list[int], int, list[tuple[str, str, int, int, int]], int]:
    # returns:
    # - run_lengths: 연속 0-run length 목록 (length >= min_run_length)
    # - zero_to_zero_transitions: 0->0 transition 총 횟수 (run_length - 1 누적)
    # - longest_runs: (user_id, news_id, run_length, start_imp_index, end_imp_index)
    # - max_run_length
    run_lengths: list[int] = []
    zero_to_zero_transitions = 0
    max_run_length = 0
    longest_runs: list[tuple[str, str, int, int, int]] = []

    for user_id, records in user_records.items():
        # news_id -> (current_run_length, start_imp_index, end_imp_index)
        active_runs: dict[str, tuple[int, int, int]] = {}

        for imp_index, (_, _, labels_by_news) in enumerate(records):
            current_zero_news = {
                news_id for news_id, label in labels_by_news.items() if label == "0"
            }
            next_active_runs: dict[str, tuple[int, int, int]] = {}

            for news_id in current_zero_news:
                if news_id in active_runs:
                    prev_len, start_imp, _ = active_runs[news_id]
                    run_len = prev_len + 1
                    zero_to_zero_transitions += 1
                    next_active_runs[news_id] = (run_len, start_imp, imp_index)
                else:
                    next_active_runs[news_id] = (1, imp_index, imp_index)

            for news_id, (run_len, start_imp, end_imp) in active_runs.items():
                if news_id in next_active_runs:
                    continue
                max_run_length = finalize_run(
                    run_len,
                    user_id,
                    news_id,
                    start_imp,
                    end_imp,
                    min_run_length,
                    run_lengths,
                    longest_runs,
                    max_run_length,
                )

            active_runs = next_active_runs

        for news_id, (run_len, start_imp, end_imp) in active_runs.items():
            max_run_length = finalize_run(
                run_len,
                user_id,
                news_id,
                start_imp,
                end_imp,
                min_run_length,
                run_lengths,
                longest_runs,
                max_run_length,
            )

    return run_lengths, zero_to_zero_transitions, longest_runs, max_run_length


def make_hist_counts(values: list[int], bin_width: int) -> tuple[list[int], list[int]]:
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0")
    if not values:
        return [0], [0, max(1, bin_width)]

    max_v = max(values)
    upper = ((max_v + bin_width) // bin_width) * bin_width
    bin_count = max(1, upper // bin_width)
    edges = [i * bin_width for i in range(bin_count + 1)]
    counts = [0] * bin_count
    for v in values:
        idx = v // bin_width
        if idx >= bin_count:
            idx = bin_count - 1
        counts[idx] += 1
    return counts, edges


def save_svg_hist(
    counts: list[int],
    edges: list[int],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 100, 30, 70, 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_count = max(counts) if counts else 1
    bar_w = plot_w / max(1, len(counts))

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="black"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="black"/>'
    )

    for i, c in enumerate(counts):
        x = margin_left + i * bar_w
        bar_h = 0 if max_count == 0 else (c / max_count) * plot_h
        y = margin_top + (plot_h - bar_h)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1, bar_w-1):.2f}" height="{bar_h:.2f}" fill="#4e79a7"/>'
        )

    for i in range(6):
        frac = i / 5
        y = margin_top + plot_h - frac * plot_h
        label = int(round(frac * max_count))
        parts.append(
            f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{margin_left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{label}</text>'
        )

    edge_n = len(edges)
    tick_stride = max(1, (edge_n + 39) // 40)
    for i, edge in enumerate(edges):
        if i % tick_stride != 0 and i != edge_n - 1:
            continue
        x = margin_left + (i / (edge_n - 1)) * plot_w if edge_n > 1 else margin_left
        parts.append(
            f'<line x1="{x:.2f}" y1="{height-margin_bottom}" x2="{x:.2f}" y2="{height-margin_bottom+5}" stroke="black"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{height-margin_bottom+20}" text-anchor="middle" font-size="10" font-family="Arial">{edge}</text>'
        )

    parts.append(
        f'<text x="{width/2}" y="{height-26}" text-anchor="middle" font-size="16" font-family="Arial">{xlabel}</text>'
    )
    parts.append(
        f'<text x="24" y="{height/2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 24 {height/2})">{ylabel}</text>'
    )
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = (len(arr) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure consecutive -0 appearance run lengths across user timelines "
            "(0->0 once means run length 2), then save histogram figure."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("../MIND-small"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--min-run-length", type=int, default=2)
    parser.add_argument("--bin-width", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_zero_negative_run_length_hist.svg"),
    )
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=None,
        help="Optional TSV path to save run_length/count table",
    )
    args = parser.parse_args()

    if args.min_run_length < 1:
        raise ValueError("--min-run-length must be >= 1")

    behaviors_path = args.dataset_root / args.split / "behaviors.tsv"
    if not behaviors_path.exists():
        raise FileNotFoundError(f"Missing file: {behaviors_path}")

    user_records, row_count = load_user_impressions(behaviors_path)
    run_lengths, zero_to_zero_transitions, longest_runs, max_run_length = collect_zero_run_lengths(
        user_records=user_records,
        min_run_length=args.min_run_length,
    )

    if not run_lengths:
        raise ValueError(
            f"No run found with length >= {args.min_run_length}. "
            "Try a lower --min-run-length."
        )

    counts, edges = make_hist_counts(run_lengths, args.bin_width)
    save_svg_hist(
        counts=counts,
        edges=edges,
        output_path=args.output,
        title=f"{args.split}: Consecutive -0 Run Length Distribution",
        xlabel="run length (0->0 once == length 2)",
        ylabel="frequency",
    )

    length_counter = Counter(run_lengths)
    if args.counts_output is not None:
        with args.counts_output.open("w", encoding="utf-8") as f:
            f.write("run_length\tcount\n")
            for run_length in sorted(length_counter):
                f.write(f"{run_length}\t{length_counter[run_length]}\n")

    print(f"behaviors_path: {behaviors_path}")
    print(f"rows: {row_count}")
    print(f"users: {len(user_records)}")
    print(f"min_run_length: {args.min_run_length}")
    print(f"runs_kept: {len(run_lengths)}")
    print(f"zero_to_zero_transitions: {zero_to_zero_transitions}")
    print(f"max_run_length: {max_run_length}")
    print(f"mean_run_length: {sum(run_lengths) / len(run_lengths):.4f}")
    print(f"median_run_length: {percentile(run_lengths, 0.5):.4f}")
    print(f"p90_run_length: {percentile(run_lengths, 0.9):.4f}")
    print(f"output_figure: {args.output}")
    if args.counts_output is not None:
        print(f"output_counts: {args.counts_output}")

    print("top_run_length_counts:")
    for run_length, cnt in sorted(length_counter.items(), key=lambda x: x[0], reverse=True)[:15]:
        print(f"  length={run_length}: {cnt}")

    if longest_runs:
        print("examples_of_longest_runs:")
        for user_id, news_id, run_length, start_imp, end_imp in longest_runs[:10]:
            print(
                f"  user={user_id} news={news_id} length={run_length} "
                f"impression_idx=[{start_imp}..{end_imp}]"
            )


if __name__ == "__main__":
    main()
