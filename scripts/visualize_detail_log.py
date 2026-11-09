#!/usr/bin/env python3
"""Export channel-aligned PNG/TXT from stage1_controlled_scm.detail.log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


def _configure_cjk_font() -> None:
    """Prefer a CJK-capable sans font so Chinese labels render (not boxes)."""
    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Droid Sans Fallback",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


_configure_cjk_font()

EVENT_COLORS = {
    "stable": "#9AA1A8",
    "trend_up": "#A86F62",
    "trend_down": "#527A98",
    "spike": "#A88450",
    "valley": "#81718F",
    "level_shift_up": "#668777",
    "level_shift_down": "#5F8A84",
    "periodic": "#A88450",
    "frequency_change": "#81718F",
}


def iter_detail_records(path: Path) -> Iterator[Dict[str, Any]]:
    buffer: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "---":
                record = _parse_buffer(buffer)
                if record is not None:
                    yield record
                buffer = []
                continue
            buffer.append(line)
    record = _parse_buffer(buffer)
    if record is not None:
        yield record


def _parse_buffer(lines: List[str]) -> Dict[str, Any] | None:
    text = "".join(lines).strip()
    if not text:
        return None
    if text.startswith("#"):
        text = text.split("\n", 1)[1].strip() if "\n" in text else ""
    if not text:
        return None
    return json.loads(text)


def _event_color(event_type: str) -> str:
    return EVENT_COLORS.get(str(event_type), "#7f8c8d")


def _format_type_probs(event: Mapping[str, Any]) -> str:
    probs = event.get("type_probs") or {}
    if not isinstance(probs, Mapping):
        return ""
    ordered = sorted(
        ((name, float(score)) for name, score in probs.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{name}={score:.3f}" for name, score in ordered[:3])


def _format_task_label(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, (int, float, str, bool)):
        return str(label)
    if isinstance(label, dict) and "split" in label and "event_types" in label:
        return "(SCM batch metadata; 见 events_gt)"
    return str(label)[:120]


def _series_by_channel(record: Mapping[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    by_channel = record.get("time_series_by_channel")
    if isinstance(by_channel, dict) and by_channel:
        return {str(key): value for key, value in by_channel.items()}
    primary = int(record.get("primary_channel", 0))
    return {str(primary): record.get("time_series") or {}}


def _channels_to_plot(record: Mapping[str, Any], series: Mapping[str, Mapping[str, Any]]) -> List[str]:
    listed = record.get("channels_plotted")
    if isinstance(listed, list) and listed:
        return [str(channel) for channel in listed if str(channel) in series]
    return list(series)


def _events_on_channel(events: Sequence[Mapping[str, Any]], channel: Any) -> List[Mapping[str, Any]]:
    text = str(channel)
    if text.startswith("-"):
        text = text[1:]
    if not text.isdigit():
        return []
    target = int(text)
    return [event for event in events if int(event.get("channel", -1)) == target]


def format_sample_text(record: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("样本明细（按 channel 对齐）")
    lines.append("=" * 72)
    lines.append(f"sample_id       : {record.get('sample_id', '')}")
    lines.append(f"epoch / step    : {record.get('epoch')} / {record.get('step')}")
    lines.append(f"valid_length    : {record.get('valid_length', '')}")
    lines.append(f"channels_plotted: {record.get('channels_plotted', record.get('primary_channel'))}")
    if "task_label" in record:
        lines.append(f"task_label      : {_format_task_label(record['task_label'])}")
    if record.get("events_valid") is False:
        lines.append("events_valid    : 无有效事件 (existence 均 < 0.5 或后处理为空)")
    lines.append("")

    for section, title in (("events", "预测事件（后处理）"), ("events_gt", "Ground Truth 事件")):
        items = record.get(section) or []
        lines.append("-" * 72)
        lines.append(title)
        lines.append("-" * 72)
        if section == "events" and record.get("events_valid") is False:
            lines.append("(无有效事件)")
            lines.append("")
            continue
        if not items:
            lines.append("(无)")
            lines.append("")
            continue
        for event in items:
            prefix = f"slot {event.get('slot')}" if section == "events" else f"gt {event.get('index')}"
            layer = event.get("layer")
            layer_text = f" [{layer}]" if layer else ""
            lines.append(
                f"[{prefix}] ch{event.get('channel')} {event.get('type')}{layer_text} "
                f"[{event.get('start', 0):.4f}, {event.get('end', 0):.4f}] "
                f"(index {event.get('start_index')} ~ {event.get('end_index')})"
            )
            if section == "events":
                lines.append(
                    f"  existence={event.get('existence', 0):.3f}, "
                    f"confidence={event.get('confidence', 0):.3f}"
                )
                type_probs = _format_type_probs(event)
                if type_probs:
                    lines.append(f"  type_probs: {type_probs}")
            attributes = event.get("attributes") or {}
            for name, value in attributes.items():
                if isinstance(value, float):
                    lines.append(f"  - {name}: {value:.6f}")
                else:
                    lines.append(f"  - {name}: {value}")
            if event.get("amplitude") is not None:
                lines.append(f"  - amplitude: {event['amplitude']}")
            lines.append("")

    rejected = record.get("events_rejected") or []
    if rejected:
        lines.append("-" * 72)
        lines.append("被拒绝/抑制的候选事件")
        lines.append("-" * 72)
        for event in rejected[:20]:
            reason = event.get("reject_reason", "unknown")
            lines.append(
                f"slot {event.get('slot')} {event.get('type')} "
                f"[{event.get('start', 0):.4f}, {event.get('end', 0):.4f}] -> {reason}"
            )
        lines.append("")

    graph = record.get("relation_graph") or {}
    lines.append("-" * 72)
    lines.append("关系图")
    lines.append("-" * 72)
    ascii_graph = graph.get("ascii")
    if ascii_graph:
        lines.append(ascii_graph)
    else:
        edges = graph.get("edges") or []
        for edge in edges[:40]:
            lines.append(
                f"E{edge.get('source_slot')} --{edge.get('relation')}({edge.get('score', 0):.3f})--> "
                f"E{edge.get('target_slot')}"
            )
    lines.append("")
    return "\n".join(lines)


def _format_event_attrs(event: Mapping[str, Any]) -> str:
    parts: List[str] = []
    existence = event.get("existence")
    confidence = event.get("confidence")
    if existence is not None:
        parts.append(f"exist={float(existence):.3f}")
    if confidence is not None:
        parts.append(f"conf={float(confidence):.3f}")
    probs = _format_type_probs(event)
    if probs:
        parts.append(f"p={probs}")
    attributes = event.get("attributes") or {}
    for name, value in attributes.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value}")
    if event.get("amplitude") is not None:
        parts.append(f"amplitude={event['amplitude']}")
    return "\n".join(parts)


def _draw_channel_axis(
    ax: plt.Axes,
    channel: int,
    times: Sequence[float],
    flux: Sequence[float],
    pred_events: Sequence[Mapping[str, Any]],
    gt_events: Sequence[Mapping[str, Any]],
    show_attrs: bool = False,
) -> None:
    ax.plot(times, flux, color="#2c3e50", linewidth=1.1, zorder=3)
    legend_handles: List[Patch] = []
    seen: set[str] = set()
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    for idx, event in enumerate(pred_events):
        event_type = str(event.get("type", "event"))
        color = _event_color(event_type)
        start = float(event.get("start", 0.0))
        end = float(event.get("end", 0.0))
        if end < start:
            start, end = end, start
        is_background = event.get("layer") == "background" or event_type in {
            "periodic",
            "frequency_change",
        }
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.14 if is_background else 0.22,
            hatch="..." if is_background else None,
            zorder=1,
        )
        slot = event.get("slot", idx)
        label = f"E{slot} {event_type}"
        if show_attrs:
            attr_text = _format_event_attrs(event)
            if attr_text:
                label = f"{label}\n{attr_text}"
        mid_x = (start + end) / 2.0
        y_pos = y_max - 0.06 * y_span - (idx % 3) * 0.12 * y_span
        ax.text(
            mid_x,
            y_pos,
            label,
            ha="center",
            va="top",
            fontsize=6,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.85},
            zorder=4,
        )
        if event_type not in seen:
            legend_handles.append(
                Patch(facecolor=color, edgecolor=color, alpha=0.35, label=f"pred:{event_type}")
            )
            seen.add(event_type)
    for idx, event in enumerate(gt_events):
        start = float(event.get("start", 0.0))
        end = float(event.get("end", 0.0))
        if end < start:
            start, end = end, start
        ax.axvspan(
            start,
            end,
            facecolor="none",
            edgecolor="#27ae60",
            hatch="///",
            linewidth=1.0,
            alpha=0.85,
            zorder=2,
        )
        gt_type = str(event.get("type", "gt"))
        gt_label = f"gt{event.get('index', idx)} {gt_type}"
        if show_attrs:
            attr_text = _format_event_attrs(event)
            if attr_text:
                gt_label = f"{gt_label}\n{attr_text}"
        mid_x = (start + end) / 2.0
        ax.text(
            mid_x,
            y_min + 0.08 * y_span + (idx % 2) * 0.10 * y_span,
            gt_label,
            ha="center",
            va="bottom",
            fontsize=6,
            color="#27ae60",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#27ae60", "alpha": 0.85},
            zorder=4,
        )
    if gt_events:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#27ae60", hatch="///", label="gt interval")
        )
    ax.set_title(f"channel {channel}", fontsize=9)
    ax.grid(True, alpha=0.25)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7)


def plot_sample(record: Mapping[str, Any], output_path: Path, *, show_attrs: bool = False) -> None:
    series = _series_by_channel(record)
    if not series:
        raise ValueError("record missing time series")

    channels = _channels_to_plot(record, series)
    pred_events = record.get("events") or []
    gt_events = record.get("events_gt") or []
    events_valid = record.get("events_valid", bool(pred_events))
    gt_series_key = record.get("gt_series_key")

    fig_h = max(2.8, 2.4 * len(channels))
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, fig_h), dpi=120, sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for axis, channel in zip(axes, channels):
        payload = series[channel]
        times = payload.get("time") or []
        flux = payload.get("flux") or []
        gt_for_channel = (
            gt_events
            if gt_series_key and str(channel) == str(gt_series_key)
            else _events_on_channel(gt_events, channel)
        )
        _draw_channel_axis(
            axis,
            channel,
            times,
            flux,
            _events_on_channel(pred_events, channel) if events_valid else [],
            gt_for_channel,
            show_attrs=show_attrs,
        )
        if not events_valid:
            axis.text(
                0.5,
                0.95,
                "无有效事件 (existence < 0.5 或 postprocess 后为空)",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="#c0392b",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#c0392b"},
            )

    axes[-1].set_xlabel("time (normalized)")
    fig.suptitle(
        f"{record.get('sample_id')} | epoch={record.get('epoch')} step={record.get('step')}",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_samples(
    detail_log: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    start: int = 0,
    show_attrs: bool = False,
) -> int:
    count = 0
    for index, record in enumerate(iter_detail_records(detail_log)):
        if index < start:
            continue
        if limit is not None and count >= limit:
            break
        stem = f"{index + 1:04d}"
        plot_sample(record, output_dir / f"{stem}.png", show_attrs=show_attrs)
        (output_dir / f"{stem}.txt").write_text(format_sample_text(record), encoding="utf-8")
        count += 1
        if count % 25 == 0:
            print(f"exported {count} samples -> {output_dir}", flush=True)
    return count


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail-log",
        type=Path,
        default=root / "log" / "stage1_controlled_scm.detail.log",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "log" / "pic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--show-event-attrs",
        action="store_true",
        help="Annotate event boxes with existence/confidence/type-probs/attributes "
        "(default off: only draw the event-type label so labels do not cover the series)",
    )
    args = parser.parse_args()

    if not args.detail_log.is_file():
        raise FileNotFoundError(f"detail log not found: {args.detail_log}")

    total = export_samples(
        args.detail_log,
        args.output_dir,
        limit=args.limit,
        start=args.start,
        show_attrs=args.show_event_attrs,
    )
    print(f"done: {total} samples -> {args.output_dir}")


if __name__ == "__main__":
    main()
