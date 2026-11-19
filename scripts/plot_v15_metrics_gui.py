#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""V15 MAPPO/AnyLogic local plotting GUI for TRAINING and HELD-OUT logs.

Run from VS Code terminal:
    .\.venv311\Scripts\python.exe .\scripts\plot_v15_metrics_gui.py
Or headless (repeat --input to compare checkpoint files):
    python plot_v15_metrics_gui.py --input metrics.jsonl --no-gui
    python plot_v15_metrics_gui.py --input heldout_ep80.jsonl --input heldout_ep200.jsonl --no-gui

Only reads logs. Produces PNG, SVG, CSV, TXT in NEW timestamped folders.
Training episode curves are NOT evaluation. Matched seeds permit paired comparisons.
Paper training steps/lead time cannot be inferred from these logs.
"""
#=========== MODULE 00 - THƯ VIỆN, KIỂM TRA PHỤ THUỘC VÀ DANH SÁCH METRIC ===========
# Mục đích: Chuẩn bị NumPy, Matplotlib, nhãn đồ thị, kiểm tra thư viện cần thiết.
# Giải thích: Đây là script đọc log vẽ hình, không huấn luyện mạng.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Headless CLI + embedded Tk figure canvas, no GUI until needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
except ImportError as exc:
    raise SystemExit(
        "THIEU THU VIEN: numpy/matplotlib. Cài: python -m pip install numpy matplotlib\n"
        f"Lỗi: {exc}"
    )

KEYS = [
    ("jobDone", "Products Completed (jobDone)", "Products"),
    ("targetReached", "Targets Reached", "Targets"),
    ("block", "Number of Blocks", "Blocks"),
    ("liveLock", "LiveLocks", "LiveLocks"),
    ("wrongTarget", "Wrong Target", "Wrong Target"),
    ("notArrived", "Not Arrived", "Not Arrived"),
    ("taken_invalid", "Taken Invalid", "Taken Invalid"),
    ("dispatch", "Dispatch Decisions", "Dispatch"),
    ("delivered_reward", "Episode Reward (delivered_reward)", "Reward"),
]
PLOT_METRICS = {key: (title, ylabel) for key, title, ylabel in KEYS}



#======= KẾT THÚC MODULE 00 - THƯ VIỆN, KIỂM TRA PHỤ THUỘC VÀ DANH SÁCH METRIC =======




#===================== MODULE 01 - CHUYỂN GIÁ TRỊ METRIC SANG SỐ =====================
# Mục đích: Đọc một giá trị từ log; trả NaN nếu không đổi được.
# Giải thích: Không tự đổi dữ liệu thiếu thành 0.

def safe_num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else float("nan")
    except (ValueError, TypeError, OverflowError):
        return float("nan")

#================ KẾT THÚC MODULE 01 - CHUYỂN GIÁ TRỊ METRIC SANG SỐ ================






#==================== MODULE 02 - ĐỌC LOG TRAINING METRICS.JSONL ====================
# Mục đích: Chuyển từng dòng JSONL thành chuỗi theo episode: JobDone, Reward, entropy, KL...
# Giải thích: Tính trục X cumulative events để vẽ các đường huấn luyện.

def read_log(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file: {path}")
    entries, bad_lines = [], []

    #======================= PHẦN READ_LOG-01 - ĐỌC CÁC DÒNG JSONL =======================
    # Tách mỗi episode và bỏ qua dòng không hợp lệ.

    with path.open("r", encoding="utf-8-sig") as fp:
        for line_no, text in enumerate(fp, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                bad_lines.append(line_no)
                continue
            if not isinstance(row, dict) or not isinstance(row.get("environment"), dict):
                bad_lines.append(line_no)
                continue
            if not isinstance(row.get("environment", {}).get("statistics"), dict):
                bad_lines.append(line_no)
                continue
            try:
                ep = int(row["episode"])
            except (KeyError, ValueError, TypeError):
                bad_lines.append(line_no)
                continue
            if ep < 0:
                bad_lines.append(line_no)
                continue
            entries.append(row)
    if not entries:
        raise ValueError(f"Không thấy episode training hợp lệ trong: {path.name}")
    entries.sort(key=lambda r: int(r["episode"]))
    epis = [int(r["episode"]) for r in entries]
    duplicates = sorted(e for e, n in Counter(epis).items() if n > 1)
    if duplicates:
        raise ValueError(
            f"{path.name}: trùng episode {duplicates[:12]}. Dừng để tránh đếm sai."
        )
    gap = [e for e in range(min(epis), max(epis) + 1) if e not in set(epis)]
    first = entries[0]
    stat = first["environment"]["statistics"]
    fleet = first.get("fleet_size", stat.get("fleetsize", None))
    try:
        fleet = int(fleet)
    except (ValueError, TypeError):
        fleet = None
    phase = str(first.get("phase", "unknown"))
    if phase == "unknown" and fleet:
        phase = f"fleet{fleet}"
    # Distinguish same-stage runs with different file names in the legend
    label = phase + (f" ({fleet} AGV)" if fleet else "")
    data = {
        "file": path, "rows": entries, "episode": np.array(epis, dtype=int),
        "phase": phase, "fleet": fleet, "label": label,
        "bad_lines": bad_lines, "missing_episodes": gap,
        "metric": {}, "ppo": {},
    }
    for key, _, _ in KEYS:
        vals = []
        for r in entries:
            if key == "delivered_reward":
                value = r["environment"].get("delivered_reward")
            else:
                value = r["environment"]["statistics"].get(key)
            vals.append(safe_num(value))
        data["metric"][key] = np.array(vals, dtype=float)
    for role in ("routing", "dispatch_front", "dispatch_rear"):
        for field in ("entropy", "approx_kl", "value_loss", "clip_fraction"):
            vals = []
            for row in entries:
                v = row.get("policy_updates", {}).get(role, {})
                vals.append(safe_num(v.get(field)) if isinstance(v, dict) else float("nan"))
            data["ppo"][(role, field)] = np.asarray(vals, dtype=float)
    data["event_steps"] = np.cumsum(
        np.array([max(0.0, safe_num(r["environment"].get("events", 0))) for r in entries])
    )
    return data

#================ KẾT THÚC MODULE 02 - ĐỌC LOG TRAINING METRICS.JSONL ================






#=================== MODULE 03 - LỌC TRUNG BÌNH TRƯỢT CHO TRAINING ===================
# Mục đích: Làm mượt đường dữ liệu nhưng giữ NaN và độ dài ban đầu.
# Giải thích: Tương tự lọc trung bình di động trong xử lý tín hiệu.

def sma(values: np.ndarray, window: int):
    # NaNs remain absent; avoid inventing zero observations.
    a = np.asarray(values, dtype=float)
    valid = np.isfinite(a)
    numer = np.convolve(np.where(valid, a, 0.0), np.ones(window), mode="full")[:len(a)]
    denom = np.convolve(valid.astype(float), np.ones(window), mode="full")[:len(a)]
    return np.divide(numer, denom, out=np.full(len(a), np.nan), where=denom > 0)

#============== KẾT THÚC MODULE 03 - LỌC TRUNG BÌNH TRƯỢT CHO TRAINING ==============






#======================= MODULE 04 - CHỌN TRỤC X CỦA TRAINING =======================
# Mục đích: Chọn episode hoặc cumulative events theo tùy chọn người dùng.
# Giải thích: Không dùng trục này cho dữ liệu held-out.

def axis_x(d, use_events: bool):
    return d["event_steps"] if use_events else d["episode"]

#=================== KẾT THÚC MODULE 04 - CHỌN TRỤC X CỦA TRAINING ===================






#===================== MODULE 05 - CẤU HÌNH TRỤC VÀ LƯỚI ĐỒ THỊ =====================
# Mục đích: Đặt nhãn trục, lưới, định dạng chia vạch.
# Giải thích: Tập trung logic giao diện trục để các hình nhất quán.

def common_axes(ax, ylabel, use_events):
    ax.set_xlabel("Training Steps (cumulative events)" if use_events else "Training Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.45)
    if not use_events:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

#================= KẾT THÚC MODULE 05 - CẤU HÌNH TRỤC VÀ LƯỚI ĐỒ THỊ =================






#===================== MODULE 06 - VẼ ĐƯỜNG CHỈ SỐ TỪ NHIỀU RUN =====================
# Mục đích: Vẽ đường trung bình trượt và dữ liệu thô cho từng dataset.
# Giải thích: Trả về trạng thái có dữ liệu hợp lệ để tránh đồ thị rỗng.

def plot_lines(ax, datasets, name, window, use_events, ylabel=None, show_raw=True):
    ylabel = ylabel or PLOT_METRICS[name][1]
    meaningful = False
    for d in datasets:
        yy = d["metric"].get(name)
        if yy is None or not np.isfinite(yy).any():
            continue
        meaningful = True
        xx = axis_x(d, use_events)
        line = ax.plot(xx, sma(yy, window), linewidth=1.9, label=d["label"], zorder=2)[0]
        if show_raw and window > 1:
            ax.plot(xx, yy, linewidth=0.8, alpha=0.20,
                    color=line.get_color(), zorder=1)
    if not meaningful:
        ax.text(0.5, 0.5, "Metric unavailable in log", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    common_axes(ax, ylabel, use_events)

#================= KẾT THÚC MODULE 06 - VẼ ĐƯỜNG CHỈ SỐ TỪ NHIỀU RUN =================






#==================== MODULE 07 - BỐN ĐỒ THỊ CHÍNH KHI HUẤN LUYỆN ====================
# Mục đích: TargetReached, Blocks, Reward và JobDone theo episode/events.
# Giải thích: Chỉ mô tả training; không được xem là đánh giá checkpoint held-out.

def plot_training(datasets, window, use_events):
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 8.7))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.895,
                        bottom=0.135, wspace=0.23, hspace=0.32)
    specs = [
        ("targetReached", "Targets Reached", "Targets Reached / episode"),
        ("block", "Number of Blocks", "Blocks / episode"),
        ("delivered_reward", "Episode Reward", "Delivered reward / episode"),
        ("jobDone", "Products Completed", "JobDone / episode"),
    ]
    for ax, (key, title, ytitle) in zip(axs.flat, specs):
        ax.set_title(title, fontsize=12, fontweight="medium")
        plot_lines(ax, datasets, key, window, use_events, ylabel=ytitle)
    fig.suptitle("V15 Training Curves — Figure 5–7 inspired", fontweight="semibold", fontsize=15)
    fig.text(0.5, 0.025,
             "Per-episode training values; paper 'training steps' and curves are not identical to V15 logs.",
             ha="center", va="bottom", fontsize=8, color="dimgray")
    return fig

#=============== KẾT THÚC MODULE 07 - BỐN ĐỒ THỊ CHÍNH KHI HUẤN LUYỆN ===============






#================= MODULE 08 - SÁU ĐỒ THỊ CHẨN ĐOÁN GIAO THÔNG / PPO =================
# Mục đích: LiveLock, WrongTarget, Invalid và Entropy/KL/Value Loss theo role.
# Giải thích: Giúp tìm liên hệ giữa nghẽn giao thông và độ ổn định học.

def plot_diagnostics(datasets, window, use_events):
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 8.8), constrained_layout=True)
    metric_specs = [
        ("liveLock", "LiveLock", "LiveLock / episode"),
        ("wrongTarget", "WrongTarget", "WrongTarget / episode"),
        ("taken_invalid", "Invalid Actions Taken", "taken_invalid / episode"),
    ]
    for ax, (key, title, ylabel) in zip(axs.flat[:3], metric_specs):
        ax.set_title(title, fontsize=12)
        plot_lines(ax, datasets, key, window, use_events, ylabel=ylabel)
    for ax, field, title in zip(axs.flat[3:],
                               ("entropy", "approx_kl", "value_loss"),
                               ("PPO Entropy", "PPO Approx KL", "PPO Value Loss")):
        ax.set_title(title, fontsize=12)
        plotted = False
        for d in datasets:
            for role in ("routing", "dispatch_front", "dispatch_rear"):
                yy = d["ppo"].get((role, field))
                if yy is None or not np.isfinite(yy).any():
                    continue
                plotted = True
                xx = axis_x(d, use_events)
                ax.plot(xx, sma(yy, window), label=f"{d['phase']}:{role}", linewidth=1.4)
        if plotted:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No PPO data", transform=ax.transAxes, ha="center", va="center")
        common_axes(ax, field, use_events)
    fig.suptitle("V15 Diagnostics — Traffic & PPO stability", fontweight="semibold", fontsize=15)
    return fig

#============ KẾT THÚC MODULE 08 - SÁU ĐỒ THỊ CHẨN ĐOÁN GIAO THÔNG / PPO ============






#====================== MODULE 09 - TÍNH TRUNG BÌNH CUỐI CỬA SỔ ======================
# Mục đích: Lấy trung bình một đoạn cuối, bỏ qua NaN.
# Giải thích: Dùng để tổng hợp mức hiệu suất gần cuối run.

def mean_last(arr, window):
    a = np.asarray(arr[-window:], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")

#================= KẾT THÚC MODULE 09 - TÍNH TRUNG BÌNH CUỐI CỬA SỔ =================






#===================== MODULE 10 - SO SÁNH HIỆU SUẤT THEO SỐ AGV =====================
# Mục đích: Chỉ vẽ khi có tối thiểu hai fleet size khác nhau.
# Giải thích: So sánh Target/AGV, JobDone, Block, LiveLock.

def plot_fleet_compare(datasets, window):
    available = [d for d in datasets if d["fleet"] and d["fleet"] > 0]
    if len({d["fleet"] for d in available}) < 2:
        return None
    available.sort(key=lambda d: (d["fleet"], d["phase"]))
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.885,
                        bottom=0.14, wspace=0.25, hspace=0.34)
    specs = [
        ("targetReached", "Targets Reached / AGV", "Targets per AGV", True),
        ("jobDone", "Products Completed (JobDone)", "JobDone / episode", False),
        ("block", "Number of Blocks", "Blocks / episode", False),
        ("liveLock", "LiveLock (NOT Lead Time)", "LiveLock / episode", False),
    ]
    for ax, (key, title, ylabel, per_agv) in zip(axs.flat, specs):
        ax.set_title(title, fontsize=12)
        by_fleet = {}
        for d in available:
            val = mean_last(d["metric"][key], window)
            if per_agv and math.isfinite(val):
                val /= d["fleet"]
            by_fleet.setdefault(d["fleet"], []).append((val, d["label"]))
        xs, ys = [], []
        for fleet in sorted(by_fleet):
            vals = [v for v, _ in by_fleet[fleet] if math.isfinite(v)]
            if not vals:
                continue
            avg = float(np.mean(vals))
            xs.append(fleet); ys.append(avg)
            for v, label in by_fleet[fleet]:
                if math.isfinite(v):
                    ax.scatter([fleet], [v], marker="o", alpha=0.5, s=30)
        ax.plot(xs, ys, linewidth=1.8, marker="o")
        common_axes(ax, ylabel, False)
        ax.set_xlabel("Fleet Size (AGVs)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Fleet-size Comparison — Figure 8 inspired (training only)",
                 fontweight="semibold", fontsize=15)
    fig.text(0.5, 0.025,
             f"Mean of last {window} episodes for each phase. Not held-out tests. Lead time is NOT in metrics.jsonl.",
             ha="center", va="bottom", fontsize=8, color="dimgray")
    return fig

#================ KẾT THÚC MODULE 10 - SO SÁNH HIỆU SUẤT THEO SỐ AGV ================






#===================== MODULE 11 - XUẤT THỐNG KÊ TRAINING VÀ CSV =====================
# Mục đích: Lưu dữ liệu từng episode và tóm tắt về file riêng.
# Giải thích: Nhấn mạnh số liệu training không thay thế held-out evaluation.

def summary_and_csv(datasets, window, out_dir):
    out_dir = Path(out_dir)
    txt = []
    txt.append("V15 TRAINING METRICS — LOCAL REPORT")
    txt.append("WARNING: training metrics, not held-out evaluation.")
    txt.append("Fig. 5–8-inspired plots do NOT reproduce the author's experiments.")
    txt.append("Training episode != paper training step; cumulative events is a proxy only.")
    txt.append("Products Completed is mapped to statistics.jobDone. Lead time is not logged here.")
    txt.append("-" * 75)
    with (out_dir / "v15_episode_data.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        keys = [k for k, _, _ in KEYS]
        writer = csv.writer(fp)
        writer.writerow(["source_file", "phase", "fleet_size", "episode", "cumulative_events", *keys])
        for d in datasets:
            for i, row in enumerate(d["rows"]):
                writer.writerow([str(d["file"]), d["phase"], d["fleet"], row["episode"],
                                 round(d["event_steps"][i], 2),
                                 *["" if not math.isfinite(d["metric"][k][i]) else round(d["metric"][k][i], 6)
                                   for k in keys]])
    with (out_dir / "v15_fleet_summary.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["source_file", "phase", "fleet_size", "last_episode", "n_episodes", "window",
                         "jobDone_avg", "targets_avg", "targets_per_agv_avg", "block_avg", "liveLock_avg",
                         "wrongTarget_avg", "reward_avg"])
        for d in datasets:
            agg = {k: mean_last(d["metric"][k], window) for k in PLOT_METRICS}
            per_agv = agg["targetReached"] / d["fleet"] if d["fleet"] else float("nan")
            writer.writerow([str(d["file"]), d["phase"], d["fleet"], d["episode"][-1], len(d["rows"]), window,
                             agg["jobDone"], agg["targetReached"], per_agv, agg["block"],
                             agg["liveLock"], agg["wrongTarget"], agg["delivered_reward"]])
    for d in datasets:
        n, ep = len(d["rows"]), d["episode"]
        txt.append(f"\n[{d['label']}] {d['file']}")
        txt.append(f"Episode {ep[0]}–{ep[-1]} | {n} rows | window={min(n,window)}")
        if d["bad_lines"]:
            txt.append(f"WARNING malformed / skipped lines: {d['bad_lines'][:20]}")
        if d["missing_episodes"]:
            txt.append(f"WARNING missing episodes: {d['missing_episodes'][:30]}")
        for k, _, _ in KEYS:
            arr = d["metric"][k]
            if not np.isfinite(arr).any():
                continue
            latest = mean_last(arr, window)
            first = float(np.nanmean(arr[:min(window, len(arr))]))
            prev = arr[max(0,n-2*window):max(0,n-window)]
            prev_avg = float(np.nanmean(prev)) if len(prev) and np.isfinite(prev).any() else float("nan")
            text = f"{k:18s} last={latest:10.3f} first={first:10.3f}"
            if math.isfinite(prev_avg):
                text += f" prev={prev_avg:10.3f} delta={latest-prev_avg:+9.3f}"
            txt.append(text)
        envs = [row.get("environment", {}) for row in d["rows"]]
        reasons = Counter(str(e.get("end_reason", "unknown")) for e in envs)
        txt.append("End reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
        violations = sum(
            safe_num(e.get("action_diagnostics", {}).get("selected_mask_violations", 0))
            for e in envs if isinstance(e.get("action_diagnostics"), dict)
        )
        txt.append(f"Selected mask violations (sum): {violations:g}")
        audits_fail = 0
        for r in d["rows"]:
            audit = r.get("ppo_audit", {})
            if isinstance(audit, dict):
                if audit.get("passed") is False:
                    audits_fail += 1
                audits_fail += sum(
                    1 for v in audit.values() if isinstance(v, dict) and v.get("passed") is False
                )
        txt.append(f"PPO audit failures: {audits_fail}")
        for role in ("routing", "dispatch_front", "dispatch_rear"):
            updates = sum(
                1 for r in d["rows"] if role in r.get("policy_updates", {})
            )
            txt.append(f"PPO updated {role}: {updates}/{n} episode(s)")
    txt.append("\nNOTE: To compare Fig.8 statistically, evaluate on matched held-out seeds & definitions.")
    report = out_dir / "v15_report.txt"
    report.write_text("\n".join(txt) + "\n", encoding="utf-8")
    return report

#================ KẾT THÚC MODULE 11 - XUẤT THỐNG KÊ TRAINING VÀ CSV ================







# ====================== HELD-OUT EVALUATION ============================
# Evaluation has one row per SEED, not one row per training episode.
# Never draw seed order as if it was a training learning curve.

#===================== MODULE 12 - KHAI BÁO CÁC METRIC HELD-OUT =====================
# Mục đích: Liệt kê trường cần tổng hợp và so sánh ở chế độ đánh giá.
# Giải thích: Mỗi hàng held-out tương ứng một seed môi trường, không phải bước training.

HELDOUT_KEYS = (
    'jobDone', 'targetReached', 'block', 'liveLock', 'wrongTarget',
    'notArrived', 'taken_invalid', 'delivered_reward',
    'routing_stop_with_move_available', 'routing_move_available_decisions',
    'routing_total_decisions', 'routing_stop_rate', 'selected_mask_violations'
)

#================= KẾT THÚC MODULE 12 - KHAI BÁO CÁC METRIC HELD-OUT =================






#================== MODULE 13 - TỰ NHẬN BIẾT TRAINING HAY HELD-OUT ==================
# Mục đích: Xem cấu trúc JSONL thay vì tin vào tên file.
# Giải thích: Tránh vẽ nhầm seed như training timestep.

def inspect_log_kind(path: Path):
    """Determine log kind by structured fields, regardless of file name."""
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row=json.loads(raw)
            except json.JSONDecodeError as err:
                raise ValueError(f'{path}: First nonblank line is malformed JSON: {err}') from err
            if not isinstance(row,dict):
                raise ValueError(f'{path}: Expected JSON object per line')
            if ('seed' in row and 'checkpoint_sha256' in row and
                    'environment' in row and 'episode' in row):
                return 'heldout'
            if 'episode' in row and 'environment' in row:
                return 'training'
            raise ValueError(f'{path}: Not recognized as V15 training or held-out JSONL')
    raise ValueError(f'{path}: Empty JSONL')

#============== KẾT THÚC MODULE 13 - TỰ NHẬN BIẾT TRAINING HAY HELD-OUT ==============






#=================== MODULE 14 - ĐỌC VÀ KIỂM TRA DỮ LIỆU HELD-OUT ===================
# Mục đích: Lấy seed, checkpoint, số xe, các metrics và chuẩn hóa dữ liệu.
# Giải thích: Không trộn các checkpoint/phase không tương thích.

def read_heldout(path: Path):
    path=Path(path).expanduser().resolve()
    rows=[]; bad=[]
    with path.open('r',encoding='utf-8-sig') as handle:
        for no,line in enumerate(handle,1):
            if not line.strip():
                continue
            try: r=json.loads(line)
            except json.JSONDecodeError:
                bad.append(no);continue
            if not isinstance(r,dict) or not isinstance(r.get('environment'),dict) or not isinstance(r['environment'].get('statistics'),dict):
                bad.append(no);continue
            rows.append(r)
    if not rows:
        raise ValueError(f'No completed held-out rows: {path}')
    ids=[]
    for r in rows:
        try: ids.append(int(r['seed']))
        except (KeyError,ValueError,TypeError) as err:
            raise ValueError(f'{path.name}: missing or invalid seed') from err
    if len(ids)!=len(set(ids)):
        raise ValueError(f'{path.name}: duplicate held-out seed; cannot average correctly.')
    # A single file must represent exactly one policy checkpoint. Detect accidental mixes.
    identity=[]
    for r in rows:
        identity.append((str(r.get('phase','unknown')),str(r.get('checkpoint_sha256','')),
                         int(r.get('episode',-1)),int(r.get('fleet',-1)),
                         float(r.get('sim_seconds',float('nan'))),bool(r.get('greedy',False)),
                         int(r.get('ppo_updates',-1))))
    if len(set(identity))!=1:
        raise ValueError(f'{path.name}: Mixed phase/episode/checkpoint/fleet/greedy/ppo configuration.')
    phase,sha,ep,fleet,sim_seconds,greedy,ppo=identity[0]
    if not sha or fleet<1 or ep<0:
        raise ValueError(f'{path.name}: Invalid checkpoint hash, fleet or episode.')
    if not greedy or ppo!=0:
        raise ValueError(f'{path.name}: Expected held-out greedy evaluation with ppo_updates=0.')
    rows.sort(key=lambda r:int(r['seed']))
    seeds=np.asarray([int(r['seed']) for r in rows],dtype=int)
    values={k:[] for k in HELDOUT_KEYS}
    stop_mode=set();end_reasons=Counter()
    for r in rows:
        env=r['environment']; stats=env['statistics'];diag=env.get('action_diagnostics',{})
        if not isinstance(diag,dict):diag={}
        end_reasons[str(env.get('end_reason','unknown'))]+=1
        stop_mode.add(str(diag.get('stop_mask_mode','unknown')))
        for k in HELDOUT_KEYS[:7]:
            values[k].append(safe_num(stats.get(k)))
        values['delivered_reward'].append(safe_num(env.get('delivered_reward')))
        values['routing_stop_with_move_available'].append(safe_num(diag.get('routing_stop_with_move_available')))
        values['routing_move_available_decisions'].append(safe_num(diag.get('routing_move_available_decisions')))
        hist=diag.get('actions_by_role',{});
        routing=hist.get('routing',{}) if isinstance(hist,dict) else {}
        total=sum(safe_num(v) for v in routing.values()) if isinstance(routing,dict) else float('nan')
        values['routing_total_decisions'].append(total if isinstance(routing,dict) and routing else float('nan'))
        moves=safe_num(diag.get('routing_move_available_decisions'))
        stops=safe_num(diag.get('routing_stop_with_move_available'))
        values['routing_stop_rate'].append(100*stops/moves if math.isfinite(moves) and moves>0 and math.isfinite(stops) else float('nan'))
        values['selected_mask_violations'].append(safe_num(diag.get('selected_mask_violations')))
    return dict(kind='heldout',file=path,phase=phase,episode=ep,checkpoint=sha,fleet=fleet,
                sim_seconds=sim_seconds,greedy=greedy,ppo_updates=ppo,
                label=f'{phase} EP{ep} ({fleet} AGV)',seed=seeds,
                metric={k:np.asarray(v,dtype=float) for k,v in values.items()},
                rows=rows,bad_lines=bad,end_reasons=end_reasons,stop_mode=stop_mode)

#=============== KẾT THÚC MODULE 14 - ĐỌC VÀ KIỂM TRA DỮ LIỆU HELD-OUT ===============






#========================= MODULE 15 - ĐỊNH DẠNG SỐ THỐNG KÊ =========================
# Mục đích: Đổi giá trị ra dạng số có n chữ số thập phân.
# Giải thích: Dữ liệu không hợp lệ được in N/A.

def fmt(value,nd=2):
    return f'{value:.{nd}f}' if math.isfinite(value) else 'N/A'

#==================== KẾT THÚC MODULE 15 - ĐỊNH DẠNG SỐ THỐNG KÊ ====================






#=================== MODULE 16 - TÍNH TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN ===================
# Mục đích: Trả mean, sample standard deviation và số mẫu hợp lệ.
# Giải thích: Dùng để so sánh độ ổn định JobDone trên các seed.

def mean_sd(arr):
    x=np.asarray(arr,dtype=float)
    x=x[np.isfinite(x)]
    return ((float(x.mean()),float(x.std(ddof=1)) if len(x)>1 else 0.,len(x))
            if len(x) else (float('nan'),float('nan'),0))

#=============== KẾT THÚC MODULE 16 - TÍNH TRUNG BÌNH VÀ ĐỘ LỆCH CHUẨN ===============






#================= MODULE 17 - PHÂN BIỆT NHÃN CHECKPOINT TRÙNG NHAU =================
# Mục đích: Bổ sung SHA ngắn khi trùng phase/EP.
# Giải thích: Giúp tránh nhầm hai checkpoint khi so sánh hình.

def dataset_labels(ds):
    """Give distinct labels to same-phase, same-EP runs from different checkpoints."""
    counts=Counter(d['label'] for d in ds)
    for d in ds:
        if counts[d['label']]>1:
            d['label'] += ' ['+d['checkpoint'][:8]+']'
    labels=[d['label'] for d in ds]
    if len(set(labels))<len(labels):
        for i,d in enumerate(ds,1):d['label']+=f' #{i}'

#============= KẾT THÚC MODULE 17 - PHÂN BIỆT NHÃN CHECKPOINT TRÙNG NHAU =============






#=================== MODULE 18 - VẼ CỘT TRUNG BÌNH VÀ THANH SAI SỐ ===================
# Mục đích: Mỗi cột là mean theo seed, error bar là độ lệch chuẩn.
# Giải thích: Tùy chọn chia chỉ số cho số AGV.

def grouped_bars(ax,ds,key,title,ylabel,per_agv=False):
    vals=[];err=[]
    for d in ds:
        arr=d['metric'][key]
        if per_agv:arr=arr/d['fleet']
        m,s,n=mean_sd(arr)
        vals.append(m);err.append(s if n>1 else 0)
    inds=np.arange(len(ds))
    good=[i for i,v in enumerate(vals) if math.isfinite(v)]
    if good:
        bars=ax.bar([inds[i] for i in good],[vals[i] for i in good],
                    yerr=[err[i] for i in good],capsize=4,alpha=.8)
        for i,b in zip(good,bars):
            ax.annotate(f'{vals[i]:.1f}',(b.get_x()+b.get_width()/2,b.get_height()),
                        xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    else:
        ax.text(.5,.5,'Unavailable',transform=ax.transAxes,ha='center')
    ax.set_xticks(inds,[d['label'] for d in ds],rotation=15,ha='right',fontsize=8)
    ax.set_title(title);ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=.3,linestyle='--')
    # Do not force a positive y-axis for reward, which may be negative.
    if key!='delivered_reward':ax.set_ylim(bottom=0)

#============== KẾT THÚC MODULE 18 - VẼ CỘT TRUNG BÌNH VÀ THANH SAI SỐ ==============






#===================== MODULE 19 - TỔNG QUAN SÁU METRIC HELD-OUT =====================
# Mục đích: Vẽ JobDone, Target/AGV, Block, LiveLock, WrongTarget, Reward.
# Giải thích: Đây là đánh giá checkpoint cố định, không phải learning curve.

def plot_heldout_overview(ds):
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    fig.suptitle('V15 HELD-OUT — checkpoint comparison (mean ± sample SD)',fontsize=16,fontweight='bold')
    specs=[('jobDone','Products Completed','JobDone / seed',False),
           ('targetReached','Targets Reached / AGV','Targets per AGV / seed',True),
           ('block','Number of Blocks','Block / seed',False),
           ('liveLock','LiveLocks','LiveLock / seed',False),
           ('wrongTarget','Wrong Target','WrongTarget / seed',False),
           ('delivered_reward','Delivered Reward','Episode reward / seed',False)]
    for ax,(k,title,yl,per) in zip(axes.flat,specs):grouped_bars(ax,ds,k,title,yl,per)
    return fig

#================ KẾT THÚC MODULE 19 - TỔNG QUAN SÁU METRIC HELD-OUT ================






#=================== MODULE 20 - HIỆU SUẤT THEO TỪNG SEED ĐÁNH GIÁ ===================
# Mục đích: Vẽ kết quả thật theo seed cho các checkpoint.
# Giải thích: Các seed trùng nhau có thể dùng để so sánh có ghép cặp.

def plot_heldout_seedwise(ds):
    """Actual seed values; shared x keys are NOT a learning-curve x-axis."""
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    fig.suptitle('V15 HELD-OUT — results by random seed',fontsize=16,fontweight='bold')
    specs=[('jobDone','Products Completed','JobDone'),
           ('targetReached','Targets Reached','Targets Reached'),
           ('block','Blocks','Blocks'),
           ('routing_stop_rate','STOP with moves available','STOP / routing decision with move (%)')]
    all_seeds=sorted({int(v) for d in ds for v in d['seed']})
    loc={v:i for i,v in enumerate(all_seeds)}
    for ax,(k,title,yl) in zip(axs.flat,specs):
        offsets=np.linspace(-.16,.16,len(ds)) if len(ds)>1 else [0.]
        for pos,d in enumerate(ds):
            y=d['metric'][k];valid=np.isfinite(y)
            if valid.any():
                x=[loc[int(seed)]+offsets[pos] for seed in d['seed'][valid]]
                ax.scatter(x,y[valid],s=40,alpha=.85,label=d['label'])
        ax.set_xticks(np.arange(len(all_seeds)),[str(v) for v in all_seeds],rotation=35,ha='right',fontsize=8)
        ax.set_xlim(-.55,len(all_seeds)-.45)
        ax.set_title(title);ax.set_xlabel('Held-out environment seed (category)');ax.set_ylabel(yl)
        ax.grid(alpha=.3,linestyle='--')
        if any(np.isfinite(d['metric'][k]).any() for d in ds):ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Unavailable',transform=ax.transAxes,ha='center')
    return fig

#============== KẾT THÚC MODULE 20 - HIỆU SUẤT THEO TỪNG SEED ĐÁNH GIÁ ==============






#====================== MODULE 21 - PHÂN TÍCH STOP CỦA ROUTING ======================
# Mục đích: So sánh STOP rate và JobDone, các bộ đếm STOP có khả năng di chuyển.
# Giải thích: Không tự quy STOP là nguyên nhân gây hiệu suất thấp.

def plot_heldout_routing(ds):
    fig,axs=plt.subplots(1,3,figsize=(16.6,5.5),layout='constrained')
    fig.suptitle('V15 HELD-OUT — routing STOP diagnostics (observed, not reward inference)',fontsize=14,fontweight='bold')
    ax=axs[0]
    for d in ds:
        x=d['metric']['routing_stop_rate']; y=d['metric']['jobDone']; valid=np.isfinite(x)&np.isfinite(y)
        if valid.any():ax.scatter(x[valid],y[valid],s=40,alpha=.78,label=d['label'])
    ax.set_xlabel('STOP / decisions with move (%)');ax.set_ylabel('JobDone / seed')
    ax.set_title('JobDone vs STOP rate');ax.grid(alpha=.3)
    if ax.has_data():ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Routing STOP unavailable',transform=ax.transAxes,ha='center')
    ax=axs[1]
    xs=np.arange(len(ds));vs=[];ys=[]
    for d in ds:
        stop=d['metric']['routing_stop_with_move_available']
        move=d['metric']['routing_move_available_decisions']
        valid=np.isfinite(stop)&np.isfinite(move)&(move>=0)
        vs.append(100*float(np.sum(stop[valid]))/float(np.sum(move[valid])) if valid.any() and np.sum(move[valid])>0 else np.nan)
        ys.append(int(np.sum(valid)))
    ok=[i for i,v in enumerate(vs) if math.isfinite(v)]
    if ok:
        ax.bar([xs[i] for i in ok],[vs[i] for i in ok],alpha=.8)
        for i in ok:ax.text(xs[i],vs[i]+.6,f'{vs[i]:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(xs,[d['label'] for d in ds],rotation=20,ha='right',fontsize=8)
    ax.set_title('Pooled STOP rate (weighted)');ax.set_ylabel('STOP / decisions with move (%)')
    ax.set_ylim(0,max(105,max([vs[i] for i in ok],default=0)*1.16));ax.grid(axis='y',alpha=.3)
    ax=axs[2]
    for d in ds:
        x=d['metric']['routing_stop_with_move_available'];y=d['metric']['selected_mask_violations']
        valid=np.isfinite(x)&np.isfinite(y)
        if valid.any():ax.scatter(x[valid],y[valid],s=40,alpha=.78,label=d['label'])
    ax.set_xlabel('STOP with move available / seed');ax.set_ylabel('Selected mask violations / seed')
    ax.set_title('STOP ≠ action-mask violation');ax.grid(alpha=.3)
    if ax.has_data():ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Diagnostics unavailable',transform=ax.transAxes,ha='center')
    return fig

#================== KẾT THÚC MODULE 21 - PHÂN TÍCH STOP CỦA ROUTING ==================






#==================== MODULE 22 - TÌM CÁC SEED PHÙ HỢP ĐỂ SO SÁNH ====================
# Mục đích: Chỉ ghép cặp nếu phase, số AGV, thời gian và seed tương ứng.
# Giải thích: Tránh suy luận từ hai tập seed khác nhau.

def matched_seed_pairs(ds):
    """Only compare policies in the same phase, fleet, horizon, and seed IDs."""
    if len(ds)!=2:return None
    a,b=ds
    if (a['phase']!=b['phase'] or a['fleet']!=b['fleet'] or
            a['sim_seconds']!=b['sim_seconds']):return None
    sa=set(map(int,a['seed']));sb=set(map(int,b['seed']))
    overlap=sorted(sa&sb)
    if not overlap:return None
    ia={int(s):i for i,s in enumerate(a['seed'])};ib={int(s):i for i,s in enumerate(b['seed'])}
    return [(s,ia[s],ib[s]) for s in overlap]

#=============== KẾT THÚC MODULE 22 - TÌM CÁC SEED PHÙ HỢP ĐỂ SO SÁNH ===============






#=============== MODULE 23 - VẼ CHÊNH LỆCH CHỈ SỐ GIỮA HAI CHECKPOINT ===============
# Mục đích: Chênh lệch đo trên cùng seed: second - first.
# Giải thích: Thể hiện biến thiên JobDone, Block, Target, STOP rate.

def plot_heldout_paired(ds):
    matched=matched_seed_pairs(ds)
    if not matched:return None
    a,b=ds
    fig,axs=plt.subplots(2,2,figsize=(14.4,9),layout='constrained')
    fig.suptitle(f'Matched held-out seeds: {a["label"]} vs {b["label"]}',fontsize=14,fontweight='bold')
    spec=[('jobDone','JobDone','Δ JobDone (second - first)'),
          ('block','Blocks','Δ Block (second - first)'),
          ('targetReached','TargetReached','Δ targets (second - first)'),
          ('routing_stop_rate','Routing STOP rate','Δ STOP rate (pp, second - first)')]
    for ax,(key,title,ytitle) in zip(axs.flat,spec):
        seeds=[];deltas=[]
        for s,i,j in matched:
            x=a['metric'][key][i];y=b['metric'][key][j]
            if math.isfinite(x) and math.isfinite(y):seeds.append(s);deltas.append(y-x)
        ax.axhline(0,color='gray',lw=1)
        if seeds:
            ax.bar(np.arange(len(seeds)),deltas,alpha=.8)
            ax.set_xticks(np.arange(len(seeds)),[str(s) for s in seeds],rotation=35,ha='right',fontsize=8)
            ax.set_title(f'{title} | mean Δ={np.mean(deltas):+.2f} | n={len(seeds)}')
        else:
            ax.set_title(f'{title} | unavailable')
        ax.set_xlabel('Matched seed');ax.set_ylabel(ytitle);ax.grid(axis='y',alpha=.3)
    return fig

#=========== KẾT THÚC MODULE 23 - VẼ CHÊNH LỆCH CHỈ SỐ GIỮA HAI CHECKPOINT ===========






#====================== MODULE 24 - SO SÁNH THEO QUY MÔ ĐỘI AGV ======================
# Mục đích: Hiệu suất trung bình/độ lệch chuẩn theo số AGV.
# Giải thích: Chỉ vẽ nếu có nhiều fleet size.

def plot_heldout_fleet(ds):
    if len({d['fleet'] for d in ds})<2:return None
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fig.suptitle('V15 HELD-OUT — fleet-size comparison (Figure 8 inspired)',fontsize=15,fontweight='bold')
    specs=[('targetReached','Targets Reached / AGV','Targets / AGV',True),
           ('jobDone','Products Completed','JobDone',False),
           ('block','Blocks','Blocks',False),
           ('liveLock','LiveLock (NOT lead time)','LiveLock',False)]
    for ax,(key,title,ylabel,per) in zip(axs.flat,specs):
        for d in sorted(ds,key=lambda d:(d['fleet'],d['episode'])):
            vec=d['metric'][key] / d['fleet'] if per else d['metric'][key]
            m,s,n=mean_sd(vec)
            if n:ax.errorbar(d['fleet'],m,yerr=s if n>1 else 0,fmt='o',capsize=4,label=d['label'])
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('Fleet Size (AGVs)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.3)
        if ax.has_data():ax.legend(fontsize=8)
    return fig

#================= KẾT THÚC MODULE 24 - SO SÁNH THEO QUY MÔ ĐỘI AGV =================






#===================== MODULE 25 - XUẤT BÁO CÁO HELD-OUT VÀ CSV =====================
# Mục đích: Ghi thống kê mỗi seed và tổng hợp checkpoint vào báo cáo riêng.
# Giải thích: Không khẳng định đã tái lập hoàn toàn kết quả paper Tobias.

def summarize_heldout(ds,out):
    report=['V15 HELD-OUT EVALUATION — LOCAL REPORT',
            'ONE row = ONE held-out environment seed. NOT a training curve.',
            'Values are descriptive; no matched Tobias baseline/lead-time inference.',
            'STOP rate = routing_stop_with_move_available / routing_move_available_decisions.',
            'Distinct seed samples are not paired, even if row indices match.','-'*78]
    columns=['source_file','phase','checkpoint_episode','checkpoint_sha256','fleet','seed','sim_seconds',
             'greedy','ppo_updates',*HELDOUT_KEYS]
    with (out/'v15_heldout_per_seed.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp);w.writerow(columns)
        for d in ds:
            for idx,seed in enumerate(d['seed']):
                w.writerow([str(d['file']),d['phase'],d['episode'],d['checkpoint'],d['fleet'],
                            int(seed),d['sim_seconds'],d['greedy'],d['ppo_updates'],
                            *[d['metric'][k][idx] if math.isfinite(d['metric'][k][idx]) else '' for k in HELDOUT_KEYS]])
    with (out/'v15_heldout_summary.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp)
        w.writerow(['source_file','phase','episode','checkpoint_sha256','fleet','n_seeds','sim_seconds',
                    *[f'{k}_{t}' for k in HELDOUT_KEYS for t in ('mean','sd','n')], 'pooled_stop_rate_pct'])
        for d in ds:
            metrics=[v for k in HELDOUT_KEYS for v in mean_sd(d['metric'][k])]
            stop=d['metric']['routing_stop_with_move_available'];move=d['metric']['routing_move_available_decisions']
            valid=np.isfinite(stop)&np.isfinite(move)&(move>=0)
            pooled=100*sum(stop[valid])/sum(move[valid]) if valid.any() and sum(move[valid])>0 else float('nan')
            w.writerow([str(d['file']),d['phase'],d['episode'],d['checkpoint'],d['fleet'],len(d['seed']),
                        d['sim_seconds'],*metrics,pooled])
            report.extend(['',f'{d["label"]} | sha256={d["checkpoint"]}',
                           f'  seeds: {len(d["seed"])} | range {min(d["seed"])}..{max(d["seed"])} | horizon={d["sim_seconds"]}s',
                           f'  end reasons: {dict(d["end_reasons"])}',
                           f'  stop mask modes: {sorted(d["stop_mode"])}'])
            if d['bad_lines']:report.append(f'  WARNING malformed/skipped input lines: {d["bad_lines"][:12]}')
            for key in HELDOUT_KEYS:
                m,s,n=mean_sd(d['metric'][key])
                if n:report.append(f'  {key:35s} mean={fmt(m):>10s} sd={fmt(s):>10s} n={n}')
            report.append(f'  STOP pooled % (correct denominator): {fmt(pooled)}')
            bad_seeds=[int(d['seed'][i]) for i,v in enumerate(d['metric']['selected_mask_violations']) if math.isfinite(v) and v>0]
            if bad_seeds:report.append(f'  WARNING mask violations in seeds: {bad_seeds}')
    pair=matched_seed_pairs(ds)
    if pair:
        a,b=ds;report.extend(['','PAIRED COMPARISON: second file MINUS first file (matched seeds only)',
                               f'  first={a["label"]} | second={b["label"]}',
                               f'  overlap={len(pair)}/{len(a["seed"])}/{len(b["seed"])}'])
        for k in ('jobDone','targetReached','block','liveLock','wrongTarget','delivered_reward','routing_stop_rate'):
            differences=[b['metric'][k][j]-a['metric'][k][i] for _,i,j in pair
                         if math.isfinite(a['metric'][k][i]) and math.isfinite(b['metric'][k][j])]
            m,s,n=mean_sd(differences)
            if n:report.append(f'  Δ {k:30s} mean={m:+.3f}, SD={s:.3f}, n={n}')
    else:
        report.append('\nNo paired comparison: requires exactly TWO held-out checkpoints, same phase/fleet/horizon, and shared seeds.')
    report.append('\nCaution: low Block may coincide with excess STOP; low Block alone is not proof of throughput improvement.')
    path=out/'v15_heldout_report.txt'
    path.write_text('\n'.join(report)+'\n',encoding='utf-8')
    return path

#================= KẾT THÚC MODULE 25 - XUẤT BÁO CÁO HELD-OUT VÀ CSV =================






#================== MODULE 26 - ĐIỀU PHỐI TẠO HÌNH VÀ FILE BÁO CÁO ==================
# Mục đích: Phân loại các input training/held-out, gọi đúng bộ hàm vẽ.
# Giải thích: Tạo thư mục kết quả mới để tránh ghi đè biểu đồ cũ.

def generate(paths, window=10, use_events=False, out_dir=None):
    if window < 1:raise ValueError('Smooth/window must be >= 1')

    #===================== PHẦN GENERATE-01 - PHÂN LOẠI FILE ĐẦU VÀO =====================
    # Không trộn learning curve với held-out checkpoint.

    inputs=[Path(x).expanduser().resolve() for x in paths]
    if not inputs:raise ValueError('Chưa chọn file JSONL')
    if len({str(p) for p in inputs})!=len(inputs):
        raise ValueError('Cùng một file xuất hiện hai lần; hãy xóa dòng trùng.')

    #===================== PHẦN GENERATE-02 - ĐỌC DỮ LIỆU THEO LOẠI =====================
    # Chọn read_heldout hoặc read_log.

    for p in inputs:
        if not p.is_file():raise FileNotFoundError(f'File not found: {p}')
    training=[];heldout=[]
    for p in inputs:
        if inspect_log_kind(p)=='heldout':heldout.append(read_heldout(p))
        else:training.append(read_log(p))
    if heldout:dataset_labels(heldout)
    if training:
        seen=Counter(d['label'] for d in training)
        for d in training:
            if seen[d['label']]>1:d['label']+=' | '+d['file'].parent.name
    if out_dir:
        folder=Path(out_dir).expanduser().resolve()
        folder.mkdir(parents=True,exist_ok=True)
    else:
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        parent=inputs[0].parent
        folder=parent/f'V15_PLOTS_{stamp}'
        c=2
        while folder.exists():
            folder=parent/f'V15_PLOTS_{stamp}_{c}';c+=1
        folder.mkdir(parents=True)
    figures=[];reports=[]
    if training:
        figures.append(('Training Curves',plot_training(training,window,use_events)))
        figures.append(('Training Diagnostics',plot_diagnostics(training,window,use_events)))
        fig=plot_fleet_compare(training,window)
        if fig is not None:figures.append(('Training Fleet Comparison',fig))
        reports.append(summary_and_csv(training,window,folder))
    if heldout:
        figures.append(('Heldout Overview',plot_heldout_overview(heldout)))
        figures.append(('Heldout By Seed',plot_heldout_seedwise(heldout)))
        figures.append(('Heldout STOP Diagnostics',plot_heldout_routing(heldout)))
        fig=plot_heldout_paired(heldout)
        if fig is not None:figures.append(('Heldout Paired Difference',fig))
        fig=plot_heldout_fleet(heldout)
        if fig is not None:figures.append(('Heldout Fleet Comparison',fig))
        reports.append(summarize_heldout(heldout,folder))
    for n,(title,fig) in enumerate(figures,1):
        slug=re.sub(r'[^a-z0-9]+','_',title.lower()).strip('_')
        fig.savefig(folder/f'figure_{n:02d}_{slug}.png',dpi=180,bbox_inches='tight')
        fig.savefig(folder/f'figure_{n:02d}_{slug}.svg',bbox_inches='tight')
    return folder,figures,reports

#============== KẾT THÚC MODULE 26 - ĐIỀU PHỐI TẠO HÌNH VÀ FILE BÁO CÁO ==============






#========================= MODULE 27 - TẠO GIAO DIỆN TKINTER =========================
# Mục đích: Khung nhập đường dẫn, chọn file, hiển thị hình và log trạng thái.
# Giải thích: Bọc các thao tác người dùng trong callback giao diện.

def run_gui():
    import tkinter as tk
    from tkinter import filedialog,messagebox,ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk

    #========================= PHẦN RUN_GUI-01 - KHỞI TẠO CỬA SỔ =========================
    # Tạo nút chọn file, các ô nhập và thanh trạng thái.

    root=tk.Tk();root.title('V15 Graph Plotter — Training + Held-out')
    root.geometry('1000x570');root.minsize(820,460)
    frame=ttk.Frame(root,padding=16);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='V15 | VẼ ĐỒ THỊ TRAINING VÀ HELD-OUT',font=('Segoe UI',16,'bold')).pack(anchor='w')
    ttk.Label(frame,text='Dán đường dẫn metrics.jsonl hoặc heldout_*.jsonl (mỗi dòng một file). Có thể chọn nhiều checkpoint để so sánh.',
              foreground='#666666').pack(anchor='w',pady=(5,8))
    tb=tk.Text(frame,height=10,wrap='none',font=('Consolas',10));tb.pack(fill='both',expand=True)
    status=tk.StringVar(value='Chỉ đọc log; không sửa checkpoint, không sửa kết quả cũ.')
    def choose():
        fs=filedialog.askopenfilenames(title='Chọn file training/held-out JSONL',
                                        filetypes=[('JSONL logs','*.jsonl'),('All files','*.*')])
        if fs:
            cur=tb.get('1.0','end').strip()
            tb.insert('end',('\n' if cur else '')+'\n'.join(fs))
            status.set(f'Đã thêm {len(fs)} file.')
    def draw():
        paths=[s.strip().strip('"').strip("'") for s in tb.get('1.0','end').splitlines() if s.strip()]
        try:
            window=int(wvar.get().strip())
            if not 1<=window<=1000:raise ValueError('Smooth must be 1–1000')
            output,figures,reports=generate(paths,window,use_events=(xvar.get()=='Training Steps'))
        except Exception as e:
            messagebox.showerror('Lỗi đọc/vẽ log',str(e),parent=root)
            status.set('Lỗi: '+str(e));return
        status.set(f'Đã xuất {len(figures)} biểu đồ → {output}')
        popup=tk.Toplevel(root);popup.title('V15 Charts — '+output.name)
        popup.geometry('1300x850')
        notebook=ttk.Notebook(popup);notebook.pack(fill='both',expand=True)
        for title,fig in figures:
            page=ttk.Frame(notebook);notebook.add(page,text=title)
            canvas=FigureCanvasTkAgg(fig,master=page)
            canvas.draw();canvas.get_tk_widget().pack(fill='both',expand=True)
            toolbar=NavigationToolbar2Tk(canvas,page,pack_toolbar=False)
            toolbar.update();toolbar.pack(side='bottom',fill='x')
        footer=ttk.Frame(popup,padding=(9,4));footer.pack(fill='x')
        ttk.Label(footer,text=f'PNG + SVG + CSV + TXT: {output}',wraplength=1100).pack(side='left')
        def browse():
            if sys.platform=='win32':os.startfile(str(output))
            else:
                import subprocess
                subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',str(output)])
        ttk.Button(footer,text='Mở thư mục kết quả',command=browse).pack(side='right')
        def finish():
            for _,fig in figures:plt.close(fig)
            popup.destroy()
        popup.protocol('WM_DELETE_WINDOW',finish)
    ctr=ttk.Frame(frame);ctr.pack(fill='x',pady=(12,8))
    ttk.Button(ctr,text='Chọn file JSONL...',command=choose).pack(side='left',padx=(0,8))
    ttk.Button(ctr,text='Xóa danh sách',command=lambda:tb.delete('1.0','end')).pack(side='left',padx=(0,18))
    ttk.Label(ctr,text='Smooth (training):').pack(side='left')
    wvar=tk.StringVar(value='10')
    ttk.Spinbox(ctr,from_=1,to=200,textvariable=wvar,width=5).pack(side='left',padx=6)
    ttk.Label(ctr,text='Training trục X:').pack(side='left',padx=(14,5))
    xvar=tk.StringVar(value='Training Steps')
    ttk.Combobox(ctr,textvariable=xvar,values=['Episode','Training Steps'],state='readonly',width=18).pack(side='left')
    ttk.Button(ctr,text='VẼ ĐỒ THỊ',command=draw).pack(side='right')
    ttk.Label(frame,textvariable=status,wraplength=950).pack(fill='x',anchor='w')
    ttk.Label(frame,text='Held-out: mean ± sample SD, từng seed, STOP và paired EP80/EP200. Smooth và trục X chỉ áp dụng cho training.',
              foreground='#666666',wraplength=950).pack(anchor='w',pady=(8,0))
    root.mainloop()

#==================== KẾT THÚC MODULE 27 - TẠO GIAO DIỆN TKINTER ====================






#=================== MODULE 28 - CHẾ ĐỘ DÒNG LỆNH HOẶC CỬA SỔ GUI ===================
# Mục đích: Nếu có --input/--no-gui thì chạy headless, nếu không mở cửa sổ.
# Giải thích: Chỉ đọc log, không đụng checkpoint/training.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',default=[],help='JSONL path, repeat for comparisons')
    p.add_argument('--window',type=int,default=10,help='smoothing window for TRAINING only')
    p.add_argument('--xaxis',choices=['episode','events'],default='events',help='TRAINING x-axis only (default = cumulative Training Steps)')
    p.add_argument('--out-dir',help='optional custom output folder; default creates a new timestamped folder')
    p.add_argument('--no-gui',action='store_true')
    args=p.parse_args()
    if args.input or args.no_gui:
        if not args.input:p.error('--no-gui requires at least one --input PATH')
        folder,figures,reports=generate(args.input,args.window,args.xaxis=='events',args.out_dir)
        for _,fig in figures:plt.close(fig)
        print(f'Vẽ xong {len(figures)} biểu đồ: {folder}')
        for report in reports:print(f'Báo cáo: {report}')
    else:run_gui()

#=============== KẾT THÚC MODULE 28 - CHẾ ĐỘ DÒNG LỆNH HOẶC CỬA SỔ GUI ===============






#========================= MODULE 29 - ĐIỂM VÀO GUI/HEADLESS =========================
# Mục đích: Chỉ gọi main() khi chạy file trực tiếp.
# Giải thích: Cho phép import các hàm vẽ vào script khác.

if __name__=='__main__':main()

#==================== KẾT THÚC MODULE 29 - ĐIỂM VÀO GUI/HEADLESS ====================




