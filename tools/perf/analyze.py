#!/usr/bin/env python3
"""Turns a performance recording (runtime/app/perf.h: perf/<date-time> beside the game) into report.html in that folder.

    python3 sdk/tools/perf/analyze.py build/macapp/perf/20261108-201500     a recording
    python3 sdk/tools/perf/analyze.py build/macapp/perf                     the newest in a folder of them

What limited each frame is judged from what was busy for most of its interval: the GPU (its command buffers' time), the
thread that draws and presents (its CPU time), or another thread; a frame that mostly waited in the pacer had time to
spare. Standard library only."""
import csv
import html
import os
import statistics
import sys
from collections import defaultdict


def load(folder):
    with open(os.path.join(folder, "frames.csv"), newline="") as f:
        frames = [r for r in csv.DictReader(f) if r.get("marker") is not None]  # whole lines only
    threads = defaultdict(dict)  # frame -> {name: ms}
    path = os.path.join(folder, "threads.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("cpu_ms"):
                    continue  # a line cut short (the game stopped mid-write)
                d = threads[int(r["frame"])]
                d[r["name"]] = d.get(r["name"], 0.0) + float(r["cpu_ms"])
    events = []
    path = os.path.join(folder, "events.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            events = [r for r in csv.DictReader(f)]
    info = {}
    path = os.path.join(folder, "info.txt")
    if os.path.exists(path):
        for line in open(path):
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return frames, threads, events, info


def num(r, k):
    try:
        return float(r.get(k) or 0)
    except ValueError:
        return 0.0


def work(r):
    """The frame's time without its waits for the display: what headroom a frame at the target had (older recordings: worked out)."""
    return max(0.0, num(r, "interval_ms") - num(r, "pace_wait_ms") - num(r, "drawable_wait_ms") - num(r, "inflight_wait_ms"))


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def limiter(r, cpu, presenter):
    """What held the frame: the busiest of the GPU and the threads, if it filled most of the interval."""
    iv = num(r, "interval_ms")
    if iv <= 0:
        return "first frame"
    busy = {"GPU": num(r, "gpu_ms"), "drawing thread": cpu.get(presenter, 0.0)}
    others = {k: v for k, v in cpu.items() if k != presenter}
    if others:
        k = max(others, key=others.get)
        busy["thread " + k] = others[k]
    what = max(busy, key=busy.get)
    if busy[what] >= 0.8 * iv:
        return what
    if num(r, "pace_wait_ms") >= 0.2 * iv:
        return "spare time (paced)"
    # two threads that together fill the frame: one waits for the other each frame (they take turns, not overlap)
    if others:
        k = max(others, key=others.get)
        if busy["drawing thread"] + others[k] >= 0.85 * iv:
            return "drawing thread and " + k + " in turn"
    if num(r, "report_wait_ms") >= 0.3 * iv:
        return "drawing thread waiting for GPU reports"
    if num(r, "kernel_wait_ms") >= 0.3 * iv:
        return "drawing thread waiting on another thread"
    if num(r, "drawable_wait_ms") + num(r, "inflight_wait_ms") >= 0.3 * iv:
        return "waiting on GPU or display"
    if num(r, "shaders_ms") + num(r, "pipelines_ms") + num(r, "uploads_ms") + num(r, "readback_ms") >= 0.3 * iv:
        return "one-off costs (shaders, pipelines, uploads)"
    # time inside draws that was not the drawing thread's CPU: it waited there (a lock, a sync with the GPU, a compile
    # on another thread)
    if num(r, "draw_cpu_ms") - busy["drawing thread"] >= 0.3 * iv:
        return "drawing thread blocked inside draws"
    return "unclear"


def svg_lines(series, width=1100, height=260, ymax=None, hline=None, marks=(), labels=("", "")):
    """series: [(name, colour var, [(x, y)])]; one chart, shared axes."""
    xs = [x for _, _, pts in series for x, _ in pts]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs) or 1
    ys = [y for _, _, pts in series for _, y in pts]
    ymax = ymax or (max(ys) if ys else 1) or 1
    pad_l, pad_b, pad_t = 48, 22, 8
    w, h = width - pad_l - 8, height - pad_b - pad_t

    def X(x):
        return pad_l + (x - x0) / ((x1 - x0) or 1) * w

    def Y(y):
        return pad_t + h - min(y, ymax) / ymax * h

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(5):
        v = ymax * i / 4
        out.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5
        out.append(f'<text x="{X(v):.1f}" y="{height - 6}" class="tick" text-anchor="middle">{v:.0f}</text>')
    if hline:
        out.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{Y(hline):.1f}" y2="{Y(hline):.1f}" class="target"/>')
    for x, text in marks:
        out.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{pad_t}" y2="{pad_t + h}" class="mark"><title>{html.escape(text)}</title></line>')
    for name, colour, pts in series:
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="var({colour})" stroke-width="1.3"><title>{html.escape(name)}</title></polyline>')
    out.append(f'<text x="{pad_l}" y="{pad_t + 10}" class="tick">{html.escape(labels[1])}</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="key"><i style="background:var({c})"></i>{html.escape(n)}</span>' for n, c, _ in series)
    return f'<div class="legend">{legend}</div>' + "".join(out) + f'<div class="axis">{html.escape(labels[0])}</div>'


def buckets(rows, key, n=1200, how=max):
    """(seconds, value) for a chart: at most n points, each the max (or mean) of its frames."""
    if not rows:
        return []
    size = max(1, len(rows) // n)
    out = []
    for i in range(0, len(rows), size):
        part = rows[i:i + size]
        vals = [key(r) for r in part]
        out.append((num(part[0], "t_ms") / 1000.0, how(vals) if how is not None else sum(vals) / len(vals)))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def table(head, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = sys.argv[1]
    if not os.path.exists(os.path.join(folder, "frames.csv")):
        subs = sorted(d for d in os.listdir(folder) if os.path.exists(os.path.join(folder, d, "frames.csv")))
        if not subs:
            print("no recording in", folder)
            return 1
        folder = os.path.join(folder, subs[-1])
    frames, threads, events, info = load(folder)
    rows = [r for r in frames if num(r, "interval_ms") > 0]
    if not rows:
        print("no frames recorded in", folder)
        return 1
    presenter = info.get("presenting thread", "")
    passes = [k[len("draws_"):] for k in frames[0].keys() if k.startswith("draws_")]
    targets = [num(r, "target_ms") for r in rows if num(r, "target_ms") > 0]
    target = statistics.median(targets) if targets else 1000 / 60
    slow_at = 1.5 * target
    ivs = [num(r, "interval_ms") for r in rows]
    dur = (num(rows[-1], "t_ms") - num(rows[0], "t_ms")) / 1000 or 1
    for r in rows:
        r["_cpu"] = threads.get(int(r["frame"]), {})
        r["_lim"] = limiter(r, r["_cpu"], presenter)
    slow = [r for r in rows if num(r, "interval_ms") > slow_at]
    markers = [(num(e, "t_ms") / 1000, e["event"]) for e in events if e.get("event")]

    # what limited frames
    def shares(rs):
        c = defaultdict(int)
        for r in rs:
            c[r["_lim"]] += 1
        return sorted(c.items(), key=lambda kv: -kv[1])

    lim_rows = []
    all_s, slow_s = dict(shares(rows)), dict(shares(slow))
    for k in sorted(set(all_s) | set(slow_s), key=lambda k: -slow_s.get(k, 0) - all_s.get(k, 0) / 1e6):
        lim_rows.append([k, f"{100 * all_s.get(k, 0) / len(rows):.1f}%", f"{100 * slow_s.get(k, 0) / len(slow):.1f}%" if slow else "-"])

    # threads
    names = sorted({n for r in rows for n in r["_cpu"]})
    th_rows = []
    for n in names:
        per = [r["_cpu"].get(n, 0.0) for r in rows]
        per_slow = [r["_cpu"].get(n, 0.0) for r in slow]
        th_rows.append([n + (" (draws, presents)" if n == presenter else ""), f"{mean(per):.2f}", f"{mean(per_slow):.2f}" if slow else "-",
                        f"{max(per):.1f}", f"{100 * sum(per) / 1000 / dur:.0f}%"])
    th_rows.sort(key=lambda r: -float(r[1]))

    # passes
    pass_rows = []
    for p in passes:
        d = [num(r, "draws_" + p) for r in rows]
        t = [num(r, "tris_" + p) for r in rows]
        pass_rows.append([p, f"{mean(d):.0f}", f"{mean([num(r, 'draws_' + p) for r in slow]):.0f}" if slow else "-", f"{max(d):.0f}",
                          f"{mean(t) / 1000:.0f}k"])

    # GPU time by pass (vertex + fragment stages; they can overlap, so these can add up to more than busy time)
    gpu_rows = []
    for p in passes:
        v, f_ = [num(r, "gpuv_" + p) for r in rows], [num(r, "gpuf_" + p) for r in rows]
        gpu_rows.append([p, f"{mean(v):.2f}", f"{mean(f_):.2f}", f"{mean([num(r, 'gpuv_' + p) + num(r, 'gpuf_' + p) for r in slow]):.2f}" if slow else "-",
                         f"{max(a + b for a, b in zip(v, f_)):.1f}"])
    gpu_rows.sort(key=lambda r: -(float(r[1]) + float(r[2])))
    kinds = [k[len("kind_"):] for k in frames[0].keys() if k.startswith("kind_")]
    kind_rows = [[k, f"{mean([num(r, 'kind_' + k) for r in rows]):.0f}", f"{mean([num(r, 'kind_' + k) for r in slow]):.0f}" if slow else "-"] for k in kinds]

    # the drawing thread's stages, the counters, the game's functions
    stage_keys = [k for k in frames[0] if k.startswith("stage_") and k.endswith("_ms")]
    stage_rows = []
    all_draws = mean([num(r, "draws") for r in rows]) or 1
    for k in sorted(stage_keys, key=lambda k: -mean([num(r, k) for r in rows])):
        m = mean([num(r, k) for r in rows])
        stage_rows.append([k[6:-3], f"{m:.2f}", f"{mean([num(r, k) for r in slow]):.2f}" if slow else "-", f"{m * 1000 / all_draws:.2f}"])
    count_rows = []
    for k in [k for k in frames[0] if k.startswith("n_")]:
        m = mean([num(r, k) for r in rows])
        count_rows.append([k[2:], f"{m:,.0f}", f"{mean([num(r, k) for r in slow]):,.0f}" if slow else "-", f"{m / all_draws:.2f}"])
    zone_rows = []
    for k in [k for k in frames[0] if k.startswith("zone_") and k.endswith("_ms")]:
        name = k[5:-3]
        zone_rows.append([name, f"{mean([num(r, k) for r in rows]):.2f}", f"{mean([num(r, k) for r in slow]):.2f}" if slow else "-", f"{mean([num(r, 'zone_' + name + '_calls') for r in rows]):.1f}"])
    zone_rows.sort(key=lambda r: -float(r[1]))
    core_rows, sys_rows = [], []
    for col, label in (("core_drawing", "drawing thread"), ("core_game", "game thread")):
        if col in frames[0]:
            c = defaultdict(int)
            for r in rows:
                c[int(num(r, col))] += 1
            core_rows.append([label, ", ".join(f"{k}: {100 * v / len(rows):.0f}%" for k, v in sorted(c.items()))])
    for col, label in (("mem_mb", "process memory (MB)"), ("gpu_mem_mb", "GPU allocations (MB)"), ("textures", "live textures"), ("targets", "render targets"), ("ring_kb", "frame ring used (KB)"), ("thermal", "thermal state (0 nominal, 3 critical)")):
        if col in frames[0]:
            vs = [num(r, col) for r in rows]
            sys_rows.append([label, f"{mean(vs):.1f}", f"{max(vs):.1f}"])

    # each GPU encoder (one frame in ten), and each screen-space draw alone (one frame in 120)
    FORMATS = {80: "BGRA8", 81: "BGRA8 sRGB", 70: "RGBA8", 252: "Depth32F", 260: "D32 S8", 40: "RGB565", 250: "D16"}
    KIND = ["draw", "clear", "present", "copy"]
    LOAD = ["dont care", "load", "clear"]
    STORE = ["dont care", "store", "resolve", "store+resolve", "unknown"]
    enc_rows_out, probe_rows_out, other_gpu = [], [], []
    path = os.path.join(folder, "encoders.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            enc = [r for r in csv.DictReader(f) if r.get("fragment_ms") not in (None, "")]
        normal = [e for e in enc if e["probe"] == "0"]
        n_frames = len({e["frame"] for e in normal}) or 1
        group = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
        for e in normal:
            pname = passes[int(e["pass"])] if int(e["pass"]) < len(passes) else e["pass"]
            depth = "x".join((e["depth_width"], e["depth_height"])) if e["depth_width"] != "0" else "none"
            actions = "C " + LOAD[int(e["colour_load"])] + "/" + STORE[int(e["colour_store"])] + (", D " + LOAD[int(e["depth_load"])] + "/" + STORE[int(e["depth_store"])] if depth != "none" else "")
            key = (KIND[int(e["kind"])], pname, e["width"] + "x" + e["height"], FORMATS.get(int(e["format"]), e["format"]), depth, actions)
            g = group[key]
            g[0] += float(e["vertex_ms"]); g[1] += float(e["fragment_ms"]); g[2] += 1; g[3] += int(e["draws"]); g[4] += int(e["vertices"])
        for key, g in sorted(group.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:30]:
            enc_rows_out.append(list(key) + [f"{g[2] / n_frames:.1f}", f"{g[3] / n_frames:.0f}", f"{g[0] / n_frames:.2f}", f"{g[1] / n_frames:.2f}"])
        probes = [e for e in enc if e["probe"] == "1" and e["flat"] == "1" and e["kind"] == "0"]
        n_probe = len({e["frame"] for e in enc if e["probe"] == "1"}) or 1
        pg = defaultdict(lambda: [0.0, 0.0, 0, 0])
        for e in probes:
            key = (e["first_ps"], e["width"] + "x" + e["height"], (e["tex_width"] + "x" + e["tex_height"]) if e["tex_width"] != "0" else "none", FORMATS.get(int(e["tex_format"]), e["tex_format"]), "yes" if e["blend"] == "1" else "no")
            g = pg[key]
            g[0] += float(e["vertex_ms"]); g[1] += float(e["fragment_ms"]); g[2] += int(e["draws"]); g[3] += int(e["vertices"])
        for key, g in sorted(pg.items(), key=lambda kv: -kv[1][1])[:25]:
            probe_rows_out.append(list(key) + [f"{g[2] / n_probe:.1f}", f"{g[3] / max(g[2], 1):.0f}", f"{g[0] / n_probe:.2f}", f"{g[1] / n_probe:.2f}"])
    for k, label in (("gpu_clear_ms", "clears"), ("gpu_present_ms", "final pass to the screen"), ("gpu_blit_ms", "copies")):
        if k in frames[0]:
            other_gpu.append([label, f"{mean([num(r, k) for r in rows]):.2f}", f"{mean([num(r, k) for r in slow]):.2f}" if slow else "-", f"{max(num(r, k) for r in rows):.1f}"])

    # by what was being played
    scenes = defaultdict(list)
    has_state = "state_city" in frames[0]
    if has_state:
        for r in rows:
            scenes[(int(num(r, "state_city")), int(num(r, "state_replay")), int(num(r, "state_view_mode")))].append(r)
    def top_pass(rs):
        best = max(passes, key=lambda p: sum(num(r, "gpuf_" + p) + num(r, "gpuv_" + p) for r in rs))
        return best
    scene_rows = []
    for (city, replay, view), rs in sorted(scenes.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 30:
            continue
        iv_s = [num(r, "interval_ms") for r in rs]
        scene_rows.append([city, "yes" if replay else "no", view, len(rs), f"{1000 / mean(iv_s):.1f}", f"{pct(iv_s, 95):.1f}", f"{100 * sum(1 for v in iv_s if v > slow_at) / len(rs):.0f}%",
                           f"{mean([num(r, 'gpu_ms') for r in rs]):.1f}", f"{mean([r['_cpu'].get(presenter, 0) for r in rs]):.1f}", f"{mean([num(r, 'draws') for r in rs]):.0f}", top_pass(rs)])

    # the frame as a whole
    def avg(k, rs):
        return f"{mean([num(r, k) for r in rs]):.2f}" if rs else "-"

    fields = [("interval_ms", "frame time (ms)"), ("gpu_ms", "GPU busy (ms)"), ("draw_cpu_ms", "CPU in draws (ms)"),
              ("report_wait_ms", "wait for GPU reports (ms)"), ("game_capture_ms", "game thread recording for the drawing thread (ms)"), ("game_wait_next_frame_ms", "game waits for drawing thread: next frame (ms)"),
              ("game_wait_flush_ms", "game waits: flush (ms)"), ("game_wait_fence_ms", "game waits: fence (ms)"), ("game_wait_resource_ms", "game waits: resource (ms)"),
              ("game_wait_readback_ms", "game waits: read-back (ms)"), ("game_wait_copy_ms", "game waits: copy (ms)"), ("kernel_wait_ms", "drawing thread's kernel waits (ms)"), ("pace_wait_ms", "pacer wait (ms)"), ("drawable_wait_ms", "drawable wait (ms)"), ("inflight_wait_ms", "wait on frames in flight (ms)"),
              ("draws", "draws"), ("triangles", "triangles"), ("shaders_ms", "shader compiles (ms)"), ("pipelines_ms", "pipeline builds (ms)"),
              ("uploads_ms", "texture uploads (ms)"), ("uploads", "texture uploads"), ("readback_ms", "read-backs (ms)")]
    others = [r for r in rows if num(r, "interval_ms") <= slow_at]
    frame_rows = [[label, avg(k, rows), avg(k, slow), avg(k, others)] for k, label in fields]
    frame_rows.insert(3, ["drawing thread CPU (ms)", f"{mean([r['_cpu'].get(presenter, 0) for r in rows]):.2f}",
                          f"{mean([r['_cpu'].get(presenter, 0) for r in slow]):.2f}" if slow else "-",
                          f"{mean([r['_cpu'].get(presenter, 0) for r in others]):.2f}"])

    # the worst frames
    worst = sorted(rows, key=lambda r: -num(r, "interval_ms"))[:max(20, len(rows) // 100)][:60]
    worst_rows = []
    for r in worst:
        cpu = r["_cpu"]
        other = max(((k, v) for k, v in cpu.items() if k != presenter), key=lambda kv: kv[1], default=("", 0))
        t = num(r, "t_ms") / 1000
        near = min(markers, key=lambda m: abs(m[0] - t), default=None)
        worst_rows.append([r["frame"], f"{t:.1f}", f"{num(r, 'interval_ms'):.1f}", f"{num(r, 'gpu_ms'):.1f}", f"{cpu.get(presenter, 0):.1f}",
                           f"{other[0]} {other[1]:.1f}", f"{num(r, 'pace_wait_ms'):.1f}", f"{num(r, 'inflight_wait_ms') + num(r, 'drawable_wait_ms'):.1f}",
                           f"{num(r, 'shaders_ms') + num(r, 'pipelines_ms') + num(r, 'uploads_ms') + num(r, 'readback_ms'):.1f}",
                           f"{num(r, 'draws'):.0f}", r["_lim"], f"{near[1]} at {near[0]:.1f}s" if near and abs(near[0] - t) < 5 else ""])

    # frame time beside frame rate
    wks = [work(r) for r in rows]
    def rate(ms):
        return f"{1000 / ms:.1f}" if ms > 0 else "-"
    time_rows = [[name, f"{f(ivs):.2f}", rate(f(ivs)), f"{f(wks):.2f}"] for name, f in (
        ("average", mean), ("median", lambda x: pct(x, 50)), ("95th percentile", lambda x: pct(x, 95)),
        ("99th percentile", lambda x: pct(x, 99)), ("99.9th percentile", lambda x: pct(x, 99.9)), ("worst", max))]
    band_rows = []
    edges = [(0, target * 1.05, f"up to {target * 1.05:.1f} ms"), (target * 1.05, target * 1.25, ""), (target * 1.25, target * 1.5, ""), (target * 1.5, target * 2, ""), (target * 2, float("inf"), "")]
    for lo, hi, label in edges:
        n, w = sum(lo < x <= hi for x in ivs), sum(lo < x <= hi for x in wks)
        band_rows.append([label or (f"{lo:.1f} to {hi:.1f} ms" if hi != float("inf") else f"over {lo:.1f} ms"), n, f"{100 * n / len(ivs):.1f}%", w, f"{100 * w / len(wks):.1f}%"])

    # charts
    timing = svg_lines([
        ("frame time (max per point)", "--c1", buckets(rows, lambda r: num(r, "interval_ms"))),
        ("frame work, without display waits (max per point)", "--c5", buckets(rows, work)),
        ("GPU busy", "--c2", buckets(rows, lambda r: num(r, "gpu_ms"), how=None)),
        ("drawing thread CPU", "--c3", buckets(rows, lambda r: r["_cpu"].get(presenter, 0.0), how=None)),
        ("CPU in draws", "--c4", buckets(rows, lambda r: num(r, "draw_cpu_ms"), how=None)),
    ], ymax=max(4 * target, pct(ivs, 99) * 1.1), hline=target, marks=markers, labels=("seconds", "ms"))
    colours = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"]
    draws_chart = svg_lines([(p, colours[i % len(colours)], buckets(rows, lambda r, p=p: num(r, "draws_" + p), how=None))
                             for i, p in enumerate(passes)], marks=markers, labels=("seconds", "draws a frame"))
    top_threads = [r[0].replace(" (draws, presents)", "") for r in th_rows[:5]]
    thread_chart = svg_lines([(n, colours[i % len(colours)], buckets(rows, lambda r, n=n: r["_cpu"].get(n, 0.0), how=None))
                              for i, n in enumerate(top_threads)], hline=target, marks=markers, labels=("seconds", "CPU ms a frame"))

    summary = [
        ["recording", os.path.basename(folder.rstrip("/"))], ["machine", info.get("machine", "")],
        ["frames", f"{len(rows)} in {dur:.0f} s, {len(rows) / dur:.1f} a second"],
        ["target", f"{target:.2f} ms ({1000 / target:.0f} a second); slow is over {slow_at:.1f} ms"],
        ["frame time", f"median {pct(ivs, 50):.1f} ms, 95th {pct(ivs, 95):.1f}, 99th {pct(ivs, 99):.1f}, worst {max(ivs):.1f}"],
        ["slow frames", f"{len(slow)} ({100 * len(slow) / len(rows):.1f}%)"],
        ["drawing thread", presenter or "(not recorded)"],
    ]
    css = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--card:#ffffff;--c1:#d1495b;--c2:#2e86ab;--c3:#e09f3e;--c4:#6a994e;
--c5:#8e5bb5;--c6:#3d5a80;--c7:#b56576;--c8:#7d8597}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe6;--muted:#9a9a93;--line:#2c2c2a;--card:#1f1f1d;
--c1:#ef6f7f;--c2:#5fb0d6;--c3:#f0b860;--c4:#8cc16f;--c5:#b48ad6;--c6:#7f9fc9;--c7:#d68a9b;--c8:#a2a8b4}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1160px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p{color:var(--muted);margin:4px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:4px 10px 4px 0;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}.scroll{overflow-x:auto}
.chart{width:100%;height:auto}.grid{stroke:var(--line)}.tick{fill:var(--muted);font-size:11px}.target{stroke:var(--muted);stroke-dasharray:4 3}
.mark{stroke:var(--c5);stroke-width:2;opacity:.6}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted)}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}.axis{text-align:center;color:var(--muted);font-size:11px}
"""
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Performance report</title><style>{css}</style>",
             "<h1>Performance report</h1><p>What limited each frame, from one recording. A frame is limited by whatever was busy for at least 80% of its time.</p>",
             '<div class="card">' + table(["", ""], summary) + "</div>",
             "<h2>What limited the frames</h2><p>Share of all frames, and of the slow ones.</p>", table(["limit", "all frames", "slow frames"], lim_rows),
             "<h2>Frame timing</h2><p>Frame time is each point's worst frame; the rest are averages. Dashed: the target. Purple: markers and events.</p>", timing,
             "<h2>Frame time and frame rate</h2><p>Frame time is from one present to the next. Work is that without the waits for the display, so it shows the headroom of frames that reached the target.</p>",
             table(["", "frame time ms", "as fps", "work ms"], time_rows),
             table(["frame time", "frames", "share", "by work", "share"], band_rows),
             "<h2>The average frame</h2>", table(["", "all", "slow", "the rest"], frame_rows),
             "<h2>Threads</h2><p>CPU time each frame, and the share of one core over the recording.</p>", table(["thread", "ms a frame", "slow frames", "worst", "of a core"], th_rows),
             thread_chart,
             "<h2>Draws by pass</h2><p>Which part of the frame the draws were for (the game's labels).</p>", table(["pass", "draws a frame", "in slow frames", "most", "triangles a frame"], pass_rows),
             draws_chart,
             "<h2>GPU time by pass</h2><p>From the GPU's timestamps, credited to the pass of the draw that opened each encoder (so the HUD on the main target counts with whatever opened it). Vertex and fragment stages overlap, so the columns can add up to more than GPU busy.</p>",
             table(["pass", "vertex ms a frame", "fragment ms a frame", "both, slow frames", "worst"], gpu_rows),
             "<h2>GPU work outside the draw passes</h2>", table(["work", "ms a frame", "slow frames", "worst"], other_gpu),
             "<h2>GPU encoders</h2><p>One frame in ten, grouped by what they draw to and how they load and store it; the costliest 30 groups, per frame.</p>",
             table(["kind", "pass", "target", "format", "depth", "load/store", "encoders", "draws", "vertex ms", "fragment ms"], enc_rows_out),
             "<h2>Screen-space draws, one at a time</h2><p>One frame in 120 gives each screen-space draw its own encoder, so its cost shows alone (a little more than in a normal frame). Grouped by pixel shader, per probed frame.</p>",
             table(["pixel shader", "target", "first texture", "texture format", "blended", "draws", "vertices a draw", "vertex ms", "fragment ms"], probe_rows_out),
             "<h2>Where the drawing thread's time goes</h2><p>Stages of each draw on the drawing thread (send_state is on the game thread, comparing state for the hand-over).</p>",
             table(["stage", "ms a frame", "slow frames", "microseconds a draw"], stage_rows),
             "<h2>The game's own functions</h2><p>Time on the game thread, inclusive, so a function's time holds what it calls (flush and world_draw run inside cube_pass, car_cubes and game_view).</p>",
             table(["function", "ms a frame", "slow frames", "calls a frame"], zone_rows),
             "<h2>Counters</h2>", table(["counter", "a frame", "slow frames", "per draw"], count_rows),
             "<h2>Cores, memory, temperature</h2><p>Which core each thread was on when last looked at, once a frame (macOS's numbering).</p>", table(["thread", "cores"], core_rows), table(["", "average", "most"], sys_rows),
             "<h2>Draws by kind</h2>", table(["kind", "draws a frame", "in slow frames"], kind_rows),
             "<h2>By scene</h2><p>City index, replay or not, and view mode; 30 frames or more.</p>",
             table(["city", "replay", "view", "frames", "fps", "95th ms", "slow", "GPU ms", "drawing ms", "draws", "heaviest GPU pass"], scene_rows),
             "<h2>The worst frames</h2>", table(["frame", "s", "ms", "GPU", "drawing thread", "busiest other thread", "paced", "GPU/display wait", "one-off", "draws", "limit", "near"], worst_rows)]
    if events:
        parts.append("<h2>Events</h2>" + table(["s", "event"], [[f"{num(e, 't_ms') / 1000:.1f}", e.get("event", "")] for e in events]))
    out = os.path.join(folder, "report.html")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
