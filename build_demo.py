#!/usr/bin/env python3
"""Build the static demo: copy the example videos, dump the captions, emit index.html.

Reads three JSONL files from a VR-finetune-VLM checkout:

  <repo>/stage0/benchmark/caption/val.jsonl          manifest + QAed ground truth
  <repo>/stage1/preds/caption/v1_val.jsonl           Qwen3.6 predictions
  <repo>/stage1/preds/caption/cosmos3-nano_val.jsonl Cosmos3-Nano predictions

and writes, next to this script:

  videos/ep<NNN>.mp4                   the example clips (self-contained demo)
  posters/ep<NNN>.jpg                  poster frame per clip, so nothing loads as a black box
  captions/ep<NNN>/{gt,qwen,cosmos}.srt
  examples.json                        the same data the page embeds
  index.html                           the presentation page (data inlined, opens via file://)

EPISODES below is the review notes: bullet -> manifest line indices (0-based,
the same index stage0/review_server.py shows in its JUDGE queue). Edit that
table and re-run to change the demo.

  python build_demo.py                      # defaults to ../VR-finetune-VLM
  python build_demo.py --repo /path/to/VR-finetune-VLM
  python build_demo.py --no-videos          # regenerate the page only
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_cosmos as ic  # noqa: E402 — the pipeline panel is generated from
                           # the real constants, so it cannot drift from the code

# --------------------------------------------------------------------------
# The review findings. `eps` are 0-based line numbers in val.jsonl.
# --------------------------------------------------------------------------
SECTIONS = [
    {
        "id": "verdict",
        "title": "Overall verdict",
        "blurb": "Head-to-head on the val split: which model's caption a human reviewer preferred.",
        "bullets": [
            {"id": "qwen-better", "tone": "qwen", "title": "Qwen is better",
             "note": "Qwen's caption is the more faithful of the two.", "eps": [30, 33]},
            {"id": "cosmos-better", "tone": "cosmos", "title": "Cosmos is better",
             "note": "Cosmos' caption is the more faithful of the two.", "eps": [29, 36]},
            {"id": "both-bad", "tone": "bad", "title": "Both are bad",
             "note": "Neither caption is usable as a label.", "eps": [28]},
        ],
    },
    {
        "id": "strength",
        "title": "Cosmos — strengths",
        "blurb": "What the chunked Cosmos3-Nano pipeline does well.",
        "bullets": [
            {"id": "hand-identity", "tone": "good", "title": "Hand identity",
             "note": "Left/right hand attribution is quite good.", "eps": [43, 45]},
            {"id": "more-detail", "tone": "good", "title": "More detail in actions",
             "note": "More clearly separated actions and more detailed descriptions — "
                     "as detailed as Qwen or better.", "eps": [43, 45, 48]},
            {"id": "no-error-accum", "tone": "good", "title": "No error accumulation",
             "note": "Long episode, no drift: a wrong chunk does not poison the ones after it.",
             "eps": [49]},
            {"id": "no-overtime", "tone": "good", "title": "No overtime",
             "note": "The chunking strategy holds — no caption runs past the end of the video.",
             "eps": [24, 27]},
        ],
    },
    {
        "id": "weakness",
        "title": "Cosmos — weaknesses",
        "blurb": "Where it still loses to Qwen, and what a prompt or post-process would have to fix.",
        "bullets": [
            {"id": "gap-timestamp", "tone": "bad", "title": "Gap / overlap in timestamps",
             "note": "The cue track is not continuous — stretches of video carry no caption.",
             "eps": [24, 47]},
            {"id": "too-many-actions", "tone": "bad", "title": "Too many actions in one caption",
             "note": "One cue packs several distinct actions instead of splitting them.",
             "eps": [37, 38]},
            {"id": "inconsistent-context", "tone": "bad", "title": "Sometimes inconsistent context",
             "note": "The described context drifts, even within a single chunk.", "eps": [31]},
        ],
    },
]

TRACKS = [("gt", "Ground truth (QAed)", "GT"),
          ("qwen", "Qwen3.6", "Qwen"),
          ("cosmos", "Cosmos3-Nano", "Cosmos")]

SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n(.*?)"
    r"(?=\n\s*\n|\Z)", re.S)


def parse_srt(text: str) -> list[dict]:
    """SRT text -> [{s, e, t}] in seconds. Tolerates ',' or '.' as the ms separator."""
    cues = []
    for m in SRT_BLOCK_RE.finditer(text or ""):
        g = m.groups()
        start = int(g[1]) * 3600 + int(g[2]) * 60 + int(g[3]) + int(g[4]) / 1000
        end = int(g[5]) * 3600 + int(g[6]) * 60 + int(g[7]) + int(g[8]) / 1000
        cues.append({"s": round(start, 3), "e": round(end, 3), "t": g[9].strip()})
    return cues


def read_jsonl_by_uid(path: Path) -> dict[str, dict]:
    return {r["episode_uid"]: r
            for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def overtime(cues: list[dict], duration: float) -> float:
    """Seconds a track runs past the end of the video. Predictions are not
    clamped to the clip, so this is a real failure mode, not a rounding wart —
    the timeline plots it rather than hiding it."""
    return round(max(0.0, max((c["e"] for c in cues), default=0.0) - duration), 2)


def coverage_gaps(cues: list[dict], duration: float) -> dict:
    """Seconds of the episode with no cue, and seconds covered by 2+ cues.

    This is what makes the 'gap / overlap timestamp' weakness measurable rather
    than a claim: sweep the cue boundaries and total the uncovered / doubly
    covered spans."""
    if not cues or duration <= 0:
        return {"gap": round(duration, 2), "overlap": 0.0}
    marks = sorted({0.0, duration} | {c["s"] for c in cues} | {c["e"] for c in cues})
    gap = overlap = 0.0
    for a, b in zip(marks, marks[1:]):
        if b <= 0 or a >= duration:
            continue
        a, b = max(a, 0.0), min(b, duration)
        n = sum(1 for c in cues if c["s"] <= a and c["e"] >= b)
        if n == 0:
            gap += b - a
        elif n > 1:
            overlap += b - a
    return {"gap": round(gap, 2), "overlap": round(overlap, 2)}


def make_poster(video: Path, duration: float, out: Path) -> bool:
    """One frame at 40% of the clip, so a not-yet-played video shows the scene
    instead of a black rectangle. Silently skipped if ffmpeg is unavailable."""
    t = max(0.1, duration * 0.4) if duration else 0.1
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
             "-vf", "scale=640:-2", "-q:v", "4", str(out)], capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def build_examples(repo: Path, copy_videos: bool) -> dict[str, dict]:
    manifest = [json.loads(l) for l in
                (repo / "stage0/benchmark/caption/val.jsonl").read_text().splitlines() if l.strip()]
    qwen = read_jsonl_by_uid(repo / "stage1/preds/caption/v1_val.jsonl")
    cosmos = read_jsonl_by_uid(repo / "stage1/preds/caption/cosmos3-nano_val.jsonl")

    wanted = sorted({i for s in SECTIONS for b in s["bullets"] for i in b["eps"]})
    vid_dir, cap_dir, pos_dir = HERE / "videos", HERE / "captions", HERE / "posters"
    for d in (vid_dir, cap_dir, pos_dir):
        d.mkdir(exist_ok=True)

    examples = {}
    for i in wanted:
        e = manifest[i]
        uid = e["episode_uid"]
        key = f"ep{i:03d}"
        srts = {"gt": e.get("caption", ""),
                "qwen": qwen.get(uid, {}).get("pred", ""),
                "cosmos": cosmos.get(uid, {}).get("pred", "")}

        out_cap = cap_dir / key
        out_cap.mkdir(exist_ok=True)
        for name, srt in srts.items():
            (out_cap / f"{name}.srt").write_text(srt)

        duration = float(e.get("duration_s") or 0)
        video_rel = f"videos/{key}.mp4"
        poster_rel = ""
        if copy_videos:
            shutil.copyfile(e["video"], vid_dir / f"{key}.mp4")
            if make_poster(vid_dir / f"{key}.mp4", duration, pos_dir / f"{key}.jpg"):
                poster_rel = f"posters/{key}.jpg"
        elif (pos_dir / f"{key}.jpg").exists():
            poster_rel = f"posters/{key}.jpg"

        tracks = {name: parse_srt(srt) for name, srt in srts.items()}
        # the timeline is drawn over `span`, not `duration`: a caption predicted
        # past the end of the clip has to stay on the axis to be visible at all
        span = max([duration] + [c["e"] for cues in tracks.values() for c in cues])
        examples[key] = {
            "key": key, "idx": i, "uid": uid, "task": e.get("task", ""),
            "instruction": e.get("instruction", ""), "duration": duration,
            "span": round(span, 3),
            "video": video_rel, "poster": poster_rel, "tracks": tracks,
            # the served Qwen path reports its own usage; the pipeline panel
            # quotes it rather than a remembered number
            "qwen_prompt_tokens": qwen.get(uid, {}).get("prompt_tokens", 0),
            "stats": {name: dict(n=len(cues), over=overtime(cues, duration),
                                 **coverage_gaps(cues, duration))
                      for name, cues in tracks.items()},
        }
        print(f"  {key}  {uid}  cues gt={len(tracks['gt'])} "
              f"qwen={len(tracks['qwen'])} cosmos={len(tracks['cosmos'])}")
    return examples


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
SVG_CSS = """
.s-box{fill:var(--panel);stroke:var(--line)}
.s-model{fill:var(--chip);stroke:var(--cosmos)}
.s-t{fill:var(--ink);font:12px ui-sans-serif,system-ui,sans-serif}
.s-tb{fill:var(--ink);font:600 12.5px ui-sans-serif,system-ui,sans-serif}
.s-m{fill:var(--muted);font:11px ui-sans-serif,system-ui,sans-serif}
.s-mono{fill:var(--muted);font:10.5px ui-monospace,SFMono-Regular,Menlo,monospace}
.s-hd{fill:var(--cosmos);font:700 11px ui-sans-serif,system-ui,sans-serif;letter-spacing:.08em}
.s-ar{stroke:var(--muted);stroke-width:1.4;fill:none;marker-end:url(#a)}
.s-ard{stroke:var(--cosmos);stroke-width:1.4;fill:none;stroke-dasharray:4 3;marker-end:url(#ac)}
.s-chunk{fill:color-mix(in srgb,var(--cosmos) 26%,transparent);
  stroke:color-mix(in srgb,var(--cosmos) 55%,transparent)}
.s-tick{fill:var(--cosmos)}
.s-strip{fill:var(--chip)}
"""

# concrete values for the standalone .svg, which has no page to inherit from
SVG_STANDALONE_VARS = """
:root{--panel:#ffffff;--ink:#16181d;--muted:#697086;--line:#e3e6ec;
      --cosmos:#0e9f6e;--chip:#eef0f5}
"""


CSS = """
:root{
  --bg:#f6f7f9; --panel:#fff; --ink:#16181d; --muted:#697086; --line:#e3e6ec;
  --qwen:#7c5cff; --cosmos:#0e9f6e; --gt:#8a93a6; --bad:#e0435a; --good:#0e9f6e;
  --chip:#eef0f5;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#0f1116; --panel:#171a21; --ink:#e8eaf0; --muted:#98a0b5; --line:#272c38;
         --qwen:#a48cff; --cosmos:#3ddc97; --gt:#8a93a6; --bad:#ff6b81; --good:#3ddc97;
         --chip:#222736; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:inherit}
.wrap{max-width:1280px;margin:0 auto;padding:0 20px 80px}

header.top{border-bottom:1px solid var(--line);background:var(--panel);margin-bottom:28px}
header.top .wrap{padding-top:30px;padding-bottom:22px}
h1{margin:0 0 6px;font-size:26px;letter-spacing:-.02em}
.sub{color:var(--muted);margin:0 0 16px}
.legend{display:flex;flex-wrap:wrap;gap:8px}
.tag{font-size:12px;padding:3px 9px;border-radius:999px;background:var(--chip);color:var(--muted)}
.tag.qwen{color:var(--qwen)} .tag.cosmos{color:var(--cosmos)} .tag.gt{color:var(--gt)}

/* "how these captions were produced" panel */
details.how{margin:22px 0 6px;border:1px solid var(--line);border-radius:12px;
  background:var(--panel);overflow:hidden}
details.how>summary{cursor:pointer;padding:13px 18px;font-size:15px;font-weight:600;
  list-style:none;display:flex;align-items:center;gap:9px}
details.how>summary::-webkit-details-marker{display:none}
details.how>summary::before{content:"▸";color:var(--muted);font-size:12px}
details.how[open]>summary{border-bottom:1px solid var(--line)}
details.how[open]>summary::before{content:"▾"}
.how-body{padding:16px 18px 20px;font-size:13.5px;line-height:1.62;max-width:900px}
.how-body p{margin:0 0 12px}
.how-body code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
  background:var(--chip);padding:1px 5px;border-radius:4px}
.how-body .lede{color:var(--muted)}
.how-body h4.step{margin:20px 0 8px;font-size:13px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);border-top:1px solid var(--line);
  padding-top:14px}
.pipe-fig{margin:0 0 18px}
svg.pipe{width:100%;height:auto;display:block}
.pipe-fig figcaption{margin-top:7px;font-size:11.5px;color:var(--muted)}
""" + SVG_CSS + """
.how-cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:4px}
@media (max-width:720px){.how-cols{grid-template-columns:1fr}}
.how-card{border:1px solid var(--line);border-left-width:3px;border-radius:9px;padding:11px 13px}
.how-card.qwen{border-left-color:var(--qwen)}
.how-card.cosmos{border-left-color:var(--cosmos)}
.how-card h4{margin:0 0 5px;font-size:13px}
.how-card.qwen h4{color:var(--qwen)} .how-card.cosmos h4{color:var(--cosmos)}
.how-card p{margin:0;color:var(--muted)}
/* the 10 sample points sit at the midpoint of each of 10 equal spans, which is
   where ffmpeg actually seeks — so draw them as ticks on the clip, not boxes */
.frames{position:relative;height:22px;margin:0 0 14px;border-radius:5px;
  background:var(--chip)}
.frames i{position:absolute;top:3px;bottom:3px;width:3px;margin-left:-1.5px;
  border-radius:2px;background:var(--cosmos)}
.chunks{margin:0 0 14px}
.chunks .ch{position:relative;height:24px;margin-bottom:4px}
/* colour-mix rather than opacity: opacity would fade the label with the bar */
.chunks .ch i{position:absolute;top:0;bottom:0;border-radius:5px;
  background:color-mix(in srgb, var(--cosmos) 26%, transparent);
  border:1px solid color-mix(in srgb, var(--cosmos) 55%, transparent);
  font-style:normal;font-size:10.5px;color:var(--ink);display:flex;
  align-items:center;justify-content:center;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ingredients{margin:0 0 12px;padding-left:20px}
.ingredients li{margin-bottom:5px}
.how-body .decode{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:11.5px;color:var(--muted);background:var(--chip);display:inline-block;
  padding:5px 10px;border-radius:6px}
.how-body .why{margin-top:16px;border-top:1px solid var(--line);padding-top:14px}

nav.jump{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line);
  padding:10px 0;margin-bottom:26px;overflow-x:auto}
nav.jump .wrap{display:flex;gap:8px;padding-bottom:0;padding-top:0;white-space:nowrap}
nav.jump a{font-size:13px;text-decoration:none;padding:5px 11px;border-radius:8px;
  background:var(--panel);border:1px solid var(--line);color:var(--muted)}
nav.jump a:hover{color:var(--ink)}

section{margin-bottom:44px}
.sec-head{margin-bottom:14px}
.sec-head h2{margin:0 0 3px;font-size:20px;letter-spacing:-.01em}
.sec-head p{margin:0;color:var(--muted);font-size:14px}

.bullet{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);
  border-radius:12px;padding:16px 18px;margin-bottom:14px}
.bullet.qwen{border-left-color:var(--qwen)}
.bullet.cosmos{border-left-color:var(--cosmos)}
.bullet.good{border-left-color:var(--good)}
.bullet.bad{border-left-color:var(--bad)}
.bullet h3{margin:0 0 4px;font-size:16px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.count{font-size:11px;font-weight:600;color:var(--muted);background:var(--chip);
  padding:2px 8px;border-radius:999px}
.bullet .note{margin:0 0 12px;color:var(--muted);font-size:14px}

.ex{border:1px solid var(--line);border-radius:10px;margin-top:11px;background:var(--bg)}
.ex-head{padding:9px 13px;font-size:13.5px;display:flex;gap:10px;align-items:center;
  flex-wrap:wrap;border-bottom:1px solid var(--line)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.uid{color:var(--muted)}
.ex-body{padding:13px;display:grid;grid-template-columns:minmax(320px,420px) 1fr;gap:18px}
@media (max-width:960px){.ex-body{grid-template-columns:1fr}}
video{width:100%;border-radius:9px;background:#000;display:block}
.instr{font-size:12.5px;color:var(--muted);margin:9px 0 0}

/* live caption strip: what each model says at the current playhead */
.live{margin-top:10px;border:1px solid var(--line);border-radius:9px;background:var(--panel);
  overflow:hidden}
.live-head{display:flex;justify-content:space-between;align-items:center;gap:8px;
  padding:6px 10px;border-bottom:1px solid var(--line);font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em}
.live-row{display:grid;grid-template-columns:74px 1fr;gap:9px;padding:7px 10px;
  border-top:1px solid var(--line);font-size:12.5px;line-height:1.45;min-height:38px}
.live-row:first-of-type{border-top:0}
.live-lab{font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  padding-top:2px}
.live-row.gt .live-lab{color:var(--gt)}
.live-row.qwen .live-lab{color:var(--qwen)}
.live-row.cosmos .live-lab{color:var(--cosmos)}
.live-txt.silent{color:var(--bad);font-style:italic}
.live-txt.idle{color:var(--muted);font-style:italic}

/* dot timeline: one time axis, one dot per caption.
   Qwen dots above the line, Cosmos dots below it. */
.tl{position:relative;display:grid;grid-template-columns:96px 1fr;gap:0 10px;
  align-items:center;padding:14px 14px 6px;border:1px solid var(--line);
  border-radius:10px;background:var(--panel);min-width:0}
.tl-lab{font-size:10.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  text-align:right}
.tl-lab.qwen{color:var(--qwen)} .tl-lab.cosmos{color:var(--cosmos)}
.tl-lab small{display:block;font-weight:400;font-size:9.5px;color:var(--muted);
  letter-spacing:0;text-transform:none}
.tl-row{position:relative;height:40px;cursor:pointer}
.tl-line{position:relative;height:10px;border-radius:5px;background:var(--muted);
  cursor:pointer}
.dot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;
  cursor:pointer;transition:transform .08s}
.tl-row.qwen .dot{background:var(--qwen)} .tl-row.cosmos .dot{background:var(--cosmos)}
.dot:hover{transform:scale(1.2)}
.dot.active{transform:scale(1.15);box-shadow:0 0 0 1.5px var(--bg),0 0 0 3px currentColor}
.tl-row.qwen .dot.active{color:var(--qwen)} .tl-row.cosmos .dot.active{color:var(--cosmos)}
/* playhead lives inside the axis (which already spans exactly the plotted
   width) and overflows up and down across both dot lanes */
.tl-line u{position:absolute;top:-44px;bottom:-44px;width:2px;background:var(--ink);
  opacity:.6;left:0;display:none;pointer-events:none}
.tl-ticks{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;padding-top:5px}
.tl-tip{position:absolute;z-index:5;max-width:330px;padding:7px 9px;border-radius:8px;
  background:var(--panel);border:1px solid var(--line);box-shadow:0 6px 20px rgba(0,0,0,.18);
  font-size:12px;line-height:1.4;pointer-events:none;display:none}
.tl-tip .ts{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10px;color:var(--muted);margin-bottom:2px}

details.full{margin-top:11px}
details.full>summary{cursor:pointer;font-size:12px;color:var(--muted);padding:4px 2px}
.tracks{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;min-width:0;margin-top:8px}
@media (max-width:760px){.tracks{grid-template-columns:1fr}}
.track{min-width:0;border:1px solid var(--line);border-radius:9px;background:var(--panel);
  display:flex;flex-direction:column}
.track h4{margin:0;padding:8px 10px;font-size:12.5px;border-bottom:1px solid var(--line);
  display:flex;justify-content:space-between;align-items:center;gap:6px}
.track.gt h4{color:var(--gt)} .track.qwen h4{color:var(--qwen)} .track.cosmos h4{color:var(--cosmos)}
.stat{font-weight:400;font-size:11px;color:var(--muted)}
.stat b{color:var(--bad);font-weight:600}
.cues{margin:0;padding:7px 6px 9px;list-style:none;max-height:330px;overflow:auto}
.cues li{padding:5px 6px;border-radius:6px;cursor:pointer;font-size:12.5px;line-height:1.45}
.cues li:hover{background:var(--chip)}
.cues li .ts{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:10.5px;color:var(--muted);margin-bottom:1px}
.cues li.active{background:var(--chip);outline:1px solid var(--line)}
.cues .empty{color:var(--muted);font-style:italic;padding:8px 6px}
.hand{font-weight:600}
"""

JS = """
const fmt = t => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return String(m).padStart(2,'0') + ':' + s.toFixed(2).padStart(5,'0');
};
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// [left hand] / [right hand] / [both hands] / [ego] are the caption's role markers
const markRoles = s => esc(s).replace(/\\[(left hand|right hand|both hands|ego)\\]/g,
                                     '<span class="hand">[$1]</span>');

const cueAt = (cues, t) => (cues || []).find(c => t >= c.s && t < c.e);
const pct = (v, d) => (100 * v / d).toFixed(3);

// One dot per caption on a shared time axis: Qwen above the line, Cosmos below.
// The dot sits at the caption's START; its full span is in the hover tooltip.
function laneHTML(k, cues, dur){
  return (cues || []).map((c, i) =>
    `<b class="dot" data-i="${i}" style="left:${pct(c.s, dur)}%"></b>`).join('');
}

function timelineHTML(ex){
  // plot over the full span (clip length, or further if a prediction overshoots)
  // so nothing falls off the axis; the bar still ends at the real clip end, so
  // a dot sitting past the bar is genuinely past the end of the video
  const d = ex.span || ex.duration, n = k => (ex.tracks[k] || []).length;
  const flag = k => {
    const st = ex.stats[k] || {};
    return (st.gap > 0.05 ? ` · gap ${st.gap}s` : '') +
           (st.overlap > 0.05 ? ` · ovl ${st.overlap}s` : '') +
           (st.over > 0.05 ? ` · +${st.over}s past end` : '');
  };
  const ticks = Array.from({length:5}, (_, i) => `<span>${fmt(d * i / 4)}</span>`).join('');
  return `<div class="tl">
    <span class="tl-lab qwen">Qwen<small>${n('qwen')} cues${flag('qwen')}</small></span>
    <div class="tl-row qwen" data-track="qwen">${laneHTML('qwen', ex.tracks.qwen, d)}</div>

    <span></span><div class="tl-line"><u></u></div>

    <span class="tl-lab cosmos">Cosmos<small>${n('cosmos')} cues${flag('cosmos')}</small></span>
    <div class="tl-row cosmos" data-track="cosmos">${laneHTML('cosmos', ex.tracks.cosmos, d)}</div>

    <span></span><div class="tl-ticks">${ticks}</div>
    <div class="tl-tip"></div>
  </div>`;
}

function renderExample(host, ex){
  host.innerHTML = `
    <div>
      <video src="${ex.video}"${ex.poster ? ` poster="${ex.poster}"` : ''}
             controls preload="none" playsinline></video>
      <div class="live">
        <div class="live-head"><span>Captions at playhead</span>
          <span class="mono live-clock">00:00.00 / ${fmt(ex.duration)}</span></div>
        ${TRACKS.map(([k,label,short]) => `<div class="live-row ${k}">
          <span class="live-lab">${short}</span>
          <span class="live-txt idle" data-live="${k}">press play</span></div>`).join('')}
      </div>
      <p class="instr">${esc(ex.instruction)}</p>
    </div>
    <div>
      ${timelineHTML(ex)}
      <details class="full"><summary>Full caption text, all three tracks</summary>
        <div class="tracks">${TRACKS.map(([k,label]) => {
          const cues = ex.tracks[k] || [], st = ex.stats[k] || {};
          const flags = [`${st.n} cues`];
          if (st.gap > 0.05) flags.push(`gap <b>${st.gap}s</b>`);
          if (st.overlap > 0.05) flags.push(`overlap <b>${st.overlap}s</b>`);
          return `<div class="track ${k}">
            <h4>${label}<span class="stat">${flags.join(' · ')}</span></h4>
            <ul class="cues" data-track="${k}">${
              cues.length ? cues.map(c =>
                `<li data-s="${c.s}"><span class="ts">${fmt(c.s)} → ${fmt(c.e)}</span>${
                   markRoles(c.t)}</li>`).join('')
              : '<li class="empty">no caption</li>'}</ul>
          </div>`;
        }).join('')}</div>
      </details>
    </div>`;

  const video = host.querySelector('video');
  const clock = host.querySelector('.live-clock');
  const seek = t => { video.currentTime = t; video.play(); };

  host.querySelectorAll('.cues li[data-s]').forEach(li =>
    li.onclick = () => seek(parseFloat(li.dataset.s)));

  // --- timeline: click to scrub, click a dot to jump to that caption, hover to read it
  const tl = host.querySelector('.tl'), tip = host.querySelector('.tl-tip');
  const mark = host.querySelector('.tl-line u');
  const dots = {};                       // cached per lane so ticking stays cheap
  const span = ex.span || ex.duration;
  // x maps through `span` (what is drawn) but the seek clamps to `duration`
  // (what the video actually has), so clicking out past the bar parks at the end
  const atX = (el, ev) => {
    const r = el.getBoundingClientRect();
    return Math.max(0, Math.min(ex.duration, (ev.clientX - r.left) / r.width * span));
  };
  const line = tl.querySelector('.tl-line');
  line.onclick = ev => seek(atX(line, ev));
  tl.querySelectorAll('.tl-row').forEach(row => {
    const k = row.dataset.track, cues = ex.tracks[k] || [];
    row.onclick = ev => { if (!ev.target.classList.contains('dot')) seek(atX(row, ev)); };
    dots[k] = [...row.querySelectorAll('.dot')];
    dots[k].forEach(dot => {
      const c = cues[+dot.dataset.i];
      dot.onclick = () => seek(c.s);
      dot.onmouseenter = () => {
        tip.innerHTML = `<span class="ts">${fmt(c.s)} → ${fmt(c.e)}</span>${markRoles(c.t)}`;
        tip.style.display = 'block';
        const r = dot.getBoundingClientRect(), tr = tl.getBoundingClientRect();
        tip.style.left = Math.max(4, Math.min(tr.width - tip.offsetWidth - 4,
                                              r.left - tr.left - tip.offsetWidth / 2)) + 'px';
        // tips escape the widget rather than covering the other lane: the top
        // lane's tip goes up, the bottom lane's goes down
        tip.style.top = (k === 'qwen' ? r.top - tr.top - tip.offsetHeight - 8
                                      : r.bottom - tr.top + 8) + 'px';
      };
      dot.onmouseleave = () => { tip.style.display = 'none'; };
    });
  });

  // one pass per frame-ish: live caption text, playhead markers, cue highlight —
  // all three tracks read off the same clock so they can be compared directly
  const tick = () => {
    const t = video.currentTime;
    clock.textContent = `${fmt(t)} / ${fmt(ex.duration)}`;
    mark.style.display = 'block';
    mark.style.left = pct(t, span) + '%';
    TRACKS.forEach(([k]) => {
      const cues = ex.tracks[k] || [], cur = cueAt(cues, t);
      const cell = host.querySelector(`[data-live="${k}"]`);
      cell.className = 'live-txt' + (cur ? '' : ' silent');
      // a gap in the track shows up here as "— no caption —", live
      cell.innerHTML = cur ? markRoles(cur.t) : '— no caption —';
      (dots[k] || []).forEach((dot, i) => {
        const c = cues[i];
        dot.classList.toggle('active', !!c && t >= c.s && t < c.e);
      });
      host.querySelectorAll(`.cues[data-track="${k}"] li[data-s]`).forEach((li, i) => {
        const c = cues[i], on = c && t >= c.s && t < c.e;
        li.classList.toggle('active', !!on);
        if (!on) { delete li.dataset.seen; return; }
        if (li.dataset.seen) return;
        li.dataset.seen = '1';
        // scroll the cue list itself, never the page — scrollIntoView would
        // yank the viewport out from under whoever is presenting
        const ul = li.parentElement;
        if (li.offsetTop < ul.scrollTop ||
            li.offsetTop + li.offsetHeight > ul.scrollTop + ul.clientHeight)
          ul.scrollTop = Math.max(0, li.offsetTop - 8);
      });
    });
  };
  video.ontimeupdate = tick;
  video.onseeked = tick;
  // only one clip talks at a time
  video.onplay = () => document.querySelectorAll('video').forEach(v => v !== video && v.pause());
}

// every example is rendered up front; preload="none" keeps the 15 clips off the
// wire until one is actually played
document.querySelectorAll('.ex').forEach(d =>
  renderExample(d.querySelector('.ex-body'), EXAMPLES[d.dataset.ep]));
"""


def pipeline_svg(examples: dict[str, dict], standalone: bool = False) -> str:
    """The Cosmos pipeline diagram.

    Drawn from the real thing: the chunk staircase is a `plan_chunks` call on
    the longest demo episode mapped onto the track, and the frame ticks sit at
    the span midpoints `extract_n_frames` actually seeks to. Retuning the
    pipeline redraws the picture instead of invalidating it."""
    longest = max(examples.values(), key=lambda e: e["duration"])
    d = longest["duration"]
    chunks = ic.plan_chunks(d, ic.CHUNK_SECONDS, ic.OVERLAP_SECONDS)

    # chunk staircase: [0, d] mapped onto x in [180, 460], one row each. Row
    # spacing shrinks past 4 chunks so nothing below has to move.
    x0, w = 180, 280
    row_h = min(18, 72 / max(1, len(chunks)))
    bars = "".join(
        f'<text class="s-mono" x="174" y="{142 + i*row_h + 11:.1f}" text-anchor="end">'
        f'{s:g}–{e:g} s</text>'
        f'<rect class="s-chunk" x="{x0 + w*s/d:.1f}" y="{142 + i*row_h:.1f}" '
        f'width="{max(3, w*(e-s)/d):.1f}" height="{min(14, row_h - 4):.1f}" rx="4"/>'
        for i, (s, e) in enumerate(chunks))

    n = ic.GLOBAL_NUM_FRAMES
    ticks = "".join(
        f'<rect class="s-tick" x="{524 + 300*(i+0.5)/n - 1.5:.1f}" y="75" '
        f'width="3" height="12" rx="1.5"/>' for i in range(n))

    body = f"""<defs>
  <marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 z" fill="var(--muted)"/></marker>
  <marker id="ac" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 z" fill="var(--cosmos)"/></marker>
</defs>

<rect class="s-box" x="180" y="14" width="280" height="40" rx="8"/>
<text class="s-tb" x="320" y="32" text-anchor="middle">episode.mp4</text>
<text class="s-mono" x="320" y="47" text-anchor="middle">{d:g} s · one clip, one manifest line</text>
<path class="s-ar" d="M460,34 L516,34"/>
<path class="s-ar" d="M320,54 L320,92"/>

<text class="s-hd" x="524" y="14">① GLOBAL PASS</text>
<rect class="s-box" x="524" y="22" width="300" height="38" rx="8"/>
<text class="s-t" x="674" y="39" text-anchor="middle">ffmpeg — {n} frames</text>
<text class="s-mono" x="674" y="53" text-anchor="middle">midpoint of each of {n} equal spans</text>
<rect class="s-strip" x="524" y="72" width="300" height="18" rx="4"/>
{ticks}
<path class="s-ar" d="M674,90 L674,112"/>
<rect class="s-model" x="584" y="114" width="180" height="36" rx="8"/>
<text class="s-tb" x="674" y="137" text-anchor="middle">Cosmos3-Nano</text>
<path class="s-ar" d="M674,150 L674,172"/>
<rect class="s-box" x="524" y="174" width="300" height="42" rx="8"/>
<text class="s-t" x="674" y="192" text-anchor="middle">global caption</text>
<text class="s-mono" x="674" y="206" text-anchor="middle">≤ 3 sentences · no timestamps</text>

<path class="s-ard" d="M674,216 L674,268 L474,268 L474,300 L466,300"/>
<text class="s-hd" x="600" y="263" text-anchor="middle">CONTEXT FOR EVERY CHUNK</text>

<text class="s-hd" x="180" y="86">② CHUNK PASS</text>
<rect class="s-box" x="180" y="94" width="280" height="38" rx="8"/>
<text class="s-t" x="320" y="111" text-anchor="middle">ffmpeg — {ic.CHUNK_SECONDS:g} s chunks, {ic.OVERLAP_SECONDS:g} s overlap</text>
<text class="s-mono" x="320" y="125" text-anchor="middle">tail &lt; 2× overlap merges into the previous</text>
{bars}
<path class="s-ar" d="M320,214 L320,296"/>
<text class="s-m" x="330" y="240">for each chunk, {ic.CHUNK_FPS:g} fps</text>

<rect class="s-box" x="8" y="270" width="150" height="76" rx="8"/>
<text class="s-mono" x="18" y="288">system_prompt.txt</text>
<text class="s-mono" x="18" y="303">+ task instruction</text>
<text class="s-mono" x="18" y="318">prev chunk's last line</text>
<text class="s-mono" x="18" y="333">“skip the overlap”</text>
<path class="s-ar" d="M158,308 L174,308"/>
<rect class="s-model" x="180" y="298" width="280" height="38" rx="8"/>
<text class="s-tb" x="320" y="322" text-anchor="middle">Cosmos3-Nano</text>

<path class="s-ar" d="M320,336 L320,362"/>
<rect class="s-box" x="180" y="364" width="280" height="42" rx="8"/>
<text class="s-mono" x="320" y="382" text-anchor="middle">00:03.4 – 00:07.1: [right hand] …</text>
<text class="s-m" x="320" y="396" text-anchor="middle">chunk-relative timestamps</text>

<path class="s-ar" d="M320,406 L320,432"/>
<rect class="s-box" x="120" y="434" width="400" height="90" rx="8"/>
<text class="s-hd" x="136" y="453">STITCH</text>
<text class="s-mono" x="136" y="472">1 · strip &lt;/think&gt;, drop unparseable lines</text>
<text class="s-mono" x="136" y="488">2 · discard lines lying inside the overlap</text>
<text class="s-mono" x="136" y="504">3 · shift to global video time, clamp to chunk</text>
<text class="s-mono" x="136" y="517">4 · sort, rejoin actions split at a boundary</text>

<path class="s-ar" d="M320,524 L320,552"/>
<rect class="s-model" x="230" y="554" width="180" height="40" rx="8"/>
<text class="s-tb" x="320" y="572" text-anchor="middle">one .srt per episode</text>
<text class="s-mono" x="320" y="586" text-anchor="middle">same format as ground truth</text>"""

    if standalone:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 610" '
                f'width="840" height="610"><style>{SVG_STANDALONE_VARS}{SVG_CSS}</style>'
                f'<rect x="0" y="0" width="840" height="610" fill="var(--panel)"/>'
                f'{body}</svg>\n')
    return f'<svg class="pipe" viewBox="0 0 840 610" role="img" ' \
           f'aria-label="Cosmos3-Nano captioning pipeline">{body}</svg>'


def pipeline_html(examples: dict[str, dict]) -> str:
    """The "how these captions were produced" panel.

    Every number is read from `infer_cosmos` or from the data — the chunk
    diagram is a real `plan_chunks` call on the longest demo episode, and the
    Qwen token count comes from that prediction's reported usage — so retuning
    the pipeline updates the explanation instead of invalidating it."""
    longest = max(examples.values(), key=lambda e: e["duration"])
    chunks = ic.plan_chunks(longest["duration"], ic.CHUNK_SECONDS, ic.OVERLAP_SECONDS)
    d = longest["duration"]
    bars = "".join(
        f'<div class="ch"><i style="left:{100*s/d:.2f}%;width:{100*(e-s)/d:.2f}%">'
        f'{s:g}–{e:g}s</i></div>' for s, e in chunks)

    # a clip whose token count is quotable: prefer a short one, so the "frames
    # dominate the input" point lands
    tok = min((e for e in examples.values() if e["qwen_prompt_tokens"]),
              key=lambda e: e["duration"], default=None)
    qwen_cost = (f" One request, one caption track. Video frames dominate the input — "
                 f"<b>{tok['qwen_prompt_tokens']:,} prompt tokens</b> for a "
                 f"{tok['duration']:.1f}-second clip." if tok else "")

    counts = sorted({len(ic.plan_chunks(e["duration"], ic.CHUNK_SECONDS, ic.OVERLAP_SECONDS))
                     for e in examples.values()})
    n = ic.GLOBAL_NUM_FRAMES
    frames = "".join(f'<i style="left:{100*(i+0.5)/n:.2f}%"></i>' for i in range(n))

    return f"""<details class="how" open>
<summary>How these captions were produced</summary>
<div class="how-body">

<p class="lede">The two tracks below come from two different mechanisms, and most
of the findings trace back to that difference.</p>

<figure class="pipe-fig">{pipeline_svg(examples)}
  <figcaption>The Cosmos3-Nano path. Chunk boundaries and frame positions are the
  real ones, drawn for the longest clip in this demo.</figcaption>
</figure>

<div class="how-cols">
  <div class="how-card qwen"><h4>Qwen3.6 — one shot</h4>
    <p>The whole episode goes to a served Qwen3.6 in a single request, with the
    system prompt plus the episode's task instruction.{qwen_cost}</p></div>
  <div class="how-card cosmos"><h4>Cosmos3-Nano — two stages, chunked</h4>
    <p>Run locally through <code>transformers</code>, no server. A global context
    pass, then overlapping chunks stitched back into one track.</p></div>
</div>

<h4 class="step">① Global pass</h4>
<p>{ic.GLOBAL_NUM_FRAMES} frames are cut with <code>ffmpeg</code>, one at the
midpoint of each of {ic.GLOBAL_NUM_FRAMES} equal spans across the clip. They go
in as images with the ask: <em>one global caption, at most 3 sentences, no
timestamps</em> (≤{ic.GLOBAL_MAX_TOKENS} tokens). This is context for every chunk
that follows — it is how a chunk that only sees {ic.CHUNK_SECONDS:g} seconds
still knows what the episode is about.</p>
<div class="frames">{frames}</div>

<h4 class="step">② Chunk pass</h4>
<p>The clip is cut into <b>{ic.CHUNK_SECONDS:g}-second chunks with
{ic.OVERLAP_SECONDS:g} seconds of overlap</b>. If the leftover tail is shorter
than 2× the overlap it is merged into the previous chunk instead of becoming a
stub. Across this demo's {len(examples)} episodes that gives
{counts[0]}–{counts[-1]} chunks; the longest clip here
({d:g}s) splits like this:</p>
<div class="chunks">{bars}</div>
<p>Each chunk is re-encoded by <code>ffmpeg</code> and sampled at
{ic.CHUNK_FPS:g} fps. Every chunk request carries four things:</p>
<ol class="ingredients">
  <li>the system prompt (<code>system_prompt.txt</code>) plus the episode's task
      instruction</li>
  <li>the global caption from stage ①</li>
  <li>the previous chunk's last caption line, for continuity</li>
  <li>an explicit instruction not to emit any line lying entirely inside the
      overlap — plus, on every chunk but the last, <em>“if the final action is
      still in progress, omit that line, it will be captioned in the next
      chunk”</em></li>
</ol>

<h4 class="step">Stitch</h4>
<p>The model answers in <code>MM:SS.d – MM:SS.d: text</code> lines, relative to
the chunk. Any <code>&lt;/think&gt;</code> reasoning is stripped, unparseable
lines dropped, lines lying entirely in the overlap discarded, and the rest
shifted back to global video time and clamped to the chunk's bounds. Lines that
collapse to zero length are dropped, and an action the boundary cut in half is
rejoined with its continuation, so the seams do not show. Sorted by start time,
the result is written as one SRT — the same format as the ground truth.</p>

<p class="decode">Decoding is identical across chunks: bf16 ·
temperature {ic.GEN_KWARGS['temperature']} · top_p {ic.GEN_KWARGS['top_p']} ·
seed {ic.SEED}</p>

<p class="why"><b>Why it shows up below.</b> Chunking is what bounds the damage:
a bad chunk cannot poison the next one (<i>no error accumulation</i>), and no
line can be emitted past the chunk's end (<i>no overtime</i>). It is also what
causes the misses: chunks are captioned independently, so nothing enforces
continuous coverage across a boundary (<i>gap timestamps</i>) or a consistent
vocabulary between chunks (<i>inconsistent context</i>).</p>

</div>
</details>"""


def render_html(examples: dict[str, dict]) -> str:
    def summary_row(ex):
        return (f'<span class="mono">#{ex["idx"]}</span>'
                f'<span class="uid">{html.escape(ex["uid"])}</span>'
                f'<span class="mono uid">{ex["duration"]:.1f}s · '
                f'gt {ex["stats"]["gt"]["n"]} / qwen {ex["stats"]["qwen"]["n"]} / '
                f'cosmos {ex["stats"]["cosmos"]["n"]} cues</span>')

    parts = []
    for sec in SECTIONS:
        items = []
        for b in sec["bullets"]:
            exs = "".join(
                f'<div class="ex" data-ep="ep{i:03d}">'
                f'<div class="ex-head">{summary_row(examples[f"ep{i:03d}"])}</div>'
                f'<div class="ex-body"></div></div>'
                for i in b["eps"])
            items.append(
                f'<div class="bullet {b["tone"]}" id="{b["id"]}">'
                f'<h3>{html.escape(b["title"])}'
                f'<span class="count">{len(b["eps"])} example'
                f'{"s" if len(b["eps"]) > 1 else ""}</span></h3>'
                f'<p class="note">{html.escape(b["note"])}</p>{exs}</div>')
        parts.append(
            f'<section id="{sec["id"]}"><div class="sec-head">'
            f'<h2>{html.escape(sec["title"])}</h2><p>{html.escape(sec["blurb"])}</p></div>'
            f'{"".join(items)}</section>')

    nav = "".join(f'<a href="#{s["id"]}">{html.escape(s["title"])}</a>' for s in SECTIONS)
    n_ex = len({i for s in SECTIONS for b in s["bullets"] for i in b["eps"]})
    n_bul = sum(len(s["bullets"]) for s in SECTIONS)
    data = json.dumps(examples, separators=(",", ":")).replace("</", "<\\/")
    tracks = json.dumps(TRACKS)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmos3-Nano vs Qwen3.6 — egocentric caption review</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="wrap">
  <h1>Cosmos3-Nano vs Qwen3.6 — timestamped egocentric captions</h1>
  <p class="sub">Human review of the EgoDex <span class="mono">val</span> split ·
     {n_bul} findings · {n_ex} example episodes · press play to watch all three
     caption tracks run against the same clock</p>
  <div class="legend">
    <span class="tag gt">Ground truth = QAed caption</span>
    <span class="tag qwen">Qwen3.6 = served baseline (v1 prompt)</span>
    <span class="tag cosmos">Cosmos3-Nano = local chunked rollout (20s chunks, 1.5s overlap)</span>
  </div>
</div></header>
<div class="wrap">{pipeline_html(examples)}</div>
<nav class="jump"><div class="wrap">{nav}</div></nav>
<main class="wrap">{"".join(parts)}</main>
<script>
const EXAMPLES = {data};
const TRACKS = {tracks};
{JS}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=HERE.parent / "VR-finetune-VLM",
                    help="VR-finetune-VLM checkout (default: %(default)s)")
    ap.add_argument("--no-videos", action="store_true",
                    help="skip copying the .mp4 files (page/captions only)")
    args = ap.parse_args()

    print(f"reading {args.repo}")
    examples = build_examples(args.repo, copy_videos=not args.no_videos)
    (HERE / "examples.json").write_text(json.dumps(examples, indent=1))
    (HERE / "index.html").write_text(render_html(examples))
    # a runnable manifest over the clips vendored here, so infer_cosmos.py has
    # something to point at without a VR-finetune-VLM checkout
    # standalone copy of the pipeline diagram, for slides — same drawing, but
    # with concrete colours since there is no page to inherit them from
    (HERE / "pipeline.svg").write_text(pipeline_svg(examples, standalone=True))
    (HERE / "example_manifest.jsonl").write_text("".join(
        json.dumps({"episode_uid": ex["uid"], "video": ex["video"],
                    "instruction": ex["instruction"], "duration_s": ex["duration"]}) + "\n"
        for ex in examples.values()))
    print(f"\n{len(examples)} examples -> {HERE/'index.html'}")
    print(f"{len(examples)}-episode manifest -> {HERE/'example_manifest.jsonl'}")
    print(f"pipeline diagram -> {HERE/'pipeline.svg'}")


if __name__ == "__main__":
    main()
