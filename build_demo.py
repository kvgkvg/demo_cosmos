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
from pathlib import Path

HERE = Path(__file__).resolve().parent

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
    (HERE / "example_manifest.jsonl").write_text("".join(
        json.dumps({"episode_uid": ex["uid"], "video": ex["video"],
                    "instruction": ex["instruction"], "duration_s": ex["duration"]}) + "\n"
        for ex in examples.values()))
    print(f"\n{len(examples)} examples -> {HERE/'index.html'}")
    print(f"{len(examples)}-episode manifest -> {HERE/'example_manifest.jsonl'}")


if __name__ == "__main__":
    main()
