#!/usr/bin/env python3
"""Standalone Cosmos3-Nano chunked video captioning (no VR-finetune-VLM imports).

Same 2-stage mechanism the VR-finetune-VLM rollout uses, rebuilt here so this
folder runs on its own:

  Stage 1 (global context): N frames sampled uniformly across the episode's
      video -> one short global caption, used as context for every chunk.

  Stage 2 (chunk captioning): the video is cut into ~20s chunks with a 1.5s
      overlap. Each chunk goes to the model with the system prompt
      (system_prompt.txt — the "MM:SS.d - MM:SS.d: [left hand] ... | [ego] ..."
      line contract the benchmark ground truth uses), the global caption, the
      previous chunk's last line, and an instruction not to re-caption the
      overlap. Parsed lines are shifted back to global video time, merged and
      rendered as one SRT string per episode.

Input is a manifest JSONL with one episode per line; only these fields are read:

  episode_uid   str    unique id, carried into the output
  video         str    absolute path to the .mp4
  instruction   str    task description, appended to the system prompt
  duration_s    float  optional; ffprobed when absent

Output is a predictions JSONL, one row per episode:

  {"episode_uid": ..., "pred": "<SRT text>", "n_events": N, "n_chunks": M}

A failed episode is still written, with "pred": "" and an "error" field, so a
run is never silently short. Resumable: episode_uids already present in --out
are skipped, so re-running after a crash continues where it stopped.

Usage:
  python infer_cosmos.py \
      --manifest /path/to/val.jsonl \
      --prompt   system_prompt.txt \
      --out      preds/cosmos3-nano_val.jsonl

  # smoke test one episode first
  python infer_cosmos.py --manifest ... --prompt ... --out /tmp/smoke.jsonl --limit 1

  # logic-only check, no model and no ffmpeg needed
  python infer_cosmos.py --selfcheck

Requires: torch + CUDA, transformers>=5.11 (Cosmos3Omni), ffmpeg and ffprobe on PATH.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Sampling params, fixed seed: same settings the reference rollout used, so
# numbers from this script are comparable with the ones in VR-finetune-VLM.
GEN_KWARGS = dict(do_sample=True, temperature=0.7, top_p=0.8)
SEED = 0

# Pipeline defaults, here rather than inline in argparse so build_demo.py can
# import them and describe the real mechanism instead of a copy of it.
CHUNK_SECONDS = 20.0
OVERLAP_SECONDS = 1.5
CHUNK_FPS = 2.0
GLOBAL_NUM_FRAMES = 10
GLOBAL_MAX_TOKENS = 256
CHUNK_MAX_TOKENS = 1024

GLOBAL_SYSTEM_PROMPT = "You are a helpful assistant specialized in video captioning."
GLOBAL_USER_PROMPT = (
    "These images are frames sampled uniformly, in chronological order, across "
    "an entire video. Based on them, write a single global caption describing "
    "the overall content of the video. Use at most 3 sentences. Do not include "
    "timestamps."
)

# Per-chunk task text. The full output-format contract lives in --prompt.
CHUNK_USER_TEXT = "Describe the person's actions."

# "MM:SS.d - MM:SS.d: description", accepting en-dash, em-dash or hyphen.
LINE_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*[–—-]\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*:\s*(.+?)\s*$"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_system_prompt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_manifest(path: Path, sample: int | None, limit: int | None, seed: int) -> list[dict]:
    episodes = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if sample is not None:
        episodes = random.Random(seed).sample(episodes, min(sample, len(episodes)))
    elif limit is not None:
        episodes = episodes[:limit]
    return episodes


def done_uids(out: Path) -> set[str]:
    """episode_uids already written to --out, so a re-run resumes."""
    if not out.exists():
        return set()
    return {json.loads(l)["episode_uid"] for l in out.read_text().splitlines() if l.strip()}


def remap_videos(episodes: list[dict], mappings: list[str]) -> None:
    """Rewrite video path prefixes in place (OLD=NEW), longest prefix first.

    Manifests carry absolute paths from the machine that curated them; this is
    how the same manifest runs on a node that mounts the dataset elsewhere."""
    pairs = []
    for mapping in mappings:
        if "=" not in mapping:
            sys.exit(f"--video-root-map needs OLD=NEW, got {mapping!r}")
        pairs.append(tuple(mapping.split("=", 1)))
    pairs.sort(key=lambda p: -len(p[0]))
    for e in episodes:
        for old, new in pairs:
            if e["video"].startswith(old):
                e["video"] = new + e["video"][len(old):]
                break


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe
# ---------------------------------------------------------------------------
def _run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def get_duration(video_path: Path) -> float:
    out = _run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ])
    return float(out.strip())


def extract_n_frames(video_path: Path, duration: float, n: int, out_dir: Path) -> list[Path]:
    """Exactly n frames spread uniformly (midpoint of each of n equal spans) —
    a fixed-fps filter cannot guarantee an exact count."""
    frame_paths = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        out_path = out_dir / f"frame_{i:03d}.jpg"
        _run_cmd([
            "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ])
        frame_paths.append(out_path)
    return frame_paths


def extract_chunk(video_path: Path, start: float, end: float, out_path: Path) -> None:
    # -ss after -i for frame-accurate cuts (slower than -ss before -i, fine here)
    _run_cmd([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
        str(out_path),
    ])


# ---------------------------------------------------------------------------
# Timestamp parsing / SRT rendering / chunk planning
# ---------------------------------------------------------------------------
def _mmss_to_seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds)


def format_mmss(t: float) -> str:
    if t < 0:
        t = 0.0
    minutes = int(t // 60)
    return f"{minutes:02d}:{t - minutes * 60:05.2f}"


def parse_hand_ego_lines(text: str) -> list[tuple[float, float, str]]:
    """Model output -> [(start_s, end_s, caption)], junk lines dropped."""
    text = text.split("</think>", 1)[-1]  # drop reasoning if the model emitted any
    events = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        events.append((_mmss_to_seconds(m.group(1), m.group(2)),
                       _mmss_to_seconds(m.group(3), m.group(4)),
                       m.group(5).strip()))
    return events


def _stamp(t: float) -> str:  # HH:MM:SS,mmm
    """Round to whole milliseconds *first*, then split.

    Rounding the fraction on its own lets 0.9996 s round to 1000 ms and emit
    `00:00:45,1000` — a four-digit field that is not valid SRT, and that a
    reader expecting `\\d{3}` drops silently rather than rejecting. Carrying
    through integer milliseconds keeps the seconds and the remainder in step."""
    ms_total = max(0, int(round(t * 1000)))
    h, ms_total = divmod(ms_total, 3_600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def events_to_srt(events) -> str:
    return "\n\n".join(f"{i}\n{_stamp(s)} --> {_stamp(e)}\n{c}"
                       for i, (s, e, c) in enumerate(events, start=1))


def remap_events(lines, start: float, end: float, offset: float):
    """Chunk-local `(s, e, text)` lines -> episode-global events.

    Lines lying entirely inside the overlap were already captioned by the
    previous chunk, so they are dropped; the rest are shifted by the chunk's
    start and clamped to its bounds. A line that collapses to zero length under
    that clamp — or that the model emitted with end <= start — is dropped too:
    a caption covering no time is not a label, and it renders as a cue with no
    duration downstream."""
    events = []
    for local_start, local_end, caption in lines:
        if local_end <= offset:                   # already captioned upstream
            continue
        global_start = min(start + max(local_start, offset), end)
        global_end = min(max(start + local_end, global_start), end)
        if global_end <= global_start:
            continue
        events.append((global_start, global_end, caption))
    return events


def merge_boundary_splits(events, boundaries, eps: float = 1e-3):
    """Rejoin one action that got cut in half by a chunk boundary.

    A line still running when its chunk ends is clamped to that end, and the
    next chunk describes the same action again from the boundary — leaving a
    stub like (38.2, 38.5) immediately followed by (38.5, 40.3) with identical
    text. Merged only when all three hold: the text matches exactly, the cues
    are contiguous, and the join sits on a chunk boundary. Two identical
    captions the model emitted *within* one chunk are left alone — those may be
    a genuinely repeated action, and guessing is worse than keeping both."""
    out = []
    for ev in events:
        if out:
            ps, pe, pt = out[-1]
            s, e, t = ev
            on_boundary = any(abs(pe - b) < eps for b in boundaries)
            if pt == t and s <= pe + eps and on_boundary:
                out[-1] = (ps, max(pe, e), pt)
                continue
        out.append(ev)
    return out


def plan_chunks(duration: float, chunk_seconds: float, overlap_seconds: float):
    """(start, end) per chunk, consecutive chunks overlapping. A would-be tiny
    final chunk (< 2x overlap left) is merged into the previous one."""
    if chunk_seconds <= overlap_seconds:
        raise ValueError("--chunk-seconds must exceed --overlap-seconds")
    chunks = []
    start = 0.0
    while True:
        end = min(start + chunk_seconds, duration)
        if 0 < duration - end < overlap_seconds * 2:
            end = duration
        chunks.append((start, end))
        if end >= duration - 1e-6:
            break
        start = end - overlap_seconds
    return chunks


def build_chunk_prompt(global_caption: str, previous_summary: str | None,
                       offset: float, is_last_chunk: bool) -> str:
    parts = [CHUNK_USER_TEXT,
             f"\n\nGlobal context for the entire video: {global_caption}"]
    if previous_summary:
        parts.append(
            "\n\nContext from the immediately preceding chunk (for continuity "
            f"only, do not repeat it as its own line): {previous_summary}")
    if offset > 0:
        parts.append(
            f"\n\nThe first {offset:.2f} seconds of this clip overlap with the "
            "previous chunk and were already captioned there. Do NOT output "
            f"any line that lies entirely within [00:00.00, {format_mmss(offset)}); "
            f"your first timestamped line must start at or after {format_mmss(offset)}.")
    if not is_last_chunk:
        parts.append(
            "\n\nThis clip is cut from a longer video. If the final action is "
            "still in progress and does not clearly conclude before the clip "
            "ends, omit that last line entirely — it will be captioned in "
            "full in the next chunk.")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def warn_missing_decoder() -> None:
    """Say it up front if the video decoder is missing.

    transformers decodes chunk video with torchcodec and falls back to
    torchvision — but torchvision >= 0.28 dropped `io.read_video`, so the
    fallback raises. That surfaces per episode, which means loading 33 GB of
    weights and then failing on every one of them with the same message."""
    try:
        import torchcodec  # noqa: F401 — probe only
    except ImportError:
        print("WARNING: torchcodec is not installed. transformers will fall back to "
              "torchvision, whose `io.read_video` was removed in 0.28 — every episode "
              "will fail with:\n"
              "    AttributeError: module 'torchvision.io' has no attribute 'read_video'\n"
              "  Fix: pip install torchcodec   (see requirements.txt)", file=sys.stderr)


def load_model(model_id: str):
    import torch  # imported here so --selfcheck runs without torch installed
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return processor, model


_PROCESSOR_KWARGS_SUPPORTED = None  # resolved once, on first use


def _chat_template_kwargs(processor, fps: float | None) -> dict:
    """Route `fps` the way this transformers wants it.

    Newer versions take per-processor options in a `processor_kwargs` dict and
    warn when they arrive as loose **kwargs. The loose form still works today,
    but a version that stops honouring it would silently change chunk sampling
    rather than fail — so prefer the supported spelling when it exists."""
    global _PROCESSOR_KWARGS_SUPPORTED
    kw = dict(tokenize=True, add_generation_prompt=True, return_dict=True,
              return_tensors="pt")
    if fps is None:
        return kw
    if _PROCESSOR_KWARGS_SUPPORTED is None:
        import inspect  # noqa: PLC0415 — only this path needs it
        _PROCESSOR_KWARGS_SUPPORTED = "processor_kwargs" in inspect.signature(
            processor.apply_chat_template).parameters
    if _PROCESSOR_KWARGS_SUPPORTED:
        kw["processor_kwargs"] = {"fps": fps}
    else:
        kw["fps"] = fps
    return kw


def run_reasoner(processor, model, messages, max_new_tokens: int,
                 fps: float | None = None) -> str:
    import torch

    kw = _chat_template_kwargs(processor, fps)
    inputs = processor.apply_chat_template(messages, **kw).to(model.device, torch.bfloat16)
    torch.manual_seed(SEED)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **GEN_KWARGS)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def get_global_caption(processor, model, video_path: Path, duration: float,
                       num_frames: int, max_tokens: int) -> str:
    with tempfile.TemporaryDirectory(prefix="infer_cosmos_frames_") as tmp_dir:
        frames = extract_n_frames(video_path, duration, num_frames, Path(tmp_dir))
        content = [{"type": "image", "path": str(f)} for f in frames]
        content.append({"type": "text", "text": GLOBAL_USER_PROMPT})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": GLOBAL_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return run_reasoner(processor, model, messages, max_tokens).strip()


def caption_chunk(processor, model, system_content: str, chunk_path: Path,
                  global_caption: str, previous_summary, offset: float,
                  is_last_chunk: bool, max_tokens: int, fps: float) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_content}]},
        {"role": "user", "content": [
            {"type": "video", "path": str(chunk_path)},
            {"type": "text", "text": build_chunk_prompt(
                global_caption, previous_summary, offset, is_last_chunk)},
        ]},
    ]
    return run_reasoner(processor, model, messages, max_tokens, fps=fps)


# ---------------------------------------------------------------------------
# Per-episode pipeline -> SRT
# ---------------------------------------------------------------------------
def process_episode(processor, model, sys_prompt: str, e: dict, args) -> dict:
    video_path = Path(e["video"])
    duration = e.get("duration_s") or get_duration(video_path)

    system_content = sys_prompt + (
        "\n\n## Episode context\n"
        "The camera-wearer is performing this task (background reference for identifying "
        "objects and the setting):\n" + e.get("instruction", ""))

    global_caption = get_global_caption(
        processor, model, video_path, duration,
        args.global_num_frames, args.global_max_tokens)

    chunks = plan_chunks(duration, args.chunk_seconds, args.overlap_seconds)
    all_events = []
    previous_summary = None

    with tempfile.TemporaryDirectory(prefix="infer_cosmos_") as tmp_dir:
        for i, (start, end) in enumerate(chunks):
            is_last_chunk = (i == len(chunks) - 1)
            offset = args.overlap_seconds if i > 0 else 0.0

            if start == 0.0 and end >= duration - 1e-6:
                chunk_path = video_path          # whole video fits in one chunk
            else:
                chunk_path = Path(tmp_dir) / f"chunk_{i:03d}.mp4"
                extract_chunk(video_path, start, end, chunk_path)

            raw = caption_chunk(
                processor, model, system_content, chunk_path, global_caption,
                previous_summary, offset, is_last_chunk,
                args.chunk_max_tokens, args.chunk_fps)

            lines = parse_hand_ego_lines(raw)
            if not lines:
                continue
            all_events.extend(remap_events(lines, start, end, offset))
            previous_summary = lines[-1][2]

    all_events.sort(key=lambda ev: ev[0])
    all_events = merge_boundary_splits(all_events, [c[1] for c in chunks])
    if not all_events:
        raise RuntimeError("no parseable timestamped lines in any chunk")
    return {"episode_uid": e["episode_uid"], "pred": events_to_srt(all_events),
            "n_events": len(all_events), "n_chunks": len(chunks)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="episodes JSONL (episode_uid, video, instruction, duration_s)")
    ap.add_argument("--prompt", type=Path, default=Path(__file__).with_name("system_prompt.txt"),
                    help="chunk system prompt file (default: ./system_prompt.txt)")
    ap.add_argument("--out", type=Path, required=True, help="predictions JSONL (appended)")
    ap.add_argument("--model", default="nvidia/Cosmos3-Nano",
                    help="HF id or local dir (default: %(default)s)")
    ap.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    ap.add_argument("--overlap-seconds", type=float, default=OVERLAP_SECONDS)
    ap.add_argument("--chunk-fps", type=float, default=CHUNK_FPS,
                    help="fps passed to the processor for chunk videos")
    ap.add_argument("--global-num-frames", type=int, default=GLOBAL_NUM_FRAMES)
    ap.add_argument("--global-max-tokens", type=int, default=GLOBAL_MAX_TOKENS)
    ap.add_argument("--chunk-max-tokens", type=int, default=CHUNK_MAX_TOKENS)
    ap.add_argument("--sample", type=int, help="random K episodes (reproducible via --seed)")
    ap.add_argument("--limit", type=int, help="first N episodes (ignored if --sample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video-root-map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite manifest video path prefixes (repeatable)")
    args = ap.parse_args()

    sys_prompt = load_system_prompt(args.prompt)
    episodes = load_manifest(args.manifest, args.sample, args.limit, args.seed)
    remap_videos(episodes, args.video_root_map)

    missing = [e["video"] for e in episodes if not Path(e["video"]).exists()]
    if missing:
        sys.exit(f"{len(missing)}/{len(episodes)} videos do not exist, e.g. {missing[0]}\n"
                 "Manifests store absolute paths from the curating machine; remap them "
                 "with --video-root-map OLD=NEW (repeatable).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_uids(args.out)
    todo = [e for e in episodes if e["episode_uid"] not in already]
    print(f"manifest={args.manifest.name} prompt={args.prompt.name} "
          f"episodes={len(episodes)} done={len(already)} todo={len(todo)}")
    if not todo:
        print("nothing to do (all episodes already in --out)")
        return

    warn_missing_decoder()
    print(f"loading {args.model} via transformers (may take a while)...")
    processor, model = load_model(args.model)
    print(f"model on {model.device} | chunk={args.chunk_seconds}s "
          f"overlap={args.overlap_seconds}s fps={args.chunk_fps}")

    n_ok = n_err = 0
    t0 = time.time()
    with open(args.out, "a") as fh:
        for i, e in enumerate(todo, 1):
            try:
                rec = process_episode(processor, model, sys_prompt, e, args)
            except Exception as ex:
                rec = {"episode_uid": e["episode_uid"], "pred": "",
                       "error": f"{type(ex).__name__}: {ex}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()                            # crash-safe: never lose finished episodes
            if rec.get("error"):
                n_err += 1
                print(f"  [{i}/{len(todo)}] ERR {rec['episode_uid']}: {rec['error']}", flush=True)
            else:
                n_ok += 1
                print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"events={rec['n_events']} chunks={rec['n_chunks']} "
                      f"({(time.time() - t0) / i:.1f} s/ep)", flush=True)

    print(f"done: {n_ok} ok, {n_err} err -> {args.out}")


def _selfcheck():
    """assert-based check of the pure logic (no model, no ffmpeg, no torch)."""
    # chunk planning: overlap stride + tiny-tail merge + single-chunk video
    chunks = plan_chunks(24.7, 20.0, 1.5)          # tail 4.7s > 2*overlap -> own chunk
    assert chunks[0] == (0.0, 20.0) and abs(chunks[1][0] - 18.5) < 1e-9
    assert plan_chunks(21.0, 20.0, 1.5) == [(0.0, 21.0)]   # tiny tail merged
    assert plan_chunks(10.0, 20.0, 1.5) == [(0.0, 10.0)]   # shorter than one chunk

    # line parsing: dash variants, decimals, </think> stripping, junk skipped
    txt = ("preamble\n00:01.2 – 00:03.4: [left hand] grasp cup | [ego] stay still\n"
           "00:03.4 - 00:05.0: [right hand] pour | [ego] stay still\nnot a line")
    evs = parse_hand_ego_lines("thinking...</think>" + txt)
    assert len(evs) == 2 and evs[0][0] == 1.2 and evs[1][1] == 5.0

    # SRT rendering matches the benchmark's format
    srt = events_to_srt([(0.0, 2.831, "[left hand] grasp domino | [ego] stay still")])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,831\n[left hand] grasp domino")
    assert format_mmss(61.25) == "01:01.25"

    # timestamps carry instead of emitting a 4-digit millisecond field, which
    # is not valid SRT and gets dropped silently by readers
    assert _stamp(45.9996) == "00:00:46,000"
    assert _stamp(59.9999) == "00:01:00,000"
    assert _stamp(3599.9999) == "01:00:00,000"
    assert _stamp(45.166667) == "00:00:45,167"
    assert _stamp(-1.0) == "00:00:00,000"
    assert all(len(p.split(",")[1]) == 3
               for p in (_stamp(i / 997) for i in range(2000)))

    # overlap prompt only appears after the first chunk
    assert "overlap" not in build_chunk_prompt("g", None, 0.0, False)
    assert "00:01.50" in build_chunk_prompt("g", "prev line", 1.5, False)

    # chunk-local -> global remap, the step that stitches the chunks together
    ev = remap_events([(0.0, 1.0, "in overlap"),      # dropped: ends at the offset
                       (1.5, 4.0, "kept"),
                       (4.0, 99.0, "clamped to the chunk end")],
                      start=18.5, end=38.5, offset=1.5)
    assert [e[2] for e in ev] == ["kept", "clamped to the chunk end"]
    assert ev[0] == (20.0, 22.5, "kept")              # shifted by the chunk start
    assert ev[1][1] == 38.5                           # never runs past the chunk
    # a line the model emitted with no duration, and one the clamp collapses,
    # are both dropped rather than becoming zero-length cues
    assert remap_events([(2.0, 2.0, "zero")], 0.0, 10.0, 0.0) == []
    assert remap_events([(12.0, 13.0, "past the end")], 0.0, 10.0, 0.0) == []
    # first chunk has no offset, so nothing is dropped as overlap
    assert len(remap_events([(0.0, 2.0, "a")], 0.0, 20.0, 0.0)) == 1

    # boundary seam: the stub clamped to the chunk end rejoins its continuation
    seam = merge_boundary_splits(
        [(38.2, 38.5, "same"), (38.5, 40.3, "same")], boundaries=[20.0, 38.5])
    assert seam == [(38.2, 40.3, "same")]
    # identical captions *inside* a chunk are a possible repeat — left alone
    assert len(merge_boundary_splits(
        [(46.5, 49.6, "x"), (49.6, 52.6, "x")], boundaries=[20.0, 38.5])) == 2
    # different text at a boundary is never merged
    assert len(merge_boundary_splits(
        [(38.0, 38.5, "a"), (38.5, 40.0, "b")], boundaries=[38.5])) == 2

    print("infer_cosmos selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
