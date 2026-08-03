# demo_cosmos

Self-contained demo comparing **Cosmos3-Nano** against **Qwen3.6** on timestamped
egocentric video captioning (EgoDex `val` split).

Two independent things live here:

1. **`infer_cosmos.py`** — a standalone Cosmos3-Nano captioning script. Same
   chunked 2-stage mechanism as the VR-finetune-VLM rollout, but it imports
   nothing from that repo: copy this folder anywhere and it runs.
2. **`index.html`** — a static presentation page walking through the review
   findings (Qwen better / Cosmos better / both bad, plus Cosmos' strengths and
   weaknesses), each backed by real example clips and side-by-side captions.

---

## Layout

```
demo_cosmos/
├── infer_cosmos.py     standalone Cosmos3-Nano inference (no repo imports)
├── system_prompt.txt   caption-format contract fed to the model as the system prompt
├── build_demo.py       regenerates index.html + videos/ + captions/ from the repo JSONLs
├── index.html          the presentation page (open it directly, no server needed)
├── examples.json       the data the page embeds, as a standalone file
├── videos/ep<NNN>.mp4  15 example clips (~112 MB)
├── posters/ep<NNN>.jpg poster frame per clip, so nothing shows as a black box
└── captions/ep<NNN>/   gt.srt · qwen.srt · cosmos.srt  per example
```

`<NNN>` is the 0-based line number of that episode in
`stage0/benchmark/caption/val.jsonl` — the same index the review server shows in
its JUDGE queue, so a number in the page maps straight back to the manifest.

---

## 1. View the demo

```bash
xdg-open index.html          # or just double-click it
```

No server, no build step, no network — the page inlines its data and reads the
clips from `videos/`. Keep `index.html`, `videos/` and `captions/` together.

How to drive it during a presentation:

- Sections at the top: **Overall verdict → Strengths → Weaknesses**.
- Every finding shows its example episodes inline — the real clip, playable on
  the spot, nothing to expand first.
- Each example is: **video + live caption strip** on the left, the **dot
  timeline** on the right.
- **The dot timeline is the main visual.** One shared time axis per episode:

  ```
  Qwen     •      •        •     •  •
  ═══════════════════════════════════════
  Cosmos  •    •         •        •
  ```

  **One dot = one caption**, placed at that caption's start timestamp. Qwen above the
  line, Cosmos below it, one shared time axis. Density and alignment read in one
  glance: a cluster of dots = too many captions, a bare stretch = a gap, dots
  that don't line up across the axis = the two models disagree about when
  something happened. The caption's exact span is in its hover tooltip.
- **The axis is normalized to the longest timestamp in the episode, not to the
  clip.** Models routinely predict cues that run past the end of the video, and
  clipping the axis at the clip length would push those dots off the widget
  entirely. So the axis covers `max(clip length, latest predicted end)` and the
  line runs the full width of it. Dot positions stay linear and true throughout;
  nothing is clamped or faked, and the tick labels are the real times. How far a
  track overruns the clip is called out in its lane label — `+30s past end` on a
  10-second clip.
- **Press play and the captions run with the video.** The strip under the player
  shows what each of the three says *at the current frame*, side by side; on the
  timeline a playhead sweeps across and the dot currently speaking lights up on
  each lane. When a model has nothing at that moment the strip says
  **“— no caption —”** in red — that is the gap weakness, live, not a claim in a
  table.
- **Hover a dot to read its caption**, **click a dot to jump the video to it**,
  and **click anywhere on a lane or the axis to scrub**.
- Ground truth is not on the timeline (two lanes only, by design). It is still
  in the live caption strip and in the full text under *“Full caption text, all
  three tracks”*, one click away, if someone asks to see everything.
- Playing one clip pauses any other, so nothing talks over you.
- `[left hand]` / `[right hand]` / `[both hands]` / `[ego]` are bolded — that is
  the hand-identity claim, readable at a glance.

Clips use `preload="none"` and a poster frame, so the page opens instantly and
only the clip you actually play is read off disk.

---

## 2. Run Cosmos inference

### Requirements

| Need | Detail |
|---|---|
| GPU | CUDA. Cosmos3-Nano in bf16 fits on one RTX 5090 (32 GB); `device_map="auto"` shards if it doesn't. |
| Python | `torch` + `transformers>=5.11` (needs `Cosmos3OmniForConditionalGeneration`). |
| Binaries | `ffmpeg` and `ffprobe` on `PATH` (frame sampling and chunk cutting). |
| Weights | `nvidia/Cosmos3-Nano` — **33 GB**, see below. |
| Disk | ~35 GB free for the weights cache. |

On this machine the `cosmos` conda env has all of it (torch 2.13.0+cu130,
transformers 5.14.1):

```bash
conda activate cosmos
```

Elsewhere:

```bash
conda create -n cosmos python=3.11 -y && conda activate cosmos
pip install "torch>=2.4" "transformers>=5.11" accelerate av huggingface_hub
sudo apt install ffmpeg      # provides ffprobe too
```

### The model to download

**One model, and only one: [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano).**
Public and **not gated** — no HF token, no licence click-through.

```bash
hf download nvidia/Cosmos3-Nano
```

`hf` is the huggingface_hub CLI (`pip install huggingface_hub`). On
huggingface_hub 1.x the old `huggingface-cli` is gone — it exits with
*“`huggingface-cli` is deprecated and no longer works. Use `hf` instead.”*
Version-independent alternative:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos3-Nano')"
```

Or skip it entirely: `infer_cosmos.py` downloads the weights on first run.

That lands in `~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano`, or set
`HF_HOME=/some/big/disk` first. `infer_cosmos.py` resolves the same cache, so
once it is there just run the script; `--model /path/to/local/dir` points it at
an explicit directory instead.

Verified against the Hub on 2026-08-04: 68 files, revision
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, which is what is cached on this
machine. Measured on disk:

| Folder | Size | What it is |
|---|---|---|
| `transformer/` | 29 GB | the omni transformer — the part that writes the captions |
| `sound_tokenizer/` | 1.9 GB | audio, unused by this script |
| `vae/` | 1.4 GB | video generation, unused by this script |
| `vision_encoder/` | 1.1 GB | encodes the sampled frames and chunk video |
| `assets/` + `images/` | 62 MB | model-card media, unused |
| `text_tokenizer/`, `scheduler/`, configs | 16 MB | — |
| **total** | **33 GB** | |

Cosmos3-Nano ships as one omni checkpoint (text/image/video/audio/action), so
the video-generation and audio experts come down with it even though captioning
never calls them. Budget the full 33 GB.

Load class is `Cosmos3OmniForConditionalGeneration` — note the `Omni`. The
checkpoint's `config.json` names `Cosmos3ForConditionalGeneration`, which does
**not** exist in transformers 5.14.1; loading by that name fails.

Nothing else needs downloading. The Qwen3.6 side of the comparison is already in
this repo as prediction files — the demo never runs a Qwen model.

### Check the logic first (no GPU, no model, no ffmpeg)

```bash
python infer_cosmos.py --selfcheck
# infer_cosmos selfcheck OK
```

This asserts the chunk planner, the timestamp parser, the SRT writer and the
overlap prompt. Run it after editing the script.

### Smoke test one episode

```bash
python infer_cosmos.py \
    --manifest ../VR-finetune-VLM/stage0/benchmark/caption/val.jsonl \
    --prompt   system_prompt.txt \
    --out      preds/smoke.jsonl \
    --limit    1
```

### Full run

```bash
python infer_cosmos.py \
    --manifest ../VR-finetune-VLM/stage0/benchmark/caption/val.jsonl \
    --prompt   system_prompt.txt \
    --out      preds/cosmos3-nano_val.jsonl
```

Progress prints per episode (`ok=`, `err=`, events, chunks, seconds/episode).

### Input format

One JSON object per line. Only four fields are read:

```json
{
  "episode_uid": "assemble_jenga_v2.1/episode_000519",
  "video": "/mnt/SSD4/dataset/egodex/.../episode_000519.mp4",
  "instruction": "assemble_jenga: Assemble block pieces into a Jenga tower ...",
  "duration_s": 10.0
}
```

`duration_s` is optional — it is ffprobed when missing. Anything else in the
line is ignored, so the benchmark manifest works as-is.

### Output format

One JSON object per line, appended:

```json
{"episode_uid": "...", "pred": "1\n00:00:00,100 --> 00:00:01,100\n[right hand] ...", "n_events": 5, "n_chunks": 1}
```

`pred` is a complete SRT string in the same format as the ground truth, so it
drops straight into `stage0/eval.py --pred` or `stage0/review_server.py --pred`.

A failed episode is still written, as `{"episode_uid": ..., "pred": "", "error": "..."}`
— a run is never silently short. **The run is resumable**: `episode_uid`s
already present in `--out` are skipped, so re-running after a crash continues
where it stopped. To redo failed episodes, delete their lines from the output
file and re-run.

### How it works

**Stage 1 — global context.** 10 frames sampled uniformly across the whole
video (`ffmpeg -ss` at the midpoint of each of 10 equal spans) go to the model
as images; it returns a ≤3-sentence global caption, no timestamps.

**Stage 2 — chunk captioning.** The video is cut into 20s chunks with a 1.5s
overlap; a would-be tiny final chunk is merged into the previous one. Each chunk
is captioned with:

- the system prompt (`system_prompt.txt`) plus the episode's `instruction`,
- the global caption from stage 1,
- the previous chunk's last line, for continuity,
- an instruction not to emit any line lying entirely inside the overlap.

Model output lines (`MM:SS.d - MM:SS.d: [left hand] ... | [ego] ...`) are
parsed, lines inside the overlap are dropped, the rest are shifted back to
global video time, clamped to the chunk, sorted and rendered as one SRT.

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--manifest` | *required* | episodes JSONL |
| `--out` | *required* | predictions JSONL (appended; resumable) |
| `--prompt` | `./system_prompt.txt` | chunk system prompt |
| `--model` | `nvidia/Cosmos3-Nano` | HF id or local directory |
| `--chunk-seconds` | `20.0` | chunk length |
| `--overlap-seconds` | `1.5` | overlap between consecutive chunks |
| `--chunk-fps` | `2.0` | fps the processor samples chunk video at |
| `--global-num-frames` | `10` | frames for the stage-1 global caption |
| `--global-max-tokens` | `256` | token budget, stage 1 |
| `--chunk-max-tokens` | `1024` | token budget per chunk |
| `--limit N` | — | first N episodes |
| `--sample K` | — | random K episodes (`--seed`, default 42) |
| `--video-root-map OLD=NEW` | — | rewrite video path prefixes (repeatable) |
| `--selfcheck` | — | logic assertions only, then exit |

Sampling is fixed at `temperature=0.7, top_p=0.8, seed=0` to match the reference
rollout — change it in `GEN_KWARGS` at the top of the script, not on the CLI.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `N/M videos do not exist` | The manifest holds absolute paths from the machine that curated it. Pass `--video-root-map /old/root=/new/root` (repeatable). |
| `command failed: ffprobe ...` | `ffmpeg`/`ffprobe` not on `PATH`. |
| `ImportError: Cosmos3OmniForConditionalGeneration` | `transformers` older than 5.11. |
| `no parseable timestamped lines in any chunk` | The model answered in prose. Check `system_prompt.txt` is the one being passed. |
| CUDA OOM | Lower `--chunk-seconds` or `--chunk-fps` (fewer video tokens per request). |

---

## 3. Rebuild the demo page

`build_demo.py` regenerates `index.html`, `videos/`, `posters/` and `captions/`
from a VR-finetune-VLM checkout:

```bash
python build_demo.py                              # defaults to ../VR-finetune-VLM
python build_demo.py --repo /path/to/VR-finetune-VLM
python build_demo.py --no-videos                  # page + captions only, skip the 112 MB copy
```

Poster frames need `ffmpeg`; without it the build still succeeds, the clips just
start out black. `--no-videos` reuses whatever is already in `videos/` and
`posters/`.

It reads:

| Source | Role |
|---|---|
| `stage0/benchmark/caption/val.jsonl` | manifest + QAed ground-truth captions |
| `stage1/preds/caption/v1_val.jsonl` | Qwen3.6 predictions |
| `stage1/preds/caption/cosmos3-nano_val.jsonl` | Cosmos3-Nano predictions |

Only stdlib — no GPU, no model, seconds to run.

**To change which episodes appear**, edit the `SECTIONS` table at the top of
`build_demo.py` (each bullet's `eps` is a list of 0-based `val.jsonl` line
numbers) and re-run. The page is generated from that table, nothing is
hand-written in the HTML.

---

## 4. The findings and their examples

Indices are `val.jsonl` line numbers.

**Overall verdict**

| Finding | Episodes |
|---|---|
| Qwen is better | 30, 33 |
| Cosmos is better | 29, 36 |
| Both are bad | 28 |

**Cosmos — strengths**

| Finding | Episodes |
|---|---|
| Hand identity is quite good | 43, 45 |
| More clearly separated actions, more detailed descriptions (≥ Qwen) | 43, 45, 48 |
| No error accumulation | 49 |
| Chunking strategy holds — no overtime | 24, 27 |

**Cosmos — weaknesses**

| Finding | Episodes |
|---|---|
| Gap / overlap in timestamps | 24, 47 |
| Too many actions in one caption | 37, 38 |
| Sometimes inconsistent context (even within a chunk) | 31 |

Two numbers worth pointing at while presenting: on episodes **49** and **31**,
Qwen degenerates into 205 and 189 cues respectively against 16 and 5 in the
ground truth, while Cosmos stays at 9 and 6 — that is the "no error
accumulation" claim, visible in the cue counts on the example headers.
