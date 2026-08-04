# demo_cosmos

Standalone **Cosmos3-Nano** timestamped video captioning, plus a static page
comparing it against **Qwen3.6** on 15 EgoDex clips.

```
infer_cosmos.py         the inference script — imports nothing from other repos
requirements.txt        Python deps (ffmpeg is separate, see setup)
system_prompt.txt       output-format contract, sent as the system prompt
example_manifest.jsonl  15 episodes pointing at videos/ — ready to run
videos/                 the clips (~112 MB)
index.html              the comparison page — open it directly, no server
build_demo.py           regenerates the page + captions + the manifest
```

---

## Setup

**1. Environment** — needs a CUDA GPU, `torch`, and `transformers>=5.11` (for
`Cosmos3OmniForConditionalGeneration`). On this machine that is the `cosmos`
conda env (torch 2.13.0+cu130, transformers 5.14.1):

```bash
conda activate cosmos
```

Elsewhere:

```bash
conda create -n cosmos python=3.13 -y && conda activate cosmos
pip install -r requirements.txt
```

**2. ffmpeg** — `ffmpeg` and `ffprobe` must be on `PATH` (frame sampling, chunk
cutting):

```bash
sudo apt install ffmpeg
```

**3. Model** — one model, [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano).
Public, not gated, **33 GB**, so budget ~35 GB of disk:

```bash
hf download nvidia/Cosmos3-Nano
```

Set `HF_HOME` first to put the cache elsewhere. Skipping this step is fine too —
the script downloads on first run.

**4. Check the install** — no GPU, model or ffmpeg needed:

```bash
python infer_cosmos.py --selfcheck
# infer_cosmos selfcheck OK
```

---

## Run

From the repo root (`example_manifest.jsonl` uses paths relative to it).

One episode first:

```bash
python infer_cosmos.py --manifest example_manifest.jsonl --out preds/smoke.jsonl --limit 1
```

All 15 (5.4 minutes of video):

```bash
python infer_cosmos.py --manifest example_manifest.jsonl --out preds/cosmos3-nano_val.jsonl
```

Your own videos: pass your own manifest (format below).

```bash
python infer_cosmos.py --manifest /path/to/mine.jsonl --out preds/mine.jsonl
```

Progress prints per episode. Re-running skips episodes already in `--out`, so a
killed run resumes.

---

## Manifest format

JSONL, one episode per line:

```json
{"episode_uid": "assemble_disassemble_soft_legos_v2.1/episode_000137", "video": "videos/ep024.mp4", "instruction": "assemble_disassemble_soft_legos: Disassemble the soft lego tower back into separate pieces on a white background while sitting.", "duration_s": 10.0}
```

| Field | Required | Notes |
|---|---|---|
| `episode_uid` | **yes** | unique; identifies the episode in the output and drives resume |
| `video` | **yes** | absolute, or relative to where you run from. Missing file aborts at startup |
| `instruction` | no | task description, appended to the system prompt |
| `duration_s` | no | `ffprobe`d when absent |

Other keys are ignored. For a manifest written on another machine,
`--video-root-map /old/root=/new/root` rewrites the paths (repeatable).

## Output format

JSONL, appended, one line per episode:

```json
{"episode_uid": "...", "pred": "1\n00:00:00,100 --> 00:00:01,100\n[right hand] pick up a block | [ego] stay still\n\n2\n...", "n_events": 5, "n_chunks": 1}
```

`pred` is SRT text. A failed episode is still written, as
`{"episode_uid": ..., "pred": "", "error": "..."}` — the run is never silently
short.

## Flags

| Flag | Default | |
|---|---|---|
| `--manifest` | *required* | episodes JSONL |
| `--out` | *required* | predictions JSONL |
| `--prompt` | `./system_prompt.txt` | system prompt file |
| `--model` | `nvidia/Cosmos3-Nano` | HF id or local dir |
| `--chunk-seconds` | `20.0` | chunk length |
| `--overlap-seconds` | `1.5` | overlap between chunks |
| `--chunk-fps` | `2.0` | fps sampled from chunk video |
| `--global-num-frames` | `10` | frames for the global caption pass |
| `--global-max-tokens` / `--chunk-max-tokens` | `256` / `1024` | token budgets |
| `--limit N` / `--sample K` | — | first N / random K episodes |
| `--video-root-map OLD=NEW` | — | rewrite video path prefixes |
| `--selfcheck` | — | logic assertions only |

Sampling is fixed at `temperature=0.7, top_p=0.8, seed=0` in `GEN_KWARGS`.

## How it works

1. **Global pass** — 10 frames sampled evenly across the clip → one short
   caption used as context for every chunk.
2. **Chunk pass** — 20s chunks with 1.5s overlap. Each goes to the model with
   the system prompt + task instruction, the global caption, the previous
   chunk's last line, and an instruction not to re-caption the overlap.
3. Model lines (`MM:SS.d - MM:SS.d: text`) are parsed, overlap duplicates
   dropped, shifted back to global video time, and written as one SRT.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `N/M videos do not exist` | wrong paths — run from the repo root, or use `--video-root-map` |
| `command failed: ffprobe ...` | `ffmpeg`/`ffprobe` not on `PATH` |
| `ImportError: Cosmos3OmniForConditionalGeneration` | `transformers` older than 5.11 |
| `Cosmos3ForConditionalGeneration` not found | that name is in the checkpoint's `config.json` but does not exist — load the `Omni` class |
| `no parseable timestamped lines in any chunk` | model answered in prose; check `--prompt` points at `system_prompt.txt` |
| CUDA OOM | lower `--chunk-seconds` or `--chunk-fps` |

---

## The comparison page

```bash
xdg-open index.html
```

No server or build step. Each finding (Qwen better / Cosmos better / both bad,
plus Cosmos' strengths and weaknesses) shows its example clips inline: press
play and the ground-truth, Qwen and Cosmos captions run against one clock, over
a timeline where each dot is one caption at its start time — Qwen above the
line, Cosmos below.

To change which episodes appear, edit `SECTIONS` at the top of `build_demo.py`
(`eps` are 0-based line numbers in the source benchmark manifest) and rerun it.
It needs a VR-finetune-VLM checkout for the clips and predictions:

```bash
python build_demo.py --repo /path/to/VR-finetune-VLM
python build_demo.py --no-videos     # page only, skip the 112 MB copy
```
