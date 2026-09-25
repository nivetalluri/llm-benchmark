# ⚡ LLM Bench — 1B–14B Parameter LLM Benchmarking (llama.cpp)

A **live, real-working web app** to download, load and benchmark **1B–14B parameter GGUF LLMs**
through **llama-cpp (llama-cpp-python)** with **Q3/Q4 quantization** — with custom prompts,
uploaded-file analysis, real hardware telemetry and **CSV / Excel export**.

Open the app: **http://localhost:8000** (start with `python3 -m uvicorn app:app --host 0.0.0.0 --port 8000`).

---

## 🚀 Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # or your existing venv
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

GPU inference: build llama-cpp-python with your backend (Jetson/CUDA:
`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`, or use the `cu124` prebuilt wheel index).
The app always requests full GPU offload (`n_gpu_layers=-1`); a CPU build simply runs on CPU.

## ✨ Features

- **13 GGUF models, 1B–14B** (Qwen 2.5 1.5B/3B/7B, Gemma 2 2B/9B, Llama 3.2 3B + Llama 3.2 1B
  diagnostic, Llama 3.1 8B, Phi-3.5 mini 3.8B, Qwen 3 4B, Mistral 7B, Mistral Nemo 12B, Phi-4 14B),
  one-click download with progress + resume (Q3_K_S/M/L, Q4_0/K_S/K_M).
- **Custom prompts** (one per line) replacing the built-in 12-prompt pool.
- **File upload analysis**: txt/md/csv/json/log/code, PDF (pypdf), Word (python-docx),
  Excel (openpyxl), images (Pillow metadata + OCR if tesseract is installed).
  Files are analysed **first in the queue**; documents larger than the context window are
  **automatically split into labelled parts** so the entire document is analysed — nothing is
  silently truncated.
- **Full answers**: inference runs through each model's **chat template** and stops only at the
  model's natural end-of-turn (EOS/EOT) when “Response length = Full” — or at an explicit token
  cap you choose (64…2048). The **complete response** is viewable per row and included in both exports.
- **Prompt count reached → “Add more prompts or Exit”** modal; batches append to one session.
- **Metrics + telemetry** per prompt (details below), live chips while running, AVERAGE row,
  CSV/Excel download (copies also saved to `exports/`).

## 📐 How every displayed parameter is measured

All timings use Python's `time.perf_counter()` (monotonic, ns-resolution). Hardware metrics are
sampled by a background thread at **5 Hz**; each row averages the samples whose timestamps fall
inside that prompt's inference window `[t_start − 0.5 s, t_end + 0.3 s]`.

| # | Parameter | Exactly how it is calculated/measured |
|---|---|---|
| 1 | **Prompt tokens** | `llm.n_tokens` (KV-cache token count after the run) − generated tokens. This is the *exact* count of prompt tokens llama.cpp evaluated, **including the chat-template wrapper** (system/user scaffolding). |
| 2 | **Gen tokens** | The complete streamed output text is re-tokenised: `len(llm.tokenize(output, add_bos=False))`. |
| 3 | **Prompt tok/s** | `prompt_tokens ÷ TTFT`. TTFT (time-to-first-token) is the wall time from request start until the first streamed token is produced — i.e. the prompt-processing (prefill) phase. |
| 4 | **Gen tok/s** | `(gen_tokens − 1) ÷ (t_end − t_first_token)` — decode throughput over the exact streaming window (the −1 removes the first token, which belongs to TTFT). |
| 5 | **TTFT (s)** | Wall time from request dispatch to first content token received. |
| 6 | **Total inference (s)** | Wall time for the whole request: end of stream − start (prefill + decode + sampling overhead). |
| 7 | **CPU util %** | `psutil.cpu_percent()` every 0.2 s (system-wide, 0–100 % across all cores), averaged over the prompt window. Note: first sample after idle can read low; values during inference reflect true load. |
| 8 | **RAM util % / RAM (GB)** | `psutil.virtual_memory().percent` and `.used/1e9` sampled at 5 Hz — full-system memory pressure (model is mmap'ed, page cache counts). |
| 9 | **GPU util %** | Priority chain: ① NVML `nvmlDeviceGetUtilizationRates().gpu` (desktop NVIDIA) ② Jetson `tegrastats` field `GR3D_FREQ <n>%` ③ Jetson sysfs `/sys/devices/gpu.0/load` (permille ÷ 10). First available source wins per sample. |
| 10 | **Power (W)** | ① NVML `nvmlDeviceGetPowerUsage()/1000` ② Jetson `tegrastats` `VDD_IN <n>mW ÷ 1000` (whole-module input rail) ③ Intel RAPL energy deltas `Δenergy_uj/Δt` ④ INA3221 hwmon rails (`power*_input` µW summed ÷ 1e6). |
| 11 | **Temp (°C)** | Max of available sensors per sample: NVML GPU temp, Linux `/sys/class/thermal/thermal_zone*/temp`, `psutil.sensors_temperatures()`, Jetson `tegrastats GPU@xxC`. Row shows the window **peak**. |
| 12 | **Achieved TOPS** | `2 × params × (prompt_tokens + gen_tokens) ÷ total_inference_seconds ÷ 10¹²`. Rationale: a forward pass of a dense transformer costs ≈ 2·N FLOPs per token (one multiply + one accumulate per parameter — the standard Kaplan/Chinchilla accounting; attention/KV-cache terms are <5 % at these context lengths and quantization does not change the *operation count*, only operand width). This is the dense-equivalent compute actually sustained over the whole request — the exact-by-measurement form used when hardware FLOP counters don't exist on consumer chips. |
| 13 | **Device peak TOPS** | Static vendor spec resolved from `/proc/device-tree/model` (Jetson AGX Orin 275, Orin NX 16 GB 100 / 8 GB 70, Orin Nano 8 GB 40 / 4 GB 20, Xavier NX 21, AGX Xavier 32 — INT8 sparse marketing peak). Unknown device → “—”. |
| 14 | **TOPS util %** | `achieved TOPS ÷ device peak TOPS × 100`. Caveat: achieved FLOPs are dense-equivalent while the peak is INT8 sparse — the percentage is an honest *indicator of headroom*, not a claim of inefficiency. |
| 15 | **AVERAGE row** | Column-wise arithmetic mean over all recorded rows (both exports and table foot). |

**Sampling/accuracy notes.** Prompt tok/s measures wall-clock TTFT, which includes llama.cpp's
automatic KV **prefix-cache reuse** (the shared system prompt is evaluated once and reused across
rows — you'll see prompt tok/s rise on later rows; that is real llama.cpp behaviour, not an error).
Generation tok/s excludes it. Seeds: fix the seed + temperature 0 for reproducible rows.

## 🗂 File analysis pipeline

1. Upload (≤ 40 MB each, ≤ 10 files) → text extraction per type (see table in UI).
2. At run time each document is tokenised with the loaded model's tokenizer and, if it exceeds the
   free context budget (`n_ctx − response_reserve − 240`), split into **labelled parts** —
   every part becomes its own analysis row in the table (part i/N), so the whole document is covered.
3. Images: GGUF models here are text-only — images contribute format/dimension metadata, plus OCR
   text when `tesseract-ocr` + `pytesseract` are installed; otherwise the row clearly says
   “metadata only”.

## 📁 Layout

```
llm-bench/
├── app.py              # FastAPI backend: catalog, downloader, chat inference, telemetry, exports
├── static/index.html   # Web UI (single file, no external assets)
├── requirements.txt
├── models/             # Downloaded .gguf files (+ .part resumes)
├── uploads/            # Uploaded documents
└── exports/            # Saved CSV / Excel copies (full Response column included)
```

**REST API**: `GET /api/models` · `POST /api/download` · `POST /api/load` · `POST /api/run`
(`max_tokens: -1` = full answer) · `POST /api/run/stop` · `POST /api/run/finish` ·
`POST /api/prompts` · `POST /api/upload` · `DELETE /api/custom` · `GET /api/status` ·
`GET /api/results` · `GET /api/export.csv` · `GET /api/export.xlsx`.
