# ⚡ LLM Bench — 1B–14B Parameter LLM Benchmarking (llama.cpp)

A **live, real-working web app** to download, load and benchmark **1B–14B parameter GGUF LLMs**
through **llama-cpp (llama-cpp-python)** with **Q3/Q4 quantization**, recording real hardware and
inference metrics per prompt, with one-click **CSV / Excel export**.

Open the app: **http://localhost:8000** (start with `python3 -m uvicorn app:app --host 0.0.0.0 --port 8000`).

---

## 🚀 Quick start (fix for “Download failed: No module named …”)

The app no longer depends on `huggingface_hub` at all. Install these into **the same venv that runs the server**:

```bash
python3 -m venv .venv && source .venv/bin/activate     # or use your existing venv
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

If a package is missing, a red banner at the top of the UI now tells you exactly what to install.
Downloads use direct HTTPS streaming with **resume** (`.part` files) and automatic retries — interrupted
downloads continue where they left off. GPU inference: use a CUDA build of llama-cpp-python instead
(`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` or the `cu124` prebuilt wheel index).

## ✨ Features

- **Model catalog (13 models, 1B–14B)** — one-click download from Hugging Face (bartowski GGUF mirrors):

  | Group | Model | Params | Q3 sizes | Q4 sizes |
  |---|---|---|---|---|
  | 1B–4B | Qwen 2.5 1.5B Instruct | 1.54B | 0.76–0.88 GB | 0.94–0.99 GB |
  | 1B–4B | Gemma 2 2B IT | 2.61B | 1.55 GB (K_L) | 1.64–1.71 GB |
  | 1B–4B | Qwen 2.5 3B Instruct | 3.09B | 1.45–1.71 GB | 1.83–1.93 GB |
  | 1B–4B | Llama 3.2 3B Instruct | 3.21B | 1.82 GB (K_L) | 1.92–2.02 GB |
  | 1B–4B | Phi-3.5 Mini Instruct | 3.8B | 1.68–2.09 GB | 2.18–2.39 GB |
  | 1B–4B | Qwen 3 4B | 4.02B | 1.89–2.24 GB | 2.38–2.50 GB |
  | 6B–8B | Mistral 7B Instruct v0.3 | 7.25B | 3.17–3.83 GB | 4.14–4.37 GB |
  | 6B–8B | Qwen 2.5 7B Instruct | 7.62B | 3.49–4.09 GB | 4.46–4.68 GB |
  | 8B–10B | Llama 3.1 8B Instruct | 8.03B | 3.66–4.32 GB | 4.69–4.92 GB |
  | 8B–10B | Gemma 2 9B IT | 9.24B | 4.34–5.13 GB | 5.48–5.76 GB |
  | 10B–14B | Mistral Nemo 12B Instruct | 12.2B | 5.53–6.56 GB | 7.12–7.48 GB |
  | 10B–14B | Phi-4 | 14.7B | 6.50–7.93 GB | 8.44–9.05 GB |
  | Diagnostic | Llama 3.2 1B Instruct | 1.24B | 0.73 GB | 0.77–0.81 GB |

  *(Quant choices: Q3_K_S / Q3_K_M / Q3_K_L and Q4_0 / Q4_K_S / Q4_K_M — only quants that actually
  exist in each repo are listed. Sizes verified against the HF API.)*

- **Quantisation: Q3 or Q4** as requested.
- **Enter the number of prompts** → runs them one-by-one from a 12-prompt mixed pool.
- **Prompt count reached → modal asks “Add more prompts or Exit”** (as specified).
- **Layout**: sticky control sidebar (model setup + run settings) beside a full-height results area
  with live metric chips in the toolbar. Temperature & seed controls for reproducible runs.
- **Pre-load safety checks**: refuses to load a model larger than available RAM, with a clear message.
- **Live telemetry + metrics table**: `GPU util %`, `CPU util %`, `Prompt tok/s`, `Generation tok/s`,
  `Total inference time (s)`, `RAM util %` (+ GB), `Power (W)`, `Temperature (°C)`, `TOPS`,
  plus timestamps, token counts and TTFT. Click a prompt row to see the full generated output.
- **Export: CSV and Excel (.xlsx)** — browser download *and* a copy saved in `llm-bench/exports/`.

## 📐 How metrics are measured

| Metric | Source |
|---|---|
| Prompt tokens/s | prompt tokens ÷ time-to-first-token (perf_counter wall clock) |
| Generation tokens/s | streamed tokens ÷ generation window |
| Total inference time | end-to-end wall clock per prompt |
| CPU / RAM utilisation | `psutil`, sampled at 5 Hz during each prompt |
| GPU util / power / temperature | NVIDIA **NVML** (`pynvml`) when a GPU is present |
| CPU power / temperature | Intel **RAPL** (`/sys/class/powercap`) + `/sys` thermal zones when available |
| TOPS | `2 × params × gen tok/s ÷ 10¹²` (effective dense-model compute) |

A bold **AVERAGE row** is appended in the table and both exports. GPU/power/temperature show “—”
when no GPU/sensors exist — nothing is simulated.

## 📁 Layout

```
llm-bench/
├── app.py              # FastAPI backend: catalog, streaming downloader, engine, telemetry, exports
├── static/index.html   # Web UI (single file, no external assets)
├── requirements.txt
├── models/             # Downloaded .gguf files (+ .part resumes)
└── exports/            # Saved CSV / Excel copies
```

**REST API**: `GET /api/models` · `POST /api/download` · `POST /api/load` · `POST /api/run` ·
`POST /api/run/stop` · `POST /api/run/finish` · `GET /api/status` · `GET /api/results` ·
`GET /api/export.csv` · `GET /api/export.xlsx`.
