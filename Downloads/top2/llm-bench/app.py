"""
LLM Bench — live benchmarking app for 1B–14B GGUF LLMs via llama.cpp.

Backend: FastAPI + llama-cpp-python.
Downloads: direct HTTPS streaming (huggingface.co /resolve/) with resume + retries.
Prompts: built-in pool, user-typed prompts, and uploaded files (txt/md/csv/log, PDF, DOCX,
         XLSX, images) — extracted text is analysed by the loaded model.
Metrics: psutil (CPU/RAM); NVIDIA NVML, Jetson tegrastats, Jetson sysfs, RAPL, /sys thermal.
TOPS: achieved = 2·params·(prompt+gen tokens)/inference seconds (dense-equivalent),
      shown next to the device peak (spec) read from /proc/device-tree/model.
Exports: CSV / Excel.
"""
import os
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import csv
import gc
import glob
import io
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
for d in (MODELS_DIR, EXPORTS_DIR, UPLOADS_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Optional-dependency self-check (surfaced in /api/status and the UI banner)
# --------------------------------------------------------------------------------------
def _opt(modname):
    try:
        m = __import__(modname)
        return getattr(m, "__version__", "ok")
    except Exception:
        return None

ENV = {k: _opt(k) for k in ("llama_cpp", "openpyxl", "multipart", "pypdf", "docx", "PIL")}
ENV["missing"] = [k for k in ("llama_cpp", "openpyxl") if ENV[k] is None]
ENV_FILE_MISSING = [k for k in ("pypdf", "docx", "PIL", "multipart") if ENV[k] is None]

# --------------------------------------------------------------------------------------
# Model catalog — file names + sizes verified against the Hugging Face API.
# --------------------------------------------------------------------------------------
CATALOG = {
    "qwen2.5-1.5b-instruct": {
        "name": "Qwen 2.5 1.5B Instruct", "params_b": 1.54, "group": "1B–4B",
        "repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF", "base": "Qwen2.5-1.5B-Instruct",
        "quants": {"Q3_K_S": 0.76, "Q3_K_M": 0.82, "Q3_K_L": 0.88, "Q4_0": 0.94, "Q4_K_S": 0.94, "Q4_K_M": 0.99},
    },
    "gemma-2-2b-it": {
        "name": "Gemma 2 2B IT", "params_b": 2.61, "group": "1B–4B",
        "repo": "bartowski/gemma-2-2b-it-GGUF", "base": "gemma-2-2b-it",
        "quants": {"Q3_K_L": 1.55, "Q4_K_S": 1.64, "Q4_K_M": 1.71},
    },
    "qwen2.5-3b-instruct": {
        "name": "Qwen 2.5 3B Instruct", "params_b": 3.09, "group": "1B–4B",
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF", "base": "Qwen2.5-3B-Instruct",
        "quants": {"Q3_K_S": 1.45, "Q3_K_M": 1.59, "Q3_K_L": 1.71, "Q4_0": 1.83, "Q4_K_S": 1.83, "Q4_K_M": 1.93},
    },
    "llama-3.2-3b-instruct": {
        "name": "Llama 3.2 3B Instruct", "params_b": 3.21, "group": "1B–4B",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF", "base": "Llama-3.2-3B-Instruct",
        "quants": {"Q3_K_L": 1.82, "Q4_0": 1.92, "Q4_K_S": 1.93, "Q4_K_M": 2.02},
    },
    "phi-3.5-mini-instruct": {
        "name": "Phi-3.5 Mini 3.8B Instruct", "params_b": 3.8, "group": "1B–4B",
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF", "base": "Phi-3.5-mini-instruct",
        "quants": {"Q3_K_S": 1.68, "Q3_K_M": 1.96, "Q3_K_L": 2.09, "Q4_0": 2.18, "Q4_K_S": 2.19, "Q4_K_M": 2.39},
    },
    "qwen3-4b": {
        "name": "Qwen 3 4B", "params_b": 4.02, "group": "1B–4B",
        "repo": "bartowski/Qwen_Qwen3-4B-GGUF", "base": "Qwen_Qwen3-4B",
        "quants": {"Q3_K_S": 1.89, "Q3_K_M": 2.08, "Q3_K_L": 2.24, "Q4_0": 2.38, "Q4_K_S": 2.38, "Q4_K_M": 2.50},
    },
    "mistral-7b-instruct-v0.3": {
        "name": "Mistral 7B Instruct v0.3", "params_b": 7.25, "group": "6B–8B",
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "base": "Mistral-7B-Instruct-v0.3",
        "quants": {"Q3_K_S": 3.17, "Q3_K_M": 3.52, "Q3_K_L": 3.83, "Q4_K_S": 4.14, "Q4_K_M": 4.37},
    },
    "llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B Instruct", "params_b": 8.03, "group": "8B–10B",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "base": "Meta-Llama-3.1-8B-Instruct",
        "quants": {"Q3_K_S": 3.66, "Q3_K_M": 4.02, "Q3_K_L": 4.32, "Q4_K_S": 4.69, "Q4_K_M": 4.92},
    },
    "qwen2.5-7b-instruct": {
        "name": "Qwen 2.5 7B Instruct", "params_b": 7.62, "group": "6B–8B",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF", "base": "Qwen2.5-7B-Instruct",
        "quants": {"Q3_K_S": 3.49, "Q3_K_M": 3.81, "Q3_K_L": 4.09, "Q4_K_S": 4.46, "Q4_K_M": 4.68},
    },
    "gemma-2-9b-it": {
        "name": "Gemma 2 9B IT", "params_b": 9.24, "group": "8B–10B",
        "repo": "bartowski/gemma-2-9b-it-GGUF", "base": "gemma-2-9b-it",
        "quants": {"Q3_K_S": 4.34, "Q3_K_M": 4.76, "Q3_K_L": 5.13, "Q4_K_S": 5.48, "Q4_K_M": 5.76},
    },
    "mistral-nemo-12b-instruct": {
        "name": "Mistral Nemo 12B Instruct", "params_b": 12.2, "group": "10B–14B",
        "repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "base": "Mistral-Nemo-Instruct-2407",
        "quants": {"Q3_K_S": 5.53, "Q3_K_M": 6.08, "Q3_K_L": 6.56, "Q4_K_S": 7.12, "Q4_K_M": 7.48},
    },
    "phi-4-14b": {
        "name": "Phi-4 14B", "params_b": 14.7, "group": "10B–14B",
        "repo": "bartowski/phi-4-GGUF", "base": "phi-4",
        "quants": {"Q3_K_S": 6.50, "Q3_K_M": 7.36, "Q3_K_L": 7.93, "Q4_K_S": 8.44, "Q4_K_M": 9.05},
    },
    "llama-3.2-1b-diagnostic": {
        "name": "Llama 3.2 1B Instruct (diagnostic — for very low-RAM smoke tests)",
        "params_b": 1.24, "group": "Diagnostic",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF", "base": "Llama-3.2-1B-Instruct",
        "quants": {"Q3_K_L": 0.73, "Q4_0": 0.77, "Q4_K_S": 0.78, "Q4_K_M": 0.81},
    },
}

PROMPT_POOL = [
    "Explain the difference between TCP and UDP to a junior network engineer in about 120 words.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain its time complexity.",
    "Summarize the key causes and consequences of the French Revolution in five bullet points.",
    "Write a short product description (about 80 words) for noise-cancelling headphones aimed at remote workers.",
    "What is the time complexity of merge sort? Derive it step by step.",
    "Draft a polite email to a colleague asking to reschedule Thursday's design review to next week.",
    "List and briefly explain four common techniques to reduce overfitting in neural networks.",
    "Explain what a hash collision is and describe two ways a hash table resolves collisions.",
    "Write a SQL query to find the top 5 customers by total order value from tables customers(id,name) and orders(id,customer_id,amount), and explain it.",
    "Give a concise comparison of electric cars versus hybrid cars in terms of cost, range, and emissions.",
    "Explain the greenhouse effect in simple terms suitable for a 12-year-old student.",
    "Write a bash one-liner that finds the ten largest files under /var and explain how it works.",
]

ANALYSIS_TEMPLATE = (
    "You are a meticulous data analyst. Analyse the document excerpt below and respond with:\n"
    "1) a concise summary,\n2) key facts or numbers,\n3) notable patterns or anomalies,\n"
    "4) one actionable insight.\n\n"
    "### DOCUMENT: {name} ({kind})\n{body}\n### END OF DOCUMENT\n\nAnalysis:"
)

# --------------------------------------------------------------------------------------
# Global state
# --------------------------------------------------------------------------------------
LLM = None
LOADED = None
RESULTS = []
DL = {}
RUN = {
    "phase": "idle",
    "target": 0, "done_in_batch": 0, "total_done": 0,
    "current_prompt": None, "live_gen_tps": None, "live_elapsed": 0.0,
    "stop": False, "error": None, "max_tokens": 128,
}
PROMPT_CACHE = {}
LAST_OUTPUTS = {}
CUSTOM = {"prompts": [], "files": []}   # user prompts + uploaded/extracted documents

# --------------------------------------------------------------------------------------
# Hardware monitor
# --------------------------------------------------------------------------------------
SAMPLES = deque(maxlen=1800)
LATEST = {}
GPU = {"available": False, "name": None, "handle": None}
TEGRA = {"active": False, "gpu_pct": None, "power_w": None, "temp_c": None}
TEGRA_SYS = {"checked": False, "load_path": None, "power_paths": []}
PEAK = None                     # {"model": str, "tops": float, "precision": str}


def _init_gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        GPU.update({"available": True, "name": name, "handle": h, "pynvml": pynvml})
    except Exception:
        GPU.update({"available": False})


def _detect_peak():
    """Device theoretical peak TOPS (vendor spec) — Jetson SoCs via device-tree model."""
    global PEAK
    try:
        model = Path("/proc/device-tree/model").read_bytes().decode("utf-8", "ignore").strip("\x00 ")
    except Exception:
        return
    table = [
        ("agx orin 64gb", 275), ("agx orin 32gb", 275), ("agx orin industrial", 248), ("agx orin", 275),
        ("orin nx 16gb", 100), ("orin nx 8gb", 70), ("orin nx", 100),
        ("orin nano 8gb", 40), ("orin nano 4gb", 20), ("orin nano", 40),
        ("jetson orin", 40),
        ("xavier nx", 21), ("agx xavier", 32),
    ]
    low = model.lower()
    for pat, tops in table:
        if pat in low:
            PEAK = {"model": model, "tops": tops, "precision": "INT8 (vendor sparse peak)"}
            return
    if "orin" in low:
        PEAK = {"model": model, "tops": 40, "precision": "INT8 (vendor sparse peak)"}


def _tegra_reader():
    try:
        if not shutil.which("tegrastats"):
            return
        rex_gpu = re.compile(r"GR3D_FREQ\s+(\d+)%")
        rex_pwr = re.compile(r"VDD_IN\s+(\d+)mW")
        rex_gtc = re.compile(r"GPU@([\d.]+)C")
        proc = subprocess.Popen(["tegrastats", "--interval", "300"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, bufsize=1)
        TEGRA["proc"] = proc
        TEGRA["active"] = True
        for line in proc.stdout:
            m = rex_gpu.search(line)
            if m:
                TEGRA["gpu_pct"] = int(m.group(1))
            m = rex_pwr.search(line)
            if m:
                TEGRA["power_w"] = round(int(m.group(1)) / 1000.0, 2)
            m = rex_gtc.search(line)
            if m:
                TEGRA["temp_c"] = float(m.group(1))
    except Exception:
        TEGRA["active"] = False


def _tegra_sysfs_probe():
    """Locate Jetson sysfs GPU-load + INA3221 power-rail files once."""
    for p in ["/sys/devices/gpu.0/load", "/sys/devices/platform/gpu.0/load",
              "/sys/devices/17000000.gpu/load"] + glob.glob("/sys/devices/platform/*.gpu/load"):
        if Path(p).exists():
            TEGRA_SYS["load_path"] = p
            break
    paths = set()
    for pat in ("/sys/bus/i2c/drivers/ina3221x/*/in_power*_input",
                "/sys/class/hwmon/hwmon*/power*_input",
                "/sys/bus/i2c/devices/*/hwmon/hwmon*/power*_input"):
        paths.update(glob.glob(pat))
    TEGRA_SYS["power_paths"] = sorted(paths)
    TEGRA_SYS["checked"] = True


def _tegra_sysfs():
    util = watts = None
    if not TEGRA_SYS["checked"]:
        _tegra_sysfs_probe()
    if TEGRA_SYS["load_path"]:
        try:
            util = round(int(Path(TEGRA_SYS["load_path"]).read_text().strip()) / 10, 1)  # permille → %
        except Exception:
            pass
    total = 0
    for p in TEGRA_SYS["power_paths"]:
        try:
            total += int(Path(p).read_text().strip())
        except Exception:
            pass
    if total:
        watts = round(total / 1e6, 2)   # µW → W
    return util, watts


def _read_rapl_energy_uj():
    for p in glob.glob("/sys/class/powercap/intel-rapl:0/energy_uj") + \
              glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"):
        try:
            return int(Path(p).read_text().strip())
        except Exception:
            continue
    return None


def _read_cpu_temp_c():
    try:
        temps = []
        for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                temps.append(int(Path(z).read_text().strip()) / 1000.0)
            except Exception:
                pass
        if temps:
            return max(temps)
        st = getattr(psutil, "sensors_temperatures", lambda: {})()
        for arr in st.values():
            for e in arr:
                temps.append(e.current)
        return max(temps) if temps else None
    except Exception:
        return None


def _sampler():
    global LATEST
    psutil.cpu_percent(None)
    e0, t0 = _read_rapl_energy_uj(), time.time()
    while True:
        try:
            vm = psutil.virtual_memory()
            s = {
                "t": time.time(), "tp": time.perf_counter(),
                "cpu_pct": psutil.cpu_percent(None),
                "ram_pct": vm.percent,
                "ram_used_gb": round(vm.used / 1e9, 2),
                "gpu_pct": None, "power_w": None, "temp_c": _read_cpu_temp_c(),
            }
            if GPU["available"]:
                pynvml = GPU["pynvml"]
                try:
                    s["gpu_pct"] = pynvml.nvmlDeviceGetUtilizationRates(GPU["handle"]).gpu
                    s["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(GPU["handle"]) / 1000.0, 1)
                    gt = pynvml.nvmlDeviceGetTemperature(GPU["handle"], 0)
                    s["temp_c"] = max(gt, s["temp_c"] or 0)
                except Exception:
                    pass
            else:
                e1, t1 = _read_rapl_energy_uj(), time.time()
                if e0 is not None and e1 is not None and t1 > t0 and e1 >= e0:
                    s["power_w"] = round((e1 - e0) / 1e6 / (t1 - t0), 1)
                if e1 is not None:
                    e0, t0 = e1, t1
            if TEGRA["active"]:                       # Jetson: tegrastats fills NVML gaps
                if s["gpu_pct"] is None and TEGRA.get("gpu_pct") is not None:
                    s["gpu_pct"] = TEGRA["gpu_pct"]
                if s["power_w"] is None and TEGRA.get("power_w") is not None:
                    s["power_w"] = TEGRA["power_w"]
                if TEGRA.get("temp_c"):
                    s["temp_c"] = max(s["temp_c"] or 0, TEGRA["temp_c"])
            if s["gpu_pct"] is None or s["power_w"] is None:   # Jetson sysfs fallback
                u, w = _tegra_sysfs()
                if s["gpu_pct"] is None:
                    s["gpu_pct"] = u
                if s["power_w"] is None:
                    s["power_w"] = w
            LATEST = s
            SAMPLES.append(s)
        except Exception:
            pass
        time.sleep(0.2)


def window_stats(tp_start, tp_end):
    rows = [s for s in list(SAMPLES) if tp_start - 0.5 <= s["tp"] <= tp_end + 0.3]

    def avg(key):
        vals = [s[key] for s in rows if s.get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    temps = [s["temp_c"] for s in rows if s.get("temp_c") is not None]
    return {
        "cpu_pct": avg("cpu_pct"), "ram_pct": avg("ram_pct"), "ram_gb": avg("ram_used_gb"),
        "gpu_pct": avg("gpu_pct"), "power_w": avg("power_w"),
        "temp_c": round(max(temps), 1) if temps else None,
    }


# --------------------------------------------------------------------------------------
# File ingestion (txt / pdf / docx / xlsx / images)
# --------------------------------------------------------------------------------------
def _extract_text(path: Path):
    ext = path.suffix.lower()
    name = path.name
    if ext in (".txt", ".md", ".csv", ".tsv", ".json", ".log", ".py", ".js", ".html", ".xml"):
        return "text", path.read_text(errors="ignore")[:400_000]
    if ext == ".pdf":
        if ENV["pypdf"] is None:
            raise RuntimeError("pip install pypdf  (needed for PDF files)")
        from pypdf import PdfReader
        r = PdfReader(str(path))
        parts = []
        for pg in r.pages[:30]:
            parts.append(pg.extract_text() or "")
        text = "\n".join(parts).strip()
        if not text:
            raise RuntimeError("no extractable text layer (scanned PDF — needs OCR)")
        return "pdf", text[:400_000]
    if ext in (".docx",):
        if ENV["docx"] is None:
            raise RuntimeError("pip install python-docx  (needed for Word files)")
        import docx
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for tbl in d.tables[:10]:
            for row in tbl.rows[:100]:
                parts.append(" | ".join(c.text for c in row.cells))
        return "word", "\n".join(parts)[:400_000]
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        from openpyxl import load_workbook
        wb = load_workbook(str(path), read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets[:5]:
            parts.append(f"[sheet: {ws.title}]")
            for i, row in enumerate(ws.iter_rows(max_row=200, max_col=20, values_only=True)):
                vals = ["" if v is None else str(v) for v in row]
                if any(vals):
                    parts.append("\t".join(vals))
                if i >= 200:
                    parts.append("…(truncated)…")
                    break
        return "excel", "\n".join(parts)[:400_000]
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"):
        if ENV["PIL"] is None:
            raise RuntimeError("pip install pillow  (needed for images)")
        from PIL import Image
        im = Image.open(str(path))
        meta = (f"Image metadata: format={im.format}, size={im.size[0]}x{im.size[1]} px, "
                f"mode={im.mode}, file_bytes={path.stat().st_size}.")
        text = ""
        if shutil.which("tesseract"):
            try:
                import pytesseract
                text = pytesseract.image_to_string(im).strip()
            except Exception:
                text = ""
        if text:
            return "image+OCR", meta + "\nOCR text:\n" + text[:200_000]
        return "image-metadata", (meta + "\n(No OCR available — install tesseract + pytesseract "
                                  "to extract text from images. These GGUF models are text-only and "
                                  "cannot view pixels; the analysis is based on metadata/OCR text.)")
    raise RuntimeError(f"unsupported file type: {ext}")


def _file_prompt(frec):
    budget = max(256, (LOADED["n_ctx"] if LOADED else 2048) - RUN["max_tokens"] - 160)
    body = frec["text"]
    try:
        toks = LLM.tokenize(body.encode("utf-8"), add_bos=False, special=False)
        if len(toks) > budget:
            body = LLM.detokenize(toks[:budget]).decode("utf-8", "ignore") + "\n…(truncated)…"
    except Exception:
        if len(body) > budget * 4:
            body = body[: budget * 4] + "\n…(truncated)…"
    return ANALYSIS_TEMPLATE.format(name=frec["name"], kind=frec["kind"], body=body)


# --------------------------------------------------------------------------------------
# Inference engine
# --------------------------------------------------------------------------------------
def _tok_count(text, bos=False):
    return len(LLM.tokenize(text.encode("utf-8"), add_bos=bos, special=True))


def _infer_one(prompt, max_tokens, temperature, seed, params_b):
    if prompt not in PROMPT_CACHE:
        PROMPT_CACHE[prompt] = _tok_count(prompt, bos=True)
    prompt_tokens = PROMPT_CACHE[prompt]

    t_start = time.perf_counter()
    t_first = None
    pieces, n_chunks = [], 0
    RUN["current_prompt"] = prompt[:80]

    kwargs = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.95, stream=True)
    if seed is not None and seed >= 0:
        kwargs["seed"] = seed

    for chunk in LLM(prompt, **kwargs):
        if t_first is None:
            t_first = time.perf_counter()
        pieces.append(chunk["choices"][0].get("text", ""))
        n_chunks += 1
        if t_first:
            gt = time.perf_counter() - t_first
            RUN["live_gen_tps"] = round(max(n_chunks - 1, 0) / gt, 2) if gt > 0.02 else None
        RUN["live_elapsed"] = round(time.perf_counter() - t_start, 1)
        if RUN["stop"]:
            break
    t_end = time.perf_counter()

    text = "".join(pieces)
    gen_tokens = _tok_count(text) if text else max(n_chunks - 1, 0)
    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    gen_time = (t_end - t_first) if t_first else 0.0
    total_time = t_end - t_start

    prompt_tps = round(prompt_tokens / ttft, 2) if ttft > 0 else None
    gen_tps = round((gen_tokens - 1) / gen_time, 2) if gen_time > 0.02 and gen_tokens > 1 else None
    # Achieved dense-equivalent compute over the WHOLE request (prefill + generation):
    tops = round(2 * params_b * 1e9 * (prompt_tokens + gen_tokens) / total_time / 1e12, 4) \
        if total_time > 0 else None
    tops_util = round(tops / PEAK["tops"] * 100, 2) if (tops and PEAK) else None
    hw = window_stats(t_start, t_end)

    return {
        "prompt": prompt, "output": text,
        "prompt_tokens": prompt_tokens, "completion_tokens": gen_tokens,
        "prompt_tps": prompt_tps, "gen_tps": gen_tps,
        "ttft_s": round(ttft, 2), "total_time_s": round(total_time, 2),
        "tops": tops, "peak_tops": PEAK["tops"] if PEAK else None, "top_util_pct": tops_util,
        **hw,
    }


def _build_queue(n):
    """Queue for this batch: pending uploaded-file analyses first, then custom prompts
    (or the built-in pool) cycling."""
    cycle = CUSTOM["prompts"][:] or PROMPT_POOL
    files = [f for f in CUSTOM["files"] if f.get("ok")]
    start = len(RESULTS)
    q = []
    for i in range(n):
        pos = start + i
        if pos < len(files):
            q.append(_file_prompt(files[pos]))
        else:
            q.append(cycle[(pos - len(files)) % len(cycle)])
    return q


def _run_engine(n_prompts, max_tokens, temperature, seed):
    try:
        RUN.update({"phase": "running", "target": n_prompts,
                    "done_in_batch": 0, "stop": False, "error": None})
        for prompt in _build_queue(n_prompts):
            if RUN["stop"]:
                break
            row = _infer_one(prompt, max_tokens, temperature, seed,
                             CATALOG[LOADED["model"]]["params_b"])
            row.update({
                "idx": len(RESULTS) + 1,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": CATALOG[LOADED["model"]]["name"],
                "quant": LOADED["quant"],
            })
            LAST_OUTPUTS[row["idx"]] = row.pop("output")
            RESULTS.append(row)
            RUN["done_in_batch"] += 1
            RUN["total_done"] = len(RESULTS)
        RUN["phase"] = "awaiting"
    except Exception:
        RUN["error"] = traceback.format_exc(limit=3)
        RUN["phase"] = "awaiting"
    finally:
        RUN["current_prompt"] = None
        RUN["live_gen_tps"] = None


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="LLM Bench")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def model_file(key, quant):
    return CATALOG[key]["base"] + "-" + quant + ".gguf"


class DownloadReq(BaseModel):
    model: str
    quant: str


class LoadReq(BaseModel):
    model: str
    quant: str
    n_ctx: int = 2048
    n_threads: int = 0


class RunReq(BaseModel):
    n_prompts: int = 3
    max_tokens: int = 128
    temperature: float = 0.7
    seed: int = -1


class PromptsReq(BaseModel):
    prompts: list


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/models")
def api_models():
    out = []
    for key, e in CATALOG.items():
        quants = {}
        for q, gb in e["quants"].items():
            part = MODELS_DIR / (model_file(key, q) + ".part")
            final = MODELS_DIR / model_file(key, q)
            quants[q] = {"size_gb": gb, "downloaded": final.exists(),
                         "partial_bytes": part.stat().st_size if part.exists() else 0}
        out.append({"key": key, "name": e["name"], "params_b": e["params_b"],
                    "group": e["group"], "repo": e["repo"], "quants": quants})
    return out


def _download_worker(key, quant):
    d = DL[f"{key}/{quant}"]
    url = f"https://huggingface.co/{CATALOG[key]['repo']}/resolve/main/{model_file(key, quant)}"
    dst = MODELS_DIR / model_file(key, quant)
    part = MODELS_DIR / (model_file(key, quant) + ".part")
    for attempt in range(3):
        try:
            resume = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={resume}-"} if resume else {}
            with requests.get(url, stream=True, headers=headers,
                              timeout=(20, 90), allow_redirects=True) as r:
                if r.status_code == 416:
                    part.rename(dst)
                    d.update({"active": False, "done": True, "pct": 100.0, "path": str(dst)})
                    return
                if r.status_code == 200 and resume:
                    resume = 0
                r.raise_for_status()
                length = int(r.headers.get("Content-Length") or 0)
                total = length + resume
                if total:
                    d["total"] = total
                with open(part, "ab" if resume and r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            d["done_bytes"] = part.stat().st_size
                            if d.get("total"):
                                d["pct"] = round(100.0 * d["done_bytes"] / d["total"], 1)
            if d.get("total") and part.stat().st_size < d["total"] - 1024:
                raise IOError(f"incomplete download ({part.stat().st_size}/{d['total']} bytes)")
            part.rename(dst)
            d.update({"active": False, "done": True, "pct": 100.0,
                      "path": str(dst), "done_bytes": dst.stat().st_size})
            return
        except Exception as ex:
            d["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            time.sleep(1.5 * (attempt + 1))
    d.update({"active": False, "done": False})


@app.post("/api/download")
def api_download(req: DownloadReq):
    if req.model not in CATALOG or req.quant not in CATALOG[req.model]["quants"]:
        return JSONResponse({"error": "unknown model/quant"}, 400)
    k = f"{req.model}/{req.quant}"
    path = MODELS_DIR / model_file(req.model, req.quant)
    if path.exists():
        DL[k] = {"active": False, "done": True, "pct": 100.0, "path": str(path),
                 "total": path.stat().st_size, "done_bytes": path.stat().st_size}
        return {"ok": True, "already": True}
    if DL.get(k, {}).get("active"):
        return {"ok": True, "already_running": True}
    DL[k] = {"active": True, "done": False, "pct": 0.0, "error": None, "done_bytes": 0,
             "total": int(CATALOG[req.model]["quants"][req.quant] * 1e9)}
    threading.Thread(target=_download_worker, args=(req.model, req.quant), daemon=True).start()
    return {"ok": True}


def _load_worker(key, quant, n_ctx, n_threads):
    global LLM, LOADED
    try:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run:  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
        if LLM is not None:
            LLM = None
            LOADED = None
            gc.collect()
        path = str(MODELS_DIR / model_file(key, quant))
        size_gb = Path(path).stat().st_size / 1e9
        if size_gb > psutil.virtual_memory().available / 1e9 * 0.98:
            raise RuntimeError(
                f"Not enough free RAM: model needs ~{size_gb:.1f} GB but only "
                f"{psutil.virtual_memory().available/1e9:.1f} GB is available.")
        LLM = Llama(
            model_path=path, n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 2)),
            n_gpu_layers=-1, verbose=False,
        )
        LOADED = {"model": key, "quant": quant, "n_ctx": n_ctx, "path": path,
                  "n_threads": n_threads or os.cpu_count()}
        RUN["phase"] = "idle"
    except Exception:
        RUN["phase"] = "idle"
        RUN["error"] = traceback.format_exc(limit=3)


@app.post("/api/load")
def api_load(req: LoadReq):
    if RUN["phase"] in ("loading", "running"):
        return JSONResponse({"error": "busy"}, 409)
    path = MODELS_DIR / model_file(req.model, req.quant)
    if not path.exists():
        return JSONResponse({"error": "model file not downloaded yet"}, 400)
    RUN["phase"] = "loading"
    threading.Thread(target=_load_worker,
                     args=(req.model, req.quant, req.n_ctx, req.n_threads), daemon=True).start()
    return {"ok": True}


@app.post("/api/run")
def api_run(req: RunReq):
    if LLM is None:
        return JSONResponse({"error": "no model loaded"}, 400)
    if RUN["phase"] == "running":
        return JSONResponse({"error": "run already in progress"}, 409)
    n = max(1, min(200, req.n_prompts))
    RUN["max_tokens"] = max(1, min(2048, req.max_tokens))
    temp = min(2.0, max(0.0, req.temperature))
    threading.Thread(target=_run_engine,
                     args=(n, RUN["max_tokens"], temp, req.seed), daemon=True).start()
    return {"ok": True}


@app.post("/api/run/stop")
def api_stop():
    RUN["stop"] = True
    return {"ok": True}


@app.post("/api/run/finish")
def api_finish():
    if RUN["phase"] == "awaiting":
        RUN["phase"] = "finished"
    return {"ok": True}


@app.post("/api/reset")
def api_reset():
    if RUN["phase"] == "running":
        return JSONResponse({"error": "stop the run first"}, 409)
    RESULTS.clear()
    LAST_OUTPUTS.clear()
    RUN.update({"phase": "idle", "total_done": 0, "done_in_batch": 0, "error": None})
    return {"ok": True}


# ------------------------- custom prompts & file uploads -------------------------
def _file_meta(f):
    return {"name": f["name"], "kind": f["kind"], "chars": f["chars"],
            "tokens_est": f["chars"] // 4, "ok": f["ok"], "error": f.get("error")}


@app.post("/api/prompts")
def api_prompts(req: PromptsReq):
    cleaned = [str(p).strip()[:8000] for p in req.prompts if str(p).strip()][:50]
    CUSTOM["prompts"] = cleaned
    return {"ok": True, "count": len(cleaned)}


@app.delete("/api/custom")
def api_custom_clear():
    CUSTOM["prompts"] = []
    CUSTOM["files"] = []
    return {"ok": True}


if ENV["multipart"]:
    from fastapi import File, UploadFile

    @app.post("/api/upload")
    async def api_upload(files: list[UploadFile] = File(...)):
        results = []
        for uf in files[:10]:
            name = Path(uf.filename or "file").name
            dst = UPLOADS_DIR / name
            data = await uf.read()
            if len(data) > 40 * 1024 * 1024:
                results.append({"name": name, "ok": False, "error": "file larger than 40 MB"})
                continue
            dst.write_bytes(data)
            try:
                kind, text = _extract_text(dst)
                rec = {"name": name, "kind": kind, "chars": len(text),
                       "ok": True, "text": text, "path": str(dst)}
                CUSTOM["files"] = [f for f in CUSTOM["files"] if f["name"] != name]
                CUSTOM["files"].append(rec)
                results.append(_file_meta(rec))
            except Exception as ex:
                results.append({"name": name, "ok": False,
                                "error": f"{type(ex).__name__}: {str(ex)[:160]}"})
        return {"ok": True, "files": results, "total": len(CUSTOM["files"])}
else:
    @app.post("/api/upload")
    async def api_upload_stub():
        return JSONResponse({"error": "file uploads need python-multipart — run: pip install python-multipart"}, 501)


@app.get("/api/status")
def api_status():
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE_DIR))
    return {
        "env": ENV,
        "hardware": {
            "cpu_physical": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_avail_gb": round(vm.available / 1e9, 2),
            "disk_free_gb": round(disk.free / 1e9, 1),
            "gpu": ({"available": True, "name": GPU["name"], "via": "NVML"}
                    if GPU["available"] else
                    ({"available": True, "name": (PEAK or {}).get("model", "GPU"), "via": "tegrastats"}
                     if TEGRA["active"] else {"available": False})),
            "peak": PEAK,
            "telemetry": {
                "nvml": GPU["available"],
                "tegrastats": TEGRA["active"],
                "jetson_sysfs": bool(TEGRA_SYS["checked"] and
                                     (TEGRA_SYS["load_path"] or TEGRA_SYS["power_paths"])),
                "rapl": _read_rapl_energy_uj() is not None,
            },
        },
        "sample": LATEST,
        "loaded": LOADED,
        "run": {k2: RUN[k2] for k2 in
                ("phase", "target", "done_in_batch", "total_done",
                 "current_prompt", "live_gen_tps", "live_elapsed", "error", "max_tokens")},
        "downloads": {k: {kk: d.get(kk) for kk in ("active", "done", "pct", "error", "total", "done_bytes")}
                      for k, d in DL.items()},
        "custom": {"prompts": len(CUSTOM["prompts"]),
                   "files": [_file_meta({**f, "chars": f.get("chars", 0)}) for f in CUSTOM["files"]]},
        "n_results": len(RESULTS),
        "prompt_pool_size": len(PROMPT_POOL),
    }


@app.get("/api/results")
def api_results():
    return {"rows": RESULTS}


@app.get("/api/results/{idx}/output")
def api_output(idx: int):
    return {"idx": idx, "output": LAST_OUTPUTS.get(idx, "")}


COLUMNS = [
    ("idx", "#"), ("ts", "Timestamp"), ("model", "Model"), ("quant", "Quant"),
    ("prompt", "Prompt"), ("prompt_tokens", "Prompt tokens"),
    ("completion_tokens", "Gen tokens"), ("prompt_tps", "Prompt tok/s"),
    ("gen_tps", "Gen tok/s"), ("ttft_s", "TTFT (s)"),
    ("total_time_s", "Total inference (s)"), ("cpu_pct", "CPU util %"),
    ("gpu_pct", "GPU util %"), ("ram_pct", "RAM util %"), ("ram_gb", "RAM used (GB)"),
    ("power_w", "Power (W)"), ("temp_c", "Temp (°C)"),
    ("tops", "Achieved TOPS"), ("peak_tops", "Device peak TOPS"), ("top_util_pct", "TOPS util %"),
]


def _avg_row():
    avg = {"prompt": "AVERAGE", "idx": "", "ts": "", "model": "", "quant": "",
           "prompt_tokens": "", "completion_tokens": "", "peak_tops": ""}
    for key, _ in COLUMNS:
        if key in avg:
            continue
        vals = [r[key] for r in RESULTS if isinstance(r.get(key), (int, float))]
        avg[key] = round(sum(vals) / len(vals), 2) if vals else None
    return avg


@app.get("/api/export.csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in COLUMNS])
    for r in RESULTS:
        w.writerow(["" if r.get(k) is None else (r.get(k) if k != "prompt" else r["prompt"][:120])
                    for k, _ in COLUMNS])
    if RESULTS:
        a = _avg_row()
        w.writerow(["" if a.get(k) is None else a.get(k) for k, _ in COLUMNS])
    fname = f"llm_bench_{datetime.now():%Y%m%d_%H%M%S}.csv"
    (EXPORTS_DIR / fname).write_text(buf.getvalue())
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/api/export.xlsx")
def export_xlsx():
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        return JSONResponse({"error": "openpyxl not installed — run: pip install openpyxl"}, 500)
    wb = Workbook()
    ws = wb.active
    ws.title = "benchmark"
    ws.append([h for _, h in COLUMNS])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    for r in RESULTS:
        ws.append([r.get(k) if k != "prompt" else r["prompt"][:120] for k, _ in COLUMNS])
    if RESULTS:
        a = _avg_row()
        ws.append([a.get(k) for k, _ in COLUMNS])
        for c in ws[ws.max_row]:
            c.font = Font(bold=True)
    for i, (k, _) in enumerate(COLUMNS, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = 34 if k == "prompt" else 14
    ws.freeze_panes = "A2"
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    fname = f"llm_bench_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    (EXPORTS_DIR / fname).write_bytes(bio.getvalue())
    return Response(
        bio.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.on_event("startup")
def _startup():
    _init_gpu()
    _detect_peak()
    _tegra_sysfs_probe()
    threading.Thread(target=_sampler, daemon=True).start()
    threading.Thread(target=_tegra_reader, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
