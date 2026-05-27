#!/usr/bin/env python3
"""
Automated PSNR/SSIM evaluation for Gaussian Splatting compression.

Usage:
    python eval_compression.py
"""

import os
import sys
import csv
import time
import multiprocessing
import numpy as np
import glfw
import OpenGL.GL as gl
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import imageio

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
os.chdir(dir_path)

import util
import util_gau
from gausCoder import GaussEncoder, GaussDecoder
from coderUtils import GaussLoader
from renderer_ogl import OpenGLRenderer

# ── Config ────────────────────────────────────────────────────────────────────

RENDER_W      = 1280
RENDER_H      = 720
RENDERS_DIR   = Path("./renders")
CAMERA_ANGLES = [0, 120, 240]

MAX_WORKERS = 2

COMPRESSION_CONFIGS = [
    ("q8",
     {"q": 8,    "hilbert": False, "po": None, "vq": None,  "compress": False}),
    ("q16",
     {"q": 16,   "hilbert": False, "po": None, "vq": None,  "compress": False}),
    ("hb-compress",
     {"q": None, "hilbert": True,  "po": None, "vq": None,  "compress": False}),
    ("vq256",
     {"q": None, "hilbert": False, "po": None, "vq": 256,   "compress": False}),
    ("vq4096",
     {"q": None, "hilbert": False, "po": None, "vq": 4096,  "compress": False}),
    ("q16-hb",
     {"q": 16,   "hilbert": True,  "po": None, "vq": None,  "compress": False}),
    ("q16-vq256",
     {"q": 16,   "hilbert": False, "po": None, "vq": 256,   "compress": False}),
    ("q16-vq4096",
     {"q": 16,   "hilbert": False, "po": None, "vq": 4096,  "compress": False}),
    ("hb-vq256",
     {"q": None, "hilbert": True,  "po": None, "vq": 256,   "compress": False}),
    ("hb-vq4096",
     {"q": None, "hilbert": True,  "po": None, "vq": 4096,  "compress": False}),
    ("q16-hb-vq256",
     {"q": 16,   "hilbert": True,  "po": None, "vq": 256,   "compress": False}),
    ("q16-hb-vq4096",
     {"q": 16,   "hilbert": True,  "po": None, "vq": 4096,  "compress": False}),
    ("po0.08-q16-hb",
     {"q": 16,   "hilbert": True,  "po": 0.08, "vq": None,  "compress": False}),
    ("po0.08-hb-vq256",
     {"q": None, "hilbert": True,  "po": 0.08, "vq": 256,   "compress": False}),
    ("po0.08-hb-vq4096",
     {"q": None, "hilbert": True,  "po": 0.08, "vq": 4096,  "compress": False}),    
     ("po0.12-hb-vq256",
     {"q": None, "hilbert": True,  "po": 0.12, "vq": 256,   "compress": False}),
    ("po0.12-hb-vq4096",
     {"q": None, "hilbert": True,  "po": 0.12, "vq": 4096,  "compress": False}),
]

# ── GL helpers (called inside each child process) ─────────────────────────────

def _init_gl():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(RENDER_W, RENDER_H, "eval", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    return window


def _make_camera(angle_deg):
    cam = util.Camera(RENDER_H, RENDER_W)
    r = 3.0
    a = np.deg2rad(angle_deg)
    cam.position = np.array([r * np.sin(a), 0.0, r * np.cos(a)], dtype=np.float32)
    cam.target   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam.up       = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cam.is_pose_dirty   = True
    cam.is_intrin_dirty = True
    return cam


def _capture(renderer, gaussians, cam, window):
    renderer.update_gaussian_data(gaussians)
    renderer.sort_and_update(cam)
    renderer.update_camera_pose(cam)
    renderer.update_camera_intrin(cam)
    renderer.set_scale_modifier(1.0)
    renderer.set_render_mod(3)
    renderer.set_render_reso(RENDER_W, RENDER_H)

    gl.glClearColor(0, 0, 0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    renderer.draw()
    glfw.swap_buffers(window)

    gl.glReadBuffer(gl.GL_FRONT)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
    buf = gl.glReadPixels(0, 0, RENDER_W, RENDER_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(buf, np.uint8).reshape(RENDER_H, RENDER_W, 3)[::-1]
    return img.copy()


def _png_paths(out_dir, prefix):
    return [out_dir / f"{prefix}_angle{a:03d}.png" for a in CAMERA_ANGLES]


def _all_exist(paths):
    return all(p.exists() for p in paths)


def _render_and_save(renderer, gaussians, window, out_dir, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for angle in CAMERA_ANGLES:
        cam = _make_camera(angle)
        img = _capture(renderer, gaussians, cam, window)
        imageio.imwrite(str(out_dir / f"{prefix}_angle{angle:03d}.png"), img)
        imgs.append(img)
    return imgs


def _load_from_disk(paths):
    return [imageio.imread(str(p)) for p in paths]


def _time_decode_and_upload(renderer, comp_path):
    """
    Time decompression (CPU) and GPU buffer upload separately.
    glFinish() after upload ensures the GPU transfer is complete before
    we stop the clock - without it we'd only time the CPU submission.
    Returns (gaussians, dec_s, gpu_s).
    """
    t0 = time.perf_counter()
    gaussians = GaussDecoder.decode(str(comp_path))
    dec_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    renderer.update_gaussian_data(gaussians)
    gl.glFinish()
    gpu_s = time.perf_counter() - t1

    return gaussians, dec_s, gpu_s

# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(ref_imgs, cmp_imgs):
    psnrs, ssims = [], []
    for ref, cmp in zip(ref_imgs, cmp_imgs):
        psnrs.append(peak_signal_noise_ratio(ref, cmp, data_range=255))
        ssims.append(structural_similarity(ref, cmp, channel_axis=2, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims)), psnrs, ssims

# ── Per-PLY worker (runs in child process) ────────────────────────────────────

def _worker(ply_path_str):
    """
    Entry point for each child process.
    Returns a list of result-row dicts for this .ply file.
    """
    ply_path = Path(ply_path_str)
    name     = ply_path.stem
    ply_size = ply_path.stat().st_size
    tag      = f"[{name}|pid={os.getpid()}]"

    print(f"{tag} started", flush=True)

    window   = _init_gl()
    renderer = OpenGLRenderer(RENDER_W, RENDER_H)
    results  = []

    # ── Reference renders ─────────────────────────────────────────────────────
    ref_dir   = RENDERS_DIR / name / "reference"
    ref_pngs  = _png_paths(ref_dir, "ref")

    if _all_exist(ref_pngs):
        print(f"{tag} reference PNGs found on disk, loading...", flush=True)
        ref_imgs = _load_from_disk(ref_pngs)
    else:
        print(f"{tag} rendering reference views...", flush=True)
        gaussians = util_gau.load_ply(str(ply_path))
        ref_imgs  = _render_and_save(renderer, gaussians, window, ref_dir, "ref")

    # Load raw once (shared across all encoder configs for this file)
    raw_gau = GaussLoader.load_ply_raw(str(ply_path))

    # ── Per-config loop ───────────────────────────────────────────────────────
    for cfg_name, cfg_args in COMPRESSION_CONFIGS:
        comp_path = ply_path.parent / f"{name}-{cfg_name}.ply.comp"
        comp_dir  = RENDERS_DIR / name / cfg_name
        comp_pngs = _png_paths(comp_dir, cfg_name)

        # Compression
        enc_time = 0.0
        if comp_path.exists():
            print(f"{tag} [{cfg_name}] .comp exists - skipping compression", flush=True)
        else:
            t0 = time.time()
            try:
                encoder = GaussEncoder(bits=cfg_args.get("q"))
                encoder.encode(raw_gau, cfg_args, str(comp_path))
                enc_time = time.time() - t0
            except Exception as e:
                print(f"{tag} [{cfg_name}] ENCODE ERROR: {e}", flush=True)
                continue

        if not comp_path.exists():
            print(f"{tag} [{cfg_name}] .comp missing after encode - skipping", flush=True)
            continue

        comp_size = comp_path.stat().st_size
        ratio     = ply_size / comp_size

        try:
            decoded, dec_s, gpu_s = _time_decode_and_upload(renderer, comp_path)
        except Exception as e:
            print(f"{tag} [{cfg_name}] DECODE/UPLOAD ERROR: {e}", flush=True)
            continue

        # Renders: load from disk if available, otherwise render now.
        if _all_exist(comp_pngs):
            print(f"{tag} [{cfg_name}] render PNGs found on disk, loading...", flush=True)
            cmp_imgs = _load_from_disk(comp_pngs)
        else:
            try:
                cmp_imgs = _render_and_save(renderer, decoded, window, comp_dir, cfg_name)
            except Exception as e:
                print(f"{tag} [{cfg_name}] RENDER ERROR: {e}", flush=True)
                continue

        psnr_mean, ssim_mean, psnrs, ssims = _metrics(ref_imgs, cmp_imgs)

        enc_label = f"t={enc_time:.1f}s" if enc_time > 0 else "t=cached"
        print(
            f"{tag} [{cfg_name:22s}] {enc_label:10s}  "
            f"size={comp_size/1024/1024:.2f}MB  ratio={ratio:.2f}x  "
            f"dec={dec_s:.3f}s  gpu={gpu_s:.3f}s  "
            f"PSNR={psnr_mean:.2f}dB  SSIM={ssim_mean:.4f}",
            flush=True
        )

        row = {
            "model":     name,
            "config":    cfg_name,
            "ply_mb":    round(ply_size / 1024 / 1024, 3),
            "comp_mb":   round(comp_size / 1024 / 1024, 3),
            "ratio":     round(ratio, 3),
            "enc_s":     round(enc_time, 2),
            "dec_s":     round(dec_s, 4),
            "gpu_s":     round(gpu_s, 4),
            "load_s":    round(dec_s + gpu_s, 4),
            "psnr_mean": round(psnr_mean, 4),
            "ssim_mean": round(ssim_mean, 6),
        }
        for angle, p, s in zip(CAMERA_ANGLES, psnrs, ssims):
            row[f"psnr_{angle}deg"] = round(p, 4)
            row[f"ssim_{angle}deg"] = round(s, 6)

        results.append(row)

    glfw.terminate()
    print(f"{tag} done ({len(results)} configs evaluated)", flush=True)
    return results

# ── Output helpers ────────────────────────────────────────────────────────────

def _save_csv(results, path):
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)


def _save_txt(results, path):
    fieldnames = list(results[0].keys())
    col_w = {
        k: max(len(k), max(len(str(r[k])) for r in results))
        for k in fieldnames
    }
    header = "  ".join(k.ljust(col_w[k]) for k in fieldnames)
    sep    = "  ".join("-" * col_w[k]    for k in fieldnames)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(sep    + "\n")
        for r in results:
            f.write("  ".join(str(r[k]).ljust(col_w[k]) for k in fieldnames) + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    ply_dir = filedialog.askdirectory(title="Select folder containing .ply files")
    root.destroy()

    if not ply_dir:
        print("No folder selected.")
        return

    ply_files = sorted(Path(ply_dir).glob("*.ply"))
    if not ply_files:
        print(f"No .ply files found in {ply_dir}")
        return

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ply_files)
    workers = min(MAX_WORKERS, n)
    print(f"Found {n} .ply file(s). Running with {workers} parallel worker(s).")
    print(f"(Adjust MAX_WORKERS in script if you hit VRAM limits.)\n")

    paths = [str(p) for p in ply_files]

    # Each child process handles exactly one .ply file end-to-end.
    with multiprocessing.Pool(processes=workers) as pool:
        per_file_results = pool.map(_worker, paths)

    # Flatten list-of-lists
    results = [row for file_rows in per_file_results for row in file_rows]

    if not results:
        print("\nNo results to save.")
        return

    ts      = time.strftime("%Y%m%d_%H%M%S")
    csv_out = Path(f"eval_results_{ts}.csv")
    txt_out = Path(f"eval_results_{ts}.txt")

    _save_csv(results, csv_out)
    _save_txt(results, txt_out)

    print(f"\nDone. Results saved to:")
    print(f"  {csv_out.resolve()}")
    print(f"  {txt_out.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()