#!/usr/bin/env python3
"""
clipper.py - YouTube link in, the most engaging vertical clips out.

What it does
  1. Finds the most engaging moments (best first, never overlapping):
       - finished live streams: the windows with the most live-chat messages
       - normal videos: YouTube's "Most replayed" heatmap
       - fallback: loudest parts of the audio
  2. Downloads only those sections.
  3. Builds 1080x1920 clips: gameplay on top, your webcam (centered on your face) below,
     or only the webcam, or only the gameplay. Optional captions and channel logo.

Setup
  pip install -U "yt-dlp[default]" faster-whisper numpy "opencv-python-headless<5"
  + install ffmpeg and Deno
  Optional GPU: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

Usage
  python clipper.py "https://youtube.com/watch?v=XXXX"      # asks you the questions
  python clipper.py URL --count 5 --duration 1m --show both --subs yes --yes   # no questions
  python clipper.py URL --show cam                          # only the webcam   (also: game, both)
  python clipper.py URL --duration 5m                       # 5 minute clips (also 90s, 1m30s)
  python clipper.py URL --subs no                           # no subtitles
  python clipper.py URL --logo logo.png --logo-pos bottom-right
  python clipper.py URL --font-file C:\\fonts\\Anton.ttf --sub-color FFD700
  python clipper.py URL --source heatmap                    # skip the slow chat download
  python clipper.py URL --start 1830                        # one clip from 30:30
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yt_dlp
from yt_dlp.utils import download_range_func

W_OUT, H_OUT = 1080, 1920
MAX_CLIPS = 10
MIN_DURATION, MAX_DURATION = 10, 900  # seconds
YDL_EXTRA = {}  # extra yt-dlp options (set in main)


def log(msg):
    print(f"[clipper] {msg}", flush=True)


def mmss(t):
    t = int(t)
    return f"{t // 3600}:{(t // 60) % 60:02d}:{t % 60:02d}"


def fmt_len(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m{s}s" if m and s else (f"{m}m" if m else f"{s}s")


def ydl_opts(**kw):
    return {**YDL_EXTRA, **kw}


# ----------------------------------------------------------------- detection

def get_info(url):
    with yt_dlp.YoutubeDL(ydl_opts(quiet=True, skip_download=True, noplaylist=True)) as ydl:
        return ydl.extract_info(url, download=False)


def score_from_chat(url, cache, n):
    """Chat messages per second, from the live-chat replay of a finished stream.
    The replay is saved in `cache`, so only the first run is slow."""
    def cached_files():
        return [f for f in cache.glob("chat*live_chat*") if not f.name.endswith(".part")]

    files = cached_files()
    if files:
        log("using the saved chat replay from an earlier run")
    else:
        log("downloading the chat replay: every message of the stream, one batch at a time. "
            "On a long stream this takes several minutes, but it is saved for next time. "
            "(Skip it with --source heatmap)")
        opts = ydl_opts(quiet=True, no_warnings=True, skip_download=True, writesubtitles=True,
                        subtitleslangs=["live_chat"], outtmpl=str(cache / "chat"), noplaylist=True)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log(f"chat download failed: {e}")
            return None
        files = cached_files()
    if not files:
        return None
    score, count = np.zeros(n), 0
    for line in files[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            action = json.loads(line).get("replayChatItemAction")
        except json.JSONDecodeError:
            continue
        if not action:
            continue
        t = int(action.get("videoOffsetTimeMsec", -1)) / 1000
        if t < 0 or t >= n:
            continue
        if any("addChatItemAction" in a for a in action.get("actions", [])):
            score[int(t)] += 1
            count += 1
    log(f"read {count} chat messages")
    return score if count >= 20 else None


def score_from_heatmap(info, n):
    """YouTube's 'Most replayed' graph (only exists on videos with enough views)."""
    heatmap = info.get("heatmap")
    if not heatmap:
        return None
    score = np.zeros(n)
    for seg in heatmap:
        a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
        if a < n:
            score[a:min(b, n)] = seg["value"]
    return score


def score_from_audio(url, work, n):
    """Fallback: loudness per second."""
    log("downloading audio for loudness analysis (this can take a while on long videos)")
    opts = ydl_opts(quiet=True, no_warnings=True, format="ba/b",
                    outtmpl=str(work / "audio.%(ext)s"), noplaylist=True)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    f = next(work.glob("audio.*"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(f), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    secs = len(x) // 8000
    rms = np.sqrt((x[: secs * 8000].reshape(secs, 8000) ** 2).mean(axis=1))
    out = np.zeros(n)
    out[: min(n, secs)] = rms[:n]
    return out


def top_windows(score, win, count):
    """Start times of the `count` best non-overlapping windows, best first."""
    n = len(score)
    if n <= win:
        return [0]
    cs = np.concatenate([[0], np.cumsum(score)])
    sums = (cs[win:] - cs[:-win]).astype(float)
    starts = []
    for _ in range(count):
        i = int(np.argmax(sums))
        if sums[i] <= 0:  # nothing engaging left
            break
        # several windows can tie (a flat peak): take the middle one so the peak sits mid-clip
        j = i
        while j + 1 < len(sums) and sums[j + 1] >= sums[i] - 1e-9:
            j += 1
        i = (i + j) // 2
        starts.append(i)
        sums[max(0, i - win + 1): i + win] = -np.inf
    return starts or [0]


def find_windows(url, info, work, cache, total, win, count, source):
    order = {
        "auto": (["chat"] if info.get("was_live") else []) + ["heatmap", "audio"],
        "chat": ["chat", "heatmap", "audio"],
        "heatmap": ["heatmap", "audio"],
        "audio": ["audio"],
    }[source]
    for method in order:
        if method == "chat":
            s, label = score_from_chat(url, cache, total), "most chat activity"
        elif method == "heatmap":
            s, label = score_from_heatmap(info, total), "most replayed"
        else:
            s, label = score_from_audio(url, work, total), "loudest audio"
        if s is not None:
            return top_windows(s, win, count), label
        log(f"no usable {method} data, trying the next method")
    return [0], "start of the video"


# ------------------------------------------------------------------ download

def has_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def download_clip(url, start, end, cdir):
    """Download one section. Returns (file, description of the chosen format)."""
    opts = ydl_opts(
        quiet=True,
        no_warnings=True,
        noprogress=True,
        # no catch-all fallback: it silently picks an audio-only file when video formats are missing
        format="bv*[height<=1080]+ba/b[height<=1080]",
        download_ranges=download_range_func(None, [(start, end)]),
        force_keyframes_at_cuts=True,
        merge_output_format="mp4",
        outtmpl=str(cdir / "clip.%(ext)s"),
        noplaylist=True,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        result = ydl.extract_info(url, download=True)
    clip = cdir / "clip.mp4"
    if not clip.exists():
        others = [p for p in cdir.glob("clip.*") if p.suffix in (".mp4", ".mkv", ".webm")]
        if not others:
            raise RuntimeError("download produced no clip file")
        clip = others[0]
    if not has_video(clip):
        raise RuntimeError(
            "YouTube gave only audio for this section (the video formats are missing). "
            'Fix: python -m pip install -U "yt-dlp[default]" and make sure "deno --version" works, '
            "then run the tool again.")
    return clip, result.get("format") or result.get("format_id") or "unknown format"


# ------------------------------------------------------------ webcam / face

def probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def detect_webcam_face(clip, work, every=6):
    """Find the streamer's face: the face that shows up at the same spot in most sampled frames.
    Returns ((x, y, w, h) in pixels, (frame_w, frame_h)) or None."""
    try:
        import cv2
    except ImportError:
        log('OpenCV is not installed. Run: pip install "opencv-python-headless<5"')
        return None

    cascades = []
    for name in ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"):
        path = os.path.join(cv2.data.haarcascades, name)
        if os.path.exists(path):
            cascades.append(cv2.CascadeClassifier(path))
    if not cascades:
        log('Your OpenCV has no face detector files (OpenCV 5 dropped them). Run: '
            'pip install "opencv-python-headless<5"')
        return None

    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"fps=1/{every}", "-q:v", "2",
             str(frames_dir / "f_%03d.jpg")],
            check=True,
        )
    except subprocess.CalledProcessError:
        log("could not read frames from the clip for face detection")
        return None
    images = sorted(frames_dir.glob("f_*.jpg"))
    if not images:
        return None

    detections, size = [], None
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        for c in cascades:
            for (x, y, w, h) in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)):
                detections.append((i, int(x), int(y), int(w), int(h)))
    if not detections:
        return None

    best = None  # (frames_with_hit, median_width, group)
    for d in detections:
        cx, cy, tol = d[1] + d[3] / 2, d[2] + d[4] / 2, 0.6 * d[3]
        group = [e for e in detections
                 if abs(e[1] + e[3] / 2 - cx) < tol and abs(e[2] + e[4] / 2 - cy) < tol]
        score = (len({e[0] for e in group}), float(np.median([e[3] for e in group])))
        if best is None or score > best[0]:
            best = (score, group)

    (hits, _), group = best
    need = max(2, int(0.3 * len(images)))
    if hits < need:
        log(f"only {hits}/{len(images)} sampled frames had a steady face, not enough to trust")
        return None
    face = tuple(int(np.median([e[k] for e in group])) for k in (1, 2, 3, 4))
    log(f"found your face in {hits}/{len(images)} sampled frames at x={face[0]}, y={face[1]}, size={face[2]}px")
    return face, size


def cam_box_from_face(face, frame_size, panel_h, zoom):
    """Crop box (fractions of the frame) centered on the face, shaped like the webcam panel."""
    fx, fy, fw, fh = face
    frame_w, frame_h = frame_size
    cw = fw * zoom
    ch = cw * panel_h / W_OUT
    if cw > frame_w:
        s = frame_w / cw
        cw, ch = cw * s, ch * s
    if ch > frame_h:
        s = frame_h / ch
        cw, ch = cw * s, ch * s
    cx, cy = fx + fw / 2, fy + fh / 2
    x0 = min(max(cx - cw / 2, 0), frame_w - cw)
    y0 = min(max(cy - ch * 0.45, 0), frame_h - ch)  # face a bit above the middle
    return tuple(round(v, 4) for v in (x0 / frame_w, y0 / frame_h, cw / frame_w, ch / frame_h))


# ----------------------------------------------------------------- subtitles

def add_cuda_dll_paths():
    """On Windows, let Python find CUDA libraries installed with:
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"""
    if sys.platform != "win32":
        return
    import site
    import sysconfig

    roots = set(site.getsitepackages() + [site.getusersitepackages(), sysconfig.get_paths()["purelib"]])
    for root in roots:
        for sub in ("cublas", "cudnn", "cuda_runtime"):
            d = Path(root) / "nvidia" / sub / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def load_audio(path):
    """Decode audio with ffmpeg (16 kHz mono float32) so we don't depend on PyAV versions."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def has_gpu():
    add_cuda_dll_paths()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def pick_model(model, on_gpu):
    """'auto' = the accurate large-v3-turbo on a GPU, medium on the CPU."""
    if model != "auto":
        return model
    return "large-v3-turbo" if on_gpu else "medium"


_MODELS = {}


def run_whisper(audio, model_size, device, compute_type, language=None, prompt=None):
    add_cuda_dll_paths()
    from faster_whisper import WhisperModel

    key = (model_size, device, compute_type)
    if key not in _MODELS:  # load once, reuse for every clip
        log(f"loading speech model '{model_size}' on {device}")
        _MODELS[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    model = _MODELS[key]
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        condition_on_previous_text=False,       # stops one mistake from snowballing
        temperature=[0.0, 0.2, 0.4],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        hallucination_silence_threshold=2.0,    # drops made-up text over silence/music
        initial_prompt=prompt,
    )
    if language is None:
        log(f"detected language: {info.language} ({info.language_probability:.0%})")
    words = []
    for seg in segments:
        if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
            continue  # probably game audio or music, not speech
        if seg.compression_ratio > 2.4:
            continue  # repetitive garbage
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append((w.start, w.end, text))
    return words


def transcribe_words(clip, model="auto", device="auto", language=None, prompt=None):
    audio = load_audio(clip)
    gpu = device == "cuda" or (device == "auto" and has_gpu())
    if gpu:
        try:
            return run_whisper(audio, pick_model(model, True), "cuda", "auto", language, prompt)
        except RuntimeError as e:
            log(f"GPU transcription failed ({e}); retrying on CPU")
    return run_whisper(audio, pick_model(model, False), "cpu", "int8", language, prompt)


# ---------------------------------------------------------- fonts / captions

def font_name_from_file(path):
    """Read the font's full name from a .ttf/.otf so the caption style can ask for it by name."""
    data = Path(path).read_bytes()
    base = struct.unpack(">I", data[12:16])[0] if data[:4] == b"ttcf" else 0  # .ttc: first font
    num_tables = struct.unpack(">H", data[base + 4:base + 6])[0]
    name_off = None
    for i in range(num_tables):
        tag, _, off, _length = struct.unpack(">4sIII", data[base + 12 + 16 * i: base + 28 + 16 * i])
        if tag == b"name":
            name_off = off
            break
    if name_off is None:
        raise ValueError(f"{Path(path).name} does not look like a font file")
    _fmt, count, str_off = struct.unpack(">HHH", data[name_off:name_off + 6])
    found = {}
    for i in range(count):
        pid, _eid, lid, nid, ln, so = struct.unpack(">HHHHHH", data[name_off + 6 + 12 * i: name_off + 18 + 12 * i])
        if nid not in (1, 4):
            continue
        raw = data[name_off + str_off + so: name_off + str_off + so + ln]
        try:
            text = raw.decode("utf-16-be") if pid in (0, 3) else raw.decode("mac_roman")
        except UnicodeDecodeError:
            continue
        rank = 0 if (pid == 3 and lid == 0x409) else 1  # prefer Windows / English
        if nid not in found or rank < found[nid][0]:
            found[nid] = (rank, text.strip())
    for nid in (4, 1):  # full name first, then family
        if nid in found and found[nid][1]:
            return found[nid][1]
    raise ValueError(f"could not read a name from {Path(path).name}")


def ass_color(hex_rgb):
    h = hex_rgb.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError("colors look like FFFFFF or #FFD700")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def ass_time(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def write_ass(words, path, sub_y, font="Arial", size=88, color="&H00FFFFFF", bold=True,
              max_words=3, max_gap=0.6):
    font = font.replace(",", " ")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W_OUT}
PlayResY: {H_OUT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{color},&H000000FF,&H00000000,&H64000000,{-1 if bold else 0},0,0,0,100,100,0,0,1,7,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w[0] - cur[-1][1] > max_gap or cur[-1][2][-1] in ".?!"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)

    pos = "{\\an5\\pos(%d,%d)}" % (W_OUT // 2, sub_y)
    lines = []
    for i, c in enumerate(chunks):
        start, end = c[0][0], c[-1][1]
        nxt = chunks[i + 1][0][0] if i + 1 < len(chunks) else None
        # hold the caption until the next one starts if the gap is tiny, so text doesn't flicker
        end = nxt if nxt is not None and nxt - end < 0.35 else end + 0.15
        text = " ".join(w[2] for w in c).upper().replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{pos}{text}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- compose

def layout(show, mode, cam_h, game_h):
    """Geometry of the 1080x1920 canvas.
    both = gameplay on top, webcam at the bottom; cam = only webcam; game = only gameplay."""
    H = H_OUT
    if show == "cam":
        return dict(cam_h=H, cam_y=0, game_y=0, top_h=0, sub_y=int(H * 0.72))
    if show == "game":
        if mode == "fill":
            return dict(cam_h=0, cam_y=H, game_y=0, top_h=H, sub_y=int(H * 0.78))
        gy = (H - game_h) // 2
        return dict(cam_h=0, cam_y=H, game_y=gy, top_h=H, sub_y=gy + game_h + 150)
    if mode == "fit":
        cam_h = min(cam_h, H - game_h)  # never cover the gameplay
    top_h = H - cam_h
    if mode == "fill":
        return dict(cam_h=cam_h, cam_y=top_h, game_y=0, top_h=top_h, sub_y=top_h - 150)
    gy = (top_h - game_h) // 2
    return dict(cam_h=cam_h, cam_y=top_h, game_y=gy, top_h=top_h, sub_y=(gy + game_h + top_h) // 2)


def compose(clip_name, ass_name, logo, out, show, cam, geo, mode, game_h, focus, work, fonts_dir=None):
    """cam = (x, y, w, h) as fractions of the source frame, or None."""
    W, H = W_OUT, H_OUT
    use_game = show in ("both", "game")
    use_cam = show in ("both", "cam") and cam is not None
    need_bg = use_game and mode == "fit"
    users = [n for n, flag in (("bg", need_bg), ("game", use_game), ("cam", use_cam)) if flag]

    parts = []
    if len(users) > 1:
        parts.append(f"[0:v]split={len(users)}" + "".join(f"[in_{u}]" for u in users))
        src = {u: f"[in_{u}]" for u in users}
    else:
        src = {users[0]: "[0:v]"}

    if need_bg:  # blurred, darkened copy of the gameplay behind everything
        parts.append(f"{src['bg']}scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                     f"boxblur=12:3,scale={W}:{H},eq=brightness=-0.12[bg]")
    if use_game:
        if mode == "fit":
            parts.append(f"{src['game']}scale={W}:{game_h}[game]")
        else:
            th = geo["top_h"]
            parts.append(f"{src['game']}scale=-2:{th},crop={W}:{th}:(iw-{W})*{focus}:0[game]")
    if use_cam:
        x, y, w, h = cam
        ch = geo["cam_h"]
        parts.append(f"{src['cam']}crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},"
                     f"scale={W}:{ch}:force_original_aspect_ratio=increase,crop={W}:{ch}[cam]")

    if show == "cam":
        parts.append("[cam]null[stack]")
    elif mode == "fit":
        parts.append(f"[bg][game]overlay=0:{geo['game_y']}[t1]")
        parts.append(f"[t1][cam]overlay=0:{geo['cam_y']}[stack]" if use_cam else "[t1]null[stack]")
    else:
        parts.append("[game][cam]vstack=inputs=2[stack]" if use_cam else "[game]null[stack]")

    inputs = ["-i", clip_name]
    last = "stack"
    if logo:
        inputs += ["-i", str(logo["path"])]
        x_expr = {"bottom-left": "60", "bottom-center": "(main_w-overlay_w)/2",
                  "bottom-right": "main_w-overlay_w-60"}[logo["pos"]]
        parts.append(f"[1:v]format=rgba,scale={logo['size']}:-1[logo]")
        parts.append(f"[stack][logo]overlay={x_expr}:main_h-overlay_h-{logo['margin']}[branded]")
        last = "branded"

    if ass_name:
        sub = f"ass={ass_name}" + (f":fontsdir={fonts_dir}" if fonts_dir else "")
        parts.append(f"[{last}]{sub}[v]")
    else:
        parts.append(f"[{last}]null[v]")

    cmd = [
        "ffmpeg", "-v", "error", "-y", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, cwd=work, check=True)  # cwd=work so the ass= path needs no escaping


# ------------------------------------------------------------ options / input

def parse_count(v):
    try:
        n = int(str(v).strip())
    except ValueError:
        raise ValueError(f"Please type a whole number from 1 to {MAX_CLIPS}.")
    if not 1 <= n <= MAX_CLIPS:
        raise ValueError(f"Please type a number from 1 to {MAX_CLIPS}.")
    return n


def parse_duration(v, bare="seconds"):
    """'2m', '90s', '1m30s', '5 minutes'; a bare number is seconds (or minutes if bare='minutes')."""
    t = str(v).strip().lower().replace(" ", "")
    if re.fullmatch(r"\d+(\.\d+)?", t):
        secs = float(t) * (60 if bare == "minutes" else 1)
    else:
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)(?:m|min|mins|minute|minutes))?"
                         r"(?:(\d+(?:\.\d+)?)(?:s|sec|secs|second|seconds))?", t)
        if not m or not (m.group(1) or m.group(2)):
            raise ValueError("Please type a length like 1m, 2m, 5m, 90s or 1m30s.")
        secs = float(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    secs = int(round(secs))
    if not MIN_DURATION <= secs <= MAX_DURATION:
        raise ValueError(f"Clip length must be between {MIN_DURATION} seconds and {MAX_DURATION // 60} minutes "
                         "(for example 90s, 2m or 5m).")
    return secs


def parse_show(v):
    table = {"1": "both", "both": "both", "b": "both",
             "2": "cam", "cam": "cam", "webcam": "cam", "facecam": "cam", "face": "cam",
             "3": "game", "game": "game", "gameplay": "game"}
    t = str(v).strip().lower()
    if t not in table:
        raise ValueError("Please choose 1 (both), 2 (only webcam) or 3 (only gameplay).")
    return table[t]


def parse_yes_no(v):
    t = str(v).strip().lower()
    if t in ("y", "yes", "true", "1", "on"):
        return "yes"
    if t in ("n", "no", "false", "0", "off"):
        return "no"
    raise ValueError("Please type y or n.")


def as_arg(fn):
    def wrapper(value):
        try:
            return fn(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
    wrapper.__name__ = fn.__name__
    return wrapper


def ask(question, parse):
    """Ask until the answer is valid. Enter = None (use the default)."""
    while True:
        raw = input(question).strip()
        if not raw:
            return None
        try:
            return parse(raw)
        except ValueError as e:
            print(e)


def fill_options(args):
    """Ask about anything not given on the command line (Enter keeps the default)."""
    interactive = sys.stdin.isatty() and not args.yes
    if interactive and args.start is None and args.count is None:
        args.count = ask(f"How many clips do you want? (1-{MAX_CLIPS}, Enter for {MAX_CLIPS}): ", parse_count)
    if interactive and args.duration is None:
        args.duration = ask("How long should each clip be? (1m, 2m, 5m, 90s ... a plain number = minutes, "
                            "Enter for 2m): ", lambda r: parse_duration(r, bare="minutes"))
    if interactive and args.show is None:
        print("What should the clips show?\n  1 = gameplay on top + webcam below\n"
              "  2 = only the webcam\n  3 = only the gameplay")
        args.show = ask("Choose 1, 2 or 3 (Enter for 1): ", parse_show)
    if interactive and args.subs is None:
        args.subs = ask("Add subtitles? (y/n, Enter for y): ", parse_yes_no)
    args.count = args.count or MAX_CLIPS
    args.duration = args.duration or 120
    args.show = args.show or "both"
    args.subs = args.subs or "yes"


def build_parser():
    ap = argparse.ArgumentParser(description="Make the most engaging vertical clips from a YouTube link.")
    ap.add_argument("url")
    ap.add_argument("--count", type=as_arg(parse_count), help=f"how many clips, 1-{MAX_CLIPS}")
    ap.add_argument("--duration", type=as_arg(parse_duration),
                    help="clip length: 1m, 2m, 5m, 90s, 1m30s (default 2m; a plain number here = seconds)")
    ap.add_argument("--show", type=as_arg(parse_show),
                    help="both = gameplay on top + webcam below, cam = only webcam, game = only gameplay")
    ap.add_argument("--subs", type=as_arg(parse_yes_no), help="subtitles: yes or no")
    ap.add_argument("--no-subs", dest="subs", action="store_const", const="no", help="same as --subs no")
    ap.add_argument("--yes", action="store_true", help="don't ask questions, use defaults for anything not given")
    ap.add_argument("--start", type=float, help="skip detection and make ONE clip starting at this second")
    ap.add_argument("--source", choices=["auto", "chat", "heatmap", "audio"], default="auto",
                    help="how to find the best moments. 'heatmap' is much faster than 'chat' on long streams")
    ap.add_argument("--out-dir", help="folder for the clips (default: clips_<video id>)")

    g = ap.add_argument_group("webcam and gameplay")
    g.add_argument("--cam", help="webcam box x:y:w:h as fractions of the frame (default: find your face)")
    g.add_argument("--no-cam", action="store_true", help="same as --show game")
    g.add_argument("--cam-height", type=int, default=800, help="height of the webcam panel in px (default 800)")
    g.add_argument("--cam-zoom", type=float, default=3.0,
                   help="space around the face: crop width = zoom x face width (smaller = closer)")
    g.add_argument("--game-mode", choices=["fit", "fill"], default="fit",
                   help="fit = whole gameplay screen visible (default); fill = crop it to fill the space")
    g.add_argument("--game-focus", type=float, default=0.5,
                   help="with --game-mode fill: crop position, 0=left 1=right")

    g = ap.add_argument_group("logo")
    g.add_argument("--logo", help="path to your logo image (PNG with a transparent background works best)")
    g.add_argument("--logo-size", type=int, default=200, help="logo width in px (default 200)")
    g.add_argument("--logo-pos", choices=["bottom-left", "bottom-center", "bottom-right"],
                   default="bottom-center", help="where the logo goes (default bottom-center)")
    g.add_argument("--logo-margin", type=int, default=60, help="distance from the bottom edge in px (default 60)")

    g = ap.add_argument_group("subtitles")
    g.add_argument("--font", help="installed font name for the subtitles, e.g. Impact")
    g.add_argument("--font-file", help="path to a .ttf/.otf font file to use for the subtitles")
    g.add_argument("--sub-size", type=int, default=88, help="subtitle size (default 88)")
    g.add_argument("--sub-color", default="FFFFFF", help="subtitle color as hex, e.g. FFD700 (default FFFFFF)")
    g.add_argument("--model", default="auto",
                   help="speech model: auto (large-v3-turbo on GPU, medium on CPU), or tiny/base/small/"
                        "medium/large-v3/large-v3-turbo")
    g.add_argument("--language", help="spoken language code, e.g. en or bn (default: detect per clip)")
    g.add_argument("--prompt", help="words the speech model should know: names, game terms, slang")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="where to run speech recognition (default auto, falls back to cpu)")

    ap.add_argument("--remote-solver", action="store_true",
                    help="let yt-dlp download its official YouTube JS solver from GitHub "
                         '(alternative to: pip install -U "yt-dlp[default]")')
    return ap


# ---------------------------------------------------------------------- main

def main():
    args = build_parser().parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found. Install ffmpeg and make sure it is on your PATH.")

    if args.no_cam:
        args.show = "game"
    fill_options(args)

    manual_cam = None
    if args.cam:
        manual_cam = tuple(float(v) for v in args.cam.split(":"))
        if len(manual_cam) != 4:
            sys.exit("--cam must look like x:y:w:h, e.g. 0.75:0.65:0.25:0.35")

    logo = None
    if args.logo:
        p = Path(args.logo).expanduser()
        if not p.is_file():
            sys.exit(f"logo file not found: {p}")
        logo = dict(path=p.resolve(), size=args.logo_size, pos=args.logo_pos, margin=args.logo_margin)

    font_name, font_src = args.font or "Arial", None
    if args.font_file:
        fp = Path(args.font_file).expanduser()
        if not fp.is_file():
            log(f"font file not found: {fp}. Using the default font ({font_name}) instead.")
        else:
            try:
                font_name = font_name_from_file(fp)
                font_src = fp
                log(f"using the font '{font_name}' from {fp.name}")
            except (ValueError, struct.error, OSError):
                log(f"{fp.name} is not a readable .ttf/.otf font. Using the default font ({font_name}) instead.")
    bold = args.font is None and font_src is None  # custom fonts are used as they are
    try:
        color = ass_color(args.sub_color)
    except ValueError as e:
        sys.exit(f"--sub-color: {e}")

    if args.remote_solver:
        YDL_EXTRA["remote_components"] = ["ejs:github"]
    elif importlib.util.find_spec("yt_dlp_ejs") is None:
        log('tip: if downloads fail or only give audio, run: python -m pip install -U "yt-dlp[default]"')

    shows = {"both": "gameplay + webcam", "cam": "only the webcam", "game": "only the gameplay"}
    log(f"settings: {'1 clip' if args.start is not None else str(args.count) + ' clips'} of "
        f"{fmt_len(args.duration)}, {shows[args.show]}, subtitles {args.subs}"
        + (", logo on" if logo else ""))

    log("reading video info")
    info = get_info(args.url)
    if info.get("is_live"):
        sys.exit("This stream is still live. Run the tool again once it has ended and the VOD is available.")
    total = int(info["duration"])
    win = min(args.duration, total)

    cache = Path.home() / ".clipper_cache" / info["id"]
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="clipper_"))
    out_dir = Path(args.out_dir or f"clips_{info['id']}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results, starts = [], []
    pool = None
    face_info, cam_tries = None, 0
    try:
        if args.start is not None:
            starts, method = [args.start], "manual start"
        else:
            starts, method = find_windows(args.url, info, work, cache, total, win, args.count, args.source)
        starts = [max(0, min(s, total - win)) for s in starts]
        log(f"found {len(starts)} clip(s) by {method}; saving to {out_dir}")
        for rank, s in enumerate(starts, 1):
            log(f"  #{rank}: {mmss(s)} to {mmss(s + win)}")

        # download the next clips in the background while the current one is being processed
        cdirs = {}
        for rank in range(1, len(starts) + 1):
            cdirs[rank] = work / f"c{rank:02d}"
            cdirs[rank].mkdir()
        pool = ThreadPoolExecutor(max_workers=2)
        downloads = {rank: pool.submit(download_clip, args.url, s, s + win, cdirs[rank])
                     for rank, s in enumerate(starts, 1)}

        for rank, start in enumerate(starts, 1):
            cdir = cdirs[rank]
            log(f"--- clip {rank}/{len(starts)}: {mmss(start)} to {mmss(start + win)} ---")
            try:
                if not downloads[rank].done():
                    log("downloading")
                clip, fmt = downloads[rank].result()
                log(f"downloaded ({fmt})")

                src_w, src_h = probe_size(clip)
                game_h = int(round(W_OUT * src_h / src_w / 2)) * 2

                if args.show != "game" and manual_cam is None and face_info is None and cam_tries < 3:
                    cam_tries += 1
                    log("looking for your face in the webcam")
                    face_info = detect_webcam_face(clip, cdir)

                show = args.show
                if show != "game" and manual_cam is None and face_info is None:
                    log("no webcam face found, so this clip shows only the gameplay "
                        "(set the webcam by hand with --cam x:y:w:h)")
                    show = "game"
                geo = layout(show, args.game_mode, args.cam_height, game_h)
                cam = None
                if show != "game":
                    cam = manual_cam or cam_box_from_face(face_info[0], face_info[1], geo["cam_h"], args.cam_zoom)

                ass_name, fonts_dir = None, None
                if args.subs == "yes":
                    log("transcribing")
                    words = transcribe_words(clip, args.model, args.device, args.language, args.prompt)
                    if words:
                        write_ass(words, cdir / "subs.ass", geo["sub_y"], font_name, args.sub_size, color, bold)
                        ass_name = "subs.ass"
                        if font_src:
                            (cdir / "fonts").mkdir(exist_ok=True)
                            shutil.copy(font_src, cdir / "fonts" / font_src.name)
                            fonts_dir = "fonts"
                    else:
                        log("no speech detected, skipping subtitles")

                out = out_dir / f"{rank:02d}_clip_{info['id']}_{int(start)}s.mp4"
                log("rendering")
                compose(clip.name, ass_name, logo, out, show, cam, geo, args.game_mode, game_h,
                        args.game_focus, cdir, fonts_dir)
                log(f"saved {out.name}")
                results.append((rank, start, out))
            except Exception as e:  # one bad clip should not stop the rest
                log(f"clip {rank} failed: {e}")
            finally:
                shutil.rmtree(cdir, ignore_errors=True)
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(work, ignore_errors=True)

    log(f"done: {len(results)} of {len(starts)} clips saved in {out_dir}")
    for rank, start, out in results:
        log(f"  #{rank}  {mmss(start)}  {out.name}")


if __name__ == "__main__":
    main()
