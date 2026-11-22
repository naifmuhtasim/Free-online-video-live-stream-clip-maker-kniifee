# Free-online-video-live-stream-clip-maker-kniifee
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

# clipper

Give it a YouTube link. It finds the most engaging moments of the video or stream and saves them as vertical (1080x1920) clips, ready for Shorts, Reels or TikTok.

- Picks the best moments for you: most chat activity on a finished live stream, YouTube's "Most replayed" graph, or the loudest parts of the audio.
- Layout: **gameplay on top, your webcam below** (cropped around your face), or only the webcam, or only the gameplay.
- Optional captions (English speech only), your own font, and a small channel logo.
- You choose how many clips (1 to 10) and how long each one is (10 seconds to 15 minutes).

> Use it on your own videos or ones you have permission to clip. Downloading other people's videos can break YouTube's terms and copyright rules.

---

## 1. What you need to install (once)

These steps are for Windows. Open **PowerShell** for each command.

| Thing | Why | How to install |
|---|---|---|
| **Python 3.10 or newer** | runs the tool | Download from python.org. Tick **"Add Python to PATH"** in the installer. |
| **ffmpeg** (includes ffprobe) | cuts and renders the video | `winget install ffmpeg` |
| **Deno** | lets yt-dlp solve YouTube's JavaScript check | `winget install DenoLand.Deno` |
| **Python packages** | download, speech recognition, face detection | the command below |

After installing Python, ffmpeg or Deno, **close PowerShell and open it again**, so it can find them.

Install the Python packages with this one command:

```
python -m pip install -U "yt-dlp[default]" faster-whisper numpy "opencv-python-headless<5"
```

Two details matter here:

- `yt-dlp[default]` (not plain `yt-dlp`) includes the piece YouTube downloads need. Without it you can get audio-only downloads.
- `opencv-python-headless<5` keeps OpenCV on version 4. Version 5 removed the face-detection files this tool uses.

Check that everything is found:

```
python --version
ffmpeg -version
ffprobe -version
deno --version
```

Each command should print a version number.

## 2. Put the script in a folder

Save `clipper.py` in a folder, for example `C:\clipper`. Open PowerShell in that folder (in File Explorer, click the address bar, type `powershell`, press Enter).

Check you have the right file:

```
python clipper.py --help
```

The list should include `--show`, `--logo` and `--range`. If not, you are running an old copy. Your browser may have saved the new one as `clipper (1).py`.

## 3. Make your first clips

```
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Always keep the quotes around the link.** Without them PowerShell breaks the link at the `&` character.

The tool asks four questions. Press **Enter** to accept the default:

1. How many clips? (1 to 10, default 10)
2. How long should each clip be? (`1m`, `2m`, `5m`, `90s` ... a plain number means minutes here, default 2m)
3. What should the clips show? (1 = gameplay + webcam, 2 = only webcam, 3 = only gameplay)
4. Subtitles? (y or n, default y)

To skip the questions, give the answers up front and add `--yes`:

```
python clipper.py "LINK" --count 3 --duration 1m --show both --subs yes --yes
```

The clips are saved in a folder called `clips_<video id>` in the folder you ran the command from. They are named `01_clip_...mp4`, `02_clip_...mp4` and so on. **`01` is the most engaging.** The tool prints the full path when it finishes.

**The first run takes longer.** It downloads the speech model (roughly 1 to 2 GB, only once). On a long video it also analyzes the whole thing first (see section 7).

---

## 4. Options

### Basics

| Option | What it does | Default |
|---|---|---|
| `--count N` | Number of clips, 1 to 10 | asks you (10) |
| `--duration 2m` | Clip length. `1m`, `2m`, `5m`, `90s`, `1m30s`. On the command line a plain number means seconds. Range: 10 seconds to 15 minutes. | asks you (2m) |
| `--show both\|cam\|game` | `both` = gameplay on top, webcam below. `cam` = only the webcam. `game` = only the gameplay. | asks you (both) |
| `--subs yes\|no` | Add subtitles or not. `--no-subs` works too. | asks you (yes) |
| `--yes` | Don't ask questions, use defaults for anything not given | off |
| `--start 1:23:45` | Skip the search and make **one** clip from this time. Also accepts `1h23m` or seconds. | off |
| `--out-dir FOLDER` | Where to save the clips | `clips_<video id>` |

### Finding the moments

| Option | What it does |
|---|---|
| `--source auto\|chat\|heatmap\|audio` | How to pick moments. `auto` uses chat for finished live streams, else the "Most replayed" graph, else loudness. `heatmap` is the fastest on long videos. |
| `--range 2h-4h` | Only search this part of a long video. Also `1:30:00-3:00:00`, `2h-` (from 2h to the end) or `-1h` (the first hour). |

### Webcam and gameplay

| Option | What it does | Default |
|---|---|---|
| `--cam x:y:w:h` | Set the webcam box by hand, as fractions of the frame (for example `0.75:0.65:0.25:0.35` for the bottom-right corner) | finds your face automatically |
| `--no-cam` | Same as `--show game` | off |
| `--cam-height 800` | Height of the webcam panel in pixels | 800 |
| `--cam-zoom 3.0` | How much space around your face. Smaller is closer (try `2`), bigger shows more. | 3.0 |
| `--game-mode fit\|fill` | `fit` shows the whole gameplay screen over a blurred background. `fill` crops it to fill the space. | fit |
| `--game-focus 0.5` | With `fill`: which part to keep, 0 = left, 1 = right | 0.5 |

### Logo

| Option | What it does | Default |
|---|---|---|
| `--logo "path\to\logo.png"` | Your channel logo. A PNG with a transparent background looks best. | none |
| `--logo-size 200` | Logo width in pixels | 200 |
| `--logo-pos` | `bottom-left`, `bottom-center` or `bottom-right` | bottom-center |
| `--logo-margin 60` | Distance from the bottom edge in pixels | 60 |

### Subtitles

| Option | What it does | Default |
|---|---|---|
| `--font-file "C:\fonts\Anton.ttf"` | Use a font file you imported. If the file is missing or unreadable, the default font is used. | Arial |
| `--font Impact` | Use an installed font by name | none |
| `--sub-size 88` | Caption size | 88 |
| `--sub-color FFD700` | Caption color in hex | FFFFFF |
| `--language en` | Force the spoken language instead of detecting it | detect per clip |
| `--any-language` | Add subtitles in any language (by default, clips that are not in English get none) | off |
| `--prompt "Valorant, Jett"` | Words the speech model should know (names, game terms, slang) | none |
| `--model` | `auto` (large-v3-turbo on a GPU, medium on the CPU), or `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` | auto |
| `--device auto\|cpu\|cuda` | Where speech recognition runs. `auto` uses the GPU if it works and falls back to the CPU. | auto |

### Other

| Option | What it does |
|---|---|
| `--remote-solver` | Lets yt-dlp download its official YouTube solver from GitHub. An alternative to installing `yt-dlp[default]`. |

Paths with spaces need quotes, for example `--logo "My Logo 2026.png"`.

---

## 5. Examples

Five 1-minute clips with a logo, a custom font and gold captions:

```
python clipper.py "LINK" --count 5 --duration 1m --show both --subs yes --logo "logo.png" --logo-pos bottom-right --font-file "C:\fonts\Anton.ttf" --sub-color FFD700 --yes
```

Three 5-minute gameplay-only clips with no subtitles:

```
python clipper.py "LINK" --count 3 --duration 5m --show game --subs no --yes
```

Only the webcam, a bit closer to your face:

```
python clipper.py "LINK" --show cam --cam-zoom 2 --count 3 --yes
```

One clip from a moment you already know:

```
python clipper.py "LINK" --start 1:23:45 --duration 90s --yes
```

A long stream, fast: search only hours 2 to 4 using the "Most replayed" graph:

```
python clipper.py "LINK" --range 2h-4h --source heatmap --count 5 --yes
```

---

## 6. How it works

1. **Finds the moments.** It scores every second of the video, then picks the best 2-minute (or your length) windows. They never overlap, and each one is centered on its peak.
   - Finished live stream: the busiest chat moments.
   - Normal video: the "Most replayed" graph.
   - Neither available: the loudest audio.
2. **Downloads only those sections**, up to 1080p.
3. **Finds your face** by checking sample frames for a face that stays in the same spot (game characters move, so they are ignored), then crops the webcam panel around it. If no face is found, that clip shows only the gameplay and the tool says so.
4. **Captions.** Speech recognition (faster-whisper) listens to the clip and the captions appear three words at a time, between the gameplay and the webcam. **If the speech is not English, that clip gets no captions** (override with `--language` or `--any-language`).
5. **Renders** the final 1080x1920 video with the logo on top.

Files the tool keeps between runs are in `C:\Users\<you>\.clipper_cache\<video id>`. They hold the downloaded chat replay and loudness analysis, so running the same video again skips the slowest step. You can delete that folder at any time.

---

## 7. Long videos and live streams

Finding the moments is the slow part on a long stream.

- **Chat mode** downloads every chat message of the stream, one batch after another. YouTube allows no shortcut. The first run can take many minutes, but it is saved, so the next run on that video is quick.
- **Loudness mode** downloads the audio track.

To go faster:

- `--source heatmap`: instant, if the video has a "Most replayed" graph.
- `--range 2h-4h`: only analyze part of the video.
- `--start 1:23:45`: skip searching entirely.
- `--subs no`, `--model small`, a lower `--count`, or a working GPU make the clips themselves faster.

The tool prints how long each clip took and the total.

A stream that is **still live** can't be clipped. Run the tool again after it ends, once the recording is available.

---

## 8. Optional: use your NVIDIA GPU for captions

Captions are much faster on a GPU. This is optional. Without it the tool uses the CPU.

1. Install the current NVIDIA driver. Run `nvidia-smi`. If it prints a table, the driver is there.
2. Install the CUDA libraries with pip:
   ```
   python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
3. Run the tool. It finds those libraries by itself. You should see the speech model load "on cuda".

If the GPU fails for any reason, the tool says so and continues on the CPU. To skip the GPU on purpose, add `--device cpu`.

---

## 9. Troubleshooting

| What you see | What to do |
|---|---|
| `python` is not recognized | Python is not on PATH. Reinstall it with "Add Python to PATH" ticked, then reopen PowerShell. |
| `ffmpeg not found` | Run `winget install ffmpeg`, then close and reopen PowerShell. |
| `unrecognized arguments` or `--show` is missing from `--help` | You are running an old `clipper.py`. Replace it with the newest one. |
| "YouTube gave only audio for this section" | Run `python -m pip install -U "yt-dlp[default]"` and check that `deno --version` works. Alternatively add `--remote-solver`. |
| Warnings about "n challenge solving failed" | Same fix as above. |
| "no webcam face found" | The face may be turned away, covered, too dark or too small. Set the box by hand with `--cam x:y:w:h`, or use `--show game`. |
| "Your OpenCV has no face detector files" | Run `python -m pip install "opencv-python-headless<5"`. |
| `cublas64_12.dll` is not found | Follow section 8, or add `--device cpu`. |
| Webcam crop too tight or too wide | Change `--cam-zoom` (smaller = closer). |
| Webcam looks blurry | Your webcam area in the original video is small. Lower `--cam-height` (try 600) or use a bigger `--cam-zoom`. |
| No captions on a clip | The speech was not English, no speech was found, or you chose `--subs no`. The tool prints which one. |
| Logo or font path with spaces fails | Put the whole path in quotes. |
| "no usable chat data" / "no usable heatmap data" | The video has none, so the tool moves to the next method. This is normal. |
| Clips are not where you expected | Check the folder `clips_<video id>` in the folder you ran the command from. |

If a single clip fails, the others still run and the tool lists what was saved at the end.

---

## 10. Notes

- Clips longer than 3 minutes are not YouTube Shorts, but they upload fine as normal vertical videos.
- Keep yt-dlp updated (`python -m pip install -U "yt-dlp[default]"`). YouTube changes often and old versions stop working.
- Everything runs on your own computer. Nothing is uploaded anywhere.
