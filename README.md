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
