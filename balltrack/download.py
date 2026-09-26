"""Download match footage from YouTube as a single 720p video-only file.

Usage:
    python -m balltrack.download URL [URL ...]

Video-only streams are used so ffmpeg is not needed for merging.
"""
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent.parent / "videos"


def download(url: str) -> Path:
    import yt_dlp

    VIDEO_DIR.mkdir(exist_ok=True)
    opts = {
        # Prefer a single mp4 stream <=720p; fall back to any single stream <=720p.
        "format": "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/bv*[height<=720]/b",
        "outtmpl": str(VIDEO_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,  # the links carry &list=..., only grab the one video
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for u in sys.argv[1:]:
        print(download(u))
