"""Download a match (or part of one) from YouTube as an mp4 with sound, for track_ball.py and demo.py.

Usage:
    python -m balltrack.download URL                       # whole video
    python -m balltrack.download URL --from 6:00 --to 7:30  # just that stretch
"""
import argparse
import os
import tempfile
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent.parent / "videos"


def _seconds(text):
    parts = [float(p) for p in str(text).split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def download(url: str, start=None, end=None, max_height=1080) -> Path:
    import imageio_ffmpeg
    import yt_dlp
    from yt_dlp.utils import download_range_func

    VIDEO_DIR.mkdir(exist_ok=True)
    # yt-dlp only recognises a binary called "ffmpeg"; link to the one bundled with imageio-ffmpeg
    ffmpeg_dir = Path(tempfile.gettempdir()) / "hackgt_ffmpeg"
    ffmpeg_dir.mkdir(exist_ok=True)
    if not (ffmpeg_dir / "ffmpeg").exists():
        (ffmpeg_dir / "ffmpeg").symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"  # partial downloads look it up here
    name = "%(id)s" + (f"_{int(_seconds(start))}-{int(_seconds(end))}" if start is not None else "")
    opts = {
        # H.264 video up to max_height plus AAC sound, merged into one mp4 (ffmpeg from imageio-ffmpeg)
        "format": f"bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]/b[height<={max_height}]",
        "merge_output_format": "mp4",
        "ffmpeg_location": str(ffmpeg_dir),
        "outtmpl": str(VIDEO_DIR / f"{name}.%(ext)s"),
        "noplaylist": True,  # the links carry &list=..., only grab the one video
    }
    if start is not None:
        opts["download_ranges"] = download_range_func(None, [(_seconds(start), _seconds(end))])
        opts["force_keyframes_at_cuts"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--from", dest="start", default=None, help="start time, e.g. 6:00")
    ap.add_argument("--to", dest="end", default=None, help="end time, e.g. 7:30")
    ap.add_argument("--max-height", type=int, default=1080)
    a = ap.parse_args()
    if (a.start is None) != (a.end is None):
        ap.error("give both --from and --to, or neither")
    print(download(a.url, a.start, a.end, a.max_height))
