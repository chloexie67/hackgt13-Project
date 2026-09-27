"""Demo: play a pre-processed clip with its sound and send the ball data to the ESP32 in sync.

Each message is one text line:  t,valid,x,y,vx,vy
    t      video time (s)
    valid  1 = ball position known, 0 = no data (paused, lost, in the air): hold or stop the motors
    x, y   metres from the centre spot (+x toward the right-hand goal, +y toward the near touchline)
    vx, vy m/s; constant for the whole of each pass

Usage:
    python demo.py clip.mp4 --udp 192.168.1.50:5005          # ESP32 over Wi-Fi
    python demo.py clip.mp4 --serial /dev/cu.usbserial-0001   # ESP32 over USB or Bluetooth serial
    python demo.py clip.mp4                                   # just play and print what would be sent
Press q in the video window (or Ctrl+C) to stop.
"""
import argparse
import bisect
import csv
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("timeline", nargs="?", help="default: <video>.timeline.csv from track_ball.py")
    p.add_argument("--udp", help="send to HOST:PORT over UDP")
    p.add_argument("--serial", help="send to a serial port (USB, or a paired Bluetooth serial device)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--rate", type=float, default=20.0, help="messages per second")
    p.add_argument("--start", type=float, default=None, help="video time to start at (default: start of the timeline)")
    p.add_argument("--end", type=float, default=None, help="video time to stop at (default: end of the timeline)")
    p.add_argument("--audio-delay", type=float, default=0.0,
                   help="seconds to delay the picture and data if the sound runs ahead (or negative if behind)")
    p.add_argument("--no-sound", action="store_true")
    p.add_argument("--fullscreen", action="store_true")
    args = p.parse_args()

    timeline_path = Path(args.timeline or Path(args.video).with_suffix(".timeline.csv"))
    rows = list(csv.DictReader(open(timeline_path)))
    if not rows:
        raise SystemExit(f"{timeline_path} is empty")
    times = [float(r["time_s"]) for r in rows]
    start = times[0] if args.start is None else args.start
    end = times[-1] if args.end is None else args.end

    udp_sock = udp_addr = serial_port = None
    if args.udp:
        host, port = args.udp.rsplit(":", 1)
        udp_sock, udp_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))
    if args.serial:
        import serial
        serial_port = serial.Serial(args.serial, args.baud)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    window = "demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    audio = None
    if not args.no_sound:
        # cut the same stretch of sound and play it with macOS's built-in player
        import imageio_ffmpeg
        wav = Path(tempfile.gettempdir()) / "hackgt_demo_audio.wav"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end - start), "-i", args.video, "-vn", "-ac", "2", "-ar", "44100", str(wav)],
                       check=False)
        if wav.exists() and wav.stat().st_size > 1000:
            audio = subprocess.Popen(["afplay", str(wav)])
        else:
            print("no sound track found; playing without sound")

    t0 = time.perf_counter() + args.audio_delay
    frame_t = start - 1 / fps
    next_send = 0.0
    last_print = 0.0
    try:
        while True:
            now = start + time.perf_counter() - t0
            if now > end:
                break

            # video: show the frame for the current time, skipping frames if we fall behind
            if now >= frame_t + 1 / fps:
                ok = True
                while ok and frame_t + 1 / fps <= now - 1 / fps:
                    ok = cap.grab()
                    frame_t += 1 / fps
                ok, frame = cap.read()
                if not ok:
                    break
                frame_t += 1 / fps
                cv2.imshow(window, frame)

            # data: the timeline row for the current time
            if now >= next_send:
                next_send = now + 1 / args.rate
                row = rows[max(bisect.bisect_right(times, now) - 1, 0)]
                valid = row["valid"] == "1"
                values = [row[k] if valid and row[k] != "" else "0" for k in ("x_m", "y_m", "vx_ms", "vy_ms")]
                line = f"{now:.2f},{int(valid)}," + ",".join(values) + "\n"
                if udp_sock is not None:
                    udp_sock.sendto(line.encode(), udp_addr)
                if serial_port is not None:
                    serial_port.write(line.encode())
                if now - last_print >= 0.5:
                    print(line, end="")
                    last_print = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if audio is not None:
            audio.terminate()
        cap.release()
        cv2.destroyAllWindows()
        if serial_port is not None:
            serial_port.close()


if __name__ == "__main__":
    main()
