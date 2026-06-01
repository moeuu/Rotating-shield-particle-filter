"""Build RA-L supplementary video assets from paper figures or Isaac Sim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "ral_supplementary_video" / "ral_supplementary_storyboard.mp4"
DEFAULT_FRAME_ROOT = ROOT / "results" / "ral_supplementary_video" / "frames"
SANS_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class VideoSegment:
    """One image-backed segment in the supplementary storyboard video."""

    title: str
    caption: str
    image_path: Path


def _resample_filter() -> int:
    """Return a high-quality Pillow resampling filter across Pillow versions."""
    resampling = getattr(Image, "Resampling", Image)
    return int(getattr(resampling, "LANCZOS"))


def _font(size_px: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a readable sans-serif font for video overlays."""
    font_path = SANS_BOLD if bold else SANS_REGULAR
    try:
        return ImageFont.truetype(font_path, size_px)
    except OSError:
        return ImageFont.load_default()


def discover_storyboard_segments(root: Path = ROOT) -> list[VideoSegment]:
    """Return available figure assets for a default RA-L supplementary video."""
    figure_root = root / "results" / "ral_isaac_figures"
    latex_root = root.parent / "latex" / "projects" / "ieee-ra-l-letter"
    candidates = [
        VideoSegment(
            title="Problem setting",
            caption=(
                "A mobile robot carries an energy-resolving nondirectional detector "
                "and pose-adjustable Fe/Pb octant shields in a known 3-D environment."
            ),
            image_path=figure_root / "problem_setting.png",
        ),
        VideoSegment(
            title="Detector and shield module",
            caption=(
                "The CeBr3 detector is surrounded by independently rotated Fe and Pb "
                "octant shells, producing posture-dependent spectral count signatures."
            ),
            image_path=figure_root / "detector_module.png",
        ),
        VideoSegment(
            title="Temporal shield program",
            caption=(
                "At each robot station, a short sequence of Fe/Pb shield postures "
                "creates discriminative temporal response patterns."
            ),
            image_path=figure_root / "shield_selection.png",
        ),
        VideoSegment(
            title="Obstacle-aware transport",
            caption=(
                "Known obstacles are used by Geant4 and by the PF expected-count "
                "kernel through material-dependent attenuation."
            ),
            image_path=figure_root / "geant4_obstacle_scatter.png",
        ),
        VideoSegment(
            title="MIX-9 PF output",
            caption=(
                "The final figure summarizes source-term estimates, ground truth, "
                "PF support, and the robot trajectory for the main multi-isotope run."
            ),
            image_path=latex_root
            / "sections/05_experiments/figures/ral_result_overview.png",
        ),
    ]
    return [segment for segment in candidates if segment.image_path.exists()]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width_px: int,
) -> list[str]:
    """Wrap text to fit a fixed pixel width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width_px:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _fit_image_to_box(
    image: Image.Image,
    box_size: tuple[int, int],
    *,
    zoom: float = 1.0,
) -> Image.Image:
    """Resize and center-crop an image so it fills a fixed box."""
    box_w, box_h = box_size
    src_w, src_h = image.size
    scale = max(box_w / max(src_w, 1), box_h / max(src_h, 1)) * max(float(zoom), 1.0)
    resized = image.resize(
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        _resample_filter(),
    )
    left = max(0, (resized.size[0] - box_w) // 2)
    top = max(0, (resized.size[1] - box_h) // 2)
    return resized.crop((left, top, left + box_w, top + box_h))


def render_storyboard_frame(
    segment: VideoSegment,
    output_path: Path,
    *,
    resolution: tuple[int, int] = (1280, 720),
    progress: float = 0.0,
) -> Path:
    """Render one storyboard frame for a video segment."""
    width, height = resolution
    frame = Image.new("RGB", resolution, (248, 248, 246))
    draw = ImageDraw.Draw(frame)
    title_font = _font(max(18, min(34, int(height * 0.047))), bold=True)
    caption_font = _font(max(12, min(21, int(height * 0.029))))
    small_font = _font(max(10, min(15, int(height * 0.021))))
    margin = max(18, int(width * 0.042))
    header_h = max(44, min(76, int(height * 0.11)))
    image_top = header_h + max(10, int(height * 0.022))
    caption_top = height - max(54, int(height * 0.145))
    image_box = (margin, image_top, width - margin, caption_top - 24)
    if image_box[3] <= image_box[1]:
        image_box = (margin, image_top, width - margin, max(image_top + 1, caption_top - 8))
    image_size = (image_box[2] - image_box[0], image_box[3] - image_box[1])
    progress_clamped = min(max(float(progress), 0.0), 1.0)
    zoom = 1.0 + 0.025 * progress_clamped

    with Image.open(segment.image_path) as opened:
        source = opened.convert("RGB")
        fitted = _fit_image_to_box(source, image_size, zoom=zoom)
    frame.paste(fitted, image_box[:2])
    draw.rectangle(image_box, outline=(52, 58, 64), width=2)

    draw.rectangle((0, 0, width, header_h), fill=(28, 33, 38))
    draw.text(
        (margin, max(10, int(header_h * 0.28))),
        segment.title,
        fill=(255, 255, 255),
        font=title_font,
    )

    caption_lines = _wrap_text(
        draw,
        segment.caption,
        caption_font,
        width - 2 * margin,
    )[:2]
    draw.multiline_text(
        (margin, caption_top),
        "\n".join(caption_lines),
        fill=(28, 33, 38),
        font=caption_font,
        spacing=6,
    )
    bar_left = margin
    bar_right = width - margin
    bar_y = height - 22
    draw.line((bar_left, bar_y, bar_right, bar_y), fill=(190, 194, 198), width=5)
    draw.line(
        (bar_left, bar_y, bar_left + int((bar_right - bar_left) * progress_clamped), bar_y),
        fill=(0, 113, 188),
        width=5,
    )
    draw.text(
        (bar_right - max(94, int(width * 0.073)), max(10, int(header_h * 0.26))),
        "RA-L video",
        fill=(196, 202, 208),
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output_path)
    return output_path


def write_storyboard_frames(
    segments: Sequence[VideoSegment],
    output_dir: Path,
    *,
    fps: int = 24,
    seconds_per_segment: float = 3.0,
    resolution: tuple[int, int] = (1280, 720),
) -> list[Path]:
    """Write a numbered PNG frame sequence for the storyboard video."""
    if not segments:
        raise ValueError("At least one storyboard segment is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frames_per_segment = max(1, int(round(float(fps) * float(seconds_per_segment))))
    frame_index = 0
    for segment in segments:
        for local_index in range(frames_per_segment):
            progress = local_index / max(frames_per_segment - 1, 1)
            frame_path = output_dir / f"frame_{frame_index:05d}.png"
            render_storyboard_frame(
                segment,
                frame_path,
                resolution=resolution,
                progress=progress,
            )
            frame_paths.append(frame_path)
            frame_index += 1
    return frame_paths


def build_ffmpeg_command(
    frames_dir: Path,
    output_path: Path,
    *,
    fps: int = 24,
    overwrite: bool = True,
) -> list[str]:
    """Return the ffmpeg command used to encode an MP4 from PNG frames."""
    command = ["ffmpeg"]
    command.append("-y" if overwrite else "-n")
    command.extend(
        [
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def encode_mp4(
    frames_dir: Path,
    output_path: Path,
    *,
    fps: int = 24,
    overwrite: bool = True,
) -> Path:
    """Encode a PNG frame sequence into an H.264 MP4 file."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the supplementary video.")
    if not any(frames_dir.glob("frame_*.png")):
        raise ValueError(f"No PNG frames found in {frames_dir}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(frames_dir, output_path, fps=fps, overwrite=overwrite)
    subprocess.run(command, check=True)
    return output_path


def build_storyboard_video(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    frame_dir: Path | None = None,
    fps: int = 24,
    seconds_per_segment: float = 3.0,
    resolution: tuple[int, int] = (1280, 720),
    keep_frames: bool = False,
) -> Path:
    """Build the default figure-based RA-L supplementary storyboard video."""
    segments = discover_storyboard_segments(ROOT)
    if not segments:
        raise RuntimeError("No RA-L figure images were found for the storyboard.")
    work_dir = frame_dir or (DEFAULT_FRAME_ROOT / "storyboard")
    if work_dir.exists() and not keep_frames:
        shutil.rmtree(work_dir)
    write_storyboard_frames(
        segments,
        work_dir,
        fps=fps,
        seconds_per_segment=seconds_per_segment,
        resolution=resolution,
    )
    video_path = encode_mp4(work_dir, output_path, fps=fps, overwrite=True)
    if not keep_frames:
        shutil.rmtree(work_dir, ignore_errors=True)
    return video_path


def _interpolate_polyline(
    points: Sequence[tuple[float, float]],
    t_norm: float,
) -> tuple[float, float]:
    """Return a point along a polyline for normalized time in [0, 1]."""
    if not points:
        raise ValueError("At least one point is required.")
    if len(points) == 1:
        return points[0]
    t = min(max(float(t_norm), 0.0), 1.0) * (len(points) - 1)
    left = min(int(t), len(points) - 2)
    frac = t - left
    x0, y0 = points[left]
    x1, y1 = points[left + 1]
    return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))


def capture_isaac_motion_frames(
    output_dir: Path,
    *,
    frame_count: int = 96,
    resolution: tuple[int, int] = (1280, 720),
) -> list[Path]:
    """Capture a robot-and-shield motion frame sequence from Isaac Sim."""
    from scripts.render_isaac_ral_figures import (  # noqa: WPS433
        IsaacSimApplication,
        _app_config,
        _capture,
        _command,
        _scene_description,
        _set_camera,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    app = IsaacSimApplication(use_mock=False, app_config=_app_config())
    frame_paths: list[Path] = []
    path = (
        (1.2, 1.4),
        (2.4, 3.1),
        (4.3, 5.9),
        (5.5, 8.2),
        (7.1, 10.4),
        (8.0, 12.2),
    )
    try:
        app.reset(_scene_description())
        _set_camera(
            app,
            "/World/SimBridge/View/SupplementaryVideoCamera",
            eye=(5.0, -4.7, 8.9),
            target=(5.0, 8.0, 0.1),
            focal_length_mm=25.0,
        )
        for index in range(max(1, int(frame_count))):
            t_norm = index / max(int(frame_count) - 1, 1)
            pose_xy = _interpolate_polyline(path, t_norm)
            yaw = 0.35 + 1.1 * t_norm
            fe_index = (index // 4) % 8
            pb_index = (index // 6 + 3) % 8
            app.step(
                _command(
                    step_id=index,
                    pose_xy=pose_xy,
                    yaw=yaw,
                    fe=fe_index,
                    pb=pb_index,
                )
            )
            captured = _capture(
                camera_path="/World/SimBridge/View/SupplementaryVideoCamera",
                output_dir=output_dir,
                name=f"isaac_raw_{index:05d}",
                resolution=resolution,
            )
            frame_path = output_dir / f"frame_{index:05d}.png"
            shutil.copy2(captured, frame_path)
            frame_paths.append(frame_path)
    finally:
        app.close()
    return frame_paths


def build_isaac_motion_video(
    output_path: Path,
    *,
    frame_dir: Path | None = None,
    fps: int = 24,
    frame_count: int = 96,
    resolution: tuple[int, int] = (1280, 720),
    keep_frames: bool = False,
) -> Path:
    """Capture and encode a supplementary Isaac Sim motion video."""
    work_dir = frame_dir or (DEFAULT_FRAME_ROOT / "isaac_motion")
    if work_dir.exists() and not keep_frames:
        shutil.rmtree(work_dir)
    capture_isaac_motion_frames(work_dir, frame_count=frame_count, resolution=resolution)
    video_path = encode_mp4(work_dir, output_path, fps=fps, overwrite=True)
    if not keep_frames:
        shutil.rmtree(work_dir, ignore_errors=True)
    return video_path


def _parse_resolution(raw: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT CLI resolution text."""
    if "x" not in raw.lower():
        raise argparse.ArgumentTypeError("resolution must be formatted as WIDTHxHEIGHT.")
    left, right = raw.lower().split("x", 1)
    width = int(left)
    height = int(right)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive.")
    return width, height


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for supplementary video generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("storyboard", "isaac"),
        default="storyboard",
        help="Video source: existing figure storyboard or Isaac Sim motion capture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument(
        "--seconds-per-segment",
        type=float,
        default=3.0,
        help="Storyboard seconds per still figure.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=96,
        help="Number of Isaac Sim frames for motion capture.",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(1280, 720),
        help="Video resolution formatted as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Optional frame working directory.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the generated PNG frame sequence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the supplementary video builder from the command line."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "storyboard":
        path = build_storyboard_video(
            output_path=args.output,
            frame_dir=args.frame_dir,
            fps=args.fps,
            seconds_per_segment=args.seconds_per_segment,
            resolution=args.resolution,
            keep_frames=args.keep_frames,
        )
    else:
        path = build_isaac_motion_video(
            output_path=args.output,
            frame_dir=args.frame_dir,
            fps=args.fps,
            frame_count=args.frame_count,
            resolution=args.resolution,
            keep_frames=args.keep_frames,
        )
    print(f"Supplementary video written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
