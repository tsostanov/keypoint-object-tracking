"""Создать копии исходников и результатов в H.264 для просмотра в VS Code."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def find_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ValueError(
            "Установите инструмент конвертации: "
            "python -m pip install -r requirements-preview.txt"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    try:
        sources = sorted(
            path for path in (ROOT / "data/videos").glob("*")
            if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".webm"}
        )
        sources += sorted((ROOT / "results/videos").glob("*_tracked.mp4"))
        if not sources:
            raise ValueError("Нет видео в data/videos/ или результатов в results/videos/.")
        names = [path.stem.casefold() for path in sources]
        if len(names) != len(set(names)):
            raise ValueError("У видео одинаковые имена без расширения. Переименуйте их перед конвертацией.")
        ffmpeg = find_ffmpeg()
        output_directory = ROOT / "results/videos/playback"
        output_directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            output = output_directory / (source.stem + ".mp4")
            # Заменяем готовую копию только после успешного завершения FFmpeg.
            with tempfile.NamedTemporaryFile(dir=output_directory, suffix=".mp4", delete=False) as file:
                temporary = Path(file.name)
            try:
                subprocess.run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(source), "-map", "0:v:0", "-an",
                    "-vf", "scale=iw:ih:in_range=auto:out_range=tv,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-color_range", "tv",
                    "-movflags", "+faststart", str(temporary),
                ], check=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Готово: {output.relative_to(ROOT)}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
