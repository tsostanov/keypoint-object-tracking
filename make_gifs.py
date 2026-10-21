"""Сделать GIF для README из трёх обработанных тестовых видео."""

from pathlib import Path
import subprocess
import sys
import tempfile

from prepare_previews import find_ffmpeg


ROOT = Path(__file__).resolve().parent
NAMES = ("mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit")


def main():
    try:
        sources = [ROOT / "results/videos" / f"{name}_tracked.mp4" for name in NAMES]
        for source in sources:
            if not source.is_file():
                raise ValueError(f"Сначала обработайте исходное видео: нет {source.relative_to(ROOT)}")
        ffmpeg = find_ffmpeg()
        directory = ROOT / "results/videos/gifs"
        directory.mkdir(parents=True, exist_ok=True)
        # Полная длительность; размер и FPS уменьшены для загрузки на GitHub.
        filters = (
            "fps=6,scale=512:-1:flags=lanczos,split[frames][palette_source];"
            "[palette_source]palettegen=max_colors=96:stats_mode=diff[palette];"
            "[frames][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        )
        for source, name in zip(sources, NAMES):
            output = directory / f"{name}.gif"
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".gif", delete=False) as file:
                temporary = Path(file.name)
            try:
                subprocess.run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(source), "-filter_complex", filters,
                    "-an", "-loop", "0", str(temporary),
                ], check=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Готово: {output.relative_to(ROOT)} ({output.stat().st_size / 1024**2:.2f} МиБ)", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
