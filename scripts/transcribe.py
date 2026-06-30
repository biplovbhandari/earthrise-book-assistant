"""Download and transcribe YouTube videos from the EarthRISE playlist.

Usage:
    uv run --group indexer python scripts/transcribe.py
    uv run --group indexer python scripts/transcribe.py --force
    uv run --group indexer python scripts/transcribe.py --video-id <id>
    uv run --group indexer python scripts/transcribe.py --whisper-model medium
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path so the repo is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logger = logging.getLogger(__name__)

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLKlxghiZuIM5YyM92lDtsJT7p0Aln3R_k"
TRANSCRIPT_DIR = Path("data/transcripts")
AUDIO_DIR = Path("data/audio")
LOG_DIR = Path("logs")
DEFAULT_WHISPER_MODEL = "large-v3"
_YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v="


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"transcription_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def _get_playlist_videos(playlist_url: str) -> list[dict]:
    """Extract video metadata from a YouTube playlist.

    Returns list of dicts with keys: id, title, duration, url.
    """
    import yt_dlp  # type: ignore[import-not-found]  # noqa: PLC0415

    ydl_opts = {
        "extract_flat": False,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(playlist_url, download=False)

    if info is None:
        return []

    videos = []
    for entry in info.get("entries", []):
        if entry is None:
            continue
        videos.append(
            {
                "id": entry.get("id", ""),
                "title": entry.get("title", ""),
                "duration": float(entry.get("duration") or 0.0),
                "url": entry.get("webpage_url") or f"{_YOUTUBE_WATCH_URL}{entry.get('id', '')}",
            }
        )
    return videos


def _get_video_info(video_id: str) -> dict:
    """Fetch metadata for a single YouTube video.

    Returns dict with keys: id, title, duration, url.
    """
    import yt_dlp  # type: ignore[import-not-found]  # noqa: PLC0415

    url = f"{_YOUTUBE_WATCH_URL}{video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=False)

    if info is None:
        raise RuntimeError(f"Could not fetch metadata for video {video_id!r}")

    return {
        "id": info.get("id", video_id),
        "title": info.get("title", ""),
        "duration": float(info.get("duration") or 0.0),
        "url": info.get("webpage_url") or url,
    }


def _download_audio(video_id: str, output_dir: Path) -> Path:
    """Download best-quality audio for a YouTube video and return the path to the mp3 file.

    Raises RuntimeError if the download fails or the output file cannot be found.
    """
    import yt_dlp  # type: ignore[import-not-found]  # noqa: PLC0415

    url = f"{_YOUTUBE_WATCH_URL}{video_id}"
    outtmpl = str(output_dir / f"{video_id}.%(ext)s")
    expected_path = output_dir / f"{video_id}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        error_count = ydl.download([url])

    if error_count != 0:
        raise RuntimeError(
            f"yt-dlp download failed for video {video_id!r} ({error_count} error(s))"
        )

    if expected_path.exists():
        return expected_path

    # FFmpeg may have kept a non-mp3 extension in edge cases -- find any matching file
    matches = list(output_dir.glob(f"{video_id}.*"))
    if not matches:
        raise RuntimeError(f"Downloaded audio file not found for video {video_id!r}")
    return matches[0]


def _transcribe_audio(audio_path: Path, model_name: str) -> list[dict]:
    """Transcribe an audio file using Whisper.

    Returns list of segment dicts: [{"start": float, "end": float, "text": str}, ...].
    """
    import whisper  # type: ignore[import-not-found]  # noqa: PLC0415

    logger.info("Loading Whisper model %r", model_name)
    model = whisper.load_model(model_name)
    logger.info("Transcribing %s", audio_path)
    result = model.transcribe(str(audio_path))
    raw_segments: list[dict] = result.get("segments", [])  # type: ignore[assignment]
    return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in raw_segments]


def _save_transcript(
    video_id: str,
    title: str,
    duration: float,
    url: str,
    segments: list[dict],
    output_dir: Path,
) -> Path:
    """Save transcript data as JSON and return the output path."""
    data = {
        "video_id": video_id,
        "title": title,
        "duration_seconds": duration,
        "url": url,
        "segments": segments,
    }
    output_path = output_dir / f"{video_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def main() -> int:
    _setup_logging()

    default_model = os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)

    parser = argparse.ArgumentParser(
        description="Download and transcribe YouTube videos from the EarthRISE playlist."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe videos that already have a saved transcript.",
    )
    parser.add_argument(
        "--video-id",
        metavar="ID",
        help="Transcribe a single video by YouTube ID instead of the full playlist.",
    )
    parser.add_argument(
        "--whisper-model",
        default=default_model,
        metavar="MODEL",
        help=f"Whisper model to use for transcription (default: {default_model}).",
    )
    args = parser.parse_args()

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if args.video_id:
        logger.info("Fetching metadata for video %r", args.video_id)
        try:
            video = _get_video_info(args.video_id)
        except Exception as e:
            logger.error("Failed to fetch metadata for video %r: %s", args.video_id, e)
            return 1
        videos = [video]
    else:
        logger.info("Fetching playlist: %s", PLAYLIST_URL)
        try:
            videos = _get_playlist_videos(PLAYLIST_URL)
        except Exception as e:
            logger.error("Failed to fetch playlist: %s", e)
            return 1
        logger.info("Found %d videos in playlist", len(videos))

    total = len(videos)
    transcribed = 0
    skipped = 0
    failed = 0

    for video in videos:
        video_id = video["id"]
        title = video["title"]
        transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"

        if transcript_path.exists() and not args.force:
            logger.info("Skipping %r (%s) -- transcript already exists", title, video_id)
            skipped += 1
            continue

        try:
            logger.info("Downloading audio for %r (%s)", title, video_id)
            audio_path = _download_audio(video_id, AUDIO_DIR)

            logger.info("Transcribing %r (%s)", title, video_id)
            segments = _transcribe_audio(audio_path, args.whisper_model)

            saved_path = _save_transcript(
                video_id=video_id,
                title=title,
                duration=video["duration"],
                url=video["url"],
                segments=segments,
                output_dir=TRANSCRIPT_DIR,
            )
            logger.info(
                "Saved transcript for %r (%s): %d segments -> %s",
                title,
                video_id,
                len(segments),
                saved_path,
            )
            transcribed += 1
        except Exception as e:
            logger.error("Failed to process video %r (%s): %s", title, video_id, e)
            failed += 1

    logger.info("=== Transcription Complete ===")
    logger.info(
        "Total: %d | Transcribed: %d | Skipped: %d | Failed: %d",
        total,
        transcribed,
        skipped,
        failed,
    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
