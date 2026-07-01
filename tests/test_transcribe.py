import json
import sys
from unittest.mock import MagicMock

from scripts.transcribe import _save_transcript, main


def test_save_transcript_writes_json(tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "text": "Hello."}]
    path = _save_transcript(
        video_id="abc123",
        title="Test Video",
        duration=120.5,
        url="https://www.youtube.com/watch?v=abc123",
        segments=segments,
        output_dir=tmp_path,
    )

    assert path == tmp_path / "abc123.json"
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["video_id"] == "abc123"
    assert data["title"] == "Test Video"
    assert data["duration_seconds"] == 120.5
    assert data["url"] == "https://www.youtube.com/watch?v=abc123"
    assert data["segments"] == segments


def test_incremental_skip(tmp_path, monkeypatch):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Pre-create an existing transcript so the script treats this video as done.
    (transcript_dir / "vid1.json").write_text("{}")

    monkeypatch.setattr("scripts.transcribe.TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setattr("scripts.transcribe.AUDIO_DIR", audio_dir)
    monkeypatch.setattr("scripts.transcribe._setup_logging", lambda: None)
    monkeypatch.setattr(
        "scripts.transcribe._get_playlist_videos",
        lambda url: [{"id": "vid1", "title": "Video 1", "duration": 100.0, "url": "http://yt.com"}],
    )
    download_mock = MagicMock()
    monkeypatch.setattr("scripts.transcribe._download_audio", download_mock)
    monkeypatch.setattr(sys, "argv", ["transcribe.py"])

    result = main()

    download_mock.assert_not_called()
    assert result == 0


def test_force_reprocesses(tmp_path, monkeypatch):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Pre-create an existing transcript; --force should ignore it and re-process.
    (transcript_dir / "vid1.json").write_text("{}")

    monkeypatch.setattr("scripts.transcribe.TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setattr("scripts.transcribe.AUDIO_DIR", audio_dir)
    monkeypatch.setattr("scripts.transcribe._setup_logging", lambda: None)
    monkeypatch.setattr(
        "scripts.transcribe._get_playlist_videos",
        lambda url: [{"id": "vid1", "title": "Video 1", "duration": 100.0, "url": "http://yt.com"}],
    )
    download_mock = MagicMock(return_value=audio_dir / "vid1.mp3")
    transcribe_mock = MagicMock(return_value=[{"start": 0.0, "end": 10.0, "text": "Hi."}])
    save_mock = MagicMock(return_value=transcript_dir / "vid1.json")
    monkeypatch.setattr("scripts.transcribe._download_audio", download_mock)
    monkeypatch.setattr("scripts.transcribe._transcribe_audio", transcribe_mock)
    monkeypatch.setattr("scripts.transcribe._save_transcript", save_mock)
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--force"])

    result = main()

    download_mock.assert_called_once_with("vid1", audio_dir)
    transcribe_mock.assert_called_once()
    assert result == 0


def test_error_isolation(tmp_path, monkeypatch):
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    monkeypatch.setattr("scripts.transcribe.TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setattr("scripts.transcribe.AUDIO_DIR", audio_dir)
    monkeypatch.setattr("scripts.transcribe._setup_logging", lambda: None)
    monkeypatch.setattr(
        "scripts.transcribe._get_playlist_videos",
        lambda url: [
            {"id": "fail_vid", "title": "Fails", "duration": 100.0, "url": "http://yt.com"},
            {"id": "ok_vid", "title": "Works", "duration": 200.0, "url": "http://yt.com"},
        ],
    )

    def fake_download(video_id, output_dir):
        if video_id == "fail_vid":
            raise RuntimeError("Download failed")
        audio_file = output_dir / f"{video_id}.mp3"
        audio_file.write_text("fake audio")
        return audio_file

    monkeypatch.setattr("scripts.transcribe._download_audio", fake_download)
    monkeypatch.setattr(
        "scripts.transcribe._transcribe_audio",
        lambda *args, **kwargs: [{"start": 0.0, "end": 10.0, "text": "Hi."}],
    )
    monkeypatch.setattr(sys, "argv", ["transcribe.py"])

    result = main()

    # One video failed so the exit code is non-zero.
    assert result == 1
    # ok_vid was processed successfully and has a saved transcript.
    assert (transcript_dir / "ok_vid.json").exists()
    # fail_vid never reached _save_transcript so no transcript was written.
    assert not (transcript_dir / "fail_vid.json").exists()
