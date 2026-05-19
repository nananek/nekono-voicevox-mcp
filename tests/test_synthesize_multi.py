"""Unit tests for voicevox_synthesize_multi (Issue #5 注文②).

Strategy:
- voiced segments are mocked via respx so /audio_query + /synthesis return
  predictable WAV bytes
- silence segments don't touch the engine, so they're a pure offline check
- Final concat is read back with `wave` module and frame count is asserted
  against the expected sum (= per-segment frames + silence frames)
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
import pytest
import respx

from nekono_voicevox_mcp import server

# Mirror of test_synthesize_overrides._BASE_QUERY (= minimal AudioQuery shape).
_BASE_QUERY: dict = {
    "accent_phrases": [],
    "speedScale": 1.0,
    "pitchScale": 0.0,
    "intonationScale": 1.0,
    "volumeScale": 1.0,
    "prePhonemeLength": 0.1,
    "postPhonemeLength": 0.1,
    "pauseLengthScale": 1.0,
    "pauseLength": None,
    "outputSamplingRate": 24000,
    "outputStereo": False,
    "kana": "",
}


def _wav_24khz_mono(duration_sec: float) -> bytes:
    """Build a tiny but valid 24kHz mono 16-bit WAV (silence) of given length."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00" * int(24000 * 2 * duration_sec))
    return buf.getvalue()


def _wav_format(sample_rate: int, channels: int, sample_width: int) -> bytes:
    """Build a tiny WAV with non-default format (= for mismatch test)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00" * (sample_rate * sample_width * channels // 10))
    return buf.getvalue()


def test_silence_frames_sizing() -> None:
    """zero-fill bytes count = sec * 24000 * 1 ch * 2 byte."""
    frames = server._silence_frames(0.5)
    assert frames == b"\x00" * (24000 * 2 // 2)  # 0.5 sec → 24000 bytes
    assert all(b == 0 for b in frames)


def test_silence_frames_rejects_negative() -> None:
    with pytest.raises(ValueError, match="silence_seconds must be >= 0"):
        server._silence_frames(-0.5)


def test_wav_frames_extracts_pcm() -> None:
    wav = _wav_24khz_mono(0.1)
    frames = server._wav_frames(wav)
    # 0.1 sec * 24000 Hz * 2 byte/sample = 4800 bytes
    assert len(frames) == 4800
    assert all(b == 0 for b in frames)


def test_wav_frames_rejects_wrong_sample_rate() -> None:
    bad = _wav_format(sample_rate=48000, channels=1, sample_width=2)
    with pytest.raises(ValueError, match="unexpected wav format: 48000Hz"):
        server._wav_frames(bad)


def test_wav_frames_rejects_stereo() -> None:
    bad = _wav_format(sample_rate=24000, channels=2, sample_width=2)
    with pytest.raises(ValueError, match=r"unexpected wav format: 24000Hz/2ch"):
        server._wav_frames(bad)


@respx.mock(base_url=server.ENGINE_URL)
def test_synthesize_multi_voice_silence_voice(tmp_path: Path, respx_mock: respx.Router) -> None:
    """voice + silence + voice → concat、 frame 数の合算を確認。"""
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=dict(_BASE_QUERY)))
    respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono(0.1))
    )

    out = tmp_path / "out.wav"
    result = server._synthesize_multi_impl(
        segments=[
            {"text": "A", "speaker_id": 14},
            {"silence_seconds": 0.5},
            {"text": "B", "speaker_id": 3, "speed_scale": 1.5},
        ],
        output_path=str(out),
    )

    assert result["segment_count"] == 3
    assert result["voiced_count"] == 2
    assert result["silence_count"] == 1
    assert out.exists()

    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        # 0.1s voice + 0.5s silence + 0.1s voice = 0.7s @ 24kHz
        expected = int(24000 * 0.1) + int(24000 * 0.5) + int(24000 * 0.1)
        assert w.getnframes() == expected


@respx.mock(base_url=server.ENGINE_URL)
def test_synthesize_multi_rejects_malformed_segment(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=dict(_BASE_QUERY)))
    respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono(0.1))
    )

    with pytest.raises(ValueError, match="segment 1: must have either"):
        server._synthesize_multi_impl(
            segments=[
                {"text": "A", "speaker_id": 14},
                {"foo": "bar"},  # neither silence nor voiced
            ],
            output_path=str(tmp_path / "out.wav"),
        )


@respx.mock(base_url=server.ENGINE_URL)
def test_synthesize_multi_passes_per_segment_overrides(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    """各 voiced segment の override が /synthesis body に反映されるか。"""
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=dict(_BASE_QUERY)))
    synth_route = respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono(0.05))
    )

    server._synthesize_multi_impl(
        segments=[
            {"text": "fast", "speaker_id": 14, "speed_scale": 2.0},
            {"text": "slow", "speaker_id": 3, "speed_scale": 0.5},
        ],
        output_path=str(tmp_path / "out.wav"),
    )

    import json as _json

    assert len(synth_route.calls) == 2
    body0 = _json.loads(synth_route.calls[0].request.read())
    body1 = _json.loads(synth_route.calls[1].request.read())
    assert body0["speedScale"] == 2.0
    assert body1["speedScale"] == 0.5
