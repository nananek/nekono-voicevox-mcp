"""Unit tests for AudioQuery override behaviour (Issue #5 注文①).

Mock the engine HTTP API with respx and assert that:
- `voicevox_synthesize` calls /audio_query then /synthesis
- The /synthesis request body has the user-supplied override fields written
  in (e.g. speedScale=1.4 when speed_scale=1.4 is passed)
- Fields the caller did NOT touch are passed through from /audio_query
  unchanged (= engine defaults preserved)
- Override helper is independent of engine network (= pure dict mutation)
"""

from __future__ import annotations

import wave

import httpx
import pytest
import respx

from nekono_voicevox_mcp import server

# Minimal AudioQuery response shape the engine actually returns.
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


def _wav_24khz_mono(duration_sec: float = 0.05) -> bytes:
    """Build a tiny but valid 24kHz mono 16-bit WAV (silence) for /synthesis mock."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00" * int(24000 * 2 * duration_sec))
    return buf.getvalue()


def test_apply_query_overrides_writes_only_non_none() -> None:
    """Helper writes camelCase fields for non-None overrides only."""
    query = dict(_BASE_QUERY)
    server._apply_query_overrides(
        query,
        speed_scale=1.4,
        pitch_scale=None,  # unchanged
        pause_length_scale=2.0,
    )
    assert query["speedScale"] == 1.4
    assert query["pauseLengthScale"] == 2.0
    assert query["pitchScale"] == 0.0  # untouched
    assert query["intonationScale"] == 1.0  # untouched


def test_apply_query_overrides_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown AudioQuery override"):
        server._apply_query_overrides({}, bogus_field=1.0)


@respx.mock(base_url=server.ENGINE_URL)
def test_synthesize_to_bytes_sends_overrides(respx_mock: respx.Router) -> None:
    """_synthesize_to_bytes hits /audio_query then /synthesis with overrides applied.

    Tests the internal helper rather than the @mcp.tool wrapper to avoid
    FastMCP-version-specific decorator-attribute coupling.
    """
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=dict(_BASE_QUERY)))
    synth_route = respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono())
    )

    wav = server._synthesize_to_bytes(
        text="テスト",
        speaker_id=14,
        speed_scale=1.4,
        pause_length_scale=2.0,
    )

    assert wav.startswith(b"RIFF")  # valid WAV header
    assert synth_route.called
    import json as _json

    body = _json.loads(synth_route.calls.last.request.read())
    assert body["speedScale"] == 1.4
    assert body["pauseLengthScale"] == 2.0
    # Untouched fields stay at engine defaults.
    assert body["pitchScale"] == 0.0
    assert body["intonationScale"] == 1.0


@respx.mock(base_url=server.ENGINE_URL)
def test_synthesize_to_bytes_no_overrides_preserves_defaults(
    respx_mock: respx.Router,
) -> None:
    """No override kwargs = /synthesis body == /audio_query response verbatim."""
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=dict(_BASE_QUERY)))
    synth_route = respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono())
    )

    server._synthesize_to_bytes(text="テスト", speaker_id=14)
    import json as _json

    body = _json.loads(synth_route.calls.last.request.read())
    for k, v in _BASE_QUERY.items():
        assert body[k] == v, f"field {k} drifted from default"
