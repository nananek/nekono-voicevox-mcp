"""nekono-voicevox-mcp: MCP server for VOICEVOX engine HTTP API.

Tools (MVP):
- voicevox_list_speakers() — `/speakers`, name keyed dict of style ids
- voicevox_synthesize(text, speaker_id, output_path, **overrides) — wav file output
- voicevox_play(text, speaker_id, **overrides) — pipe through pw-cat to local PipeWire

Engine URL is read from $VOICEVOX_ENGINE_URL (default http://127.0.0.1:50021).
The engine itself is out of scope for this server — point it at any running
VOICEVOX engine, local or remote.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP

DEFAULT_ENGINE_URL = "http://127.0.0.1:50021"
ENGINE_URL = os.environ.get("VOICEVOX_ENGINE_URL", DEFAULT_ENGINE_URL)
PW_CAT = shutil.which("pw-cat") or "/usr/bin/pw-cat"

# Mapping from this server's snake_case override names to VOICEVOX engine
# AudioQuery JSON camelCase fields. Issue #5 注文① — exposed so callers can
# tweak engine defaults without going through the editor.
_QUERY_OVERRIDE_FIELDS: dict[str, str] = {
    "speed_scale": "speedScale",
    "pitch_scale": "pitchScale",
    "intonation_scale": "intonationScale",
    "volume_scale": "volumeScale",
    "pre_phoneme_length": "prePhonemeLength",
    "post_phoneme_length": "postPhonemeLength",
    "pause_length_scale": "pauseLengthScale",
    "pause_length": "pauseLength",
}

mcp = FastMCP("nekono-voicevox")


def _client() -> httpx.Client:
    """httpx client pinned to the configured engine base URL."""
    return httpx.Client(base_url=ENGINE_URL, timeout=httpx.Timeout(60.0, read=120.0))


def _apply_query_overrides(query: dict[str, Any], **overrides: float | None) -> dict[str, Any]:
    """Apply non-None overrides to the AudioQuery dict in place and return it.

    Override names are this server's snake_case form (`speed_scale` etc.) and
    are mapped to engine camelCase fields. None values are skipped so engine
    defaults remain unchanged.
    """
    for snake, value in overrides.items():
        if value is None:
            continue
        camel = _QUERY_OVERRIDE_FIELDS.get(snake)
        if camel is None:
            raise ValueError(f"unknown AudioQuery override: {snake}")
        query[camel] = value
    return query


def _synthesize_to_bytes(
    text: str,
    speaker_id: int,
    *,
    speed_scale: float | None = None,
    pitch_scale: float | None = None,
    intonation_scale: float | None = None,
    volume_scale: float | None = None,
    pre_phoneme_length: float | None = None,
    post_phoneme_length: float | None = None,
    pause_length_scale: float | None = None,
    pause_length: float | None = None,
) -> bytes:
    """Run /audio_query + /synthesis and return raw WAV bytes.

    Shared by `voicevox_synthesize`, `voicevox_play`, and the (forthcoming)
    multi-segment connector. Override kwargs default to None — only non-None
    values are written into the AudioQuery JSON before /synthesis, so engine
    defaults are preserved otherwise.
    """
    with _client() as client:
        query_response = client.post("/audio_query", params={"text": text, "speaker": speaker_id})
        query_response.raise_for_status()
        query = _apply_query_overrides(
            query_response.json(),
            speed_scale=speed_scale,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            volume_scale=volume_scale,
            pre_phoneme_length=pre_phoneme_length,
            post_phoneme_length=post_phoneme_length,
            pause_length_scale=pause_length_scale,
            pause_length=pause_length,
        )
        synth = client.post(
            "/synthesis",
            params={"speaker": speaker_id},
            json=query,
            headers={"Accept": "audio/wav", "Content-Type": "application/json"},
        )
        synth.raise_for_status()
    return synth.content


@mcp.tool
def voicevox_list_speakers() -> dict[str, list[dict[str, Any]]]:
    """List all available VOICEVOX speakers and their style ids.

    Returns: {speaker_name: [{style_name, style_id}, ...], ...}

    Use this first to find the style_id for the voice you want, then pass
    that style_id as `speaker_id` to the synthesize / play tools below.
    """
    with _client() as client:
        response = client.get("/speakers")
        response.raise_for_status()
        speakers = response.json()
    return {
        s["name"]: [{"style_name": st["name"], "style_id": st["id"]} for st in s["styles"]]
        for s in speakers
    }


@mcp.tool
def voicevox_synthesize(
    text: str,
    speaker_id: int,
    output_path: str,
    speed_scale: float | None = None,
    pitch_scale: float | None = None,
    intonation_scale: float | None = None,
    volume_scale: float | None = None,
    pre_phoneme_length: float | None = None,
    post_phoneme_length: float | None = None,
    pause_length_scale: float | None = None,
    pause_length: float | None = None,
) -> dict[str, Any]:
    """Synthesize text to a WAV file with the given speaker style.

    Engine pipeline: /audio_query (= phoneme + intonation inference) →
    /synthesis (= waveform generation); saves the WAV to output_path.

    Use voicevox_list_speakers() to find a valid speaker_id (= style_id).

    Optional AudioQuery overrides (None = use engine default):
    - speed_scale: speech rate multiplier (1.0 = default)
    - pitch_scale: pitch shift in semitones (0 = default)
    - intonation_scale: intonation depth multiplier (1.0 = default)
    - volume_scale: output volume multiplier (1.0 = default)
    - pre_phoneme_length / post_phoneme_length: silence padding before/after
      the utterance, in seconds
    - pause_length_scale: multiplier for all 句点/読点 pauses (1.0 = default)
    - pause_length: fixed 句点 pause in seconds (takes precedence over
      pause_length_scale on the engine side)

    Engine 0.14+ required for pause_length_scale / pause_length.
    Returns the written path and byte count for readback.
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wav = _synthesize_to_bytes(
        text,
        speaker_id,
        speed_scale=speed_scale,
        pitch_scale=pitch_scale,
        intonation_scale=intonation_scale,
        volume_scale=volume_scale,
        pre_phoneme_length=pre_phoneme_length,
        post_phoneme_length=post_phoneme_length,
        pause_length_scale=pause_length_scale,
        pause_length=pause_length,
    )
    output.write_bytes(wav)
    return {"output_path": str(output), "size_bytes": output.stat().st_size}


@mcp.tool
def voicevox_play(
    text: str,
    speaker_id: int,
    speed_scale: float | None = None,
    pitch_scale: float | None = None,
    intonation_scale: float | None = None,
    volume_scale: float | None = None,
    pre_phoneme_length: float | None = None,
    post_phoneme_length: float | None = None,
    pause_length_scale: float | None = None,
    pause_length: float | None = None,
) -> dict[str, Any]:
    """Synthesize text and play it through the local PipeWire default sink.

    Pipes the synthesized WAV to `pw-cat -p` (= pipewire-pulse playback) and
    blocks until playback finishes. For long passages, prefer
    voicevox_synthesize() + separate playback so the tool doesn't tie up the
    MCP loop.

    Accepts the same AudioQuery override kwargs as voicevox_synthesize() —
    see that tool's docstring for the parameter list. Engine 0.14+ required
    for pause_length_scale / pause_length.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        wav = _synthesize_to_bytes(
            text,
            speaker_id,
            speed_scale=speed_scale,
            pitch_scale=pitch_scale,
            intonation_scale=intonation_scale,
            volume_scale=volume_scale,
            pre_phoneme_length=pre_phoneme_length,
            post_phoneme_length=post_phoneme_length,
            pause_length_scale=pause_length_scale,
            pause_length=pause_length,
        )
        tmp_path.write_bytes(wav)
        result = subprocess.run(
            [PW_CAT, "-p", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {
                "played": False,
                "error": result.stderr.strip() or "pw-cat failed",
            }
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"played": True}


def main() -> None:
    """Entry point for the `nekono-voicevox-mcp` console script."""
    mcp.run()


if __name__ == "__main__":
    main()
