"""nekono-voicevox-mcp: MCP server for VOICEVOX engine HTTP API.

Tools (MVP):
- voicevox_list_speakers() — `/speakers`, name keyed dict of style ids
- voicevox_synthesize(text, speaker_id, output_path) — wav file output
- voicevox_play(text, speaker_id) — pipe through pw-cat to local PipeWire

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

mcp = FastMCP("nekono-voicevox")


def _client() -> httpx.Client:
    """httpx client pinned to the configured engine base URL."""
    return httpx.Client(base_url=ENGINE_URL, timeout=httpx.Timeout(60.0, read=120.0))


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
) -> dict[str, Any]:
    """Synthesize text to a WAV file with the given speaker style.

    Engine pipeline: /audio_query (= phoneme + intonation inference) →
    /synthesis (= waveform generation); saves the WAV to output_path.

    Use voicevox_list_speakers() to find a valid speaker_id (= style_id).
    Returns the written path and byte count for readback.
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _client() as client:
        query = client.post("/audio_query", params={"text": text, "speaker": speaker_id})
        query.raise_for_status()
        synth = client.post(
            "/synthesis",
            params={"speaker": speaker_id},
            json=query.json(),
            headers={"Accept": "audio/wav", "Content-Type": "application/json"},
        )
        synth.raise_for_status()
    output.write_bytes(synth.content)
    return {"output_path": str(output), "size_bytes": output.stat().st_size}


@mcp.tool
def voicevox_play(text: str, speaker_id: int) -> dict[str, Any]:
    """Synthesize text and play it through the local PipeWire default sink.

    Pipes the synthesized WAV to `pw-cat -p` (= pipewire-pulse playback) and
    blocks until playback finishes. For long passages, prefer
    voicevox_synthesize() + separate playback so the tool doesn't tie up the
    MCP loop.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        voicevox_synthesize(text, speaker_id, str(tmp_path))
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
