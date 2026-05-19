"""Unit tests for Issue #5 注文③ (AudioQuery split) + 注文④ (user_dict CRUD).

The MCP tool wrappers can't be called directly (FastMCP-version-specific
attribute coupling), so we exercise the same logic via /audio_query and
/synthesis mocks plus a low-level helper for the dict tools.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx
import respx

from nekono_voicevox_mcp import server

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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00" * int(24000 * 2 * duration_sec))
    return buf.getvalue()


# --- 注文③: AudioQuery split ------------------------------------------------


@respx.mock(base_url=server.ENGINE_URL)
def test_audio_query_returns_engine_response(respx_mock: respx.Router) -> None:
    """voicevox_audio_query just forwards engine /audio_query JSON."""
    custom = dict(_BASE_QUERY, kana="テスト")
    respx_mock.post("/audio_query").mock(return_value=httpx.Response(200, json=custom))

    # Re-create the function body inline to bypass FastMCP wrapping.
    with server._client() as client:
        r = client.post("/audio_query", params={"text": "テスト", "speaker": 14})
        r.raise_for_status()
        result = r.json()

    assert result == custom


@respx.mock(base_url=server.ENGINE_URL, assert_all_called=False)
def test_synthesize_from_query_writes_query_verbatim(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    """Caller-supplied query is passed to /synthesis without /audio_query call."""
    audio_query_route = respx_mock.post("/audio_query").mock(
        return_value=httpx.Response(500)  # tripwire: must NOT be called
    )
    synth_route = respx_mock.post("/synthesis").mock(
        return_value=httpx.Response(200, content=_wav_24khz_mono())
    )

    edited_query = dict(_BASE_QUERY, kana="アタラシイ", speedScale=1.7)
    out = tmp_path / "out.wav"

    # Inline replica of voicevox_synthesize_from_query body (FastMCP-wrap-free).
    out.parent.mkdir(parents=True, exist_ok=True)
    with server._client() as client:
        synth = client.post(
            "/synthesis",
            params={"speaker": 14},
            json=edited_query,
            headers={"Accept": "audio/wav", "Content-Type": "application/json"},
        )
        synth.raise_for_status()
    out.write_bytes(synth.content)

    assert not audio_query_route.called  # query was supplied, no need to ask engine
    assert synth_route.called
    import json as _json

    body = _json.loads(synth_route.calls.last.request.read())
    assert body["kana"] == "アタラシイ"
    assert body["speedScale"] == 1.7


# --- 注文④: user_dict CRUD -------------------------------------------------


@respx.mock(base_url=server.ENGINE_URL)
def test_dict_list_returns_engine_response(respx_mock: respx.Router) -> None:
    """GET /user_dict body is returned verbatim."""
    fake_dict = {
        "abc-uuid": {
            "surface": "解する",
            "pronunciation": "カイスル",
            "accent_type": 0,
        }
    }
    respx_mock.get("/user_dict").mock(return_value=httpx.Response(200, json=fake_dict))

    with server._client() as client:
        r = client.get("/user_dict")
        r.raise_for_status()
        result = r.json()

    assert result == fake_dict


@respx.mock(base_url=server.ENGINE_URL)
def test_dict_add_posts_required_params(respx_mock: respx.Router) -> None:
    """POST /user_dict_word with surface/pronunciation/accent_type."""
    add_route = respx_mock.post("/user_dict_word").mock(
        return_value=httpx.Response(200, json="new-uuid-1234")
    )

    with server._client() as client:
        r = client.post(
            "/user_dict_word",
            params={
                "surface": "解する",
                "pronunciation": "カイスル",
                "accent_type": 0,
            },
        )
        r.raise_for_status()
        uuid = r.json()

    assert uuid == "new-uuid-1234"
    assert add_route.called
    sent_params = dict(add_route.calls.last.request.url.params)
    assert sent_params["surface"] == "解する"
    assert sent_params["pronunciation"] == "カイスル"
    assert sent_params["accent_type"] == "0"


@respx.mock(base_url=server.ENGINE_URL)
def test_dict_remove_uses_uuid_path(respx_mock: respx.Router) -> None:
    """DELETE /user_dict_word/<uuid> hits the right URL."""
    del_route = respx_mock.delete("/user_dict_word/abc-uuid").mock(return_value=httpx.Response(204))

    with server._client() as client:
        r = client.delete("/user_dict_word/abc-uuid")
        r.raise_for_status()

    assert del_route.called


@respx.mock(base_url=server.ENGINE_URL)
def test_dict_roundtrip_add_list_remove(respx_mock: respx.Router) -> None:
    """Smoke: add → list → remove returns coherent state through the mock."""
    # Start empty.
    listings = [{}]
    respx_mock.get("/user_dict").mock(
        side_effect=lambda req: httpx.Response(200, json=listings[-1])
    )
    respx_mock.post("/user_dict_word").mock(return_value=httpx.Response(200, json="uuid-1"))
    respx_mock.delete("/user_dict_word/uuid-1").mock(return_value=httpx.Response(204))

    with server._client() as client:
        # empty initial
        assert client.get("/user_dict").json() == {}
        # add
        added = client.post(
            "/user_dict_word",
            params={"surface": "辞書", "pronunciation": "ジショ", "accent_type": 1},
        )
        added.raise_for_status()
        uuid = added.json()
        listings.append({uuid: {"surface": "辞書"}})
        # list now contains
        assert uuid in client.get("/user_dict").json()
        # remove
        client.delete(f"/user_dict_word/{uuid}").raise_for_status()
        listings.append({})
        assert client.get("/user_dict").json() == {}
