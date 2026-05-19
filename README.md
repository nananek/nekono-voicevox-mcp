# nekono-voicevox-mcp

VOICEVOX engine HTTP API を MCP tool として expose する Claude 用 stdio server。
TTS 合成や PipeWire 経由の再生を Claude の自然言語で扱えるようにする。

engine 本体は内包せず、 別途起動された VOICEVOX engine (= ローカル or remote host) の
HTTP API (`/audio_query` + `/synthesis` 等) を叩く設計。

## tools (MVP)

- `voicevox_list_speakers()` — `/speakers` を name keyed dict で返す。 style_id を引くために最初に呼ぶ
- `voicevox_synthesize(text, speaker_id, output_path)` — `/audio_query` + `/synthesis` で wav を `output_path` に保存
- `voicevox_play(text, speaker_id)` — wav を temp file 化 → `pw-cat -p` で local PipeWire default sink で再生 (= 短い text 用、 長文は synthesize + 別途再生 推奨)

## config

engine URL は env で override 可能 (default は localhost):

```sh
export VOICEVOX_ENGINE_URL=http://127.0.0.1:50021   # default
# remote host 上の engine を使う場合は IP / FQDN を指定
```

## install

```sh
pipx install ~/repos/nekono-voicevox-mcp
# ~/.local/bin/nekono-voicevox-mcp が配置される
```

## Claude への登録

`~/.claude/settings.json` の `mcpServers` に追加:

```json
{
  "mcpServers": {
    "nekono-voicevox": {
      "command": "~/.local/bin/nekono-voicevox-mcp",
      "env": { "VOICEVOX_ENGINE_URL": "http://127.0.0.1:50021" }
    }
  }
}
```

Claude Code 再起動後、 任意 speaker / text で TTS を依頼して動作確認。

## 前提

- Linux + PipeWire 環境 (`/usr/bin/pw-cat` 必須、 = pipewire package)
- VOICEVOX engine が HTTP API を listen 中 (local or remote)
- Python 3.11+
