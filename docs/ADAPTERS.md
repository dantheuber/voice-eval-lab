# Swapping a model or provider

Every model sits behind one of five small interfaces, and a YAML profile picks the adapter for each role. Swapping a role means editing a profile, plus at most one small adapter. Engine, agent, and eval code never import an adapter.

The code is in `src/voicelab/providers/base.py` (interfaces and registry), `src/voicelab/providers/adapters/` (adapters), and `src/voicelab/core/profiles.py` (profile loading).

## Profile format

A profile is one YAML file in `config/profiles/`. All of them load at startup, and one that fails to parse is skipped with a logged error. Pick one with `./run.sh --profile <id>`, or on the settings screen at `/settings`.

```yaml
id: my-profile                  # what --profile and the dashboard call it
label: "Local: my stack"
description: One line shown in the run launcher.
live_capable: true              # false makes "auto" choose turn-based calls
untested: false                 # true puts an "untested" mark on every call that uses it
support_engine: cascade         # or realtime, which uses the realtime role for the bot
roles:
  stt_live: {type: http_transcriptions, base_url: "http://127.0.0.1:18101", name: my-stt}
  stt_eval: {type: whisper_cpp, base_url: "http://127.0.0.1:18104", name: whisper-large-v3-q5}
  tts_bot: {type: phonon_tts, base_url: "http://127.0.0.1:18103", name: my-bot, voice_id: Easton}
  tts_caller: {type: pocket_tts, base_url: "http://127.0.0.1:18106", name: my-caller}
  llm_support: {type: openai_compat, base_url: "http://127.0.0.1:18102/v1", name: my-llm, timeout_s: 120}
  llm_caller: {type: openai_compat, base_url: "http://127.0.0.1:18102/v1", name: my-llm-caller, timeout_s: 120}
  llm_judge: {type: openai_compat, base_url: "http://127.0.0.1:18105/v1", name: my-judge, timeout_s: 600}
services:                       # local servers voicelab starts and watches. Leave out for a cloud stack
  - name: llm
    port: 18102                 # a role is tied to a server by this port appearing in its URL
    startup_timeout_s: 180
    env: {LD_LIBRARY_PATH: .runtime/llama-vulkan/llama-b12017}
    command: [".runtime/llama-vulkan/llama-b12017/llama-server", "--port", "18102", "..."]
```

Under `roles`, `type` picks the adapter and every other key goes to its constructor.

| Role | Interface | Required | Used for |
| --- | --- | --- | --- |
| `stt_live` | `STT` | yes | What each bot hears during a call |
| `stt_eval` | `STT` | yes | The independent post-call transcript. Keep it a different model family from `stt_live`, because pronunciation checks compare the two |
| `tts_bot` | `TTS` | yes | The support bot's voice |
| `tts_caller` | `TTS` | yes | Persona voices |
| `llm_support` | `LLM` | yes | The support bot, also in scenario dry runs |
| `llm_caller` | `LLM` | yes | The caller bot, also in scenario dry runs |
| `llm_judge` | `LLM` | yes | Post-call judging and scenario drafts |
| `realtime` | `RealtimeVoice` | no | Speech-to-speech in place of the cascade, when `support_engine: realtime` |
| `audio_analyst` | `AudioAnalyst` | no | Advisory listening. Every model tested scored chance, so the local profiles leave it out |

`voicelab` starts the servers in `services` with the repo root as the working directory, one process group each, and logs to `data/logs/<name>.log`. Health is `GET /health` unless `health_path` says otherwise. A server already answering on the port is used and never touched.

## Swap one role without writing code

Either edit the YAML, or use the settings screen, which lists every adapter any profile configures for that role. The screen rewrites the active profile's YAML, so commit or discard that change on purpose. It refuses while a run is in progress.

To use any OpenAI-compatible server for the support LLM, change one line. This is an example shape, not a tested setup:

```yaml
llm_support: {type: openai_compat, base_url: "http://127.0.0.1:11434/v1", model: "your-model", name: my-llm, timeout_s: 120}
```

`openai_compat` also takes `api_key_env` for a hosted provider, and `structured_output` for a server that lacks `json_schema`.

Registered adapters today:

| Interface | Types |
| --- | --- |
| `STT` | `http_transcriptions`, `whisper_cpp`, `transcribe`, `fake` |
| `TTS` | `phonon_tts`, `pocket_tts`, `llama_tts`, `polly`, `fake` |
| `LLM` | `openai_compat`, `bedrock_converse`, `fake` |
| `RealtimeVoice` | `lfm3_realtime`, `nova_sonic` |
| `AudioAnalyst` | `llama_audio_analyst`, `fake` |

## The five interfaces

These are Python protocols in `providers/base.py`. Audio is always `AudioClip`, mono 16-bit PCM at a known rate.

```python
class STT(Protocol):
    name: str
    async def transcribe(self, audio, *, language="en", hints=None) -> Transcript: ...

class TTS(Protocol):
    name: str
    sample_rate: int
    def stream(self, text, *, voice_id, style=None) -> AsyncIterator[AudioClip]: ...
    async def synthesize(self, text, *, voice_id, style=None) -> AudioClip: ...
    async def voices(self) -> list[VoiceInfo]: ...

class LLM(Protocol):
    name: str
    async def chat(self, messages, *, tools=None, temperature=0.3, max_tokens=512, json_schema=None) -> LLMResponse: ...
    def stream_chat(self, messages, *, tools=None, temperature=0.3, max_tokens=512) -> AsyncIterator[LLMDelta]: ...

class RealtimeVoice(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int
    async def connect(self, *, instructions, tools, voice_id=None) -> RealtimeSession: ...

class AudioAnalyst(Protocol):
    name: str
    async def analyze(self, audio, prompt, *, json_schema=None) -> str: ...
```

Rules that matter:

- `Transcript.words` stays empty if the provider can't time words. Don't invent timings.
- Word `confidence` is `None` if the provider has no real figure. Parakeet reports 1.0 on every word, so its adapter passes `trust_confidence: false`.
- With `json_schema`, `LLM.chat` must return text that matches the schema. The judge depends on this.
- `hints` biases recognition. Eval code always passes `None`, because hints can hide a mispronunciation.
- `RealtimeSession.interrupt` gets the milliseconds actually heard. If your model can't drop unheard text from its context, document that in the adapter.
- `stream` is an async generator, and a barge-in cancels it. Clean up in `async with` or `finally`.

## Write an adapter

This TTS adapter talks to a made-up server with `POST /speak` that streams 16-bit mono PCM. I ran it against a mocked server, through `build()` and a profile. The server itself doesn't exist, so replace the request with your provider's.

```python
"""src/voicelab/providers/adapters/my_speaker.py"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from voicelab.core.audio import AudioClip
from voicelab.providers.adapters.http_client import PerLoopClient
from voicelab.providers.base import VoiceInfo, register


@register("tts", "my_speaker")
class MySpeakerTTS:
    def __init__(self, base_url: str, *, name: str = "my-speaker", sample_rate: int = 16000) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.sample_rate = sample_rate
        self._http = PerLoopClient(timeout=httpx.Timeout(60.0))

    async def stream(
        self, text: str, *, voice_id: str, style: str | None = None
    ) -> AsyncIterator[AudioClip]:
        body = {"text": text, "voice": voice_id}
        async with self._http.get().stream("POST", f"{self.base_url}/speak", json=body) as reply:
            reply.raise_for_status()
            async for chunk in reply.aiter_bytes(4096):
                if len(chunk) >= 2:
                    yield AudioClip(chunk[: len(chunk) // 2 * 2], self.sample_rate)

    async def synthesize(self, text: str, *, voice_id: str, style: str | None = None) -> AudioClip:
        chunks = [clip.pcm async for clip in self.stream(text, voice_id=voice_id)]
        return AudioClip(b"".join(chunks), self.sample_rate)

    async def voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="narrator", label="Narrator", gender="female")]
```

Then:

1. Add `my_speaker` to the import list in `src/voicelab/providers/adapters/__init__.py`, which is what registers it.
2. Use it in a profile: `tts_bot: {type: my_speaker, base_url: "http://127.0.0.1:18200", name: my-voice}`.
3. If voicelab should start the server, add it under `services` with port 18200.
4. Test it with a mocked server, as `tests/providers/` does for the others, using `respx`.

Adapters stay thin. Local models run as separate processes, so the app never imports torch. If a model needs Python packages, run it as a server in its own venv, the way `scripts/phonon_server.py` does.

Seeded personas name abstract voices such as `default_female_1`. Accept a `voice_map` option, as `polly` and `pocket_tts` do, to turn them into your provider's voices. Otherwise your server gets those literal names.

## Move to Bedrock

`config/profiles/bedrock.example.yaml` is a full AWS stack: Claude through Converse, Polly, Transcribe, and Nova Sonic.

**None of this has run against live AWS.** No AWS account exists on the machine this was written on. Requests and parsing follow AWS documentation and were checked against stubbed clients only, so expect to fix things on first contact. Every adapter carries `untested = True`, and calls made with them show an untested mark.

Steps:

1. Install the AWS packages. `./run.sh` keeps them, since it syncs with `--inexact`.

   ```bash
   uv sync --inexact --extra aws
   ```

2. Export credentials in the shell that starts `./run.sh`. Use `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` with its secret, plus `AWS_REGION`. The settings screen mentions `.env`, but nothing in the app loads that file, so export them instead.

3. In the AWS console, enable the models you plan to use in `us-east-1`, or change `region` in the profile. The example needs `global.anthropic.claude-haiku-5-5`, `global.anthropic.claude-opus-5-5`, and a Sonic model.

4. Decide on the Sonic id, which is the one open question in the file. The profile says `amazon.nova-2-5-sonic`, which was the requested id, but **it could not be confirmed in AWS documentation.** The documented ids are `amazon.nova-sonic-v1:0` and `amazon.nova-2-sonic-v1:0`. Check your console. The Sonic event protocol in the adapter comes from the documented models, and `amazon.nova-2-5-sonic` is assumed to match.

5. Start it and test the connection:

   ```bash
   ./run.sh --profile bedrock-example
   ```

   Open `/settings`. **Test connection** calls STS `GetCallerIdentity`, which proves the credentials and nothing about Bedrock.

6. Run one call, turn-based. The profile has `live_capable: false`, so `auto` chooses turn-based. Use one scenario, one persona, and one repetition, and read `data/logs/serve.log` afterward.

7. Move one role at a time if something fails. To start with Claude on a local stack, copy `local-default.yaml`, give it a new `id`, and replace `llm_support` with the block from the example. Local STT and TTS keep running.

What is untested or limited:

| Item | Status |
| --- | --- |
| All four adapters | Never run against live AWS |
| `amazon.nova-2-5-sonic` | Could not be confirmed in AWS documentation. Protocol assumed from the documented Sonic models |
| Nova Sonic as the bot | The example sets the `realtime` role but not `support_engine: realtime`, so calls use the Claude cascade. Add that line to use Sonic |
| Claude model ids | Never sent to AWS. Check them in your account |
| Transcribe and Sonic streaming | Use AWS's experimental `aws-sdk-*` Python clients over the CRT transport. AWS says not to run those in production. Check them first if a stream fails |
| `stt_eval` independence | Transcribe fills both STT roles, so eval is not independent of live STT as the design wants. Eval never passes hints, and no custom vocabulary is set |
| Eval speed | Transcribe streams at real time, so eval of a long call takes as long as the call. Raising `speed` is unverified |
| `temperature` on Claude | These models reject non-default values, so the profile sets `send_temperature: false` |
| Token limits | These models always think, and thinking counts against `maxTokens`, so the profile sets `min_max_tokens` as a floor |
| Polly voices | The persona-to-voice map in the example has never been heard |
