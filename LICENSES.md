# Licenses

Every model, voice, and code dependency the profiles use, read from each repository's metadata or license file on 2026-10-31. This is an engineer's reading to help a decision, not legal advice. Confirm it before shipping anything.

## What needs attention

| Item | Applies to | What it means |
| --- | --- | --- |
| LFM Open License v1.0, USD 10 million limit | `local-fast`, `local-realtime` | Commercial use is licensed only while you and everything under common control have under USD 10 million in annual revenue. Above that you need a license from Liquid AI. `local-default` has no LFM model |
| CC-BY-4.0 attribution | Pocket weights and VCTK voices in `local-default` and `local-realtime`. Parakeet in `local-fast` | Credit the source. Suggested text is at the end of this file |
| Pocket `cosette` and `jean` | Nothing, by design | Non-commercial. They are in the upstream repo, and the adapter refuses them |
| Pocket `alba` | Loaded at startup, never speaks | `pocket-tts serve` loads it as its default voice. Its terms are unclear. See below |
| Pocket prohibited uses | Pocket weights | The model card bans voice impersonation or cloning without consent, fraudulent calls, and presenting generated content as genuine recordings. Fine for test calls |
| PyAV wheel | All profiles, through `aiortc` | PyAV is BSD-3, but its wheel bundles `libx264` and `libx265`, which are GPL. This project sends audio only and never loads video codecs. Check this before redistributing |
| NVIDIA CUDA packages | The Phonon and Pocket venv | PyPI's torch pulls in 15 `nvidia-*` packages and `cuda-toolkit` under NVIDIA's proprietary license. They are 3 GB, and they never load on this AMD machine. CPU torch wheels would avoid them, which is untested |

## Models

Revisions are pinned in `scripts/setup/manifest.tsv`.

| Model | Repository | Used by | Runtime license | Weights license | Conditions |
| --- | --- | --- | --- | --- | --- |
| Qwen3-ASR 0.6B Q8_0 | `ggml-org/Qwen3-ASR-0.6B-GGUF` | default, realtime | llama.cpp, MIT | Apache-2.0 | The GGUF repo has no license tag. Apache-2.0 comes from upstream `Qwen/Qwen3-ASR-0.6B` |
| Qwen3.5-4B Q4_K_M | `unsloth/Qwen3.5-4B-GGUF` | default, realtime | llama.cpp, MIT | Apache-2.0 | |
| Qwen3.8-27B UD-Q4_K_M | `unsloth/Qwen3.8-27B-GGUF` | all three, as judge | llama.cpp, MIT | Apache-2.0 | |
| Whisper large-v3 q5_0 | `ggerganov/whisper.cpp` | all three, for eval STT | whisper.cpp, MIT | MIT | |
| Phonon 31m q8 | `gradium/phonon-31m-en` | all three | `ptts`, MIT or Apache-2.0 | Apache-2.0 | Don't confuse it with `gradium/phonon-40m-*`, which is CC-BY-NC-4.0 |
| Pocket TTS 18M | `kyutai/pocket-tts-without-voice-cloning` | default, realtime | `pocket-tts`, MIT | CC-BY-4.0 | Attribution and the prohibited uses above. The benchmark doc's table lists Apache-2.0 for this role, but the model card says CC-BY-4.0, so this file follows the card |
| Parakeet TDT 0.6B v3 Q8_0 | `nvidia/parakeet-tdt-0.6b-v3` | fast | NeMo-Speech.cpp, Apache-2.0 | CC-BY-4.0 | Attribution |
| LFM3-1.7B Q4_K_M | `LiquidAI/LFM3-1.7B-GGUF` | fast | llama.cpp, MIT | LFM Open License v1.0 | USD 10 million limit |
| LFM3-Audio-2B Q4_K_M | `LiquidAI/LFM3-Audio-2B-GGUF` | realtime | llama.cpp, MIT | LFM Open License v1.0 | USD 10 million limit |
| Silero VAD | Bundled in Pipecat | live calls in all three | Pipecat, BSD-2 | MIT | |
| Smart Turn v3.2 | Bundled in Pipecat as `smart-turn-v3.2-cpu.onnx`, from `pipecat-ai/smart-turn-v3` | live calls in all three | Pipecat, BSD-2 | BSD-2 | |

The rest of the LFM terms:

- The limit also covers derivative works, including fine-tunes.
- Internal research, testing, and evaluation need no commercial license, per Liquid's user guide. Using it inside a business can count as commercial even with no direct revenue.
- Qualified non-profits are exempt for non-commercial or research use.
- The license ends automatically on breach, and all copies must then be deleted.

Other models in `.runtime/models`, such as Gemma 4, Voxtral, Nemotron, and Kokoro, were research downloads. No profile uses them, and `scripts/setup.sh` doesn't fetch them. See the benchmark doc for those.

`bedrock.example.yaml` uses AWS services under your AWS agreement. Its models and voices are not covered here.

## Voices

I checked every voice in `seeds/` and `config/profiles/`. All of them are allowed.

Seeded personas name abstract voices, and the adapters map them in `src/voicelab/providers/adapters/local_speech.py`. No seed names a real voice or a provider.

| Persona | `voice_id` | Pocket preset, default and realtime | Pocket source | License | Phonon voice, fast |
| --- | --- | --- | --- | --- | --- |
| Chatty, polite | `default_female_1` | `mary` | VCTK p333 | CC-BY-4.0 | Juniper |
| Confused retiree | `default_female_2` | `fantine` | VCTK p244 | CC-BY-4.0 | Everly |
| Soft-spoken user | `default_female_3` | `jane` | VCTK p339 | CC-BY-4.0 | Hadley |
| Rushed interrupter | `default_female_4` | `eve` | VCTK p361 | CC-BY-4.0 | Harper |
| Skeptical bookkeeper | `default_female_5` | `vera` | VCTK p229 | CC-BY-4.0 | Delia |
| Impatient owner | `default_male_1` | `marius` | `voice-donations/Selfie.wav` | CC0 | Preston |
| Angry customer | `default_male_2` | `george` | VCTK p315 | CC-BY-4.0 | Russell |
| Sysadmin | `default_male_3` | `javert` | `voice-donations/Butter.wav` | CC0 | Miles |
| Noisy cafe student | `default_male_4` | `michael` | VCTK p360 | CC-BY-4.0 | Micah |

The bot's voice is Phonon `Easton` in all three profiles. All Phonon voices come from `gradium/phonon-voices` under Apache-2.0. The card says nothing about who the voices came from, and I did not look further.

The persona editor also offers Pocket's `anna`, `charles`, `paul`, `eponine`, and `azelma` (all VCTK, CC-BY-4.0), and on `local-fast` Phonon's `Trevor`, `Maeve`, `Hugh`, and `Edwin`.

Upstream Pocket presets, and how this project treats them:

| Preset | Source folder | Terms | Here |
| --- | --- | --- | --- |
| `marius`, `javert` | `voice-donations/` | CC0 | Allowed |
| `anna`, `vera`, `fantine`, `charles`, `paul`, `eponine`, `azelma`, `george`, `mary`, `jane`, `michael`, `eve` | `vctk/` | CC-BY-4.0 | Allowed, with attribution |
| `cosette` | `expresso/` | CC-BY-NC-4.0 | Never used |
| `jean` | `ears/` | Non-commercial | Never used |
| `alba` | `alba-mackenna/` | Unclear | Loaded, never speaks |
| `bill_boerst`, `peter_yearsley`, `stuart_bell`, `caro_davy` | `voice-zero/` | Unclear | Never used |
| `estelle` | `unmute-prod-website/` | Unclear | Never used |

The allow list is `POCKET_VOICES` in `local_speech.py`, and `PocketTTS._preset` raises an error for any name outside it, so a persona can't reach `cosette` or `jean`.

**`alba`.** `pocket-tts serve` loads its default voice at startup, and the profiles don't name one, so the `alba` embedding gets downloaded and loaded. No call uses it, because the adapter names a voice on every request. Passing `--default-voice marius` in the profiles' `pocket-tts` command removes it, which I tested with the `alba` file deleted. The change would also let `alba` leave the setup manifest.

Pocket voice cloning needs the gated `kyutai/pocket-tts` weights. This project doesn't use cloning.

## Code

### Model runtimes

| Component | Version | License | Notes |
| --- | --- | --- | --- |
| llama.cpp | b12017 | MIT | Official Vulkan archive |
| NeMo-Speech.cpp | 0.3.0 | Apache-2.0 | `local-fast` only. Bundled notices are in its `share/licenses` |
| whisper.cpp | v1.9.5 | MIT | Built locally |
| `ptts` | 0.6.0 | MIT or Apache-2.0 | Its SBOM lists 327 Rust crates, all permissive except two MPL-2.0: `colored` and `option-ext` |
| `pocket-tts` | 4.0.1 | MIT | The license text is in the package. Its metadata names none |
| PyTorch | 2.15.0 | BSD-3-Clause | Pulls the NVIDIA packages noted above |
| conda toolchain | see `scripts/setup/toolchain-explicit.txt` | Mixed | GCC is GPL with the runtime exception. `whisper-server` loads `libstdc++`, `libgcc_s`, and `libgomp` from it, and the Vulkan loader (Apache-2.0) |
| micromamba | 2.10.0 | BSD-3-Clause, MIT, OpenSSL | Setup only |

The venv at `.runtime/venvs/bench` on the original machine also holds benchmark packages. `kokoro-onnx` there pulls in `phonemizer` (GPL-3) and `espeakng-loader`. No profile imports them, and `scripts/setup.sh` doesn't install them.

### Python app

99 packages get installed from `uv.lock`. These are the direct dependencies, plus anything with stronger terms:

| Package | Version | License |
| --- | --- | --- |
| `pipecat-ai` | 1.15.0 | BSD-2-Clause |
| `fastapi` | 0.162.0 | MIT |
| `uvicorn` | 0.54.0 | BSD-3-Clause |
| `pydantic` | 2.14.2 | MIT |
| `numpy` | 2.5.4 | BSD-3-Clause, with MIT, 0BSD, Zlib, and CC0 parts |
| `httpx` | 0.28.1 | BSD-3-Clause |
| `websockets` | 17.4 | BSD-3-Clause |
| `pyyaml` | 6.0.3 | MIT |
| `python-multipart` | 0.0.32 | Apache-2.0 |
| `cryptography` | 50.0.2 | Apache-2.0 or BSD-3-Clause |
| `aiortc` | 1.15.0 | BSD-3-Clause |
| `onnxruntime` | 1.31.1 | MIT |
| `soundfile` | 0.13.1 | BSD-3-Clause. Its wheel bundles libsndfile, which is LGPL-2.1 |
| `soxr` | 1.0.0 | LGPL-2.1-or-later, flagged because it's copyleft for changes to the library itself |
| `num2words` | 0.5.14 | LGPL, flagged the same way. Comes in through Pipecat |
| `certifi` | 2026.10.23 | MPL-2.0 |
| `av` | 17.1.0 | BSD-3-Clause, with bundled GPL codecs as noted above |

`boto3`, `aiobotocore`, `awscrt`, and the `aws-sdk-*` packages are Apache-2.0. They come with the `aws` extra, and some with the dev group.

### Dashboard

`pnpm licenses list --prod` gives 82 packages: 77 MIT, 3 BSD-2-Clause (`@pipecat-ai/client-js`, `@pipecat-ai/small-webrtc-transport`, `@daily-co/daily-js`), 1 Apache-2.0 (`rxjs`), and 1 0BSD (`tslib`).

### This repo

The repo is private and has no license file. Pick one before sharing it.

## Attribution text

Use this wherever generated audio or the models get redistributed. Keep the Parakeet line only if you use `local-fast`.

> Caller voices use Pocket TTS by Kyutai, licensed under CC BY 4.0.
>
> Caller voices anna, vera, fantine, charles, paul, eponine, azelma, george, mary, jane, michael, and eve derive from the CSTR VCTK Corpus version 0.92 by Junichi Yamagishi, Christophe Veaux, and Kirsten MacDonald, University of Edinburgh, doi:10.7488/ds/2645, licensed under CC BY 4.0. Kyutai enhanced the clips and computed the embeddings.
>
> Speech recognition uses Parakeet TDT 0.6B v3 by NVIDIA, licensed under CC BY 4.0.
