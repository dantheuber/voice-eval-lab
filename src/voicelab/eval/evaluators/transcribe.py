"""Independent post-call transcript, the trusted record of what was said."""

from __future__ import annotations

from voicelab.core.audio import AudioClip
from voicelab.core.models import Speaker, Utterance, Word
from voicelab.eval.config import TranscribeConfig
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.transcript import estimate_words, speech_chunks
from voicelab.eval.vad import Region
from voicelab.providers.base import STT, Transcript


class TranscribeEvaluator:
    name = "transcribe"

    async def run(self, ctx: EvalContext, out: EvaluatorOutput) -> None:
        stt = ctx.deps.stt
        if stt is None:
            raise Skip("no STT provider configured")
        out.model = stt.name
        stored: dict[str, dict] = {}
        estimated: list[str] = []
        failures: list[str] = []
        for speaker in (Speaker.CALLER, Speaker.BOT):
            # One channel failing still leaves the other for the evaluators that follow.
            try:
                transcript, timing_estimated = await _transcribe(ctx, stt, speaker)
            except Exception as exc:
                failures.append(f"{speaker.value}: {exc}")
                continue
            ctx.transcripts[speaker] = transcript
            stored[speaker.value] = transcript.model_dump(mode="json")
            out.model = transcript.model or out.model
            if timing_estimated:
                estimated.append(speaker.value)
        out.metrics["transcript"] = stored
        out.metrics["word_counts"] = {k: len(v["words"]) for k, v in stored.items()}
        if estimated:
            out.metrics["estimated_word_timing"] = estimated
        if failures:
            raise RuntimeError("; ".join(failures))


async def _transcribe(ctx: EvalContext, stt: STT, speaker: Speaker) -> tuple[Transcript, bool]:
    regions = ctx.regions(speaker)
    if not regions:
        # Whisper-style models invent text when fed silence, and this channel has nothing to hear.
        return Transcript(text="", model=stt.name), False
    audio = ctx.audio(speaker)
    if getattr(stt, "speech_only", False):
        scripted = _scripted_turns(ctx, speaker)
        return await _transcribe_speech(stt, audio, regions, scripted, ctx.config.transcribe)
    transcript = await stt.transcribe(audio, language="en", hints=None)
    if transcript.words or not transcript.text.strip():
        return transcript, False
    words = estimate_words(transcript.text, regions, audio.duration_s)
    return transcript.model_copy(update={"words": words}), True


def _scripted_turns(ctx: EvalContext, speaker: Speaker) -> list[Region]:
    """When a side with a script was talking. A human's side has none, so nothing narrows it."""
    return [
        Region(event.start_s, event.end_s)
        for event in ctx.artifact.timeline
        if isinstance(event, Utterance) and event.speaker == speaker and event.intended_text
    ]


def _heard_in(chunk: Region, turns: list[Region]) -> bool:
    return any(turn.start_s < chunk.end_s and turn.end_s > chunk.start_s for turn in turns)


async def _transcribe_speech(
    stt: STT,
    audio: AudioClip,
    regions: list[Region],
    scripted: list[Region],
    cfg: TranscribeConfig,
) -> tuple[Transcript, bool]:
    """One request per stretch of speech, with word times moved back onto the call's clock.

    Speech-like line noise passes the energy VAD, and Whisper writes words for it. A stretch that
    overlaps none of a scripted side's turns can only be that noise, so it isn't sent.
    """
    words: list[Word] = []
    estimated = False
    model = stt.name
    chunks = speech_chunks(
        regions, gap_s=cfg.chunk_gap_s, pad_s=cfg.chunk_pad_s, end_s=audio.duration_s
    )
    for chunk in chunks:
        if scripted and not _heard_in(chunk, scripted):
            continue
        part = await stt.transcribe(
            audio.slice(chunk.start_s, chunk.end_s), language="en", hints=None
        )
        model = part.model or model
        if part.words:
            words.extend(_on_call_clock(part.words, chunk))
        elif part.text.strip():
            inside = [r for r in regions if chunk.start_s <= r.start_s < chunk.end_s]
            words.extend(estimate_words(part.text, inside, chunk.end_s))
            estimated = True
    text = " ".join(word.text for word in words)
    return Transcript(text=text, words=words, language="en", model=model), estimated


def _on_call_clock(words: list[Word], chunk: Region) -> list[Word]:
    return [
        word.model_copy(
            update={
                "start_s": min(chunk.end_s, chunk.start_s + word.start_s),
                "end_s": min(chunk.end_s, chunk.start_s + word.end_s),
            }
        )
        for word in words
    ]
