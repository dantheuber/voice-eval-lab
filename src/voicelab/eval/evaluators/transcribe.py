"""Independent post-call transcript, the trusted record of what was said."""

from __future__ import annotations

from voicelab.core.models import Speaker
from voicelab.eval.context import EvalContext, EvaluatorOutput, Skip
from voicelab.eval.transcript import estimate_words
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
    transcript = await stt.transcribe(audio, language="en", hints=None)
    if transcript.words or not transcript.text.strip():
        return transcript, False
    words = estimate_words(transcript.text, regions, audio.duration_s)
    return transcript.model_copy(update={"words": words}), True
