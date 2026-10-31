"""Importing this package registers every adapter with the provider registry."""

from voicelab.providers.adapters import (  # noqa: F401
    bedrock,
    fake,
    lfm3_realtime,
    local_speech,
    openai_compat,
)
