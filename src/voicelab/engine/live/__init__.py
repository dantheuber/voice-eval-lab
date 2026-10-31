"""Live calls on Pipecat: mic calls and real-time bot-to-bot calls. See docs/LIVE_ENGINE.md.

Nothing here is imported until a live call starts, so API paths that don't need Pipecat don't pay
for it. Keep it that way: import from these modules only inside functions in `engine/service.py`.
"""
