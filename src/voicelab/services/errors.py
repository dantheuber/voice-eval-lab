"""Failures a client can act on. The API turns these into the documented error body."""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    code = "internal"
    status = 500

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFound(ServiceError):
    code = "not_found"
    status = 404


class Invalid(ServiceError):
    code = "invalid"
    status = 400


class Conflict(ServiceError):
    code = "conflict"
    status = 409


class ProfileBusy(Conflict):
    """Every call this profile's servers can hold is on the line. `detail` says how long to wait."""

    code = "profile_busy"


class NotInstalled(Invalid):
    """A profile needs files setup hasn't put here. `detail` names them and the command to run."""

    code = "not_installed"


class ProfileFailed(ServiceError):
    """A profile's servers didn't come up, and the ones before them were put back."""

    code = "profile_failed"
    status = 503


class ModelNotReady(ServiceError):
    code = "model_not_ready"
    status = 503

    def __init__(
        self,
        role: str,
        model: str = "",
        *,
        elapsed_s: float | None = None,
        usual_s: float | None = None,
        reason: str = "",
    ) -> None:
        detail = {"role": role, "model": model, "elapsedS": elapsed_s, "usualS": usual_s}
        super().__init__(reason or f"{role} is not ready", detail)


class EngineUnavailable(ServiceError):
    """`voicelab.engine.service` isn't importable, so calls can't run."""

    status = 503
