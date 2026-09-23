"""FastAPI application exposing the bracket-macro repair endpoint."""

from __future__ import annotations

from fastapi import FastAPI

from .repair import pairing, solve
from .schemas import Change, RepairRequest, RepairResponse

app = FastAPI(
    title="Assembly Macro Repair API",
    version="1.0.0",
    summary="Repairs transcribed bracket macros into balanced nested sequences.",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/repair", response_model=RepairResponse)
def repair(request: RepairRequest) -> RepairResponse:
    chars = [t.char for t in request.tokens]
    locked = [t.locked for t in request.tokens]

    result = solve(chars, locked)
    if result is None:
        return RepairResponse(
            status="NO_REPAIR",
            repaired=None,
            pairs=None,
            changes=None,
            cost=None,
        )

    cost, text = result
    changes = [
        Change(index=i, before=chars[i], after=text[i])
        for i in range(len(chars))
        if chars[i] != text[i]
    ]
    return RepairResponse(
        status="OK",
        repaired=text,
        pairs=pairing(text),
        changes=changes,
        cost=cost,
    )
