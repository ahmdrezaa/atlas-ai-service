"""Shared assertion for the P-B1 output policy.

A forbidden item may make the whole turn fail OR be dropped from the Solution;
either way none of its text may reach the returned protobuf.
"""

import json

from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from atlas_ai.consultation.assessment import ConsultationFailure


def rejected_or_absent(produce, *forbidden: str):
    """Run produce(); return the error if rejected, else assert texts are absent."""
    try:
        result = produce()
    except (ConsultationFailure, ValidationError) as exc:
        return exc
    dumped = json.dumps(MessageToDict(result), ensure_ascii=False)
    for text in forbidden:
        text = text.strip()
        if text:
            assert text not in dumped, f"forbidden text reached the Solution: {text!r}"
    return result
