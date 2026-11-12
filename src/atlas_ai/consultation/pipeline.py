import time
from collections.abc import Callable

import grpc

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import (
    ConsultationFailure,
    ContextAssessor,
    GeminiSettings,
    must_generate,
    validate_question,
)
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.prompts import PROMPT_VERSION, SOLUTION_PROMPT_VERSION
from atlas_ai.consultation.solution import SolutionGenerator


class ConsultationPipeline:
    """Pipeline Guided AI Consultation berbasis Gemini."""

    def __init__(self, assessor: ContextAssessor | None = None):
        self.assessor = assessor

    def run(
        self,
        request: ai_pb2.ProcessConsultationTurnRequest,
        *,
        time_remaining: float | None = None,
        is_active: Callable[[], bool] | None = None,
    ) -> ai_pb2.ProcessConsultationTurnResponse:
        started = time.monotonic()
        context = prepare_context(request)
        assessor = self.assessor or ContextAssessor(GeminiSettings.from_env())
        # One total turn budget, shared by assessment and generation.
        budget = assessor.settings.timeout_seconds
        if time_remaining is not None:
            budget = min(budget, time_remaining)

        def remaining() -> float:
            return budget - (time.monotonic() - started)

        def finish_response(response):
            # Local guards/mapping also consume the shared budget, including the
            # question path after the provider has already returned successfully.
            if remaining() <= 0:
                raise ConsultationFailure(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Consultation deadline exceeded before response",
                )
            if is_active is not None and not is_active():
                raise ConsultationFailure(
                    grpc.StatusCode.CANCELLED, "Consultation cancelled"
                )
            return response

        def solution_response() -> ai_pb2.ProcessConsultationTurnResponse:
            return finish_response(
                ai_pb2.ProcessConsultationTurnResponse(
                    solution=SolutionGenerator(assessor).generate(
                        context,
                        time_remaining=remaining(),
                        is_active=is_active,
                    ),
                    model_id=assessor.settings.model,
                    prompt_version=SOLUTION_PROMPT_VERSION,
                )
            )

        if must_generate(context):
            return solution_response()
        assessment = assessor.assess(
            context,
            time_remaining=remaining(),
            is_active=is_active,
        )
        if must_generate(context) or assessment.decision == "GENERATE_SOLUTION":
            return solution_response()
        return finish_response(
            ai_pb2.ProcessConsultationTurnResponse(
                next_question=validate_question(assessment.question, context),
                model_id=assessor.settings.model,
                prompt_version=PROMPT_VERSION,
            )
        )
