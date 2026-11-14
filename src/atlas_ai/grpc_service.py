import grpc

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai import observability
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.feedback.pipeline import FeedbackPipeline
from atlas_ai.llm import AIFailure


def _request_id(context) -> str:
    """Go sends "x-request-id" (the job/turn ID) for cross-service correlation."""
    for key, value in context.invocation_metadata() or ():
        if key == "x-request-id":
            return str(value)[:100]
    return ""


class AtlasAIService(ai_pb2_grpc.AIServiceServicer):
    """Menghubungkan RPC Go ke pipeline AI."""

    def __init__(self):
        self.feedback_pipeline = FeedbackPipeline()
        self.consultation_pipeline = ConsultationPipeline()

    def _run(self, rpc, context, subject_id, call):
        # One structured log line per RPC (IDs, timings, tokens, status; no text).
        with observability.request(rpc, _request_id(context), subject_id) as record:
            try:
                response = call()
                record.status = "OK"
                return response
            except AIFailure as exc:
                record.status = exc.code.name
                context.abort(exc.code, str(exc))
            except ValueError as exc:
                record.status = grpc.StatusCode.INVALID_ARGUMENT.name
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except NotImplementedError as exc:
                record.status = grpc.StatusCode.UNIMPLEMENTED.name
                context.abort(grpc.StatusCode.UNIMPLEMENTED, str(exc))

    def ProcessFeedback(self, request, context):
        return self._run(
            "ProcessFeedback",
            context,
            request.feedback.id,
            lambda: self.feedback_pipeline.run(
                request,
                time_remaining=context.time_remaining(),
                is_active=context.is_active,
            ),
        )

    def ProcessConsultationTurn(self, request, context):
        return self._run(
            "ProcessConsultationTurn",
            context,
            request.session_id,
            lambda: self.consultation_pipeline.run(
                request,
                time_remaining=context.time_remaining(),
                is_active=context.is_active,
            ),
        )
