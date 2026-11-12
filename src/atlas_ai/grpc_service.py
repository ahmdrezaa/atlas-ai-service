import grpc

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.pipeline import ConsultationPipeline
from atlas_ai.nlp.pipeline import FeedbackPipeline


class AtlasAIService(ai_pb2_grpc.AIServiceServicer):
    """Menghubungkan RPC Go ke pipeline AI."""

    def __init__(self):
        self.feedback_pipeline = FeedbackPipeline()
        self.consultation_pipeline = ConsultationPipeline()

    def ProcessFeedback(self, request, context):
        try:
            return self.feedback_pipeline.run(request)
        except NotImplementedError as exc:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                str(exc),
            )

    def ProcessConsultationTurn(self, request, context):
        try:
            return self.consultation_pipeline.run(
                request,
                time_remaining=context.time_remaining(),
                is_active=context.is_active,
            )

        except ConsultationFailure as exc:
            context.abort(exc.code, str(exc))

        except ValueError as exc:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                str(exc),
            )

        except NotImplementedError as exc:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                str(exc),
            )
