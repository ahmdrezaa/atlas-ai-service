
from atlas.ai.v1 import ai_pb2


class FeedbackPipeline:
    """Pipeline NLP untuk menganalisis feedback pelanggan."""

    def run(
        self,
        request: ai_pb2.ProcessFeedbackRequest,
    ) -> ai_pb2.ProcessFeedbackResponse:

        # TODO (Ucan):
        # 1. Aspect-based opinion extraction
        # 2. Sentiment classification
        # 3. Severity classification
        # 4. Multilingual embeddings
        # 5. Issue clustering

        raise NotImplementedError(
            "Feedback NLP pipeline is not implemented yet."
        )
