import hmac
import logging
import os
from concurrent import futures

import grpc

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai.grpc_service import AtlasAIService

logger = logging.getLogger(__name__)


class BearerAuthInterceptor(grpc.ServerInterceptor):
    """Memvalidasi token pada setiap request gRPC."""

    def __init__(self, token: str):
        self.expected_token = f"Bearer {token}"

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        authorization = metadata.get("authorization", "")

        if hmac.compare_digest(authorization, self.expected_token):
            return continuation(handler_call_details)

        def reject(request, context):
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "Missing or invalid service token",
            )

        return grpc.unary_unary_rpc_method_handler(reject)


def serve():
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("GRPC_PORT", "50051"))
    token = os.getenv("GRPC_AUTH_TOKEN")

    if not token:
        raise RuntimeError("GRPC_AUTH_TOKEN must be configured")

    address = f"{host}:{port}"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        interceptors=[BearerAuthInterceptor(token)],
    )

    ai_pb2_grpc.add_AIServiceServicer_to_server(
        AtlasAIService(),
        server,
    )

    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Cannot bind to {address}")

    server.start()
    logger.info("ATLAS AI gRPC server running on %s", address)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5).wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
