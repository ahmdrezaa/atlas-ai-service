import hmac
import logging
import os
import signal
import threading
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import AIFailure

logger = logging.getLogger(__name__)

SERVICE_NAME = "atlas.ai.v1.AIService"
# Health reports liveness/readiness only, so it is reachable without the token
# (orchestrator probes). Every AIService method still requires the bearer token.
HEALTH_PREFIX = "/grpc.health.v1.Health/"


class BearerAuthInterceptor(grpc.ServerInterceptor):
    """Memvalidasi token pada setiap request gRPC."""

    def __init__(self, token: str):
        self.expected_token = f"Bearer {token}"

    def intercept_service(self, continuation, handler_call_details):
        if handler_call_details.method.startswith(HEALTH_PREFIX):
            return continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or [])
        authorization = metadata.get("authorization", "")

        if hmac.compare_digest(
            authorization.encode("utf-8"), self.expected_token.encode("utf-8")
        ):
            return continuation(handler_call_details)

        def reject(request, context):
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "Missing or invalid service token",
            )

        return grpc.unary_unary_rpc_method_handler(reject)


def create_server(
    token: str,
    address: str,
    *,
    service: AtlasAIService | None = None,
    max_workers: int = 8,
) -> tuple[grpc.Server, health.HealthServicer, AtlasAIService]:
    service = service or AtlasAIService()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=[BearerAuthInterceptor(token)],
    )
    ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.NOT_SERVING)
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Cannot bind to {address}")
    return server, health_servicer, service


def warm_up(service: AtlasAIService) -> bool:
    """Load the E5 model before serving. Consultation does not depend on it."""
    try:
        service.feedback_pipeline.warm_up()
    except AIFailure as exc:
        # ProcessFeedback keeps failing explicitly until fixed; Path B still serves.
        logger.error("ProcessFeedback not ready: %s", exc)
        return False
    logger.info("ProcessFeedback components loaded")
    return True


def serve():
    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("GRPC_PORT", "50051"))
    token = os.getenv("GRPC_AUTH_TOKEN")

    if not token:
        raise RuntimeError("GRPC_AUTH_TOKEN must be configured")

    address = f"{host}:{port}"
    server, health_servicer, service = create_server(token, address)
    server.start()
    warm_up(service)
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    logger.info("ATLAS AI gRPC server running on %s", address)

    stopped = threading.Event()

    def shutdown(signum, frame):
        logger.info("Shutting down gRPC server")
        health_servicer.enter_graceful_shutdown()
        server.stop(grace=10).wait()
        stopped.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    stopped.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
