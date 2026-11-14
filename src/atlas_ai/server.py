import hmac
import logging
import os
import signal
import threading
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atlas.ai.v1 import ai_pb2_grpc
from atlas_ai.grpc_service import AtlasAIService
from atlas_ai.llm import AIFailure

logger = logging.getLogger(__name__)

SERVICE_NAME = "atlas.ai.v1.AIService"
# Health reports liveness/readiness only, so it is reachable without the token
# (orchestrator probes). Every AIService method still requires the bearer token.
HEALTH_PREFIX = "/grpc.health.v1.Health/"


class ServerSettings(BaseModel):
    """Transport limits. Defaults match Go's client (4 MiB messages, no TLS)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_workers: int = Field(default=8, ge=1, le=64)
    # In-flight RPC cap; beyond it gRPC answers RESOURCE_EXHAUSTED immediately
    # instead of queueing without bound (Go's feedback worker retries).
    max_concurrent_rpcs: int = Field(default=32, ge=1, le=1024)
    max_message_bytes: int = Field(
        default=4 * 1024 * 1024, ge=64 * 1024, le=64 * 1024 * 1024
    )
    tls_cert_file: str = ""
    tls_key_file: str = ""

    @model_validator(mode="after")
    def tls_pair(self):
        if bool(self.tls_cert_file) != bool(self.tls_key_file):
            raise ValueError("TLS needs both certificate and key")
        return self

    @property
    def tls(self) -> bool:
        return bool(self.tls_cert_file)

    @classmethod
    def from_env(cls) -> "ServerSettings":
        def env(name, default=""):
            return os.getenv(name, default).strip()

        try:
            return cls(
                max_workers=env("GRPC_MAX_WORKERS", "8"),
                max_concurrent_rpcs=env("GRPC_MAX_CONCURRENT_RPCS", "32"),
                max_message_bytes=env("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)),
                tls_cert_file=env("GRPC_TLS_CERT_FILE"),
                tls_key_file=env("GRPC_TLS_KEY_FILE"),
            )
        except ValidationError:
            # Never echo values (paths may be sensitive); names are enough.
            raise RuntimeError(
                "Invalid GRPC_MAX_WORKERS/GRPC_MAX_CONCURRENT_RPCS/"
                "GRPC_MAX_MESSAGE_BYTES/GRPC_TLS_CERT_FILE/GRPC_TLS_KEY_FILE"
            ) from None


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
    max_workers: int | None = None,
    settings: ServerSettings | None = None,
) -> tuple[grpc.Server, health.HealthServicer, AtlasAIService]:
    settings = settings or ServerSettings()
    if max_workers is not None:
        settings = settings.model_copy(update={"max_workers": max_workers})
    service = service or AtlasAIService()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.max_workers),
        interceptors=[BearerAuthInterceptor(token)],
        maximum_concurrent_rpcs=settings.max_concurrent_rpcs,
        options=[
            ("grpc.max_receive_message_length", settings.max_message_bytes),
            ("grpc.max_send_message_length", settings.max_message_bytes),
        ],
    )
    ai_pb2_grpc.add_AIServiceServicer_to_server(service, server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.NOT_SERVING)
    if settings.tls:
        with (
            open(settings.tls_key_file, "rb") as key,
            open(settings.tls_cert_file, "rb") as cert,
        ):
            credentials = grpc.ssl_server_credentials([(key.read(), cert.read())])
        bound = server.add_secure_port(address, credentials)
    else:
        bound = server.add_insecure_port(address)
    if bound == 0:
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
    settings = ServerSettings.from_env()
    server, health_servicer, service = create_server(token, address, settings=settings)
    server.start()
    warm_up(service)
    for name in ("", SERVICE_NAME):
        health_servicer.set(name, health_pb2.HealthCheckResponse.SERVING)
    logger.info(
        "ATLAS AI gRPC server running on %s tls=%s max_concurrent_rpcs=%d",
        address,
        settings.tls,
        settings.max_concurrent_rpcs,
    )

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
