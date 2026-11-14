"""Container health probe: ``python -m atlas_ai.healthcheck`` (exit 0 = SERVING).

Uses the standard gRPC health service on the local port (no token needed). With TLS
enabled, the server certificate is trusted as the root and the name is checked against
``GRPC_TLS_SERVER_NAME`` (the certificate's DNS name; default ``localhost``).
"""

import os
import sys

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def main() -> int:
    port = os.getenv("GRPC_PORT", "50051").strip() or "50051"
    target = f"127.0.0.1:{port}"
    cert_file = os.getenv("GRPC_TLS_CERT_FILE", "").strip()
    if cert_file:
        with open(cert_file, "rb") as handle:
            credentials = grpc.ssl_channel_credentials(root_certificates=handle.read())
        name = os.getenv("GRPC_TLS_SERVER_NAME", "").strip() or "localhost"
        channel = grpc.secure_channel(
            target, credentials, options=[("grpc.ssl_target_name_override", name)]
        )
    else:
        channel = grpc.insecure_channel(target)
    try:
        with channel:
            response = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(), timeout=3
            )
    except grpc.RpcError:
        return 1
    return 0 if response.status == health_pb2.HealthCheckResponse.SERVING else 1


if __name__ == "__main__":
    sys.exit(main())
