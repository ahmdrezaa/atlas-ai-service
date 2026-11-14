"""Fase 2B: transport limits, optional TLS and the container health probe."""

import shutil
import socket
import subprocess
from uuid import uuid4

import grpc
import pytest
from grpc_health.v1 import health_pb2

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc
from atlas_ai import healthcheck
from atlas_ai.server import ServerSettings, create_server

TOKEN = "test-only-token"
AUTH = (("authorization", f"Bearer {TOKEN}"),)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def serving(settings: ServerSettings | None = None):
    port = free_port()
    server, health, _ = create_server(
        TOKEN, f"127.0.0.1:{port}", settings=settings or ServerSettings()
    )
    health.set("", health_pb2.HealthCheckResponse.SERVING)
    server.start()
    return server, port


def test_defaults_match_go_client():
    settings = ServerSettings()
    assert settings.max_message_bytes == 4 * 1024 * 1024
    assert settings.max_concurrent_rpcs == 32 and settings.max_workers == 8
    assert settings.tls is False


@pytest.mark.parametrize(
    "name,value",
    [
        ("GRPC_MAX_CONCURRENT_RPCS", "0"),
        ("GRPC_MAX_WORKERS", "lots"),
        ("GRPC_MAX_MESSAGE_BYTES", "12"),
        ("GRPC_TLS_CERT_FILE", "/secret/path/cert.pem"),  # key missing
    ],
)
def test_invalid_server_env_fails_without_echoing_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError) as error:
        ServerSettings.from_env()
    assert value not in str(error.value)


def test_oversized_message_is_rejected(feedback_request_factory):
    server, port = serving(ServerSettings(max_message_bytes=64 * 1024))
    request = feedback_request_factory("Roti " * 30000)  # ~150 KB
    try:
        with (
            grpc.insecure_channel(f"127.0.0.1:{port}") as channel,
            pytest.raises(grpc.RpcError) as error,
        ):
            ai_pb2_grpc.AIServiceStub(channel).ProcessFeedback(
                request, timeout=10, metadata=AUTH
            )
    finally:
        server.stop(grace=0).wait(timeout=5)
    assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_healthcheck_probe_reports_serving_and_down(monkeypatch):
    server, port = serving()
    monkeypatch.setenv("GRPC_PORT", str(port))
    monkeypatch.delenv("GRPC_TLS_CERT_FILE", raising=False)
    try:
        assert healthcheck.main() == 0
    finally:
        server.stop(grace=0).wait(timeout=5)
    assert healthcheck.main() == 1


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_tls_server_requires_tls_client_and_keeps_auth(tmp_path, monkeypatch):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=atlas-ai",
            "-addext",
            "subjectAltName=DNS:atlas-ai,DNS:localhost",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check=True,
        capture_output=True,
    )
    server, port = serving(
        ServerSettings(tls_cert_file=str(cert), tls_key_file=str(key))
    )
    credentials = grpc.ssl_channel_credentials(root_certificates=cert.read_bytes())
    options = [("grpc.ssl_target_name_override", "atlas-ai")]
    bad = ai_pb2.ProcessFeedbackRequest(shop_id=str(uuid4()))
    try:
        with grpc.secure_channel(f"127.0.0.1:{port}", credentials, options) as channel:
            stub = ai_pb2_grpc.AIServiceStub(channel)
            with pytest.raises(grpc.RpcError) as no_token:
                stub.ProcessFeedback(bad, timeout=10)
            with pytest.raises(grpc.RpcError) as with_token:
                stub.ProcessFeedback(bad, timeout=10, metadata=AUTH)
        # A plaintext client cannot talk to the TLS port.
        with (
            grpc.insecure_channel(f"127.0.0.1:{port}") as channel,
            pytest.raises(grpc.RpcError) as plaintext,
        ):
            ai_pb2_grpc.AIServiceStub(channel).ProcessFeedback(
                bad, timeout=5, metadata=AUTH
            )
        monkeypatch.setenv("GRPC_PORT", str(port))
        monkeypatch.setenv("GRPC_TLS_CERT_FILE", str(cert))
        monkeypatch.setenv("GRPC_TLS_SERVER_NAME", "atlas-ai")
        assert healthcheck.main() == 0
    finally:
        server.stop(grace=0).wait(timeout=5)
    assert no_token.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert with_token.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert plaintext.value.code() == grpc.StatusCode.UNAVAILABLE
