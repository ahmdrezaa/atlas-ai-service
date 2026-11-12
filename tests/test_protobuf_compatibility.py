from pathlib import Path

import grpc_tools
from google.protobuf import descriptor_pb2
from grpc_tools import protoc

from atlas.ai.v1 import ai_pb2, ai_pb2_grpc


def test_generated_stubs_import_and_expose_official_service():
    assert ai_pb2_grpc.AIServiceStub
    service = ai_pb2.DESCRIPTOR.services_by_name["AIService"]
    assert service.full_name == "atlas.ai.v1.AIService"
    assert {method.name for method in service.methods} == {
        "ProcessFeedback",
        "ProcessConsultationTurn",
    }


def test_generated_descriptor_matches_official_proto(tmp_path):
    # Compile a descriptor into pytest temp storage; never overwrite stubs/proto.
    root = Path(__file__).resolve().parents[1]
    includes = Path(grpc_tools.__file__).parent / "_proto"
    output = tmp_path / "contract.pb"
    assert (
        protoc.main(
            [
                "protoc",
                f"-I{root / 'proto'}",
                f"-I{includes}",
                f"--descriptor_set_out={output}",
                "atlas/ai/v1/ai.proto",
            ]
        )
        == 0
    )
    compiled = descriptor_pb2.FileDescriptorSet.FromString(output.read_bytes()).file[0]
    generated = descriptor_pb2.FileDescriptorProto()
    ai_pb2.DESCRIPTOR.CopyToProto(generated)
    # protoc emits explicit JSON names while the checked-in descriptor omits them.
    for message in compiled.message_type:
        for field in message.field:
            field.ClearField("json_name")
    assert compiled == generated


def test_consultation_request_serialization(valid_request):
    assert (
        ai_pb2.ProcessConsultationTurnRequest.FromString(
            valid_request.SerializeToString(),
        )
        == valid_request
    )
