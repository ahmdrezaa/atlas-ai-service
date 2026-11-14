"""Controlled evaluation tests: Gemini and LangSmith are always mocked."""

import copy
import json
import socket
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.exceptions import OutputParserException
from langsmith.utils import (
    LangSmithAuthError,
    LangSmithConnectionError,
    LangSmithNotFoundError,
)

from scripts import consultation_m7_2_langsmith as m

# Historical harness: the M7.2 experiment is pinned (by hash) to the v8 solution
# prompt and its M7.1 fixture bundle, and the runner refuses any other prompt. The
# bakery MVP prompt (v9, 2026-11-14) supersedes it; its records stay unmodified.
pytestmark = pytest.mark.skipif(
    m.SOLUTION_PROMPT_VERSION != "consultation-solution-v8",
    reason="historical M7.2 harness bound to consultation-solution-v8",
)

PREDEFINED = {
    c["id"]: c["predefined_output_not_model_generated"]
    for c in json.loads(
        (m.ROOT / "evaluation/m7_1_consultation_quality.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
}


class FakeSmith:
    def __init__(self):
        self.dataset, self.project = None, None
        self.examples, self.runs, self.feedbacks = {}, {}, {}
        self.created_examples = self.created_runs = 0
        self.fail_upload = False

    def read_dataset(self, **kwargs):
        if self.dataset is None:
            raise LangSmithNotFoundError("test missing")
        return self.dataset

    def create_dataset(self, name, **kwargs):
        self.dataset = SimpleNamespace(
            id=uuid4(), metadata=kwargs["metadata"], url="https://smith.test/dataset"
        )
        return self.dataset

    def list_examples(self, **kwargs):
        return iter(self.examples.values())

    def create_example(self, **kwargs):
        self.created_examples += 1
        row = SimpleNamespace(
            id=kwargs["example_id"],
            inputs=copy.deepcopy(kwargs["inputs"]),
            outputs=copy.deepcopy(kwargs["reference_outputs"]),
            metadata=copy.deepcopy(kwargs["metadata"]),
        )
        self.examples[str(row.id)] = row
        return row

    def read_project(self, **kwargs):
        if self.project is None:
            raise LangSmithNotFoundError("test missing")
        return self.project

    def create_project(self, name, **kwargs):
        assert kwargs["upsert"] is False
        assert kwargs["num_examples"] == 7 and kwargs["num_repetitions"] == 1
        self.project = SimpleNamespace(
            id=uuid4(),
            reference_dataset_id=kwargs["reference_dataset_id"],
            extra={"metadata": kwargs["metadata"]},
            url="https://smith.test/experiment",
        )
        return self.project

    def read_run(self, rid, **kwargs):
        if rid not in self.runs:
            raise LangSmithNotFoundError("test missing")
        return self.runs[rid]

    def create_run(self, **kwargs):
        if self.fail_upload:
            raise LangSmithConnectionError(
                "private response includes fake-gemini-secret"
            )
        self.created_runs += 1
        assert kwargs["reference_example_id"] in self.examples
        self.runs[kwargs["id"]] = SimpleNamespace(
            inputs=copy.deepcopy(kwargs["inputs"]),
            outputs=copy.deepcopy(kwargs["outputs"]),
            reference_example_id=kwargs["reference_example_id"],
            session_id=self.project.id,
            extra=copy.deepcopy(kwargs["extra"]),
            url="https://smith.test/run",
        )

    def list_feedback(self, **kwargs):
        return iter(
            f
            for f in self.feedbacks.values()
            if f.run_id in kwargs["run_ids"] and f.key in kwargs["feedback_key"]
        )

    def create_feedback(self, **kwargs):
        assert kwargs["stop_after_attempt"] == 1
        assert (
            "project_id" not in kwargs
        )  # This selects project-level feedback in SDK 0.14.17.
        assert kwargs["session_id"] == self.project.id or str(
            kwargs["session_id"]
        ) == str(self.project.id)
        row = SimpleNamespace(
            id=kwargs["feedback_id"],
            run_id=kwargs["run_id"],
            key=kwargs["key"],
            score=kwargs["score"],
            comment=kwargs["comment"],
        )
        self.feedbacks[str(row.id)] = row
        return row


class ProviderFailure(Exception):
    def __init__(self, code):
        super().__init__("private provider body fake-gemini-secret")
        self.code = code


class FakeCapture:
    """Exercise the real production pipeline with a model-boundary fake only."""

    def __init__(self, on_attempt, mode, output):
        self.on_attempt, self.mode, self.output = (
            on_attempt,
            mode,
            copy.deepcopy(output),
        )
        self.http_events, self.responses, self.requests, self.invocations = (
            [],
            [],
            [],
            [],
        )
        self.payload, self.prompt_hash = None, None

    def model_factory(self, **kwargs):
        assert kwargs["max_retries"] == 0
        assert kwargs["model"] == "mock-configured-model"
        assert kwargs["timeout"] <= 30
        self.requests.append(
            {"max_output_tokens": kwargs["max_output_tokens"], "mock": True}
        )

        def structured(schema, **options):
            assert schema == m.solution_provider_schema()
            assert options == {"method": "json_schema"}

            def invoke(prompt, **unused):
                self.prompt_hash = m.digest(
                    [
                        {"role": msg.type, "content": msg.content}
                        for msg in prompt.to_messages()
                    ]
                )
                assert self.prompt_hash == self.expected_prompt_hash
                self.on_attempt()
                self.invocations.append({"mock": True})
                if isinstance(self.mode, int):
                    self.http_events.append({"http_status": self.mode})
                    raise ProviderFailure(self.mode)
                if self.mode == "timeout":
                    raise TimeoutError("private fake-gemini-secret")
                self.http_events.append({"http_status": 200})
                self.responses.append(
                    {
                        "model_version": "mock-returned-version",
                        "usage": {"total_token_count": 123},
                        "finish_reasons": ["STOP"],
                        "visible_text": "mock response",
                    }
                )
                if self.mode == "parse":
                    raise OutputParserException("private parse response")
                if self.mode == "pydantic":
                    self.output = {"findings": []}
                if self.mode == "grounding":
                    self.output["findings"][0]["statement"] = (
                        "Pelanggan menyebut: Mesin kasir rusak."
                    )
                if self.mode == "irrelevant":
                    self.output["recommendations"][0]["description"] = (
                        "Evaluasi orbit satelit."
                    )
                self.payload = self.output
                return self.output

            return SimpleNamespace(invoke=invoke)

        return SimpleNamespace(with_structured_output=structured)


@pytest.fixture(autouse=True)
def no_credentials_or_auto_tracing(monkeypatch):
    for name in m.ENV_NAMES | {"LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"}:
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-secret")
    monkeypatch.setenv("GEMINI_MODEL", "mock-configured-model")
    # M7.2 was recorded with a 30 s budget; the production default is now 45 s.
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-langsmith-secret")


@pytest.fixture
def registered(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = m.prepare(path)
    client = FakeSmith()
    m.ensure_dataset(ledger, client, create=True)
    m.write_json(path, ledger)
    return path, client


def invoker(mode="ok", calls=None):
    def invoke(request, settings, row, on_attempt):
        if calls is not None:
            calls.append(row["scenario_id"])
        return m.invoke_case(
            request,
            settings,
            row,
            on_attempt,
            capture_factory=lambda callback: FakeCapture(
                callback, mode, PREDEFINED[row["scenario_id"]]
            ),
        )

    return invoke


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluation", *map(str, args)])
    return m.main()


def test_exact_fixture_identities_and_no_golden_answers():
    manifest = m.load_manifest()
    assert tuple(e["scenario_id"] for e in manifest["examples"]) == m.SCENARIO_IDS
    assert manifest["aliases"]["cafe_queues"] == "cafe_queue"
    for example in manifest["examples"]:
        assert "predefined_output_not_model_generated" not in json.dumps(example)
        assert set(example["inputs"]) == {"scenario_id", "request"}
        assert len(example["reference_outputs"]["rubric"]) == 8
        assert (
            example["inputs"]["request"]["event"] == "CONSULTATION_EVENT_GENERATE_NOW"
        )


def test_offline_prepare_review_make_zero_network_calls(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network/client allowed offline")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(m, "make_client", forbidden)
    monkeypatch.setattr(m.Capture, "model_factory", forbidden)
    ledger, review = tmp_path / "ledger.json", tmp_path / "review.json"
    assert cli(monkeypatch, "prepare", "--ledger", ledger) == 0
    assert cli(monkeypatch, "review", "--ledger", ledger, "--output", review) == 0
    assert m.summary(m.read_ledger(ledger))["provider_attempts"] == 0
    assert all(
        v is None
        for c in json.loads(review.read_text())["cases"]
        for v in c["scores"].values()
    )


def test_existing_ledger_and_annotated_reviews_are_not_overwritten(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = m.prepare(path)
    with pytest.raises(m.EvaluationError, match="ALREADY_EXISTS"):
        m.prepare(path)
    review = tmp_path / "review.json"
    m.export_review(ledger, review)
    data = json.loads(review.read_text())
    data["cases"][0]["reviewer"] = "actual-test-reviewer"
    m.write_json(review, data)
    with pytest.raises(m.EvaluationError, match="ANNOTATED_REVIEW"):
        m.export_review(ledger, review)


def test_idempotent_dataset_and_stable_example_ids(registered):
    path, client = registered
    ledger = m.read_ledger(path)
    m.ensure_dataset(ledger, client, create=True)
    assert client.created_examples == 7
    assert len(set(ledger["dataset"]["example_ids"].values())) == 7
    assert ledger["dataset"]["status"] == "VERIFIED"


@pytest.mark.parametrize("change", ["metadata", "input", "output", "extra_example"])
def test_existing_dataset_content_mismatch_is_rejected_without_writes(
    registered, change
):
    path, client = registered
    if change == "metadata":
        client.dataset.metadata = {}
    elif change == "input":
        next(iter(client.examples.values())).inputs = {
            "real_customer": "must not be accepted"
        }
    elif change == "output":
        next(iter(client.examples.values())).outputs = {"golden_solution": "wrong"}
    else:
        client.examples["unexpected"] = SimpleNamespace(id="unexpected")
    with pytest.raises(m.EvaluationError, match="MISMATCH"):
        m.ensure_dataset(m.read_ledger(path), client, create=True)
    assert client.created_examples == 7


def test_partial_dataset_registration_recovers_without_duplicates(tmp_path):
    ledger = m.prepare(tmp_path / "ledger.json")
    client = FakeSmith()
    m.ensure_dataset(ledger, client, create=True)
    removed = list(client.examples)[-2:]
    for eid in removed:
        del client.examples[eid]
    m.ensure_dataset(ledger, client, create=True)
    assert set(removed) <= set(client.examples)
    assert client.created_examples == 9


def test_ledger_cannot_swap_approved_inputs_for_private_content(registered):
    path, _ = registered
    ledger = m.read_ledger(path)
    ledger["manifest"]["examples"][0]["inputs"]["request"]["context"]["shop"][
        "name"
    ] = "private changed content"
    m.write_json(path, ledger)
    with pytest.raises(m.EvaluationError, match="IDENTITY_MISMATCH"):
        m.read_ledger(path)


@pytest.mark.parametrize(
    "missing,category",
    [
        ("LANGSMITH_API_KEY", "MISSING_LANGSMITH_CONFIGURATION"),
        ("GEMINI_API_KEY", "MISSING_GEMINI_CONFIGURATION"),
        ("GEMINI_MODEL", "MISSING_GEMINI_CONFIGURATION"),
    ],
)
def test_missing_credentials_fail_before_client_or_model(
    registered, monkeypatch, missing, category
):
    configure(monkeypatch)
    monkeypatch.delenv(missing)
    path, client = registered
    with pytest.raises(m.EvaluationError, match=category):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=lambda *args: pytest.fail("No provider"),
        )
    assert m.read_ledger(path)["cases"][0]["provider_attempts"] == 0


def test_explicit_live_and_remote_flags_are_required(registered, monkeypatch):
    configure(monkeypatch)
    path, client = registered
    with pytest.raises(m.EvaluationError, match="EXPLICIT_LIVE"):
        m.run_live(path, authorized=False, phase="smoke", client_factory=lambda: client)
    with pytest.raises(m.EvaluationError, match="INVALID_LIVE_PHASE"):
        m.run_live(path, authorized=True, phase="bad", client_factory=lambda: client)
    assert cli(monkeypatch, "dataset", "--ledger", path) == 1
    assert cli(monkeypatch, "upload", "--ledger", path) == 1


def test_smoke_gate_and_single_attempt_are_durable(registered, monkeypatch):
    configure(monkeypatch)
    path, client = registered
    calls = []
    with pytest.raises(m.EvaluationError, match="SMOKE_GATE"):
        m.run_live(
            path,
            authorized=True,
            phase="remaining",
            client_factory=lambda: client,
            invoke=invoker(calls=calls),
        )
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(calls=calls),
    )
    assert calls == ["bakery_filling"]
    assert ledger["cases"][0]["fully_accepted"]
    assert ledger["cases"][0]["validation"] == {
        "pydantic": True,
        "structural": True,
        "grounding": True,
        "protobuf": True,
    }
    assert ledger["cases"][0]["returned_model_identifier"] == "mock-returned-version"
    assert ledger["cases"][0]["token_usage"]["total_token_count"] == 123
    m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(calls=calls),
    )
    assert calls == ["bakery_filling"]
    assert client.created_runs == 1


def test_total_cap_is_seven_with_no_duplicate_smoke_or_resume(registered, monkeypatch):
    configure(monkeypatch)
    path, client = registered
    calls = []
    for phase in ("smoke", "remaining", "remaining", "smoke"):
        ledger = m.run_live(
            path,
            authorized=True,
            phase=phase,
            client_factory=lambda: client,
            invoke=invoker(calls=calls),
        )
    assert calls == list(m.SCENARIO_IDS)
    assert m.summary(ledger)["provider_attempts"] == 7
    assert m.summary(ledger)["status"] == "LIVE CAPTURED — REVIEW PENDING"
    assert client.created_runs == 7
    assert len(client.feedbacks) == 42


@pytest.mark.parametrize(
    "mode,category",
    [
        (429, "RATE_LIMIT"),
        (401, "GEMINI_AUTHENTICATION_FAILURE"),
        (403, "GEMINI_AUTHENTICATION_FAILURE"),
        (504, "PROVIDER_HTTP_504"),
        (400, "GEMINI_REQUEST_REJECTED_AUTH_MODEL_OR_SCHEMA"),
        (503, "PROVIDER_HTTP_ERROR"),
        ("timeout", "CLIENT_TIMEOUT"),
    ],
)
def test_infrastructure_failure_stops_without_retry_and_is_sanitized(
    registered, monkeypatch, mode, category
):
    configure(monkeypatch)
    path, client = registered
    calls = []
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(mode, calls),
    )
    assert ledger["stop_reason"] == category
    assert calls == ["bakery_filling"]
    assert ledger["cases"][0]["provider_attempts"] == 1
    assert ledger["cases"][0]["attempt_status"] == "PROVIDER_ATTEMPTED_NO_RESPONSE"
    assert m.summary(ledger)["status"] == "LIVE PARTIAL / BLOCKED"
    assert "private provider body" not in path.read_text()
    assert "fake-gemini-secret" not in path.read_text()
    with pytest.raises(m.EvaluationError, match="EXPERIMENT_STOPPED"):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=invoker(calls=calls),
        )
    assert calls == ["bakery_filling"]


@pytest.mark.parametrize(
    "mode,category",
    [
        ("parse", "STRUCTURED_OUTPUT_PARSE_FAILURE"),
        ("pydantic", "PYDANTIC_VALIDATION_FAILURE"),
        ("grounding", "FACTUAL_EVIDENCE_REJECTION"),
    ],
)
def test_validation_failures_preserve_actual_model_capture_and_stage_separation(
    registered, monkeypatch, mode, category
):
    configure(monkeypatch)
    path, client = registered
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(mode),
    )
    row = ledger["cases"][0]
    assert row["provider_outcome"] == category
    assert row["model_response_received"] and not row["fully_accepted"]
    assert row["provider_responses"] and row["upload_status"] == "UPLOADED"
    assert row["provider_attempts"] == 1
    if mode == "grounding":
        assert row["validation"] == {
            "pydantic": True,
            "structural": True,
            "grounding": False,
            "protobuf": None,
        }
        assert (
            row["offline_revalidation"]["first_blocking_rejection"]["claim_id"]
            == "findings[0].statement"
        )


def test_upload_failure_recovers_locally_without_regeneration(registered, monkeypatch):
    configure(monkeypatch)
    path, client = registered
    client.fail_upload = True
    calls = []
    with pytest.raises(m.EvaluationError, match="UPLOAD_FAILED"):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=invoker(calls=calls),
        )
    row = m.read_ledger(path)["cases"][0]
    assert row["captured_payload"] == PREDEFINED["bakery_filling"]
    assert row["fully_accepted"] and row["upload_status"] == "PENDING"
    client.fail_upload = False
    monkeypatch.setattr(m, "make_client", lambda: client)
    assert cli(monkeypatch, "upload", "--ledger", path, "--remote-upload") == 0
    m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(calls=calls),
    )
    assert calls == ["bakery_filling"] and client.created_runs == 1


def test_rate_limit_stop_survives_a_simultaneous_upload_failure(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered
    m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(),
    )
    client.fail_upload = True
    with pytest.raises(m.EvaluationError, match="UPLOAD_FAILED"):
        m.run_live(
            path,
            authorized=True,
            phase="remaining",
            client_factory=lambda: client,
            invoke=invoker(429),
        )
    assert m.read_ledger(path)["stop_reason"] == "RATE_LIMIT"


def test_write_ahead_reservation_prevents_duplicate_after_process_interruption(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered

    def interrupted(request, settings, row, on_attempt):
        on_attempt()
        raise RuntimeError("interrupted")

    with pytest.raises(m.EvaluationError, match="LOCAL_CAPTURE_INTERRUPTED"):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=interrupted,
        )
    assert m.read_ledger(path)["cases"][0]["provider_attempts"] == 1
    with pytest.raises(m.EvaluationError, match="CAPTURE_NEEDS_UPLOAD"):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=lambda *args: pytest.fail("Must not regenerate"),
        )


def test_lock_blocks_concurrent_live_and_configuration_cannot_change(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered
    with m.locked(path), pytest.raises(m.EvaluationError, match="LEDGER_LOCKED"):
        m.run_live(path, authorized=True, phase="smoke", client_factory=lambda: client)
    m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(),
    )
    monkeypatch.setenv("GEMINI_MODEL", "silently-changed-model")
    with pytest.raises(m.EvaluationError, match="SETTINGS_CHANGED"):
        m.run_live(
            path, authorized=True, phase="remaining", client_factory=lambda: client
        )


def test_sdk_client_is_manual_nonbatched_no_retry_and_no_runtime_env_upload(
    monkeypatch,
):
    configure(monkeypatch)
    observed = {}
    monkeypatch.setattr(m, "Client", lambda **kwargs: observed.update(kwargs))
    m.make_client()
    assert observed["auto_batch_tracing"] is False
    assert observed["omit_traced_runtime_info"] is True
    assert observed["tracing_sampling_rate"] == 1
    assert observed["retry_config"].total == 0


def test_env_loader_ignores_auth_token_and_auto_tracing_flags(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.env"
    path.write_text(
        'GEMINI_MODEL=from-file\nLANGSMITH_API_KEY="test-file-key"\nGRPC_AUTH_TOKEN=must-not-load\nLANGSMITH_TRACING=true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_MODEL", "existing-model")
    m.load_environment(path)
    assert m.os.environ["GEMINI_MODEL"] == "existing-model"
    assert m.os.environ["LANGSMITH_API_KEY"] == "test-file-key"
    assert "LANGSMITH_TRACING" not in m.os.environ
    assert m.os.getenv("GRPC_AUTH_TOKEN") != "must-not-load"


def test_safe_exception_reporting_and_remote_auth_categories(
    tmp_path, monkeypatch, capsys
):
    configure(monkeypatch)
    path = tmp_path / "ledger.json"
    m.prepare(path)
    monkeypatch.setattr(
        m,
        "make_client",
        lambda: (_ for _ in ()).throw(
            LangSmithAuthError("private fake-langsmith-secret")
        ),
    )
    assert cli(monkeypatch, "dataset", "--ledger", path, "--remote-setup") == 1
    printed = capsys.readouterr().out
    assert "LANGSMITH_AUTHENTICATION_FAILURE" in printed
    assert "private" not in printed and "fake-langsmith-secret" not in printed
    assert "fake-langsmith-secret" not in path.read_text()


def scorecard(ledger):
    keys = [r["id"] for r in ledger["manifest"]["rubric"]]
    return {
        "reviewer": "test-human-reviewer",
        "scores": {k: 2 for k in keys},
        "reasons": {
            k: "Human checked the actual claim and cited synthetic source."
            for k in keys
        },
    }


def test_quality_gate_requires_actual_success_and_critical_dimensions(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(),
    )
    row, rubric = ledger["cases"][0], ledger["manifest"]["rubric"]
    review = scorecard(ledger)
    assert m.review_scorecard(row, review, rubric)["status"] == "QUALITY_PASS"
    review["scores"]["factual_accuracy_attribution"] = 1
    assert (
        m.review_scorecard(row, review, rubric)["status"] == "HUMAN_QUALITY_REJECTION"
    )
    review["scores"]["factual_accuracy_attribution"] = 0
    assert m.review_scorecard(row, review, rubric)["critical_failure_flags"] == [
        "factual_accuracy_attribution"
    ]


@pytest.mark.parametrize(
    "mutation",
    ["no_reviewer", "no_reason", "bool_score", "out_of_range", "failed_generation"],
)
def test_invalid_human_scores_and_failed_model_success_scores_are_rejected(
    registered, mutation
):
    path, _ = registered
    ledger = m.read_ledger(path)
    review = scorecard(ledger)
    row = {
        "provider_attempts": 1,
        "model_response_received": True,
        "fully_accepted": True,
    }
    if mutation == "no_reviewer":
        review["reviewer"] = None
    elif mutation == "no_reason":
        review["reasons"]["indonesian_clarity"] = None
    elif mutation == "bool_score":
        review["scores"]["indonesian_clarity"] = True
    elif mutation == "out_of_range":
        review["scores"]["indonesian_clarity"] = 3
    else:
        row["fully_accepted"] = False
    with pytest.raises(m.EvaluationError):
        m.review_scorecard(row, review, ledger["manifest"]["rubric"])


def test_unscored_review_is_pending_and_uploading_feedback_needs_no_model_rerun(
    registered, monkeypatch, tmp_path
):
    configure(monkeypatch)
    path, client = registered
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker(),
    )
    review_path = tmp_path / "review.json"
    m.export_review(ledger, review_path)
    data = json.loads(review_path.read_text())
    row = ledger["cases"][0]
    assert (
        m.review_scorecard(row, data["cases"][0], ledger["manifest"]["rubric"])[
            "status"
        ]
        == "HUMAN_REVIEW_PENDING"
    )
    data["cases"][0].update(scorecard(ledger))
    m.write_json(review_path, data)
    monkeypatch.setattr(m, "make_client", lambda: client)
    assert (
        cli(
            monkeypatch,
            "submit-review",
            "--ledger",
            path,
            "--review-file",
            review_path,
            "--remote-feedback",
        )
        == 0
    )
    assert (
        len([f for f in client.feedbacks.values() if f.key.startswith("human.")]) == 8
    )
    assert client.created_runs == 1
    assert (
        cli(
            monkeypatch,
            "submit-review",
            "--ledger",
            path,
            "--review-file",
            review_path,
            "--remote-feedback",
        )
        == 0
    )
    assert len(client.feedbacks) == 14


def test_review_export_cannot_overwrite_an_unrelated_historical_ledger(tmp_path):
    ledger = m.prepare(tmp_path / "ledger.json")
    historical = tmp_path / "historical.json"
    original = '{"milestone":"M7.1","cases":[]}'
    historical.write_text(original)
    with pytest.raises(m.EvaluationError, match="REVIEW_OUTPUT_IDENTITY_MISMATCH"):
        m.export_review(ledger, historical)
    assert historical.read_text() == original


def test_feedback_acknowledgement_loss_recovers_without_recreating_run_or_provider_call(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered
    original = client.create_feedback

    def lost_ack(**kwargs):
        original(**kwargs)
        raise LangSmithConnectionError("acknowledgement lost")

    monkeypatch.setattr(client, "create_feedback", lost_ack)
    calls = []
    with pytest.raises(m.EvaluationError, match="UPLOAD_FAILED"):
        m.run_live(
            path,
            authorized=True,
            phase="smoke",
            client_factory=lambda: client,
            invoke=invoker(calls=calls),
        )
    monkeypatch.setattr(client, "create_feedback", original)
    monkeypatch.setattr(m, "make_client", lambda: client)
    assert cli(monkeypatch, "upload", "--ledger", path, "--remote-upload") == 0
    assert calls == ["bakery_filling"]
    assert client.created_runs == 1 and len(client.feedbacks) == 6


def test_nonblocking_diagnostic_does_not_become_a_code_based_factual_failure(
    registered, monkeypatch
):
    configure(monkeypatch)
    path, client = registered
    ledger = m.run_live(
        path,
        authorized=True,
        phase="smoke",
        client_factory=lambda: client,
        invoke=invoker("irrelevant"),
    )
    row = ledger["cases"][0]
    assert row["fully_accepted"] and row["validation"]["grounding"]
    assert any(
        d["label"] not in m.ACCEPTED_LABELS
        for d in row["offline_revalidation"]["diagnostics"]
    )
    assert (
        next(f for f in client.feedbacks.values() if f.key == "atlas.grounding").score
        == 1
    )
    assert row["review"] is None


def test_installed_sdk_accepts_run_level_feedback_with_session_id_not_project_id(
    monkeypatch,
):
    # Use the actual SDK argument validation and serialization, mock HTTP only.
    from langsmith import Client
    from urllib3.util import Retry

    client = Client(
        api_key="test-only",
        auto_batch_tracing=False,
        retry_config=Retry(total=0),
        info={
            "version": "0.14.17",
            "batch_ingest_config": {"use_multipart_endpoint": False},
        },
    )
    requests = []
    monkeypatch.setattr(
        Client,
        "request_with_retries",
        lambda self, *args, **kwargs: requests.append((args, kwargs)),
    )
    run_id, session_id = uuid4(), uuid4()
    try:
        with pytest.raises(ValueError, match="project_id cannot"):
            client.create_feedback(
                run_id=run_id, project_id=session_id, key="test", score=0
            )
        feedback = client.create_feedback(
            run_id=run_id,
            session_id=session_id,
            key="test",
            score=0,
            stop_after_attempt=1,
        )
        assert feedback.run_id == run_id
        assert requests[0][0] == ("POST", "/feedback")
        body = json.loads(requests[0][1]["request_kwargs"]["data"])
        assert body["session_id"] == str(session_id)
        assert requests[0][1]["stop_after_attempt"] == 1
    finally:
        client.close()


@pytest.mark.parametrize(
    "name,value,category",
    [
        ("GEMINI_TIMEOUT_SECONDS", "9999", "INVALID_GEMINI_CONFIGURATION"),
        ("LANGSMITH_WORKSPACE_ID", "not-a-uuid", "INVALID_LANGSMITH_WORKSPACE"),
    ],
)
def test_invalid_configuration_is_not_misreported_as_a_provider_or_upload_failure(
    monkeypatch, name, value, category
):
    configure(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(m.EvaluationError, match=category):
        m.require_configuration(gemini=True)


def test_sdk_urls_allow_trace_coordinates_but_reject_credentials(monkeypatch):
    configure(monkeypatch)
    assert m.safe_url("https://smith.test/run?trace_id=synthetic")
    assert m.safe_url("https://smith.test/run?api_key=secret") is None
    assert m.safe_url("https://smith.test/run?other=fake-langsmith-secret") is None
    assert m.safe_url("https://user:password@smith.test/run") is None


def test_dataset_identity_cannot_silently_switch_workspaces_on_resume(registered):
    path, client = registered
    ledger = m.read_ledger(path)
    client.dataset.id = uuid4()
    with pytest.raises(m.EvaluationError, match="REMOTE_DATASET_IDENTITY_MISMATCH"):
        m.ensure_dataset(ledger, client)


def test_actual_capture_checks_prompt_before_any_sdk_request(monkeypatch):
    calls = []
    model = SimpleNamespace(
        client=SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kw: calls.append(kw))
        ),
        with_structured_output=lambda *a, **kw: SimpleNamespace(
            invoke=lambda *a, **kw: pytest.fail("No runnable invocation")
        ),
    )
    monkeypatch.setattr(m.Capture, "model_factory", lambda *a, **kw: model)
    capture = m.EvaluationCapture(lambda: pytest.fail("No provider attempt"))
    capture.expected_prompt_hash = "wrong"
    prompt = m.SOLUTION_PROMPT.format_prompt(context_json="{}")
    with pytest.raises(m.EvaluationError, match="ACTUAL_PROMPT_MISMATCH"):
        capture.model_factory().with_structured_output({}).invoke(prompt)
    assert not calls
