"""Offline acceptance fixtures and human review bundles, never a provider caller."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.assessment import ConsultationFailure
from atlas_ai.consultation.context import prepare_context
from atlas_ai.consultation.memory import memory_candidates_to_proto
from atlas_ai.consultation.prompts import (
    SOLUTION_PROMPT,
    SOLUTION_PROMPT_VERSION,
    SOLUTION_SYSTEM_PROMPT,
)
from atlas_ai.consultation.solution import (
    GeneratedSolutionTurn,
    solution_provider_schema,
    solution_to_proto,
)

BASELINE = {
    "commit": "d8328e618b15f8c5276a4ea07916860e75cfd10b",
    "prompt_version": "consultation-solution-v7",
    "solution_prompt_words": 675,
    "solution_prompt_bytes": 4891,
    "provider_schema_sha256": "2e79359e0584ab9e01f04fb90368ab7b540e24d027ba4e4e425d950de07d3805",
    "local_schema_sha256": "b0f9b46a99ac2d9f93a073bfaf810968bb388849a31591ca1277994b05ccaaac",
    "proto_descriptor_sha256": "1675910827873f8c8423fa7a8f9a443b15139575dcc0fcd86df4965d31e521ad",
}

SCENARIOS = [
    {
        "id": "bakery_filling",
        "category": "Bakery",
        "issue": "Isian coklat sedikit",
        "feedback": "Rotinya empuk, tapi isian coklatnya sedikit.",
        "excerpt": "isian coklatnya sedikit.",
        "second_evidence": "Isian coklatnya sedikit untuk ukuran rotinya.",
        "owner": "Kami menakar isian coklat secara manual.",
        "memory": "Produksi roti dilakukan di dapur toko.",
        "hypothesis": "Hipotesis: takaran isian coklat mungkin belum konsisten selama produksi.",
        "title": "Evaluasi takaran isian coklat",
        "action": "Evaluasi standar takaran isian coklat dan bandingkan hasilnya pada beberapa batch produksi.",
        "metric": "Catatan variasi takaran isian coklat",
        "irrelevant_action": "Evaluasi strategi jual beli valuta asing.",
        "forbidden_finding": "Mesin pengisi coklat mengalami kerusakan.",
    },
    {
        "id": "bakery_quality",
        "category": "Bakery",
        "issue": "Rasa roti tidak konsisten",
        "feedback": "Rasa rotinya tidak konsisten, kadang terlalu asin.",
        "excerpt": "Rasa rotinya tidak konsisten",
        "second_evidence": "Roti hari ini lebih asin dari biasanya.",
        "owner": "Pemeriksaan rasa roti dilakukan sebelum dipajang.",
        "memory": "Roti dibuat di dapur toko.",
        "hypothesis": "Hipotesis: proses pencampuran adonan mungkin belum konsisten.",
        "title": "Tinjau konsistensi rasa roti",
        "action": "Bandingkan rasa roti antarbatch melalui catatan pemeriksaan sederhana.",
        "metric": "Catatan konsistensi rasa roti",
        "irrelevant_action": "Ubah strategi perdagangan saham toko.",
        "forbidden_finding": "Pelanggan menyebut: Rasa rotinya konsisten.",
    },
    {
        "id": "cafe_queue",
        "category": "Kafe",
        "issue": "Antrean kasir lama",
        "feedback": "Kopinya enak. Antreannya lama sekali.",
        "excerpt": "Antreannya lama sekali.",
        "second_evidence": "Saya menunggu lama sebelum pesanan dicatat.",
        "owner": "Pencatatan pesanan dilakukan secara manual.",
        "memory": "Jumlah kasir 2 orang.",
        "hypothesis": "Hipotesis: pembagian tugas kasir mungkin belum jelas.",
        "title": "Evaluasi pembagian tugas kasir",
        "action": "Amati alur penerimaan pesanan dan catat durasi antrean kasir.",
        "metric": "Durasi antrean kasir",
        "irrelevant_action": "Evaluasi orbit satelit untuk kedai.",
        "forbidden_finding": "Mesin kasir rusak.",
    },
    {
        "id": "cafe_packaging",
        "category": "Kafe",
        "issue": "Kemasan takeaway rusak",
        "feedback": "Kemasan rusak saat dibawa pulang.",
        "excerpt": "Kemasan rusak saat dibawa pulang.",
        "second_evidence": "Tutup kemasan terbuka saat perjalanan pulang.",
        "owner": "Pesanan dibungkus sebelum diserahkan kepada pelanggan.",
        "memory": "Toko melayani pesanan untuk dibawa pulang.",
        "hypothesis": "Hipotesis: cara penutupan kemasan mungkin belum konsisten.",
        "title": "Uji penutupan kemasan",
        "action": "Uji penutupan kemasan dengan simulasi dibawa pulang.",
        "metric": "Catatan kondisi kemasan setelah simulasi",
        "irrelevant_action": "Lakukan investasi komoditas untuk memperbaiki kemasan.",
        "forbidden_finding": "Pemasok mengirim kemasan cacat.",
    },
    {
        "id": "msme_pricing",
        "category": "Kuliner",
        "issue": "Harga dibanding porsi",
        "feedback": "Harga produk terasa mahal untuk porsinya.",
        "excerpt": "Harga produk terasa mahal untuk porsinya.",
        "second_evidence": "Porsinya terasa kecil dibanding harganya.",
        "owner": "Daftar harga ditampilkan di meja pemesanan.",
        "memory": "Harga produk ditetapkan oleh pemilik toko.",
        "hypothesis": "Hipotesis: perbandingan harga dan porsi mungkin belum jelas bagi pelanggan.",
        "title": "Evaluasi persepsi harga dan porsi",
        "action": "Bandingkan persepsi harga dan porsi produk melalui catatan tanggapan pelanggan.",
        "metric": "Catatan tanggapan harga dan porsi produk",
        "irrelevant_action": "Evaluasi logo sebagai satu-satunya solusi keluhan harga.",
        "forbidden_finding": "Semua pelanggan menolak harga toko.",
    },
    {
        "id": "msme_orders",
        "category": "Kuliner",
        "issue": "Pesanan keliru",
        "feedback": "Pesanan saya keliru saat diterima.",
        "excerpt": "Pesanan saya keliru saat diterima.",
        "second_evidence": "Pesanan yang datang berbeda dari yang saya minta.",
        "owner": "Pencatatan pesanan dilakukan secara manual.",
        "memory": "Toko menerima pesanan langsung di kasir.",
        "hypothesis": "Hipotesis: pencatatan pesanan mungkin belum konsisten.",
        "title": "Tinjau pencatatan pesanan",
        "action": "Tinjau pencatatan pesanan dan uji konfirmasi ulang sebelum pesanan dibuat.",
        "metric": "Jumlah pesanan keliru dalam catatan pengamatan",
        "irrelevant_action": "Beli perangkat mahal tanpa memeriksa proses pencatatan.",
        "forbidden_finding": "Kasir mengabaikan pesanan pelanggan.",
    },
    {
        "id": "owner_constraints",
        "category": "Kafe",
        "issue": "Antrean dengan kendala sumber daya",
        "feedback": "Antrean kasir lama pada sore hari.",
        "excerpt": "Antrean kasir lama pada sore hari.",
        "second_evidence": "Saya lama menunggu sebelum membayar.",
        "owner": "Jam operasional 08.00 sampai 17.00. Anggaran perbaikan terbatas. Kami belum bisa menambah staf.",
        "memory": "Jumlah kasir 2 orang.",
        "hypothesis": "Hipotesis: alur pencatatan pesanan mungkin menghambat antrean kasir.",
        "title": "Catat durasi antrean kasir",
        "action": "Catat durasi antrean kasir secara manual tanpa pembelian alat baru.",
        "metric": "Durasi antrean kasir",
        "irrelevant_action": "Tambah staf dan beli mesin otomatis tanpa memeriksa anggaran.",
        "forbidden_finding": "Owner menyatakan: Anggaran perbaikan tidak terbatas.",
    },
]

RUBRIC = [
    {
        "id": "factual_accuracy_attribution",
        "criterion": "Every finding is entailed by the actual cited current source, correctly attributed; no invented conditions or sample-to-population claim.",
        "critical": True,
    },
    {
        "id": "hypothesis_uncertainty",
        "criterion": "Explanation remains explicitly tentative; citations establish symptoms/context rather than requiring proof of the proposed cause; no hidden factual premise.",
        "critical": True,
    },
    {
        "id": "recommendation_relevance",
        "criterion": "Primary action addresses the observed issue substantively; incidental keywords and irrelevant tangents do not supply relevance.",
    },
    {
        "id": "operational_usefulness",
        "criterion": "Action identifies what the owner can investigate or do and a feasible manual observation/measure; no guaranteed outcome.",
    },
    {
        "id": "constraint_awareness",
        "criterion": "Advice respects supplied staffing, budget, process and capacity limits; unknown resources remain unknown.",
    },
    {
        "id": "indonesian_clarity",
        "criterion": "Concise, approachable Indonesian with concrete operational meaning, coherent compound wording and no unnecessary jargon or repeated disclaimers.",
    },
    {
        "id": "reference_faithfulness",
        "criterion": "All IDs belong to the correct current namespace and actually support their content; exact quotes preserve negation; no AI/history laundering or invented memory.",
        "critical": True,
    },
    {
        "id": "generation_validation_success",
        "criterion": "Actual captured generation returns all five sections and passes local validation/protobuf mapping within the supplied turn deadline, without extra repair calls.",
    },
]


def uid(scenario, kind):
    return str(uuid5(NAMESPACE_URL, f"atlas-m7-1-synthetic/{scenario}/{kind}"))


def fixture(scenario):
    """Complete synthetic Suthe request and PREDEFINED output, not model output."""
    name = scenario["id"]
    oid, oid2, owner_id, memory_id = (
        uid(name, kind) for kind in ("opinion", "opinion2", "owner", "memory")
    )
    historical = ai_pb2.Solution(
        findings=[
            ai_pb2.Finding(statement="Mesin kasir rusak. Penjualan meningkat 80%.")
        ]
    )
    request = ai_pb2.ProcessConsultationTurnRequest(
        session_id=uid(name, "session"),
        request_id=uid(name, "request"),
        event=ai_pb2.CONSULTATION_EVENT_GENERATE_NOW,
        context=ai_pb2.ContextSnapshot(
            issue=ai_pb2.IssueSummary(id=uid(name, "issue"), title=scenario["issue"]),
            shop=ai_pb2.ShopProfile(
                id=uid(name, "shop"),
                name="Toko Sintetis",
                category=scenario["category"],
                scale="Mikro",
            ),
            stats=ai_pb2.IssueStats(
                unique_feedback_count=32,
                opinion_count=40,
                average_rating=3.8,
                negative_count=24,
                positive_count=10,
                neutral_count=6,
                severe_count=2,
                not_severe_count=38,
            ),
            evidence=[
                ai_pb2.EvidenceOpinion(
                    opinion_id=oid,
                    opinion_text=scenario["issue"],
                    evidence_text=scenario["excerpt"],
                    full_feedback=scenario["feedback"],
                    sentiment=ai_pb2.SENTIMENT_NEGATIVE,
                    severity=ai_pb2.SEVERITY_NOT_SEVERE,
                    rating=3,
                ),
                ai_pb2.EvidenceOpinion(
                    opinion_id=oid2,
                    opinion_text=scenario["issue"],
                    evidence_text=scenario["second_evidence"],
                    full_feedback=scenario["second_evidence"],
                    sentiment=ai_pb2.SENTIMENT_NEGATIVE,
                    severity=ai_pb2.SEVERITY_NOT_SEVERE,
                    rating=3,
                ),
            ],
            memories=[
                ai_pb2.BusinessMemory(id=memory_id, fact_text=scenario["memory"])
            ],
            prior_issue_solution=ai_pb2.PriorSolution(
                session_id=uid(name, "archived"), solution=historical
            ),
        ),
        messages=[
            ai_pb2.ConversationMessage(
                id=uid(name, "ai"),
                role=ai_pb2.MESSAGE_ROLE_ATLAS_AI,
                content="Bagaimana proses terkait masalah tersebut?",
                topic_key="process",
                solution=historical,
            ),
            ai_pb2.ConversationMessage(
                id=owner_id, role=ai_pb2.MESSAGE_ROLE_OWNER, content=scenario["owner"]
            ),
        ],
        previous_solution=historical,
        asked_topics=["process"],
        question_count=1,
    )
    request.context.period.start_time.FromJsonString("2026-10-01T00:00:00Z")
    request.context.period.end_time.FromJsonString("2026-10-31T23:59:59Z")
    refs = {
        "opinion_ids": [oid],
        "owner_message_ids": [owner_id],
        "memory_ids": [memory_id],
    }
    output = {
        "findings": [
            {
                "statement": "Pelanggan menyebut: " + scenario["excerpt"],
                "references": {"opinion_ids": [oid]},
            }
        ],
        "customer_evidence": [
            {"opinion_id": oid, "quote": scenario["excerpt"], "note": ""}
        ],
        "hypotheses": [
            {
                "statement": scenario["hypothesis"],
                "confidence": None,
                "references": refs,
            }
        ],
        "recommendations": [
            {
                "title": scenario["title"],
                "description": scenario["action"],
                "priority": "medium",
                "references": refs,
            }
        ],
        "success_indicators": [
            {
                "metric": scenario["metric"],
                "target": "Usulan: catat hasil pengamatan untuk membandingkan proses",
                "timeframe": "Usulan: evaluasi selama satu minggu",
            }
        ],
        "memory_candidates": [],
    }
    if name == "owner_constraints":
        output["memory_candidates"] = [
            {
                "fact_text": "Jam operasional 08.00 sampai 17.00.",
                "source_owner_message_ids": [owner_id],
            }
        ]
    return request, output


VARIANTS = (
    "valid",
    "fabricated_finding",
    "fabricated_quote",
    "wrong_namespace",
    "false_sql",
    "false_operational_number",
)


def variant_output(request, output, scenario, variant):
    result = copy.deepcopy(output)
    if variant == "fabricated_finding":
        result["findings"][0]["statement"] = scenario["forbidden_finding"]
        if scenario["forbidden_finding"].startswith("Owner menyatakan:"):
            result["findings"][0]["references"] = {
                "owner_message_ids": [request.messages[-1].id]
            }
        elif not scenario["forbidden_finding"].startswith("Pelanggan menyebut:"):
            result["findings"][0]["references"] = {
                "memory_ids": [request.context.memories[0].id]
            }
    elif variant == "fabricated_quote":
        result["customer_evidence"][0]["quote"] = (
            "Kutipan sintetis ini tidak ada di sumber pelanggan."
        )
    elif variant == "wrong_namespace":
        result["findings"][0]["references"] = {
            "opinion_ids": [request.context.memories[0].id]
        }
    elif variant == "false_sql":
        result["findings"][0]["statement"] = "Sebanyak 99 opini negatif."
    elif variant == "false_operational_number":
        result["findings"][0].update(
            statement="Jumlah kasir 9 orang.",
            references={"memory_ids": [request.context.memories[0].id]},
        )
    elif variant != "valid":
        raise ValueError("Unknown acceptance fixture")
    return result


def validate_output(request, output):
    try:
        context = prepare_context(request)
        typed = GeneratedSolutionTurn.model_validate(output)
        solution = solution_to_proto(typed.solution_output(), context)
        response = ai_pb2.ProcessConsultationTurnResponse(
            solution=solution,
            memory_candidates=memory_candidates_to_proto(
                typed.memory_candidates, context
            ),
            model_id="offline-predefined-not-Gemini",
            prompt_version=SOLUTION_PROMPT_VERSION,
        )
        assert (
            ai_pb2.ProcessConsultationTurnResponse.FromString(
                response.SerializeToString()
            )
            == response
        )
        return {
            "accepted": True,
            "response": MessageToDict(response, preserving_proto_field_name=True),
            "error": None,
        }
    except (ConsultationFailure, ValidationError) as exc:
        return {
            "accepted": False,
            "response": None,
            "error": str(exc)
            if isinstance(exc, ConsultationFailure)
            else "invalid_schema",
        }


def review_item(scenario):
    request, _ = fixture(scenario)
    data = prepare_context(request).model_dump(mode="json", exclude={"sources"})
    data["event_name"] = ai_pb2.ConsultationEvent.Name(request.event)
    messages = SOLUTION_PROMPT.format_prompt(
        context_json=json.dumps(data, ensure_ascii=False)
    ).to_messages()
    rendered = [{"role": m.type, "content": m.content} for m in messages]
    return {
        "id": scenario["id"],
        "request": MessageToDict(request, preserving_proto_field_name=True),
        "messages": rendered,
        "generation_record": {
            "request_id": request.request_id,
            "prompt_version": SOLUTION_PROMPT_VERSION,
            "rendered_prompt_sha256": hashlib.sha256(
                json.dumps(rendered, sort_keys=True).encode()
            ).hexdigest(),
            "model_id": None,
            "duration_seconds": None,
            "http_status": None,
            "grpc_status": None,
            "token_usage": None,
            "actual_provider_invocations": None,
        },
        "scores": {r["id"]: None for r in RUBRIC},
        "reviewer": None,
        "evidence_notes": None,
    }


def evaluate(captures=None):
    if captures is not None and not isinstance(captures, dict):
        raise ValueError("Captures must map scenario IDs to typed output objects")
    cases = []
    for scenario in SCENARIOS:
        request, output = fixture(scenario)
        checks = [
            {
                "variant": variant,
                "expected_accepted": variant == "valid",
                **validate_output(
                    request, variant_output(request, output, scenario, variant)
                ),
            }
            for variant in VARIANTS
        ]
        cases.append(
            {
                "id": scenario["id"],
                "customer_evidence": [
                    scenario["feedback"],
                    scenario["second_evidence"],
                ],
                "owner_context": scenario["owner"],
                "confirmed_memory": scenario["memory"],
                "factual_boundaries": [
                    "Two sampled opinions are not the full SQL population.",
                    "Historical AI output is not evidence.",
                    "Unknown resources and causes are not confirmed facts.",
                    "Never use as a finding: " + scenario["forbidden_finding"],
                ],
                "plausible_hypothesis": scenario["hypothesis"],
                "relevant_recommendation": scenario["action"],
                "irrelevant_recommendation": scenario["irrelevant_action"],
                "synthetic_request": MessageToDict(
                    request, preserving_proto_field_name=True
                ),
                "predefined_output_not_model_generated": output,
                "prompt_bytes": sum(
                    len(m["content"].encode())
                    for m in review_item(scenario)["messages"]
                ),
                "deterministic_checks": checks,
            }
        )
    capture_checks = []
    for name, output in (captures or {}).items():
        scenario = next((s for s in SCENARIOS if s["id"] == name), None)
        if scenario is None:
            raise ValueError("Capture has an unknown scenario ID")
        request, _ = fixture(scenario)
        capture_checks.append(
            {
                "id": name,
                "origin": "user-supplied capture; provider origin NOT verified by this script",
                "validation": validate_output(request, output),
                "quality_scores": {r["id"]: None for r in RUBRIC},
            }
        )

    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    return {
        "milestone": "M7.1",
        "prompt_version": SOLUTION_PROMPT_VERSION,
        "baseline": BASELINE,
        "prompt_words": len(SOLUTION_SYSTEM_PROMPT.split()),
        "prompt_bytes": len(SOLUTION_SYSTEM_PROMPT.encode()),
        "provider_schema_sha256": canonical(solution_provider_schema()),
        "local_schema_sha256": canonical(GeneratedSolutionTurn.model_json_schema()),
        "proto_descriptor_sha256": hashlib.sha256(
            ai_pb2.DESCRIPTOR.serialized_pb
        ).hexdigest(),
        "external_calls": 0,
        "evaluation_type": "PREDEFINED_OUTPUT_ACCEPTANCE_NOT_GENERATION_QUALITY",
        "deterministic_checks_passed": sum(
            c["accepted"] == c["expected_accepted"]
            for s in cases
            for c in s["deterministic_checks"]
        ),
        "deterministic_checks_total": len(cases) * len(VARIANTS),
        "rubric": RUBRIC,
        "rubric_scale": {
            "0": "substantive failure",
            "1": "partial or insufficient review evidence",
            "2": "fully meets criterion with cited output/source evidence",
        },
        "human_quality_gate": "Actual provider capture plus validation success; total >=14/16, no zero scores, all critical criteria=2. Each score requires reviewer evidence; fixture acceptance is not a quality score.",
        "live_generation_quality": "NOT VERIFIED",
        "actual_generation_scores": None,
        "cases": cases,
        "capture_validation_only": capture_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument(
        "--captures",
        type=Path,
        help="JSON object: scenario ID -> externally captured typed output; no provider call",
    )
    args = parser.parse_args()
    targets = [args.output] + ([args.review_bundle] if args.review_bundle else [])
    if any(p.exists() for p in targets) or len({p.resolve() for p in targets}) != len(
        targets
    ):
        parser.error(
            "Use distinct new output paths; existing evidence is never overwritten"
        )
    captures = (
        json.loads(args.captures.read_text(encoding="utf-8")) if args.captures else None
    )
    result = evaluate(captures)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.review_bundle:
        bundle = {
            "prompt_version": SOLUTION_PROMPT_VERSION,
            "provider_schema": solution_provider_schema(),
            "rubric": RUBRIC,
            "live_generation_quality": "NOT VERIFIED",
            "cases": [review_item(s) for s in SCENARIOS],
        }
        args.review_bundle.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "checks_passed": result["deterministic_checks_passed"],
                "checks_total": result["deterministic_checks_total"],
                "external_calls": 0,
                "live_generation_quality": "NOT VERIFIED",
            }
        )
    )
    return (
        0
        if result["deterministic_checks_passed"] == result["deterministic_checks_total"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
