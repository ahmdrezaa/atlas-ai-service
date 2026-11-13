"""Optional, extractive OWNER proposals. Nothing here confirms or stores memories.

Precision-first bounded equivalences for staffing, hours, budget and order recording.
Unknown language is omitted rather than inferred by a second model/verifier.
"""

import re
from collections import defaultdict
from typing import Annotated

from pydantic import BeforeValidator, Field, ValidationError

from atlas.ai.v1 import ai_pb2
from atlas_ai.consultation.grounding import _normal
from atlas_ai.consultation.models import ConsultationContext, ContextModel

MAX_CANDIDATES = 5


class ProposedMemoryCandidate(ContextModel):
    fact_text: str = Field(min_length=1, max_length=1000)
    source_owner_message_ids: list[str] = Field(min_length=1, max_length=20)


def _optional_proposals(value):
    # Malformed optional metadata must not invalidate an otherwise valid Solution.
    if not isinstance(value, list) or len(value) > MAX_CANDIDATES:
        return []
    result = []
    for item in value:
        try:
            result.append(ProposedMemoryCandidate.model_validate(item))
        except ValidationError:
            continue
    return result


MemoryProposals = Annotated[
    list[ProposedMemoryCandidate],
    BeforeValidator(_optional_proposals),
    Field(max_length=MAX_CANDIDATES),
]

_INJECTION_OR_PRIVATE = re.compile(
    r"\b(abaikan|ignore|instruksi|prompt|system|memory_candidates|source_owner_message_ids|"
    r"anggap|berpura|palsu|fiktif|karang|rahasia|password|kata sandi|token|api.?key|"
    r"nama pelanggan|pelanggan bernama|nomor telepon|nomor hp|nomor rekening|ktp|email)\b"
    r"|simpan.{0,30}(?:memori|memory)|masukkan.{0,30}(?:memori|memory)"
    r"|\b(menurut ai|kata ai|ai bilang|ai mengatakan|menurut pelanggan|kata pelanggan|"
    r"pelanggan mengatakan|pelanggan bilang|kutipan|mengutip|hanya contoh|bukan fakta|tidak benar)\b",
    re.IGNORECASE,
)
_NON_DURABLE = re.compile(
    r"\b(mungkin|diduga|sepertinya|perkiraan|rencana|akan|ingin|harap|tolong|buatkan|"
    r"usulan|jika|apabila|andai|tidak tahu|belum tahu|apakah|bagaimana|mengapa|"
    r"hari ini|besok|kemarin|minggu ini|bulan ini|sementara|kali ini|nanti|katanya|"
    r"menurut pelanggan|kata pelanggan|menurut ai|kata ai|ai bilang|feedback|ulasan|review|"
    r"pelanggan bilang|pelanggan mengatakan|opini|rating|statistik|komentar|keluhan|"
    r"penyebab|disebabkan|menyebabkan|karena|seandainya|pekan ini|pekan depan|minggu depan|"
    r"bulan depan|jangan|tambahkan|tuliskan|tulis|tampilkan|sebutkan|katakan|ekstrak|extract|masukkan|simpan|ingat|"
    r"tambah|kurangi|gunakan|pakai|buat|lakukan|amati|catat|evaluasi|ukur|"
    r"cari|bantu|berikan|ganti)\b",
    re.IGNORECASE,
)
_GENERIC = {
    "orang",
    "secara",
    "terdapat",
    "adalah",
    "pada",
    "ada",
    "jumlah",
    "untuk",
    "di",
    "hari",
    "berjumlah",
    "kita",
    "bukan",
    "punya",
    "hanya",
    "toko",
    "belum",
    "kedai",
    "sebanyak",
    "shift",
    "saya",
    "sangat",
    "tidak",
    "bekerja",
    "dilakukan",
    "kami",
    "memiliki",
    "yang",
}


def _atom(text: str) -> str:
    return text.strip().rstrip(".;!").strip()


def _clauses(text: str) -> list[str]:
    # Do not split decimal clock values; do not drop comma/exception qualifiers.
    return [
        _atom(part)
        for part in re.split(r"(?<!\d)[.;]|[.;](?!\d)|[\r\n]", text)
        if _atom(part)
    ]


def _signature(text: str) -> tuple[tuple[str, ...], str] | None:
    text = _atom(text)  # Sentence punctuation is not part of a budget amount.
    if "?" in text or _NON_DURABLE.search(text) or _INJECTION_OR_PRIVATE.search(text):
        return None
    normal = _normal(text)
    words = normal.split()
    negative = bool({"tidak", "bukan"} & set(words))

    def scope(exclude, *, retain_numbers=False):
        return tuple(
            sorted(
                set(words)
                - _GENERIC
                - set(exclude)
                - (set() if retain_numbers else {w for w in words if w.isdigit()})
            )
        )

    # Complete predicate + count + scope, not a number bag. A cashier is not a barista.
    roles = {
        word
        for word in words
        if word in {"barista", "kasir", "staf", "karyawan", "pegawai"}
    }
    numbers = [word for word in words if word.isdigit()]
    if (
        len(roles) == 1
        and len(numbers) == 1
        and not {"anggaran", "budget"} & set(words)
    ):
        role = next(iter(roles))
        if not (
            re.search(rf"\b\d+ (?:orang )?{role}\b", normal)
            or re.search(
                rf"\b{role}(?: (?:kami|kita|saya|ada|berjumlah|sebanyak|hanya|shift|pagi|sore|malam))* \d+\b",
                normal,
            )
            or f"jumlah {role} " in normal
        ):
            return None
        role = "staf" if role in {"pegawai", "karyawan"} else role
        return ("staff", role, *scope(roles)), ("not:" if negative else "") + numbers[0]

    clocks = re.findall(r"(?<!\d)(\d{1,2})[.:](\d{2})(?!\d)", text)
    if (
        clocks
        and len(clocks) <= 2
        and {"buka", "tutup", "operasional", "ramai", "sibuk"} & set(words)
    ):
        if any(int(h) > 23 or int(m) > 59 for h, m in clocks) or len(
            numbers
        ) != 2 * len(clocks):
            return None
        predicate = "ramai" if {"ramai", "sibuk"} & set(words) else "operasional"
        state = "at" if len(clocks) == 1 else "open"
        if len(clocks) == 1:
            if "tutup" in words:
                predicate = "tutup"
            elif "buka" in words:
                predicate = "buka"
            elif predicate != "ramai":
                return None  # One 'operating hours' clock has ambiguous meaning.
        elif "tutup" in words and "buka" not in words:
            state = "closed"
        elif "tutup" in words and words.index("tutup") < words.index("buka"):
            return None  # Do not invert a closed interval into operating hours.
        key = (
            "hours",
            predicate,
            *scope(
                {
                    "buka",
                    "tutup",
                    "operasional",
                    "ramai",
                    "sibuk",
                    "jam",
                    "pukul",
                    "sampai",
                    "hingga",
                }
            ),
        )
        return key, ("not:" if negative else "") + state + ":" + "-".join(
            f"{int(h):02}:{m}" for h, m in clocks
        )

    if "anggaran" in words or "budget" in words:
        if {"tetapi", "namun", "tapi", "meskipun"} & set(words):
            return None
        amount_pattern = r"(?:[0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)"
        amounts = list(
            re.finditer(
                rf"\brp\.?\s*(?P<prefix>{amount_pattern})(?![\w.,])|\b(?P<suffix>{amount_pattern})\s+rupiah\b",
                text,
                re.IGNORECASE,
            )
        )
        if len(amounts) == 1:
            amount = amounts[0]
            remainder = _normal(text[: amount.start()] + " " + text[amount.end() :])
            remainder_words = set(remainder.split())
            if any(word.isdigit() for word in remainder_words) or remainder_words & {
                "ribu",
                "juta",
                "miliar",
                "jt",
                "rb",
            }:
                return None
            qualifiers = tuple(
                sorted(remainder_words - _GENERIC - {"anggaran", "budget"})
            )
            value = (amount["prefix"] or amount["suffix"]).replace(".", "")
            return ("budget_amount_idr", *qualifiers), (
                "not:" if negative else ""
            ) + str(int(value))
        if numbers:
            return None
        exclude = {"anggaran", "budget", "tersedia", "terbatas", "tambahan"}
        # 'tambahan' is a real budget scope, retained separately.
        qualifier = ("tambahan",) if "tambahan" in words else ()
        key = ("budget", *qualifier, *scope(exclude))
        if (negative or "belum" in words) and {
            "tersedia",
            "ada",
            "punya",
            "memiliki",
        } & set(words):
            return key, "unavailable"
        if "terbatas" in words:
            return key, "not_limited" if negative else "limited"
        if {"tersedia", "ada", "punya", "memiliki"} & set(words) and not negative:
            return key, "available"

    if "pesanan" in words and (
        re.search(r"\bpesanan (?:kami )?(?:tidak )?dicatat\b", normal)
        or re.search(
            r"\b(?:kasir|kami|saya|staf|pegawai|barista|karyawan) (?:hanya )?mencatat pesanan\b",
            normal,
        )
        or "pencatatan pesanan" in normal
    ):
        modes = {word for word in words if word in {"manual", "otomatis", "digital"}}
        if len(modes) == 1:
            mode = next(iter(modes))
            key = (
                "order_recording",
                *scope(
                    {
                        "pesanan",
                        "manual",
                        "otomatis",
                        "digital",
                        "dicatat",
                        "mencatat",
                        "pencatatan",
                        "catat",
                        "proses",
                    },
                    retain_numbers=True,
                ),
            )
            return key, ("not:" if negative else "") + mode
    return None


def memory_candidates_to_proto(
    proposals: list[ProposedMemoryCandidate], context: ConsultationContext
) -> list[ai_pb2.MemoryCandidate]:
    """Filter metadata against OWNER-only full clauses; never modify the snapshot.

    All cited owners must support the SAME atomic fact; one must contain the exact
    candidate clause. Multiple owners may corroborate wording/number equivalences.
    Conflicting assertions anywhere in owner history suppress the whole fact slot.
    """
    if not proposals or not context.sources.owner_messages:
        return []
    owners = context.sources.owner_messages
    clauses = {
        source_id: [
            (atom, signature)
            for atom in _clauses(message.content)
            if (signature := _signature(atom)) is not None
        ]
        for source_id, message in owners.items()
        if not _INJECTION_OR_PRIVATE.search(message.content)
    }
    values = defaultdict(set)
    for atoms in clauses.values():
        for _, (key, value) in atoms:
            values[key].add(value)
    existing = {
        signature
        for memory in context.confirmed_memories
        for atom in _clauses(memory.fact_text)
        if (signature := _signature(atom)) is not None
    }
    order = {source_id: i for i, source_id in enumerate(owners)}
    result = {}
    for proposal in proposals:
        fact = proposal.fact_text.strip()
        signature = _signature(fact)
        ids = set(proposal.source_owner_message_ids)
        if (
            not signature
            or not ids
            or any(source_id not in owners for source_id in ids)
        ):
            continue
        if len(values[signature[0]]) != 1 or signature in existing:
            continue
        if not all(
            any(sig == signature for _, sig in clauses.get(source_id, []))
            for source_id in ids
        ):
            continue
        if not any(
            _atom(fact) == atom
            for source_id in ids
            for atom, _ in clauses.get(source_id, [])
        ):
            continue
        if signature in result:
            previous = result[signature]
            ids.update(previous.source_owner_message_ids)
            fact = previous.fact_text
        result[signature] = ai_pb2.MemoryCandidate(
            fact_text=fact, source_owner_message_ids=sorted(ids, key=order.__getitem__)
        )
    return list(result.values())[:MAX_CANDIDATES]
