# Milestone 4C.1 — Solution Validation Alignment

Tanggal audit: 13 November 2026, Asia/Jakarta.

**Alignment dan regression selesai; tujuan Solution live + gRPC belum tercapai.**
Satu GENERATE_NOW sintetis pada override `gemini-3.6-flash` menerima HTTP 504.
Tidak ada output model untuk divalidasi. Tidak dilakukan retry, correction invocation,
atau gRPC live karena prasyarat Solution langsung belum berhasil.
Model produksi `.env` tetap `gemini-3.8-flash`; default budget Python tetap 30 detik.

## Investigasi C5 asli

Input dan output C5 dibaca dari `evaluation/functional_live_readiness.json` tanpa
mengubah teks, kutipan, UUID, atau hasil historisnya. Audit replay terpisah tersedia
di `evaluation/solution_alignment_c5_audit.json`, termasuk hash ledger/output asli.
Replay offline bukan keberhasilan generation live baru.

C5 menerima HTTP 200. Provider-compatible schema diterima dan `GeneratedSolution`
lolos Pydantic dengan lima bagian berjumlah 3/2/1/1/1. Kegagalan awal terjadi di
`solution._check_numbers`: seluruh kalimat yang mengandung angka harus menjadi
excerpt sumber persis atau diawali `Usulan:` untuk rekomendasi. Deskripsi C5 memuat
jam ramai 17.00 sampai 19.00 yang benar-benar ada di memory yang dirujuk, tetapi
kalimat tindakan lengkap tentu bukan kutipan memory. Ini false rejection angka
operasional, bukan bukti hallucination dan bukan kegagalan schema/Pydantic.

Prompt v3 juga menyamakan setiap rekomendasi numerik dengan angka usulan. Selain
masalah itu, replay semua claim menemukan enam abstention grounding berikut:

| Claim C5 | Alasan checker |
| --- | --- |
| recommendations[0].description | Awalan `Lakukan ...` di luar pola tindakan yang didukung |
| customer_evidence[0].note | Penjelasan waktu tunggu merupakan parafrasa yang belum terbukti oleh matching terbatas |
| customer_evidence[1].note | Penjelasan kecepatan pelayanan di luar bentuk excerpt/topic yang dapat dibuktikan |
| success_indicators[0].metric | Awalan `Catatan durasi ...` di luar pola definisi metric |
| success_indicators[0].target | `Terkumpulnya data ...` di luar pola target/action yang didukung |
| success_indicators[0].timeframe | `Selama ...` di luar pola action/metric yang didukung |

Abstention tersebut bukan penetapan bahwa seluruh kalimat salah. Tidak ditambahkan
whitelist agar C5 lolos. Setelah alignment, C5 asli lolos pemeriksaan struktural
angka, tetapi tetap gagal grounding pada enam claim itu. Finding statistik C5 juga
memiliki references kosong yang bertentangan dengan validator Go saat ini.

## Kebijakan angka dan provenance

`grounding-rules-v2` menambahkan satu helper bersama untuk structural checks dan
grounding tindakan. Helper hanya mengenali proposisi operasional terbatas:

- Jam ramai/sibuk/buka/tutup/operasional dengan jam dan rentang jam.
- Jumlah barista/staf/kasir/karyawan dalam orang.
- Staffing shift pagi/sore/malam dalam orang.

Predicate, satuan, urutan dan nilai harus cocok dengan OWNER message atau confirmed
memory yang benar-benar dirujuk. `pukul` boleh hadir atau tidak; ekuivalensi angka
kata kecil mengikuti normalisasi grounding yang sudah tersedia. Ini bukan matching
sekumpulan angka: jam ramai tidak membuktikan jam buka ataupun jumlah staf. Sumber
negatif, rencana, belum diketahui, bertentangan, atau mengandung instruksi berbahaya
tidak membuktikan fakta operasional afirmatif. Unknown wording tetap ditolak.

Angka operasional yang terbukti boleh muncul dalam tindakan tanpa `Usulan:`.
Angka rencana/target baru tetap eksplisit sebagai usulan, termasuk target staffing
yang berbeda dari jumlah staf sekarang. Attributed exact excerpts tetap tersedia
untuk laporan owner, termasuk penyangkalan; grounding tetap memeriksa provenance.
Normalized `opinion_text` tidak dipakai sebagai bukti angka sumber pelanggan.

Nilai SQL tetap wajib memakai marker bernama yang cocok dengan `IssueStats`.
Pemeriksaan makna metric, denominator, generalisasi sampel dan klaim tambahan tetap
aktif. Source owner/memory tidak boleh mengganti statistik issue melalui prose
numerik tanpa marker, termasuk angka Indonesia 0–99 yang dieja sebagai kata.
Label confidence/priority tetap kualitatif, bukan probabilitas.
Target/timeframe tidak boleh disamarkan sebagai hasil historis yang sudah tercapai.

## Kutipan dan schema provider

PRD §5.4 dan validator 4A membolehkan **substring verbatim** dari `evidence_text`
atau `full_feedback` milik `opinion_id` yang dirujuk. Proto tidak mensyaratkan seluruh
feedback. Kebijakan itu dipertahankan; tidak ada rekonstruksi, normalisasi quote,
penggabungan potongan, atau penggantian UUID.

Pemeriksaan tambahan menolak potongan di tengah kata dan negasi yang dipotong tepat
sebelum excerpt. Jika excerpt ada dalam full feedback, konteks feedback dipakai agar
evidence_text yang terpotong tidak dapat menghapus negasi. Ini bounded context guard,
bukan jaminan seluruh nuansa, koreksi lintas kalimat, atau bahasa pelanggan dipahami.

Schema provider ringkas dari 4C dipertahankan. Grammar provider menghilangkan
annotations/defaults dan bounds yang memicu masalah kompatibilitas, tetapi setiap
hasil tetap masuk `GeneratedSolution.model_validate` sebelum validation/mapping.
Tests memastikan lima array wajib nonempty, batas jumlah rekomendasi/panjang teks,
unknown fields, tipe reference ID, dan label numerik tetap diperiksa secara lokal.

## Alignment dengan validator Go

Source [consultation_validate.go](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/service/consultation_validate.go)
mewajibkan setidaknya satu current reference pada setiap finding/hypothesis/action.
Mapping final sekarang menolak references kosong meski marker statistiknya benar.
Tidak ditambahkan sitasi palsu untuk memenuhi syarat itu. Bukti nilai SQL tetap
IssueStats, bukan keberadaan opinion ID yang kebetulan tercantum.

Prompt `consultation-solution-v4` membedakan angka bersumber/rencana, mempertahankan
kutipan substring, dan meminta bentuk singkat yang dapat diperiksa checker. Contoh
format prompt bukan data untuk disalin. Tidak diubah question prompt/guard,
`ai.proto`, generated stubs, autentikasi, NLP, atau state lifecycle Go.

## Hasil live dan deadline

Ledger: `evaluation/solution_alignment_live.json`. Hanya satu generateContent dikirim.

| ID | Jalur | Model override | Durasi total / provider | Hasil |
| --- | --- | --- | --- | --- |
| A1 | GENERATE_NOW langsung | gemini-3.6-flash | 29,466 / 29,375 detik | HTTP 504; DEADLINE_EXCEEDED |
| — | Python gRPC live | — | Tidak dijalankan | SKIPPED: tidak ada Solution langsung yang valid |

A1 memakai v1beta, thinking LOW, output budget 2.048 token, actual prompt text 9.745
byte, dan schema 2.409 byte. Effective SDK timeout 29,749 detik dari shared budget
30 detik dan margin 250 ms. Tidak ada response body model atau token usage yang
dilaporkan. Structural, grounding dan protobuf mapping **NOT_REACHED**. Tidak ada
klaim truncation/parsing failure, rate limit, local httpx TimeoutException, ataupun
token budget sebagai penyebab 504. Token usage tidak tersedia bukan bukti biaya nol.

Image aktual: Python 3.11, google-genai 3.2.0, langchain-google-genai 4.5.0, grpcio
1.85.1, protobuf 7.36.2, Pydantic 2.14.3. Source SDK mengubah timeout milidetik menjadi
detik untuk HTTP client, sekaligus memasang `X-Server-Timeout=ceil(timeout_seconds)`.
Untuk A1 nilai turunan itu 30; ini inferensi dari source SDK, bukan raw-header log.
Jadi HTTP 504 dekat batas tersebut dapat berkaitan dengan server deadline yang
diminta client; penyebab provider belum menyelesaikan generation tidak terbukti.

Config/client Go pada default branch dibaca ulang dan blob-nya sama dengan commit
yang dipin dalam audit 4C: timeout AI 60 detik, HTTP write 90 detik, overhead luar
panggilan AI 15 detik. Parent context yang lebih pendek tetap menang. Source:
[config](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/config/config.go#L155),
[client](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/aiclient/client.go#L53).

Override 55/60 hanya menyisakan 5 detik nominal dan **tidak dinyatakan aman**.
Pengujian ini memakai 30/60 dengan headroom nominal 30 detik untuk request mapping,
queue, transport dan return. Syarat kompatibilitas tetap: seluruh biaya tersebut
ditambah waktu Python harus muat dalam sisa deadline caller. Sisa deadline saat
server masuk perlu diukur pada round trip nyata; karena gRPC live dilewati, overhead
transport belum terukur. Runner sudah mencatat entry deadline/pipeline/RPC overhead
bila pengujian berikutnya mencapai gRPC. Environment deployed Go/proxy, latency
lintas mesin, dan Go backend end-to-end belum diverifikasi. Tidak ada kenaikan
timeout produksi atau perubahan lifecycle Go.

## Regression dan file berubah

Final pytest melalui Docker Python 3.11, tanpa jaringan: **702 passed in 3.57s**.
Regression 50 kasus memakai dataset dan label lama tanpa perubahan: 0/23 false
acceptance, 2/27 false rejection, sama dengan hasil 4B. Laporan baru:
`evaluation/grounding_alignment_rules_report.json`. Label sintetis tersebut belum
diadjudikasi independen; hasilnya bukan ukuran semantic accuracy model produksi.
Baseline structural dalam evaluator menguji claim-level checks, bukan seluruh
kelayakan wire response Go.

Ruff lint dan format checks untuk consultation, tests dan scripts lulus. Pemeriksaan
lint seluruh `src tests scripts` masih menemukan 8 issue lama pada generated stubs
dan `server.py`. Import auto-fix di luar scope sempat terjadi dan sudah dipulihkan
dengan hash baseline yang cocok; final tests dijalankan sesudah pemulihan. Generated
timestamp import tetap dipertahankan. Tidak diklaim bahwa lint seluruh repository lulus.

- `src/atlas_ai/consultation/grounding.py`: bounded operational provenance, quote
  context guard, dan rule version v2; tidak menambahkan invocation verifier.
- `src/atlas_ai/consultation/solution.py`: numeric source policy dan final Go reference
  requirement; compact provider schema serta semua lokal constraints tetap aktif.
- `src/atlas_ai/consultation/prompts.py`: solution prompt v4; assessment v3 tetap.
- `scripts/consultation_live_readiness.py`: mode 4C.1 maksimal dua request, satu per
  case, gRPC bergantung pada Solution langsung; default test budget 30, telemetry
  protobuf/deadline/overhead tanpa key.
- `tests/test_consultation_solution.py`, `tests/test_consultation_live_readiness.py`,
  `tests/test_consultation_assessment.py`: provenance benar/salah/konflik, angka
  usulan dan SQL, quote substring/negasi, C5 asli, schema bounds, Go references,
  caller deadline, dan regression kelima event/NextQuestion.
- Dokumen ini, tautan pada dokumen sebelumnya dan evaluation README; empat artifact
  alignment (`solution_alignment_*` dan `grounding_alignment_rules_report.json`).

Tidak ada dependency baru, API key dalam source/log, perubahan `.env` produksi,
Business Memory Extraction, vector retrieval, agent, atau LLM semantic verifier.

## Reproduksi terbatas dan residual risks

Dengan dependency/image yang sama, jalankan module runner menggunakan ledger baru:

```powershell
$env:PYTHONPATH = "src"
python -m scripts.consultation_live_readiness --milestone 4C.1 --case solution --id A1 --turn-budget 30 --env-file .env --output evaluation/new-alignment-live.json
# Hanya setelah Solution langsung PASSED:
python -m scripts.consultation_live_readiness --milestone 4C.1 --case grpc --id A2 --turn-budget 30 --env-file .env --output evaluation/new-alignment-live.json
```

Perintah itu menghasilkan request live baru; tidak dijalankan ulang dalam audit ini.
Regressions memakai `python -m pytest -q -p no:cacheprovider` dalam image dengan
mount workspace read-only dan jaringan dimatikan. CLI runner memakai production key
sebagai credential, tetapi seluruh prompt berasal dari fixture sintetisnya.

Penghalang demo penuh saat ini adalah belum adanya Solution live valid setelah
alignment, sehingga belum ada round trip gRPC live yang berhasil. C5 replay masih
memiliki enam abstention grounding dan satu incompatibility references Go; tidak
dihitung sebagai keberhasilan baru. Prompt compliance, reliabilitas/latensi provider,
dukungan model produksi 3.8, deployment deadlines, dan integrasi Go nyata belum
terbukti. Matcher konservatif tetap dapat menolak parafrasa valid atau melewatkan
makna bahasa yang tidak tercakup; source IDs dan quotes valid tidak menjamin semantic
grounding lengkap. Milestone 4C belum dinyatakan siap.
