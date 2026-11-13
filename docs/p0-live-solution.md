# P0 — Live Solution Generation

Audit 13 November 2026 (Asia/Jakarta), dari branch `main`, base commit
`c3e1e589fca7333dc8189bd71ea308877a1ae682`.

**Definition of Done P0 terpenuhi melalui request P3 dengan data sintetis.**
Gemini fresh menghasilkan Solution melalui Python gRPC, bukan response fixture,
cache, replay, atau output yang diperbaiki setelah generation. Ini verifikasi
fungsional development pada `gemini-3.6-flash`, bukan production readiness 3.8.

| Tahap | PASS/FAIL | Bukti P3 |
| --- | --- | --- |
| Gemini Live Generation | PASS | generateContent HTTP 200, model version 3.6 Flash |
| Pydantic Validation | PASS | GeneratedSolution dengan lima array berjumlah 3/2/1/1/1 |
| Structural Validation | PASS | Namespace/source IDs, quote, angka, hypothesis/proposal policy |
| Grounding Validation | PASS | Tiga finding SUPPORTED; enam claim dugaan/usulan NOT_APPLICABLE |
| Protobuf Mapping | PASS | Solution + metadata; serialization/deserialization identik |
| Python gRPC Live | PASS | AIService.ProcessConsultationTurn, actual status OK, 5,959920 detik |

Bukti lengkap: `evaluation/p0_solution_live.json` dan
`evaluation/p0_acceptance_report.json`. Response memiliki model_id
`gemini-3.6-flash`, prompt_version `consultation-solution-v5`, result `solution`,
serta memory_candidates kosong. Semua references finding/hypothesis/recommendation
nonkosong dan berasal dari namespace request yang benar.

## Root cause dan kebijakan statistik MVP

Versi sebelumnya meminta model menulis marker statistik, mengizinkan finding marker
tanpa sumber dalam validate_solution, kemudian menolaknya saat mapping karena Go
mewajibkan references. References tidak memiliki stats_id. Memberi opinion ID yang
tidak membuktikan statistik tidak menyelesaikan provenance.

MVP sekarang menggunakan aggregate_statistics hanya sebagai context. Output Solution
menolak marker statistik maupun prose statistik aktual tanpa sumber yang sah; tidak
ada lagi pengecualian references kosong atau rendering marker sebelum protobuf.
Go tetap pemilik data/presentasi statistik. Tidak ada perubahan proto/repository Go,
penambahan stats_id, maupun sitasi fiktif untuk melewati validator.

`_statistics` tetap tersedia untuk audit semantik 50 kasus historis, tetapi tidak
menjadi pengecualian pada wire policy. Finding SQL-only tetap ditolak sebelum
grounding/mapping. Pesan error juga tidak lagi menyuruh model memakai marker yang
sekarang dilarang pada output MVP.

## Audit enam abstention C5 asli

Output dan sumber asli lengkap tersedia di `evaluation/p0_c5_audit.json`, termasuk
hash ledger/output C5. Tidak ada ledger historis yang diubah.

| Bagian | Output asli | Sumber / jenis provenance | Kategori dan perbaikan |
| --- | --- | --- | --- |
| Recommendation.description | Lakukan pemantauan dan pencatatan manual waktu pelayanan transaksi kasir pada jam ramai 17.00 sampai 19.00 untuk mengidentifikasi titik hambatan tanpa melakukan pembelian alat baru. | Memory ...0001: Jam ramai pukul 17.00 sampai 19.00 setiap hari. Memory ...0002: Anggaran tambahan belum tersedia; langkah awal harus tanpa pembelian alat. Owner ...0001: Buat langkah observasi tanpa biaya. Penyebab antrean belum diketahui. | FALSE_REJECTION. Tindakan usulan dengan angka operasional bersumber; intent tindakan tidak membuktikan premis. |
| CustomerEvidence[0].note | Keluhan pelanggan terkait waktu tunggu antrean | Opinion ...0001: Antreannya lama sekali. | UNRESOLVED oleh bounded matcher. Tidak ditambahkan sinonim agar lolos; prompt memilih note kosong atau excerpt yang langsung dapat dibuktikan. |
| CustomerEvidence[1].note | Keluhan pelanggan terkait kecepatan pelayanan | Opinion ...0002: Pelayanannya lambat. | UNRESOLVED; penanganan sama, tidak diganti menjadi fakta baru. |
| SuccessIndicator.metric | Catatan durasi transaksi kasir harian | Definisi variabel pengukuran yang diusulkan, bukan bukti catatan historis telah ada | FALSE_REJECTION. Role metric tidak memerlukan kata pertama tertentu. |
| SuccessIndicator.target | Usulan: Terkumpulnya data hambatan alur transaksi kasir | Target pengumpulan data, bukan klaim data sudah terkumpul | FALSE_REJECTION. Role target dan framing usulan menjadi dasar klasifikasi. |
| SuccessIndicator.timeframe | Usulan: Selama 7 hari pengamatan | Jendela evaluasi yang diusulkan; memory observasi satu minggu juga tersedia dalam context | FALSE_REJECTION. Role timeframe tidak perlu diawali kata kerja. |

Keempat false rejection kini NOT_APPLICABLE karena berupa usulan, bukan karena
matcher membuktikan hasil tindakan akan terjadi. Dua note tetap INSUFFICIENT_EVIDENCE.
C5 secara keseluruhan tetap bukan Solution MVP valid: finding statistik dan dua note
tersebut belum sesuai kebijakan. Tidak dihitung sebagai keberhasilan live baru.

Pemeriksaan proposal menggunakan role field, intent tindakan/advisory, dan guard
premis/fakta historis. Lakukan/sebaiknya/nominal tindakan tidak mengesahkan kalimat
berikutnya yang mengarang mesin rusak, resource tersedia, keuntungan meningkat, atau
alasan sebab-akibat tanpa sumber. Success indicators dapat berupa definisi variabel,
target koleksi, dan jangka evaluasi tanpa satu grammar awalan yang sama. Target dan
timeframe tetap memakai Usulan:; pernyataan kondisi aktual/achievement tetap ditolak.

## Hipotesis dan provenance angka

Previously, kata mungkin/diduga saja bisa membuat hipotesis tidak relevan mendapat
NOT_APPLICABLE. Sekarang setiap clause substantif memerlukan anchor leksikal dari
sumber CURRENT yang dirujuk, selain tentativeness dan larangan premis faktual baru.
Kata generik seperti owner, pelanggan, proses, atau mungkin tidak menjadi anchor.
Hypothesis tentang Bitcoin, source ID valid tetapi topik tidak relevan, clause
spekulasi tak terkait yang ditempelkan pada clause antrean, fakta operasional baru,
dan penyebab terbukti mendapat negative tests.

Ini necessary relevance gate, bukan bukti semantic entailment/plausibility universal.
Tidak ada general-purpose regex semantic verifier, mandatory LLM verifier, ataupun
kamus khusus untuk menerima isi C5. Model tetap boleh mengusulkan dugaan yang belum
terbukti; sumber membuktikan symptom, bukan kebenaran root cause.

Angka operasional tetap membandingkan predicate, unit, nilai dan sumber. Tambahan
setelah P1 membedakan jadwal tindakan dari klaim predicate bisnis: tindakan pada jam
17.00 hingga 19.00 dapat memakai rentang dari memory jam ramai 17.00 sampai 19.00,
tanpa mengklaim itu jam buka. Alias hingga/sampai dan dash di antara clock values
lengkap hanya dinormalisasi untuk pembuktian, bukan untuk mengubah quote, UUID, atau
wire text. Jam berubah, urutan terbalik, predicate berbeda, namespace salah dan sumber
negatif/konflik tetap ditolak. Jendela baru berlabel Usulan: tetap merupakan rencana,
bukan fakta operasional historis.

## Tiga invocation aktual dan diagnosis

| ID | Jalur | HTTP | Durasi total | Hasil |
| --- | --- | --- | --- | --- |
| P1 | GENERATE_NOW langsung | 200 | 5,794 s | Lima bagian + Pydantic PASS; structural menolak jadwal numerik yang sah |
| P2 | GENERATE_NOW setelah perbaikan jadwal | 504 | 44,592 s | Provider timeout; tidak ada model output untuk divalidasi |
| P3 | GENERATE_NOW melalui Python gRPC, Gemini fresh | 200 | RPC 5,959920 s; runner 5,970 s | Seluruh enam tahap PASS, actual gRPC OK |

P1 asli menyatakan: Catat durasi transaksi kasir secara manual pada jam 17.00 hingga
19.00 tanpa membeli alat baru. Referensinya mencakup memory jam ramai yang cocok.
Perbaikan menambahkan pembuktian jadwal tindakan tersebut, dengan positive/negative
tests; tidak mengubah output P1. Replay output asli sesudah fix lolos Pydantic,
structural, grounding dan protobuf, tetapi tetap bukan request live baru yang sukses.

Karena P2 timeout, izin belanja diagnostik untuk P3 memakai revalidasi P1 itu dan
mencatatnya eksplisit sebagai NOT_NEW_LIVE_SUCCESS. Baris P1 tetap FAILED di ledger.
P3 memanggil Gemini baru, memakai actual service + BearerAuthInterceptor + generated
stub; output P1 tidak pernah dipakai sebagai response RPC. Test memastikan spend
gate tidak memperbaiki/memutasi row serta menolak output rusak/non-HTTP-200.

Jumlah generateContent aktual tepat tiga, satu HTTP response per invocation. Tidak
ada automatic retry, model correction, agent/tool call, verifier invocation, atau
evaluasi live besar. Tidak ditambahkan call sesudah P3 sukses.

| Tahap P3 | Durasi aktual |
| --- | --- |
| SDK/provider | 5,885 s |
| LangChain invoke + JSON parsing | 5,909130 s (termasuk provider) |
| Pydantic lokal | 0,031358 ms |
| Structural validation | 0,191891 ms |
| Grounding validation | 0,791098 ms |
| Protobuf mapping | 0,038560 ms |
| Pipeline total | 5,954834 s |
| gRPC call total | 5,959920 s |
| Overhead RPC teramati di luar pipeline | 5,086 ms |

Stage durations berasal dari per-request timings pada execution asli, bukan hanya
replay. Persiapan/factory tercakup di pipeline total; waktu provider dan LangChain
overlap sehingga tidak boleh dijumlahkan sebagai tahap independen.

Prompt aktual 7.775 UTF-8 byte, schema provider 2.401 byte, native v1beta, thinking LOW,
max_output_tokens 2.048. P3 finish STOP: 2.207 input + 997 output = 3.204 token.
Thinking usage tidak dilaporkan. Temperature diabaikan model yang memakai fixed
sampling defaults; tidak diklaim deterministik. P2 tidak melaporkan usage, bukan
bukti token/biaya nol. Output budget tidak terbukti menyebabkan HTTP 504.

## Deadline dan konfigurasi

Development override saja: model 3.6 Flash dan shared turn budget 45 detik dengan
Python gRPC client deadline 60 detik. CLI mensyaratkan budget 40–45, setidaknya 10
detik nominal headroom, serta caller tidak lebih panjang dari default Go yang telah
diaudit. Saat masuk server, P3 masih memiliki 59,998909 detik; headroom terhadap
budget Python 14,998909 detik. Runtime gate menolak pemanggilan dengan headroom kurang
sebelum provider diakses. Effective SDK timeout 44,749584 detik, bukan 45 detik baru
untuk setiap invocation. Assessment + generation tetap berbagi budget yang sama.

Source Go yang diaudit tetap default AI 60 detik, HTTP 90 detik, overhead 15 detik
di luar panggilan AI; lihat `evaluation/solution_alignment_deadlines.json` untuk
commit/blob sumber. Tidak ada Go process/end-to-end backend yang dijalankan, tidak
ada perubahan Go, dan environment deployment/proxy belum diverifikasi. Hasil ini
membuktikan loopback Python gRPC pada deadline tersebut, bukan batas overhead remote.

SDK meneruskan timeout sebagai HTTP client timeout dan server deadline. P2 adalah
HTTP 504 sungguhan, bukan parsing/structural/grounding failure atau local TimeoutException.
P3 sukses pada konfigurasi dan prompt yang sama. Penyebab provider melewati budget
pada P2 belum terisolasi; menaikkan timeout saja tidak dianggap menyelesaikan provider.

Model produksi tetap gemini-3.8-flash; default timeout repository tetap 30 detik.
Tidak diubah .env, .env.example, compose, dependencies, maupun output-token budget.

## Perubahan fungsi dan verifikasi

- prompts.py: SOLUTION_SYSTEM_PROMPT / SOLUTION_PROMPT versi v5, pendek dan konsisten
  dengan keputusan MVP. Assessment prompt v3 dan question guard tetap.
- solution.py: _check_numbers dan validate_solution menolak statistik output dan
  references kosong; GeneratedEvidence.note default kosong; solution_to_proto memakai
  validator tunggal sebelum mapping, tanpa marker rewriting atau output repair.
- grounding.py: operational_number_context, evaluate_claim, _related_hypothesis dan
  _action_framing; rule version v3, scope proof tetap terbatas pada provenance/role.
- assessment.invoke_structured, pipeline.run, SolutionGenerator.generate dan
  solution_to_proto menerima timings opsional per request, tanpa session state,
  prompt data tambahan atau logging isi sumber.
- scripts/consultation_live_readiness.py: bounded P0 mode (dua generation langsung,
  satu gRPC maksimum), headroom gate, stage/error telemetry, saved-output spend gate
  yang eksplisit tidak dihitung sebagai sukses live.
- server.py: import order dan module logger untuk dua issue lint lama. AST
  BearerAuthInterceptor identik dengan HEAD; log token/auth flow tidak diubah.
- tests/test_consultation_p0.py serta solution/grounding/live-readiness regressions:
  proposal positif/negatif, sumber angka/alias jam/konflik, hipotesis tidak relevan,
  note kosong, statistik MVP, semua event dan NextQuestion, timings dan replay gate.

Baseline: 702 passed in 5.23s. Final: **796 passed in 4.22s**, Docker Python 3.11,
network disabled. Ruff `check src/atlas_ai tests scripts` PASS. Format check untuk
consultation, server/grpc_service, tests dan scripts PASS (25 files). Generated stubs
dan NLP tidak diformat/diubah. git diff --check PASS.

50 label/case lama tetap utuh. Laporan baru `evaluation/p0_grounding_rules_report.json`:
0/23 false acceptance. Satu paraphrase fakta tetap semantic abstention; polite-action
false rejection lama terselesaikan. Combined structural+rules menolak empat kasus
statistik yang sebelumnya berlabel SUPPORTED karena output SQL-only sengaja dilarang
MVP, sehingga combined false_rejection=5 (1 abstention + 4 policy exclusions). Jangan
menyebut kelimanya semantic failure baru atau mengklaim angka itu model accuracy.

## Batas bukti dan residual risks

P0 functional acceptance tercapai oleh **P3 fresh**, bukan oleh unit tests atau replay.
Provider tetap pernah timeout pada konfigurasi 45 detik; tiga request bukan pengukuran
p95/reliabilitas. Production 3.8 belum diuji pada jalur ini. Go deployment, remote
transport, frontend/presentasi statistik dan backend end-to-end belum divalidasi.
Bounded matcher dan lexical relevance bukan semantic entailment menyeluruh; parafrasa
valid dapat ditolak dan makna di luar guard masih dapat terlewat. Target/hipotesis
diterima sebagai usulan, bukan janji keberhasilan atau probabilitas terkalibrasi.

Tidak ada perubahan ai.proto, generated stubs, NLP Ucan, autentikasi, Go lifecycle,
database, retrieval, embeddings, agents, Business Memory Extraction, atau aktivasi
production LLM verifier. API key dan data pengguna asli tidak dicetak/dikirim ke prompt.

Reproduksi manual dengan ledger baru akan memakai request tambahan, tidak dilakukan
lagi dalam task ini:

```powershell
$env:PYTHONPATH = "src"
python -m scripts.consultation_live_readiness --milestone P0 --case solution --id P1 --turn-budget 45 --rpc-timeout 60 --env-file .env --output evaluation/new-p0-live.json
# Setelah output valid, lakukan satu invocation fresh melalui gRPC:
python -m scripts.consultation_live_readiness --milestone P0 --case grpc --id P3 --turn-budget 45 --rpc-timeout 60 --env-file .env --output evaluation/new-p0-live.json
```
