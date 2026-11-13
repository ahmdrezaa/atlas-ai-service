# Milestone 5 — Business Memory Candidate Extraction

Implementasi dan verifikasi offline/gRPC dengan mock provider selesai. **Live M5
belum terverifikasi:** dua invocation sintetis mendapat HTTP 503 dari provider.
Tidak ada response model untuk membuktikan schema/output M5 melalui Gemini live.
Artifact mock dan live diberi label terpisah; hasil mock tidak dihitung sebagai live.

## Kontrak dan desain invocation

Input tetap `ProcessConsultationTurnRequest` resmi. Output memakai field
`ProcessConsultationTurnResponse.memory_candidates` yang sudah ada, masing-masing
berisi `fact_text` dan `source_owner_message_ids`. Proto dan generated stubs tidak
diubah. Lima komponen protobuf Solution tetap persis lima, dengan seluruh P0
structural validation, grounding, dan mapping tetap dijalankan.

| Jalur turn | Invocation model | Memory selection |
| --- | ---: | --- |
| Assessment → NextQuestion | 1 | Metadata pada assessment yang sama |
| Assessment → Solution | 2 | Metadata dari dua invocation yang memang sudah ada; digabung dan dedup |
| GENERATE_NOW / cumulative cap → Solution | 1 | Metadata pada solution invocation yang sama |

Tidak ada mandatory second call untuk extraction, repair call, semantic verifier,
embedding, vector search, database client, atau agent. Python tidak mengonfirmasi,
menyimpan, mengaktifkan, ataupun menambahkan kandidat ke ContextSnapshot.memories.

Audit architecture memilih bentuk internal flat `GeneratedSolutionTurn`: lima array
Solution ditambah optional `memory_candidates` sebagai **metadata turn**. Ini menjaga
grammar provider sederhana, tidak menambahkan wrapper/nesting besar. `GeneratedSolution`
tetap model lima field; `solution_output()` secara eksplisit memisahkan metadata
sebelum validator dan mapping Solution. Mapping response menaruh kandidat di field
saudaranya, bukan di dalam `ai_pb2.Solution`.

`SufficiencyAssessment` juga memiliki metadata optional agar question path tidak
memerlukan invocation tambahan. Candidate proposal dari assessment dan generation
boleh saling melengkapi; mereka tidak dimasukkan kembali sebagai confirmed context
untuk invocation berikutnya. Jumlah call, shared deadline dan source trust boundary
tidak berubah.

Schema provider memakai compact dialect yang sudah terbukti pada P0, sekarang
dipakai bersama melalui `compact_provider_schema()`. Full Pydantic validation tetap
lokal. Metadata candidate yang malformed dibuang secara terbatas; field/struktur
Solution yang invalid tetap menggagalkan response seperti P0.

Prompt versions: `consultation-assessment-v4`, `consultation-solution-v6`, serta
modular memory policy `memory-candidates-v1` di prompts.py. Base prompt tidak lagi
melarang proposal memory secara absolut; tetap melarang confirmation/storage.

## Provenance dan kebijakan validasi

Gemini mengusulkan maksimal lima fakta pendek dari Owner history. Validator lokal
di memory.py kemudian menerapkan aturan berikut:

1. Setiap ID wajib berada di `context.sources.owner_messages`, yang hanya diisi
   dari pesan role OWNER melalui prepare_context. AI IDs, opinion IDs, memory IDs,
   source dari sesi arsip, atau UUID hasil tebakan tidak menjadi sumber baru.
2. Semua owner yang dirujuk harus mendukung proposisi atomik yang sama. Setidaknya
   satu harus memiliki clause fact_text lengkap, verbatim. Tidak menyambung fragmen
   menjadi fakta baru, memotong negasi/exception, atau menebak konteks angka pendek.
3. Fakta yang bersifat pertanyaan, spekulasi, rencana/permintaan sementara, instruksi
   AI, kutipan AI/customer, atau mengandung secret/identity cues tidak diusulkan.
   Framing berbahaya pada satu pesan dapat membuat seluruh pesan tidak eligible.
4. Scope, predicate, unit dan kuantitas dipertahankan. Jam ramai bukan jam buka;
   interval tutup bukan interval operasional; kasir bukan barista. Variasi angka kata
   kecil, clock notation dan sejumlah bentuk aktif/pasif hanya dipakai untuk proof
   dan dedup, tidak untuk mengganti teks yang dikembalikan.
5. Perbedaan nilai dalam slot comparable pada Owner history dianggap konflik/ambigu
   dan disisihkan, meskipun model hanya mengutip pesan terakhir. Tidak ada last-write
   winner atau perubahan state di Python.
6. Equivalent proposals digabung, source IDs disatukan dalam urutan history Go.
   Equivalent confirmed memories hanya dipakai untuk menolak pengulangan, tidak
   pernah sebagai source kandidat baru.
7. Bila tidak ada kandidat yang lolos, array kosong. Kandidat invalid tidak diberi
   ID pengganti atau diperbaiki; Solution yang valid tetap dapat dikembalikan.

MVP deterministic equivalences mencakup staffing, jam operasional/ramai, budget
constraints dan pencatatan pesanan manual/otomatis/digital. Nominal IDR tunggal dengan
format eksplisit Rp/rupiah didukung, termasuk grouping ribuan yang jelas; tidak ada
inferensi currency, konversi valuta, atau perkalian singkatan ribu/juta. Bentuk yang
ambigu/di luar recognizer sengaja diomit untuk menjaga precision.

Fakta menggunakan maksimal 1.000 karakter, sesuai source Go yang diaudit:
[MaxMemoryFactLength](https://github.com/Elioth-Gray/atlas-backend-core/blob/002472705b76f1f86e0b6628fc07bd6a92fe478d/internal/service/consultation.go#L20),
blob `a42c5b6c5793a7d4f669c5cb3f33aaa72b5d436b`. Batas lima proposal per response
adalah kebijakan latency/output budget lokal, lebih kecil dari batas 20 memories
yang dapat dikonfirmasi dalam satu Save Go. Tidak ada pemotongan fact text untuk
memenuhi batas. Proposal terlalu besar/invalid diomit, bukan diubah artinya.

Nilai Owner baru yang berbeda dari confirmed memory tidak otomatis mengganti memory
lama. Nilai itu dapat menjadi proposal baru jika tidak konflik dalam Owner history;
Go dan owner tetap harus menyelesaikan confirmation/update-nya.

## Bukti respons gRPC dan sumber

`evaluation/m5_memory_mock_grpc.json` berisi response protobuf lengkap dengan kelima
bagian Solution serta metadata berikut. **Provider pada artifact ini adalah mock**,
ditandai `AUTHENTICATED_PYTHON_GRPC_WITH_MOCK_PROVIDER_NOT_LIVE`; actual provider
requests = 0. Payload Solution P0 dipakai sebagai fixture valid, bukan diklaim sebagai
generation M5 baru. Candidate berikut berasal dari Owner sintetis yang ditambahkan
ke request test, bukan dari fixture Solution itu.

```json
{
  "memory_candidates": [
    {
      "fact_text": "Pesanan dicatat manual di kasir.",
      "source_owner_message_ids": [
        "20000000-0000-4000-8000-000000000002"
      ]
    }
  ],
  "model_id": "mock-provider-no-live",
  "prompt_version": "consultation-solution-v6"
}
```

Potongan di atas hanya metadata; field `solution` lengkap ada di artifact. Owner
message dengan ID tersebut memiliki content **Pesanan dicatat manual di kasir.**
Source berbeda dari seluruh confirmed memories. Candidate melalui owner-only lookup,
complete-clause proof, conflict check dan dedup sebelum mapping. gRPC actual status
OK; protobuf serialize/deserialize identik; mock model boundary dipanggil sekali.
Server request tidak berubah dan jumlah confirmed memories tetap lima.

Tests juga membuktikan satu kandidat yang didukung dua pesan Owner (jumlah dua
barista dalam dua bentuk wording), penggabungan IDs, source salah namespace,
nominal berbeda, private/injection cues, quoted AI lintas kalimat, dan references
yang tidak semuanya mendukung fakta tersebut. Tidak ada kandidat dengan sumber
sebagian benar yang diam-diam diperbaiki dengan membuang ID salahnya.

## Tests, latency dan live smoke

Baseline P0: **796 passed in 3.54s**. Final pada workspace terbaru: **873 passed in 4.46s** melalui Docker
Python 3.11, jaringan dimatikan. Full suite mencakup semua consultation events,
NextQuestion, P0 Solution/grounding, serta authenticated mock-provider gRPC. Test
fixtures dipindah ke conftest agar dipakai bersama, tidak membuat ulang payload
Solution atau validator P0.

Ruff `check src/atlas_ai tests scripts` PASS; format check consultation/tests/scripts
PASS. Tidak ada perubahan dependency, ai.proto, generated stubs, NLP Ucan, autentikasi,
Go backend/database/lifecycle, atau artifact/ledger historis P0. Core context dan
grounding juga cocok dengan hash saat pekerjaan M5 dimulai.

Saat penutupan pekerjaan, workspace memindahkan package `nlp/` ke `feedback/`.
Verifikasi ulang mendeteksi empat collection errors karena import lama pada adapter.
`grpc_service.py` disesuaikan dari `atlas_ai.nlp.pipeline` ke
`atlas_ai.feedback.pipeline`, kemudian seluruh 873 tests dijalankan ulang dan lulus.
Isi pipeline feedback tidak diedit; SHA-256-nya tetap
`b9567efbaf8a49541996675192cd03790641886e538e451fcf9dbecea9b1d022`.
Tidak ada implementasi NLP baru atau perubahan autentikasi.

| Live ID | Jalur | HTTP / actual gRPC | Durasi runner | Tahap validasi |
| --- | --- | --- | ---: | --- |
| M1 | Fresh Gemini melalui Python gRPC | 503 / UNAVAILABLE | 2,227 s | Output model tidak ada; structural/grounding NOT_REACHED |
| M2 | Satu pengujian terakhir | 503 / UNAVAILABLE | 1,541 s | Output model tidak ada; structural/grounding NOT_REACHED |

Ledger `evaluation/m5_memory_live.json` mempertahankan dua kegagalan tersebut.
Konfigurasi test-only: gemini-3.6-flash, v1beta, thinking LOW, shared budget 45 detik,
client deadline 60 detik, max_output_tokens 2.048. Runtime headroom gate lama tetap
aktif. Tidak ada retry otomatis, warmup tambahan, correction, atau call sesudah M2.
Ini HTTP 503 aktual dari provider, bukan hasil parsing/grounding, INVALID_ARGUMENT,
atau client timeout. Akar masalah server di balik 503 belum diketahui; tidak diklaim
bahwa schema baru pasti diterima provider karena belum ada successful output.

Request text aktual M5 9.587 byte dan provider schema 2.685 byte, dibanding P0
7.775/2.401 byte. Delta text 1.812 byte juga mencakup fakta owner sintetis tambahan;
bukan controlled latency comparison. Kandidat menambah output tokens walaupun tidak
menambah invocation. Output budget 2.048 tetap sama, sehingga response kompleks
masih berisiko truncation dan tetap harus lulus schema/validator, tanpa fake success.
Token usage tidak dikembalikan pada dua kegagalan, bukan bukti biaya nol.

Pada satu mock gRPC sample, candidate validation/mapping sekitar **2,09 ms**, seluruh
RPC sekitar **29,84 ms**. Angka ini mengukur local code/mock transport, bukan latency
Gemini production. Dampak latency extraction live belum dapat diukur.

## Files dan fungsi

- `consultation/memory.py`: ProposedMemoryCandidate, optional metadata parser,
  bounded signature/durability helpers, memory_candidates_to_proto.
- `consultation/assessment.py`: metadata pada SufficiencyAssessment dan pemindahan
  compact_provider_schema agar assessment/solution berbagi dialect yang sama.
- `consultation/solution.py`: GeneratedSolutionTurn.solution_output dan output tuple
  generator untuk memisahkan Solution dari proposals, tanpa invocation kedua.
- `consultation/pipeline.py`: merge/validate/map metadata pada dua jalur, tetap dalam
  shared deadline dan tanpa menulis snapshot.
- `consultation/prompts.py`: memory policy modular dan versions v4/v6.
- `scripts/consultation_live_readiness.py`: mode M5 maksimal dua fresh gRPC calls,
  owner fixture, request-immutability/provenance telemetry dan parser metadata.
- `scripts/gemini_live_diagnostics.py`: kompatibilitas return tuple generator.
- `grpc_service.py`: satu import mengikuti pemindahan folder oleh workspace,
  tanpa mengubah implementasi pipeline feedback.
- `tests/conftest.py`, memory tests baru dan schema/prompt regression tests: reuse
  fixtures, kandidat valid/invalid, multi-source, dedup, konflik dan seluruh events.
- Dokumen ini, evaluation README, serta tiga artifact `m5_memory_*`.

## Batas bukti dan tindak lanjut

Feature sudah mengembalikan candidates yang memenuhi kontrak melalui gRPC dengan
mock provider. Gemini live M5 belum tervalidasi karena dua HTTP 503; mock bukan
pengganti bukti live tersebut. Model produksi tetap gemini-3.8-flash dan default
timeout repository tetap 30 detik; tidak ada fallback produksi yang diaktifkan.

Durability, private-data detection, semantic dedup dan conflict detection adalah
batas konservatif yang dapat menolak fakta valid atau melewatkan nuansa di luar
grammar. Jawaban fragmen/parafrasa dan proses bisnis di luar recognizer sengaja
diomit; ini tidak menjanjikan recall semua fakta owner. Full-clause copying menjaga
precision dan source meaning, bukan jaminan kebenaran bisnis independen.

Unconfirmed candidates dapat diusulkan lagi pada turn berikutnya: proto request
tidak menyediakan daftar proposal yang pernah ditampilkan, dan Python tidak membuat
cache/session state untuk menebaknya. Dedup response dan confirmed snapshot dilakukan;
dedup penyajian lintas turn serta confirmation/update tetap tanggung jawab Go/UI.

Untuk percobaan manual berikutnya gunakan ledger M5 baru; perintah ini menghasilkan
request tambahan dan tidak dijalankan lagi pada task ini:

```powershell
$env:PYTHONPATH = "src"
python -m scripts.consultation_live_readiness --milestone M5 --case grpc --id M1 --turn-budget 45 --rpc-timeout 60 --env-file .env --output evaluation/new-m5-live.json
```
