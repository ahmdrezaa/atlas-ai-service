# ATLAS — AI Service Implementation PRD v2.1 (Audited Against Original PRD)

**Status:** Technical implementation specification; original-feature traceability audited; unresolved cross-repository blockers marked explicitly  
**Tanggal:** 12 November 2026  
**Target repository:** [`ahmdrezaa/atlas-ai-service`](https://github.com/ahmdrezaa/atlas-ai-service)  
**Backend contract owner:** [`Elioth-Gray/atlas-backend-core`](https://github.com/Elioth-Gray/atlas-backend-core)  
**Implementation:** Reza (Guided Consultation, shared service/integration), Ucan (NLP/ML Feedback), Suthe (Go backend/DB), Sakha (frontend)

> **Prioritas tertinggi:** Jangan membuat protobuf, database schema, atau endpoint alternatif berdasarkan PRD lama. File `proto/atlas/ai/v1/ai.proto` di repository Go Suthe adalah *source of truth* untuk komunikasi Go–Python. Kontrak yang ditulis dalam PRD lama (`AtlasAIService`, `ConsultationTurnRequest`, `SolutionResult.solution_json`, `ExistingIssueCluster`, empat kategori severity) **bukan** kontrak implementasi yang sedang digunakan.

## 0. Tujuan revisi dan keputusan yang berlaku

Dokumen ini menggantikan *PRD teknis Python gRPC ATLAS* sebelumnya sebagai spesifikasi implementasi untuk coding agent dan kolaborasi dua pengembang AI. Konsep produk ATLAS tetap: QR feedback pelanggan → ekstraksi opini dan klaster isu → visual Galaxy → diskusi owner dengan AI → lima bagian rekomendasi → simpan hasil diskusi, bukan pemantauan penyelesaian tugas.

**Keputusan yang sudah diambil:**

1. Satu repository Python `atlas-ai-service`; satu server gRPC `atlas.ai.v1.AIService` yang mengimplementasikan `ProcessFeedback` dan `ProcessConsultationTurn`.
2. Dua modul: `src/atlas_ai/nlp/` milik Ucan, `src/atlas_ai/consultation/` milik Reza. Modul bersama untuk konfigurasi, generated stubs, gRPC server, error handling, dan observability.
3. Go Suthe adalah satu-satunya pihak yang mengelola PostgreSQL, otorisasi shop, penyimpanan feedback/opinion/issue, sesi, pesan konsultasi, dan Business Memory. Python **stateless terhadap sesi** dan tidak punya akses DB langsung.
4. Docker-first: image Python 3.11, `pyproject.toml`, `Dockerfile`, `.dockerignore`, `compose.yaml`; token/API key melalui environment, tidak di-commit.
5. Tidak ada FastAPI, REST endpoint Python, tabel tambahan, autonomous agent, aksi operasional otomatis, atau action tracking untuk MVP.
6. Konflik desain PDM: keputusan produk terakhir adalah `Issues` dan `Consultation_Sessions` tidak mempunyai `shop_id`, tetapi **kode Go saat dokumen dibuat masih memiliki `consultation_sessions.shop_id`**. Perubahan DB dan Go sepenuhnya dikoordinasikan Suthe; jangan mengubah kontrak Python untuk menyiasatinya.

### 0.1 Compatibility dan preservasi kebutuhan PRD awal

PRD ini **mempertahankan tujuan fungsional** PRD pertama: ekstraksi multi-aspek + sentiment, severity, multilingual E5, shop-scoped issue grouping, *Context Sufficiency Assessment*, satu pertanyaan per turn, maksimum empat pertanyaan per siklus, lima komponen konsultasi, claim-level grounding, dan owner-confirmed durable Business Memory. Yang berubah adalah detail kontrak dan bentuk wire response agar kompatibel dengan implementasi Go yang ada.

**Keputusan implementasi yang harus dipertahankan dari PRD awal:** Guided Consultation memakai **LangChain + Google Gemini**, dengan **Pydantic typed structured output** sebagai schema internal; setelah lolos validasi, Python memetakan hasil ke generated protobuf `NextQuestion` atau `Solution`. Jangan mengubahnya diam-diam menjadi API JSON generik, memanggil model langsung tanpa orkestrasi yang disepakati, atau membuat LangGraph/multi-agent tanpa persetujuan. LangChain digunakan untuk prompting, model invocation, dan typed output; validasi ID/quote/angka tetap deterministik dan *claim entailment* diuji secara terpisah.

**Catatan keterlacakan (traceability):** contoh JSON yang panjang dalam PRD awal adalah **desain informasi dan UX**, bukan wire format Go. Field awal yang belum memiliki padanan langsung di proto (misalnya `scope_and_period`, `limitations_note`, `verification_step`, `feasibility_assumptions`, `how_owner_measures`) tidak boleh dipura-purakan sudah ada. Pemetaan spesifik dan keputusan bersama tim dijelaskan pada §2.5 di bawah.

## 1. Tujuan dan ruang lingkup

### 1.1 Path A — `ProcessFeedback`

Menerima satu feedback berisi rating dan teks serta kandidat isu/vektor **dari shop yang sama**. Mengembalikan **0..N opini**: teks opini ter-normalisasi, kutipan asli, sentimen, flag severity biner, embedding, identitas model embedding, dan assignment ke existing issue atau judul isu baru.

**Bukan tanggung jawab Python:** menulis opini ke PostgreSQL, menghasilkan UUID final issue, mengubah shop ownership, membuat feedback record, menjadwalkan job/retry. Go worker melakukan langkah-langkah tersebut.

### 1.2 Path B — `ProcessConsultationTurn`

Menerima snapshot issue, statistik, representatif evidence, profil shop, owner messages, confirmed memories, histori, event, serta status pertanyaan yang direpresentasikan Go. Mengembalikan **tepat satu** dari:

- `NextQuestion` — satu pertanyaan bernilai tinggi; atau
- `Solution` — lima komponen terstruktur, disertai referensi grounded.

Di kedua mode, response bisa menyertakan `memory_candidates`, `model_id`, dan `prompt_version`. **Tidak ada operasi save/archive dari Python.**

## 2. Kontrak integrasi yang wajib diikuti

### 2.1 Sumber dan sinkronisasi

- Repo Go: [`proto/atlas/ai/v1/ai.proto`](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/proto/atlas/ai/v1/ai.proto).
- Protobuf package: `atlas.ai.v1`.
- Service: `AIService`.
- RPC: `ProcessFeedback(ProcessFeedbackRequest) returns (ProcessFeedbackResponse)` dan `ProcessConsultationTurn(ProcessConsultationTurnRequest) returns (ProcessConsultationTurnResponse)`.
- Setelah mengambil salinan dari Go, catat commit SHA sumber di `docs/proto-sync.md` atau README. Protobuf harus disalin **apa adanya**, kemudian generate Python `*_pb2.py` dan `*_pb2_grpc.py`. Jangan mengedit hasil generate.
- Perubahan kontrak memerlukan koordinasi Suthe; jangan mengedit proto versi Python secara sepihak.

### 2.2 Field Path A yang dipakai

| Pesan | Field utama |
| --- | --- |
| `ProcessFeedbackRequest` | `shop_id`, `feedback`, `shop`, `candidate_issues`, `candidate_vectors` |
| `Feedback` | `id`, `customer_name`, `rating`, `text`, `create_time` |
| `CandidateIssue` | `issue_id`, `title`, `opinion_count` |
| `CandidateVector` | `opinion_id`, `issue_id`, `embedding`, `embedding_model` |
| `ProcessFeedbackResponse` | `repeated ExtractedOpinion opinions` |
| `ExtractedOpinion` | `opinion_text`, `evidence_text`, `sentiment`, `severity`, `embedding`, `embedding_model`, `issue` |
| `IssueAssignment` | `existing_issue_id`, `new_issue_title` |

Enum sentiment: `SENTIMENT_NEGATIVE`, `SENTIMENT_POSITIVE`, `SENTIMENT_NEUTRAL`. Jangan kirim `SENTIMENT_UNSPECIFIED` untuk opini valid. Enum severity adalah `SEVERITY_SEVERE`, `SEVERITY_NOT_SEVERE`, `SEVERITY_UNSPECIFIED`. Go hanya menyimpan `Severe` atau NULL di DB; **tidak ada enum `LOW/MEDIUM/HIGH/CRITICAL`** di kontrak saat ini.

`IssueAssignment.new_issue_title` **selalu diisi**, sekalipun `existing_issue_id` ada, karena Go memakai judul sebagai fallback bila assignment ID ditolak. Go mengelompokkan judul isu baru yang sama di *response* yang sama; ini bukan jaminan deduplikasi lintas feedback.

`ProcessFeedbackResponse.opinions` boleh kosong ketika tidak ada opini yang dapat diekstrak. Perilaku retry untuk feedback yang tidak menghasilkan opini perlu diperhatikan pada integrasi karena pengecekan `HasOpinions` Go mengandalkan opini tersimpan.

### 2.3 Field Path B yang dipakai

| Pesan | Field utama |
| --- | --- |
| `ProcessConsultationTurnRequest` | `session_id`, `request_id`, `event`, `context`, `messages`, `asked_topics`, `skipped_topics`, `previous_solution`, `question_count` |
| `ContextSnapshot` | `issue`, `period`, `stats`, `evidence`, `shop`, `memories`, `prior_issue_solution` |
| `IssueStats` | `unique_feedback_count`, `opinion_count`, `average_rating`, `negative_count`, `positive_count`, `neutral_count`, `severe_count`, `not_severe_count` |
| `EvidenceOpinion` | `opinion_id`, `opinion_text`, `evidence_text`, `sentiment`, `severity`, `full_feedback`, `rating`, `create_time` |
| `ConversationMessage` | `id`, `role`, `content`, `solution`, `topic_key`, `create_time` |
| `ProcessConsultationTurnResponse` | `oneof result { next_question, solution }`, `memory_candidates`, `model_id`, `prompt_version` |

Event **persis** `CONSULTATION_EVENT_START`, `CONSULTATION_EVENT_OWNER_REPLY`, `CONSULTATION_EVENT_SKIP_QUESTION`, `CONSULTATION_EVENT_GENERATE_NOW`, `CONSULTATION_EVENT_CONTINUE_DISCUSSION`. Pada owner reply/continue, Go **sudah menambahkan pesan owner baru ke `messages`** sebelum mengirim request. Python tidak boleh menambahkan atau memprosesnya dua kali.

`previous_solution` adalah revisi terakhir sesi aktif jika tersedia; `context.prior_issue_solution` adalah hasil arsip terdahulu. Keduanya konteks historis, **bukan bukti baru** untuk mengklaim kondisi saat ini.

### 2.4 Format solusi adalah nested protobuf, bukan JSON string

Python mengisi `Solution` langsung melalui generated protobuf. Struktur yang wajib ada:

```text
Solution
├── findings[]            -> Finding(statement, references)
├── customer_evidence[]   -> CustomerEvidence(opinion_id, quote, note)
├── hypotheses[]          -> Hypothesis(statement, confidence, references)
├── recommendations[]     -> Recommendation(title, description, priority, references)
└── success_indicators[]  -> SuccessIndicator(metric, target, timeframe)

References
├── opinion_ids[]
├── owner_message_ids[]
└── memory_ids[]
```

Gunakan **Pydantic v2 typed internal objects** untuk validasi structured output **sebelum mapping** ke protobuf, konsisten dengan stack Guided Consultation PRD awal. Wire output tetap wajib protobuf `Solution`, bukan `solution_json`. Go yang melakukan mapping ke `consultation_messages.solution_data JSONB`.

### 2.5 Mapping detail output PRD pertama → kontrak Go (wajib dibaca Codex)

| Kebutuhan spesifik PRD pertama | Representasi di `ai.proto` Suthe | Kebijakan preservasi |
| --- | --- | --- |
| `findings.summary`, `findings.claims[]` | `findings[] -> Finding(statement, references)` | Pecah menjadi klaim singkat dan traceable; jangan tambahkan nested `summary` di protobuf. |
| `findings.scope_and_period` | `context.stats` dan `context.period` **hanya di request** | Untuk output, angka/periode bisa disebut di `Finding.statement` **hanya dengan nilai yang sama persis**; sumber angka adalah snapshot Go, bukan opinion ID rekaan. Frontend dapat menampilkan statistik dari API Go secara terpisah. |
| `findings.limitations_note` | Tidak ada field khusus | Nyatakan sebagai `Finding.statement` terpisah secara eksplisit (misalnya bukti masih terbatas) atau diskusikan perluasan proto dengan Suthe; jangan diam-diam hilang. |
| `customer_evidence.sentiment`, `.severity` | Hanya `CustomerEvidence(opinion_id, quote, note)` di output; sentiment/severity ada di `context.evidence` | Kutipan + opini ID tetap wajib; jika UI memerlukan label sentimen/severity, frontend/Go mengambilnya dari evidence asal, bukan output yang dibuat AI. |
| `hypotheses.label` dan `verification_step` | `Hypothesis(statement, confidence, references)` | Hipotesis wajib jelas dibedakan dari temuan; langkah verifikasi boleh dijelaskan dalam teks hipotesis atau tindakan rekomendasi. Tidak ada field khusus `verification_step`. |
| `recommendations.primary_action` + `alternative_options[0..2]` | `repeated Recommendation(title, description, priority, references)` | Jadikan **elemen pertama rekomendasi utama**, lalu maksimal dua opsi alternatif; `priority` bukan jaminan struktur primary-vs-alternative. **Koordinasikan aturan urutan dengan Go/Sakha** agar UI konsisten. |
| `feasibility_assumptions` dan `reason` | Tidak ada field khusus di `Recommendation` | Nyatakan asumsi terpisah dengan penanda eksplisit di `description` dan sumber faktual di `references`; jangan menampilkan asumsi sebagai fakta terkonfirmasi. |
| `success_indicators.metric_name`, `.how_owner_measures` | `SuccessIndicator(metric, target, timeframe)` | `metric_name` → `metric`. **Metode pengukuran tidak punya field khusus**; jangan menaruhnya ke `target` seakan angka sasaran. Jika UX wajib menampilkan metode terstruktur, minta Suthe menambahkan `measurement_method` pada kontrak; sementara ini dapat dijelaskan secara manusiawi dalam `metric` hanya jika disepakati tim. |
| `SolutionResult.solution_json` | `ProcessConsultationTurnResponse.solution` adalah `Solution` protobuf | Pydantic internal tetap digunakan; **tidak** mengirim JSON string melalui gRPC. Go mengonversi protobuf ke JSONB untuk persistence. |
| `owner_message` + `history` | `messages` telah mencakup pesan owner turn terbaru | Tidak memproses pesan dua kali; ID owner tersebut tetap bisa direferensikan. |
| `LOW/MEDIUM/HIGH/CRITICAL` | `SEVERITY_SEVERE`/`SEVERITY_NOT_SEVERE` | Untuk wire dan penyimpanan gunakan enum Go. Rubrik 4-level hanya boleh menjadi eksperimen internal dengan mapping teruji, tidak muncul sebagai field baru. |

**Implikasi scope:** kebutuhan bisnis lama **dipertahankan**, tetapi beberapa representasi UI tidak bisa dijamin *lossless* tanpa perubahan protobuf dan frontend. Jangan mengklaim 100% identik secara struktural. Implementasi MVP harus memilih fallback teks yang dapat diterima atau mengusulkan perubahan kontrak pada Suthe terlebih dahulu.

## 3. Struktur repository target

```text
atlas-ai-service/
├── src/
│   └── atlas_ai/
│       ├── __init__.py
│       ├── server.py                 # Registrasi kedua RPC pada satu gRPC server
│       ├── config.py                 # Typed environment settings
│       ├── grpc_service.py           # AIServiceServicer delegasi ke dua pipeline
│       ├── generated/               # Python protobuf/gRPC stubs dari Suthe
│       ├── nlp/
│       │   ├── __init__.py
│       │   ├── pipeline.py           # ProcessFeedback orchestration
│       │   ├── extraction.py         # ABSA / opinion extraction
│       │   ├── severity.py           # Severe vs not severe
│       │   ├── embedding.py          # multilingual E5 inference
│       │   └── clustering.py         # assignment shop-scoped
│       └── consultation/
│           ├── __init__.py
│           ├── pipeline.py           # ProcessConsultationTurn orchestration
│           ├── models.py             # Typed internal LLM outputs
│           ├── context.py            # Sanitized grounded context preparation
│           ├── questioning.py        # Sufficiency + question guardrails
│           ├── solution.py           # Generate/map five protobuf sections
│           ├── grounding.py          # Deterministic + semantic verification
│           └── memory.py             # Owner-sourced memory candidate extraction
├── proto/atlas/ai/v1/ai.proto        # Exact pinned copy from Go repo
├── tests/
│   ├── test_grpc_contract.py
│   ├── test_feedback.py
│   ├── test_clustering.py
│   ├── test_consultation_turn.py
│   ├── test_grounding.py
│   └── test_memory.py
├── docs/proto-sync.md
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .env.example
├── compose.yaml
├── README.md
└── .gitignore
```

Ini adalah **target struktur**, bukan klaim bahwa semua file sudah dibuat. `nlp/` dan `consultation/` dapat diimplementasikan paralel setelah skeleton service dan protobuf valid. Jangan menyalin struktur lama `nlp_ml/` atau membuat dua server gRPC independen.

## 4. Path A — Spesifikasi pipeline NLP (Ucan)

### 4.1 Ekstraksi opini

- Input teks pelanggan bisa berbahasa Indonesia, campuran Indonesia–Inggris, typo, singkatan, dan menyebut lebih dari satu aspek.
- Ulasan majemuk dipisah menjadi opini berorientasi aspek. Contoh: `Rotinya enak, tetapi pelayanannya lambat` menghasilkan dua opini dengan sentimen berbeda.
- `opinion_text` merupakan formulasi yang ringkas dan membawa konteks aspek; `evidence_text` harus berupa **kutipan verbatim** dari `feedback.text`, tidak boleh dikarang atau diterjemahkan.
- Rating merupakan sinyal pendukung, **bukan** pengganti sentimen aspek. Ulasan ber-rating tinggi dapat memiliki satu keluhan negatif.
- Jika tidak ada aspek yang benar-benar dapat dipertanggungjawabkan, response kosong lebih baik daripada klaim buatan. Jangan memaksakan satu opini pada teks kosong, emoji saja, atau komentar tanpa makna terukur.
- Model ABSA/ekstraksi (misalnya Indo LEGO-ABSA) dipasang setelah checkpoint dan cara inference dikonfirmasi. PRD tidak mengasumsikan checkpoint sudah tersedia/berhasil dilatih.

### 4.2 Severity

- Output **biner**: `SEVERITY_SEVERE` bila opini mengindikasikan urgensi serius yang dapat ditindaklanjuti; selain itu `SEVERITY_NOT_SEVERE`. Kriteria contoh: indikasi bahaya pangan/keselamatan atau dampak layanan yang sangat serius, dengan kehati-hatian terhadap konteks dan hiperbola.
- `SEVERITY_UNSPECIFIED` tidak dipakai untuk menggambarkan kelas keempat/kelas tengah; penggunaan hanya ketika benar-benar tidak dinilai dan harus disepakati dengan Go.
- Classifier XLM-R hanya dipakai jika checkpoint, label map, dan evaluasi sesuai output biner. Bila rubrik internal empat level diterapkan untuk eksperimen, pemetaan ke enum Go harus eksplisit, diuji, dan tidak mengubah kontrak tanpa Suthe.
- Severity bukan diagnosis keamanan pangan dan bukan otomatis prioritas bisnis.

### 4.3 Embedding dan clustering

- Model target: `intfloat/multilingual-e5-small`; output 384 dimensi, **dengan prefix input yang sesuai dokumentasi model** dan normalisasi vektor yang konsisten.
- Gunakan kandidat `candidate_vectors` dari request; `candidate_issues` berisi metadata (bukan centroid/embedding). Filter kandidat yang tidak kompatibel dimensi/model. Tidak ada pengambilan kandidat lintas shop dari Python.
- Similarity cosine terhadap kandidat valid; aggregasi skor per issue harus dijelaskan dan diuji (misalnya skor tertinggi antar opinion candidates atau agregasi top-k). Jangan hard-code `0.82` sebagai threshold yang sudah terbukti optimal: jadikan konfigurasi dan kalibrasi dengan sampel validasi.
- Jika cocok dan kandidat issue ID ada dalam `candidate_issues`, kirim `existing_issue_id`. Jika tidak cukup cocok, kirim `existing_issue_id=""` dan `new_issue_title` yang deterministik/ringkas.
- Selalu isi `new_issue_title` karena Go memakainya sebagai fallback untuk invalid assignment. Uji kemiripan terlalu longgar (false merge) dan terlalu ketat (issue fragmentation).
- Asumsi domain wajib: **satu `issue_id` tidak pernah mencampur opini dari dua shop**. Python hanya melihat kandidat shop yang dikirim Go, sementara Go memvalidasi ownership saat penulisan. Pastikan integrasi menguji isolasi shop.

### 4.4 Kualitas dan fallback

- Validasi string nonempty untuk setiap opini, sentiment valid, embedding finite/nonempty, model ID diisi, quote cocok dengan sumber, issue title diisi.
- Gagal memuat checkpoint/model → gRPC error eksplisit; jangan diam-diam mengembalikan prediksi acak atau fake embeddings. Mock model hanya untuk test/dev dan harus berlabel demikian.
- Batch inferensi bila relevan; model di-load sekali saat startup, bukan di-load per request. Evaluasi memory CPU container sebelum deployment.

## 5. Path B — Guided AI Consultation (Reza)

**Stack wajib sesuai PRD awal:** `langchain` + `langchain-google-genai` + **Google Gemini** untuk prompt/orchestration dan **Pydantic v2** untuk *typed structured output*. Versi model Gemini harus dikonfigurasi lewat environment; jangan menganggap satu model ID pasti tersedia di akun. Pola implementasi: protobuf request → builder konteks tepercaya → LangChain Gemini typed assessment/generation → Pydantic validation → deterministic/semantic grounding → mapping ke generated protobuf → gRPC response. LLM hanya mengusulkan; aturan batas pertanyaan, allowlist ID, angka SQL, dan quote wajib dijalankan oleh kode Python. Tidak perlu membuat autonomous agent atau retrieval DB Python.

### 5.1 Input trust model dan peran data

Snapshot Go membawa: isu, periode, statistik SQL, sejumlah opini representatif, profil shop, memory yang **sudah dikonfirmasi owner**, dan prior archived solution. Messages membawa percakapan sesi, termasuk pesan owner yang sedang diproses.

Pisahkan secara ketat:

- **Observasi:** angka dari `context.stats`, fakta yang benar-benar tertulis pada evidence, owner statement yang diatribusikan.
- **Hipotesis:** kemungkinan penyebab yang belum terbukti.
- **Asumsi:** kondisi yang dibutuhkan suatu tindakan, tetapi tidak dikonfirmasi.
- **Rekomendasi:** saran tindakan, bukan klaim bahwa owner sudah melaksanakannya.

`stats` mencakup keseluruhan opini dalam issue yang dihitung SQL, sementara `evidence` adalah **sampel**. Tidak boleh menganggap sampel mewakili semua ulasan atau membuat persentase subtopik yang tidak tersedia dalam SQL.

### 5.2 Decision policy untuk satu turn

1. Parse event dan validasi data wajib; gunakan `question_count`, `asked_topics`, `skipped_topics`, seluruh messages, dan `previous_solution` yang dikirim Go.
2. Nilai kecukupan konteks: bukti masalah, operasional owner, constraint staf/biaya/waktu, dan potensi nilai informasi tambahan. Gunakan structured LLM assessment bila dibutuhkan, lalu **deterministic guardrails** sebelum response.
3. Jika `GENERATE_NOW`: generate solusi sekarang **hanya sepanjang dapat memenuhi kontrak lima bagian dengan bukti yang ada**; jika mustahil secara valid, gunakan explicit gRPC failure, **jangan fabrikasi kutipan**.
4. Jika batas pertanyaan satu siklus tercapai: generate solusi, bukan pertanyaan lagi. Target UX 2–4 pertanyaan, bukan minimum wajib; bisa 0–1 pertanyaan saat data cukup.
5. Jika konteks sudah cukup atau jawaban tambahan tidak akan mengubah rekomendasi secara bermakna: langsung solusi.
6. Jika ada gap penting dan masih di bawah batas: ajukan **satu pertanyaan pendek, spesifik, bernilai tinggi**. Hindari multi-pertanyaan tersembunyi.
7. `SKIP_QUESTION`: jangan mengulang pertanyaan yang dilewati. Topik serupa baru boleh dipakai jika kebutuhan informasinya berbeda dan secara jelas justified.
8. `CONTINUE_DISCUSSION`: gunakan permintaan owner untuk merevisi/menajamkan solusi lama; boleh bertanya atau memberi revisi sesuai data, tetapi **Go sekarang belum reset `question_count` per siklus** (lihat Blockers).

`topic_key` spesifik, stabil, panjang maksimum 100 byte (sesuai validasi Go). Jangan bertanya tentang informasi yang sudah ada di snapshot/memory/messages. Bila owner tidak tahu, hasilkan alternatif asumsi rendah risiko atau pertanyaan lain yang lebih berguna.

### 5.3 Lima komponen wajib dalam `Solution`

Semua lima repeated fields **harus minimal memiliki satu elemen** agar lolos validator Go saat ini.

1. **`findings[]`:** fakta/evidence yang tersedia, dengan `statement` yang dapat ditelusuri. Jangan menggunakan bahasa sebab-akibat jika baru ada korelasi atau satu ulasan.
2. **`customer_evidence[]`:** `opinion_id` dari `context.evidence`, `quote` verbatim dari `evidence_text` atau `full_feedback`, dan `note` singkat.
3. **`hypotheses[]`:** kemungkinan penyebab dan apa yang belum diketahui. `confidence` adalah label kualitatif *jika ada dasar evaluasi*, bukan probabilitas kalibrasi; kosongkan atau gunakan label berhati-hati bila tidak dapat dipertanggungjawabkan.
4. **`recommendations[]`:** prioritaskan satu langkah konkret, boleh sampai dua alternatif secara kebijakan UX; setiap tindakan tidak boleh mengasumsikan sumber daya tak terbukti tersedia. `priority` label deskriptif, bukan turunan otomatis severity.
5. **`success_indicators[]`:** cara owner mengecek hasil secara manual, dengan `metric`; `target` dan `timeframe` hanya konkret jika diberikan owner atau jelas dinyatakan sebagai **usulan**, bukan data historis atau janji dampak. Tidak ada action tracking oleh ATLAS.

**Contoh konseptual:** Ulasan menyebut antrean lama → finding: “Satu opini menyebut antrean lambat,” evidence kutipan aslinya, hypothesis: “Alur penerimaan pesanan mungkin menjadi bottleneck,” recommendation: “Amati antrean pada dua periode sibuk,” indicator: “Durasi antrean sebelum pesanan dicatat.” Jangan ubah satu opini menjadi “mayoritas pelanggan mengeluh.”

### 5.4 Claim-level grounding dan validasi

Go saat ini memvalidasi **keberadaan ID** dan field wajib; Go **belum mengecek apakah arti klaim benar-benar didukung sumber yang dikutip**. Python wajib mengimplementasikan lapisan tambahan:

1. **Source registry:** daftar ID yang sah dari `context.evidence[].opinion_id`, `messages` dengan `role=MESSAGE_ROLE_OWNER`, serta `context.memories[].id`.
2. **Claim-to-reference check:** setiap pernyataan faktual harus memiliki referensi yang benar-benar menjelaskan pernyataan tersebut. Ketersediaan ID saja tidak cukup.
3. **Quote check:** `CustomerEvidence.quote` diverifikasi sebagai substring verbatim dari sumber yang dirujuk, bukan parafrasa; paling aman mengambil quote langsung dari request dan membatasi LLM memilih opinion ID.
4. **Numbers check:** hanya angka SQL dan angka yang benar-benar tertulis di evidence/owner statements; denominators dan cakupan jelas. Tidak membuat ROI/impact % tanpa data.
5. **Contradiction/entailment review:** gunakan aturan deterministik dan, jika diperlukan, model sebagai pemeriksa kedua; bila unsupported, *remove or rewrite as explicit hypothesis*, lalu validasi ulang.
6. **Guardrails final:** tepat satu field dari `oneof result`, tidak ada struktur yang kosong/invalid, panjang field aman, semua IDs sah, hanya lima field asli proto.

Jika `context.evidence` kosong, **jangan mengisi kutipan palsu**. Saat ini validator Go mensyaratkan `customer_evidence` nonempty pada solusi. Perlu keputusan Suthe: revisi validator/kontrak agar mode no-evidence legal, atau blokir solution dengan error yang jelas; **jangan diam-diam memenuhi validator dengan data fiktif**.

### 5.5 Business Memory candidates

- Hanya mengusulkan fakta bisnis **durable** dari pernyataan owner, misalnya “Jumlah barista shift sore dua orang.” Tidak mengambil fakta dari review pelanggan, output AI, hipotesis, atau rekomendasi.
- Setiap `MemoryCandidate` wajib memuat `fact_text` nonempty dan minimal satu `source_owner_message_ids` yang valid.
- Kandidat boleh dikembalikan saat `next_question` **maupun** `solution`; Go menyimpan `memory_candidates` pada AI message tersebut.
- Jangan mengaktifkan memory. Go melakukan konfirmasi saat Save; saat ini Go juga menerima fakta owner-sourced yang dikonfirmasi walaupun tidak identik dengan kandidat AI, selama IDs owner valid. UI bisa memilih membatasi sesuai kebijakan produk, tetapi Python tidak bisa mengasumsikan Go mewajibkan pencocokan candidate.
- Fakta sementara, rahasia, identitas pelanggan, dan interpretasi ambigu tidak boleh dipromosikan otomatis menjadi memory.

## 6. Persistence, state, dan tanggung jawab Go

Tabel Go relevan: `consultation_sessions`, `consultation_messages`, `business_memories`. Solution tersimpan di `consultation_messages.solution_data` JSONB bersama `revision`, `model_id`, `prompt_version`, dan `memory_candidates`. **Tidak** membuat tabel `consultation_insights` terpisah.

State: `COLLECTING_CONTEXT` → `SOLUTION_READY` → `ARCHIVED`, dengan event `START`, `OWNER_REPLY`, `SKIP_QUESTION`, `GENERATE_NOW`, `CONTINUE_DISCUSSION`. Go menangani session locking, `request_id` idempotency, version, dan transaction. `Save` mengarsipkan solusi pilihan serta membuat confirmed Business Memories dalam transaksi, **tanpa panggilan Python**. Archived bersifat read-only, bukan menyatakan issue selesai.

Backend saat dokumen dibuat masih menyimpan `shop_id` di `consultation_sessions` (bukan di `issues`); user telah memutuskan desain ideal tanpa `consultation_sessions.shop_id`. Ini **pending refactor Go** jika tim jadi menerapkannya; perubahan tidak boleh hanya menghapus kolom migration tanpa memperbarui Go repository/service/authorization. Isolasi shop tetap harus divalidasi melalui kepemilikan opinion–feedback.

## 7. Docker, config, keamanan, dan dependency

### 7.1 Kondisi yang telah diverifikasi

Local Windows environment user sudah berhasil menjalankan `docker --version`, `docker compose version`, `docker run --rm hello-world`, dan **build image `atlas-ai-service:dev` dari Dockerfile berbasis Python 3.11-slim**. Ini memverifikasi Docker daemon, build image, dan dependency installation dasar, **belum** menguji server gRPC hidup.

### 7.2 Development runtime

- `Dockerfile`: Python 3.11 slim, install package dari `pyproject.toml`, `CMD ["python", "-m", "atlas_ai.server"]`. `server.py` belum ada pada tahap awal; implementasikan sebelum `docker compose up`.
- Satu service bernama `atlas-ai` / `atlas-ai-service`; listen `0.0.0.0:50051` dalam container. Port host sebaiknya bind ke `127.0.0.1:50051` pada development agar tidak terbuka ke LAN.
- Jika Go berada di Docker network yang sama: `AI_GRPC_ADDR=atlas-ai:50051`. Jika Go berjalan native Windows dan container publish port: `AI_GRPC_ADDR=localhost:50051`.
- Environment Python minimal: `GRPC_HOST=0.0.0.0`, `GRPC_PORT=50051`, `AI_GRPC_TOKEN`, `GEMINI_API_KEY` / nama env sesuai SDK, `GEMINI_MODEL`, `LOG_LEVEL`, konfigurasi threshold/model checkpoints. Dokumentasikan secara eksplisit di `.env.example` tanpa nilai rahasia.
- Metadata Go: `authorization: Bearer <token>` dan `x-request-id`. Implementasikan unary server interceptor untuk validasi bearer ketika token dikonfigurasi; gunakan comparison constant-time. Jangan log token, API key, konten feedback/owner mentah. Untuk deployment gunakan TLS/private service network.
- `AI_CONSULTATION_TIMEOUT` Go default **60 detik**, `AI_FEEDBACK_TIMEOUT` default **30 detik**. Python harus patuh deadline/cancellation, punya model/provider timeout internal lebih pendek, dan tidak melakukan retry LLM tanpa kontrol batas. Go punya retry untuk feedback worker; Python harus aman dipanggil berulang.
- Tambahkan `.dockerignore` (`.git`, `.venv`, `__pycache__`, `.env`, caches, checkpoints lokal besar yang tidak di-copy sengaja), dan `.gitignore` secrets/artifacts.
- `pyproject.toml` sudah menggunakan `setuptools` dan optional dependency `dev`; pertahankan agar starter build tetap kompatibel. Sebelum deployment, **lock versi** (mis. `uv.lock` atau constraints hasil resolver) supaya image konsisten. Jangan install stack model Ucan besar sampai checkpoint dan CPU/GPU target dikonfirmasi; boleh ada extras `nlp`.
- Build multi-stage dan user non-root adalah hardening lanjutan, bukan blocker smoke-test pertama.

### 7.3 Proteksi prompt injection dan privasi data

- Teks pelanggan, pesan owner, dan solusi arsip merupakan **untrusted content**, walaupun dikirim lewat Go. Instruksi di dalamnya tidak boleh mengganti system policy, memerintahkan pemanggilan tool, atau mendorong bocornya API key.
- Pisahkan instruksi aplikasi dari data dengan *structured prompt inputs* dan boundary/label sumber; delimiter membantu keterbacaan, **bukan satu-satunya perlindungan**. Batasi akses ke tool; untuk MVP tidak ada tool eksekusi action atau SQL dari LLM.
- Sanitasi/potong sesuai anggaran konteks, jangan menyimpan raw customer text pada trace/log eksternal secara default. Test dengan contoh review yang berisi kalimat seperti “abaikan instruksi sebelumnya”.

### 7.4 Observability

Structured logs: `rpc_name`, `request_id`, `session_id`/`feedback_id` (jika tersedia), `model_id`, `prompt_version`, latency, status, token usage estimasi/aktual bila API mengembalikannya. Metric minimum: volume, error, latency p50/p95, outcome question vs solution, jumlah pertanyaan, invalid grounding/rejected output, estimasi biaya inference. LangSmith tracing optional dan **off by default** sampai kebijakan redaction data berlaku.

## 8. Testing dan evaluasi wajib

### 8.1 Contract tests dan runtime

- Protobuf generated Python dari file yang sama dengan Go; import/serialize/deserialize semua dua jenis request dan response.
- gRPC server benar-benar listen pada 50051; test dengan in-process client/server atau container smoke test.
- Interceptor auth: valid token sukses, invalid token `UNAUTHENTICATED`.
- Event yang tidak valid → `INVALID_ARGUMENT`; model unavailable → `UNAVAILABLE`, deadline → `DEADLINE_EXCEEDED` bila sesuai; log aman.
- Ulangi request yang sama (idempotency): Python tidak menulis state; respons stochastic bisa berbeda, tetapi tidak menimbulkan side effect di Python. Go bertanggung jawab deduplikasi persisten.

### 8.2 NLP/ML tests

- Multi-aspect ulasan: positif + negatif dalam satu kalimat → dua opini yang benar.
- Input teks kosong/tidak actionable → `opinions=[]` tanpa fabricated content.
- Quote persis ada dalam teks asal, rating tidak menghapus sentimen campuran.
- Embedding dimensi konsisten dan finite, model name benar; mismatch model/dimensi kandidat difilter.
- Existing issue sama shop masuk cluster yang sesuai; data shop lain tidak bisa dipilih; ketidakcocokan membuat judul isu baru.
- Severity `SEVERE`/`NOT_SEVERE` dengan set evaluasi berlabel; laporkan confusion matrix/precision/recall sesuai label data. Threshold similarity ditentukan lewat validasi, bukan asumsi.

### 8.3 Consultation tests

- `START`: question **atau** solution sesuai kecukupan bukti.
- `OWNER_REPLY`: pesan baru sudah ada di `messages` dan tidak diproses ganda.
- `SKIP_QUESTION`: tidak mengulang topik yang dilewati.
- `GENERATE_NOW`: tidak mengembalikan pertanyaan.
- `CONTINUE_DISCUSSION`: revisi solusi atau satu pertanyaan relevan; cek issue question-count lintas siklus dengan Suthe.
- Batas pertanyaan; tepat satu question per response; no repeated semantic question.
- Semua lima array solusi nonempty pada happy path; quote verbatim; ID sumber valid; entailment klaim diuji dengan contoh unsupported ID valid; tidak membuat persentase/statistik palsu.
- Owner fact bisa menjadi candidate dengan owner message IDs; AI hypothesis dan customer review **tidak** menjadi memory.
- Test 0 evidence, 1 evidence, konflik owner–review, malicious prompt injection dalam review, large history, provider timeout/rate limit.
- Test integrasi memakai fake/mock AI yang jelas dipisahkan dari mode real; tidak boleh mengklaim evaluation score atau live accuracy tanpa eksperimen nyata.

## 9. Acceptance criteria / Definition of Done

### Milestone M1 — Skeleton + Contract (prioritas pertama)

- [ ] Repo Python rapi dengan `src/atlas_ai/nlp`, `src/atlas_ai/consultation`, dan shared `server.py`.
- [ ] `ai.proto` persis dari Go, commit sumber didokumentasikan, stubs generated importable.
- [ ] `AIServiceServicer` menangani dua RPC dengan **mock response yang diberi label development**, dan contract tests hijau.
- [ ] Docker image berhasil build, container benar-benar listening, test gRPC Go → Python berhasil.
- [ ] Auth metadata, timeout, log minimal, `.env.example`, `.dockerignore`, README ada.

### Milestone M2 — Path B Consultation

- [ ] Structured LLM output → mapping ke protobuf `Solution` / `NextQuestion` yang tepat.
- [ ] Lima bagian grounded, sources valid+semantically relevant, verbatim quotes.
- [ ] Event + question guardrails, memory candidates owner-only.
- [ ] Unit tests dengan mocked LLM plus minimal satu real Gemini smoke test terkontrol.

### Milestone M3 — Path A NLP/ML

- [ ] Pipeline Ucan menggunakan checkpoint/model yang benar-benar tersedia.
- [ ] Aspect/sentiment, severity biner, multilingual E5, clustering shop-scoped.
- [ ] Output Go-valid, tests untuk kontrak, multi-aspect, quotes, dimensions, cluster.
- [ ] Resource/latency sesuai container target; model load sekali.

### Milestone M4 — End-to-end

- [ ] QR feedback → Go worker → Python NLP → Go persistence → planet.
- [ ] Planet → START → questions/solution → Save/Archive → history, tanpa memanggil Python saat Save.
- [ ] Test invalid response, recovery/retry, race/duplicate request, multi-shop isolation.
- [ ] Laporan hasil test yang **benar-benar dijalankan** dan known limitations tertulis.

## 10. Cross-repository blockers / decisions bersama Suthe

**BLOCKER A — Missing evidence vs wajib customer evidence**  
`validateSolution` Go saat ini mensyaratkan `len(customer_evidence)>0`. Namun `ContextSnapshot.evidence` bisa kosong. Python tidak boleh membuat fake quote. Tim harus menentukan kontrak mode tanpa evidence atau memblokir generate secara eksplisit. Jangan menyatakan sudah tertangani.

**BLOCKER B — Batas empat pertanyaan per siklus**  
Saat ini `question_count` di Go bertambah setiap pertanyaan dan tidak terlihat reset pada `CONTINUE_DISCUSSION`. Pilihan: Suthe reset counter dan asked-topic scope pada siklus baru, atau sepakati batas sepanjang sesi. Target produk adalah maksimum 4 **per siklus**; Python tidak bisa mengubah state Go.

**DECISION C — `consultation_sessions.shop_id`**  
Go migration/service sekarang masih memakainya. Keputusan desain terakhir user meniadakannya. Suthe perlu review refactor ownership/auth di Go dan implikasi issue tanpa opini. Tidak menghalangi Python memakai `ai.proto` saat ini.

**DECISION D — Validasi makna referensi**  
Go memvalidasi ID sumber ada, **bukan entailment**. Python mengerjakan validator claim-level; tim sebaiknya mendiskusikan apakah Go perlu perlindungan tambahan untuk quote non-verbatim saat menerima AI result.

**DECISION E — Runtime NLP**  
Tentukan checkpoint ABSA dan classifier, lisensi, bentuk deployment CPU/GPU, dan embedding threshold berdasar evaluasi. PRD tidak menganggap model eksperimental sudah siap inference production.

## 11. Instruksi untuk Codex / coding agent

1. **Baca `proto/atlas/ai/v1/ai.proto` yang disalin dari Go lebih dulu**, serta mapping preservasi kebutuhan awal di §2.5; implementasikan signature-nya persis dan jangan mengarang RPC/field.
2. **Implementasikan perubahan bertahap**: scaffold → proto compile → gRPC mock → Docker smoke test → Consultation → NLP → integration. Jangan menghasilkan satu PR besar tanpa test.
3. **Jangan membuat FastAPI, koneksi PostgreSQL Python, status/session manager baru, atau alter migration Go**.
4. **Jangan mengubah keputusan bisnis atau stack Guided Consultation**: LangChain + Gemini + Pydantic internal, read-only archived history, no action tracking, no autonomous business actions, no auto-confirm memory.
5. **Tidak ada fabricated evidence**, hardcoded fake LLM success, persentase tidak berdasar, confidence probabilistik palsu, atau real model results di test mock.
6. Pada setiap tahap jalankan tests terkait dan tulis: file yang diubah, perintah test yang benar-benar dieksekusi, hasil, known blockers, serta dependencies baru. Jangan menyatakan integration berhasil kalau Go belum diuji.
7. Perubahan kontrak lintas repo dan blocker di §10 harus diusulkan kepada Suthe sebagai perubahan tersendiri, **bukan diatasi dengan proto fork**.

---

## Lampiran: referensi implementasi Go yang diaudit

- [Kontrak proto aktual](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/proto/atlas/ai/v1/ai.proto)
- [Mapper Go↔proto](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/aiclient/mapping.go)
- [AI gRPC client Go](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/aiclient/client.go)
- [Validation consultation Go](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/service/consultation_validate.go)
- [Consultation service Go](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/service/consultation.go)
- [Feedback worker Go](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/internal/service/feedback.go)
- [SQL migration consultation Go](https://github.com/Elioth-Gray/atlas-backend-core/blob/main/migrations/000003_create_consultation_tables.up.sql)

**Catatan:** Referensi link menunjuk `main` dan bisa berubah; salinan `ai.proto` harus dicatat commit sumbernya agar reproducible.

### Catatan audit v2.1 terhadap dokumen awal

- **Dipertahankan:** dua jalur AI, target model ABSA/XLM-R/E5 (sesuai checkpoint yang tersedia), context sufficiency, question-limit, 5 sections, validation grounding, memory owner-only, stateless Go↔Python, observability, prompt-injection defenses.
- **Diperbaiki karena konflik dengan Go:** nama service/RPC/messages; bentuk output `Solution`; enum severity; candidate clusters menggunakan `candidate_issues`/`candidate_vectors`; owner turn sudah termasuk di `messages`.
- **Gap asli yang sengaja tidak disembunyikan:** subfield UX tidak tersedia di protobuf (lihat §2.5), no-evidence response, pertanyaan per cycle, ownership scope `shop_id`, semantic grounding yang belum divalidasi Go.
- **Yang dikembalikan eksplisit pada v2.1 setelah audit:** penggunaan wajib LangChain + Google Gemini + Pydantic, mapping field lama→baru, dan proteksi prompt injection sebagai requirement tersendiri.
