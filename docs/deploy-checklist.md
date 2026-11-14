# Checklist Deploy — ATLAS AI Service (pilot/staging)

Status: **siap untuk pilot/staging** setelah semua kotak di bawah tercentang. Untuk produksi penuh,
tunggu evaluasi final (Fase 3) dan pemantauan berjalan.

## 1. Build image

- [ ] Build target produksi dari commit yang sudah di-review:
  ```powershell
  docker build --target prod -t atlas-ai-service:<versi> .
  ```
  Target `prod`:
  - Dependency persis dari `uv.lock`, tanpa pytest/ruff.
  - Berjalan sebagai user non-root `atlas` (uid 10001).
  - Model E5 dibundel dengan cek sha256, tanpa download saat runtime.
  - Punya `HEALTHCHECK` bawaan.
- [ ] Suite lulus di image dev:
  ```powershell
  docker compose build atlas-ai
  docker run --rm --network none -v "${PWD}:/app" -w /app atlas-ai-service:dev python -m pytest -q
  ```
- [ ] Tag image dengan versi (bukan hanya `latest`), supaya rollback cukup dengan menjalankan tag sebelumnya.

## 2. Secret dan konfigurasi

| Variabel | Wajib | Catatan |
|---|---|---|
| `GRPC_AUTH_TOKEN` | ya | **Harus sama** dengan `AI_GRPC_TOKEN` di Go. Acak ≥32 byte |
| `OPENAI_API_KEY` | ya | Dari secret store platform; tidak boleh masuk image, repo, atau log |
| `CONSULTATION_MODEL` | ya | `gpt-6.1-terra` (Path B) |
| `FEEDBACK_MODEL` | ya | `gpt-6-luna` (Path A). Tidak pernah fallback ke model Path B |
| `*_PRICE_*_PER_1M_USD` | tidak | Harga resmi untuk metrik `cost_usd` di log |
| `GRPC_TLS_CERT_FILE` / `GRPC_TLS_KEY_FILE` | kondisional | Lihat §3 |

- [ ] Cek akses model sebelum go-live (`/v1/models` berisi kedua model). Kunci yang salah atau tidak punya
  akses menghasilkan `FAILED_PRECONDITION`, bukan retry tanpa akhir.
- [ ] Nilai `.env` tidak pernah dicetak ke log CI/CD.

## 3. Jaringan

Pilih salah satu:

- [ ] **Jaringan privat (default).** Port 50051 hanya bisa diakses container Go (network Docker/VPC yang
  sama) dan tidak dipublikasikan ke internet. Token tetap wajib di setiap RPC.
- [ ] **TLS.** Isi `GRPC_TLS_CERT_FILE` + `GRPC_TLS_KEY_FILE` di AI dan `AI_GRPC_TLS=true` di Go.
  - Go memverifikasi sertifikat dengan **root CA sistem**, tanpa CA kustom (`internal/grpcclient/client.go`).
    Jadi sertifikat harus dari CA publik, atau CA internal yang dipasang di trust store container Go.
  - Nama di sertifikat harus cocok dengan host di `AI_GRPC_ADDR`.
  - Untuk healthcheck container, isi `GRPC_TLS_SERVER_NAME` dengan nama DNS sertifikat.

Endpoint health (`grpc.health.v1.Health`) bisa diakses tanpa token. Endpoint ini hanya melaporkan
status `SERVING`.

## 4. Sumber daya dan batas

- [ ] RAM ±1 GB per instance (E5 + ONNX Runtime ±0,5 GB saat idle). CPU 1–2 vCPU.
- [ ] Ukuran image ±1,4 GB. Waktu start sampai `SERVING` beberapa detik karena E5 dimuat sekali di awal.
- [ ] Batas waktu selaras dengan Go:

  | Path | Go | AI |
  |---|---|---|
  | Feedback | `AI_FEEDBACK_TIMEOUT=30s` | Ekstraksi ≤20 dtk + judge ≤10 dtk, selalu dibatasi deadline RPC (margin 0,5 dtk) |
  | Konsultasi | `AI_CONSULTATION_TIMEOUT=60s` | Total 45 dtk (`CONSULTATION_TIMEOUT_SECONDS`) |

- [ ] Konkurensi: `GRPC_MAX_WORKERS=8` dan `GRPC_MAX_CONCURRENT_RPCS=32`. Di atas batas ini server langsung
  menjawab `RESOURCE_EXHAUSTED`. Worker feedback Go (`FEEDBACK_WORKER_CONCURRENCY=2`) akan retry.
- [ ] Ukuran pesan maksimal 4 MiB (`GRPC_MAX_MESSAGE_BYTES`), sama dengan default Go.
- [ ] Rate limit akun OpenAI cukup untuk beban pilot. Error 429 dipetakan ke `UNAVAILABLE` tanpa retry
  di AI (Go yang mengatur retry feedback).

## 5. Observabilitas

- [ ] Log `atlas_ai.rpc` berisi satu baris JSON per RPC:
  - `request_id` (dari header `x-request-id` Go), `rpc`, `status`, `latency_ms`;
  - `llm_calls[]` (stage, model, attempts, token, `cost_usd`, latensi);
  - `tokens`, `cost_usd`, dan `spans` (waterfall per tahap).
- [ ] Log **tidak** pernah berisi teks pelanggan, pesan owner, output model, nama customer, atau secret.
- [ ] Tracing eksternal mati: `LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`.
- [ ] Telemetri ONNX Runtime (Microsoft 1DS) mati: `ORT_DISABLE_TELEMETRY=1`. Nilai ini sudah diset di
  image dan di kode.
- [ ] Alert minimal: rasio `status != OK` per RPC, p95 `latency_ms`, dan lonjakan `cost_usd`.

## 6. Data yang dikirim ke OpenAI

| Path | Yang dikirim | Yang tidak dikirim |
|---|---|---|
| A (feedback) | Teks feedback, rating, profil toko (nama, deskripsi, kategori, skala). Untuk judge: teks opini keluhan serta ID dan judul kandidat issue | Nama customer, ID feedback/toko, vektor |
| B (konsultasi) | Profil toko, ringkasan isu, statistik agregat, kutipan opini pelanggan, riwayat pesan owner, memori bisnis yang sudah dikonfirmasi, solusi sebelumnya, serta ID sumber (UUID) untuk referensi | Nama customer, token, secret |

- [ ] Kebijakan data API OpenAI terbaru (penggunaan untuk training dan retensi) sudah dicek dan
  disetujui tim.
- [ ] Pemilik toko pilot sudah diberi tahu bahwa feedback diproses oleh layanan AI pihak ketiga.

## 7. Verifikasi setelah deploy

- [ ] Container `healthy` (`python -m atlas_ai.healthcheck` → exit 0).
- [ ] Log Go menampilkan `ai service configured addr=...`.
- [ ] Satu feedback uji → `feedback_jobs.status = DONE`, opini dan vektor tersimpan, planet terbentuk dari keluhan.
- [ ] Satu konsultasi uji sampai Solution → Save → `ARCHIVED`.
- [ ] Log RPC uji punya `status=OK`, token terisi, dan `cost_usd` masuk akal.

## 8. Batasan yang diketahui

- Threshold matching (`FEEDBACK_MATCH_THRESHOLD=0.85`) dan batas waktu masih *provisional*. Kalibrasi
  dilakukan di Fase 3 dengan dataset bakery yang direview tim.
- Angka kualitas dan latensi Gemini lama tidak berlaku untuk OpenAI dan diukur ulang di Fase 3.
- Gateway hanya retry sekali untuk 5xx/koneksi putus. Retry lebih lanjut untuk feedback dilakukan Go
  (maks. 8 percobaan). Konsultasi yang gagal harus diulang owner.
