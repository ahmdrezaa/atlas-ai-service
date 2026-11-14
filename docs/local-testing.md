# Tes Manual Lokal: AI + Go + Frontend (Docker)

Panduan ini menjalankan ketiga repo ATLAS di satu laptop Windows dengan Docker Desktop, lalu menguji
alur lengkap: feedback pelanggan → planet di Galaxy → konsultasi AI → solusi → Save.

```
Browser :5173 ──REST──> Go API :3333 ──gRPC (token)──> AI service atlas-ai:50051 ──> OpenAI
   (atlas-frontend)       (atlas-backend-core)            (repo ini)
                          └─ PostgreSQL + pgvector :5433, pgweb :8081
```

## 0. Prasyarat

- Docker Desktop (backend WSL2) sudah berjalan, dengan RAM Docker minimal ±6 GB.
- Git.
- Kunci API OpenAI dengan akses ke `gpt-6-luna` (Path A) dan `gpt-6.1-terra` (Path B).
- Disk kosong ±8 GB. Image AI ±1,5 GB karena model E5 ikut dibundel.

Susunan folder yang dipakai di panduan ini:

```
D:\atlas-ai-service      (repo ini)
D:\atlas-backend-core    git clone https://github.com/Elioth-Gray/atlas-backend-core
D:\atlas-frontend        git clone https://github.com/Elioth-Gray/atlas-frontend
```

## 1. Buat satu token rahasia untuk Go ↔ AI

Token yang sama dipakai di dua tempat: `AI_GRPC_TOKEN` di Go dan `GRPC_AUTH_TOKEN` di AI. Buat dari PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Buat juga satu nilai lagi untuk `JWT_SECRET` di Go dengan perintah yang sama. Jangan pernah commit
nilai ini.

## 2. Jalankan Go backend (pertama, karena dia yang membuat network Docker)

```powershell
cd D:\atlas-backend-core
copy .env.example .env
```

Isi di `.env` (sisanya biarkan default):

```
JWT_SECRET=<nilai acak langkah 1>
AI_GRPC_TOKEN=<token langkah 1>
DOCKER_AI_GRPC_ADDR=atlas-ai:50051
CORS_ALLOWED_ORIGINS=http://localhost:5173
```

`CORS_ALLOWED_ORIGINS` sudah ada di `.env.example`. Kalau `.env` dibuat dari nol, baris ini wajib
ditambahkan; tanpa baris ini signup dari browser gagal dengan "Terjadi kesalahan".

Jalankan:

```powershell
docker compose -f compose.dev.yaml up --build api db dbviewer
```

Service `fakeai` sengaja tidak ikut dijalankan karena kita memakai AI yang asli. Cek di log Go muncul
`ai service configured addr=atlas-ai:50051`.

- REST API: http://localhost:3333 (Swagger: http://localhost:3333/docs)
- pgweb (lihat isi database): http://localhost:8081

## 3. Jalankan AI service (repo ini)

```powershell
cd D:\atlas-ai-service
copy .env.example .env
```

Isi di `.env`:

```
GRPC_AUTH_TOKEN=<token yang SAMA dengan AI_GRPC_TOKEN di Go>
OPENAI_API_KEY=<kunci OpenAI>
CONSULTATION_MODEL=gpt-6.1-terra
FEEDBACK_MODEL=gpt-6-luna
```

Jalankan dengan overlay integrasi. Overlay ini menyambungkan container ke network Go dengan nama
`atlas-ai`:

```powershell
docker compose -f compose.yaml -f compose.integration.yaml up --build
```

Tunggu status `healthy`. Model E5 dimuat dulu, biasanya beberapa detik:

```powershell
docker inspect -f "{{.State.Health.Status}}" atlas-ai-service
```

Log yang benar: `ProcessFeedback components loaded`, lalu
`ATLAS AI gRPC server running on 0.0.0.0:50051 tls=False ...`.

## 4. Jalankan frontend

```powershell
cd D:\atlas-frontend
copy .env.example .env
```

Isi `.env` default sudah benar (`VITE_APP_NAME=ATLAS`, `VITE_BACKEND_URL=http://localhost:3333`). Jalankan:

```powershell
docker compose -f compose.dev.yaml up --build
```

Buka http://localhost:5173.

## 5. Skenario uji

| # | Langkah | Yang dicek |
|---|---|---|
| 1 | Sign up akun owner, lalu sign in | Masuk ke dashboard |
| 2 | Lengkapi profil toko di **Business Account**: nama, kategori bakery, skala | Profil tersimpan |
| 3 | Buka **Create QR** lalu salin link feedback (`/f/<shopId>`) | Link terbentuk |
| 4 | Buka link itu di tab incognito (sebagai pelanggan) dan kirim beberapa feedback (contoh di bawah) | Halaman sukses |
| 5 | Tunggu 5–30 detik (worker Go polling setiap 5 detik), lalu buka **Galaxy** | Planet muncul hanya dari keluhan; keluhan yang sama masuk planet yang sama |
| 6 | Buka satu planet → **mulai konsultasi** | AI bertanya (maks. 4 pertanyaan per siklus, 8 per sesi) |
| 7 | Jawab, lewati satu pertanyaan, lalu minta solusi sekarang | Solusi 5 bagian muncul; rekomendasi relevan dengan keluhan |
| 8 | **Save** konsultasi | Sesi menjadi arsip; fakta bisnis yang dikonfirmasi muncul di Business Memories |

Contoh feedback bakery. Kirim satu per satu; dua yang pertama sengaja masalah yang sama:

- "Croissantnya keras banget kayak kemarin sore, padahal belinya pagi."
- "Rotinya udah alot pas dibeli, ga fresh."
- "Isian coklatnya dikit banget, tapi kasirnya ramah."
- "Kue ulang tahun pesanan saya telat 1 jam dari jadwal."
- "Enak semua, mantap!" (positif: tidak membuat planet)

### Cara memeriksa hasilnya

**Database (pgweb, http://localhost:8081):**

- `feedback_jobs`: status `DONE`. Kalau `FAILED`, lihat `attempts`.
- `feedback_opinions` dan `opinion_vectors`: tiap opini punya embedding 384 dimensi.
- `issues`: satu baris per planet.
- `consultation_sessions` / `consultation_messages`: riwayat konsultasi.
- `business_memories`: fakta bisnis.

**Log AI.** Ada satu baris JSON per RPC, berisi ID, status, latensi, token, dan biaya, tanpa teks
pelanggan:

```powershell
docker logs -f atlas-ai-service 2>&1 | Select-String '"event": "rpc"'
```

Field penting: `rpc`, `request_id` (sama dengan `x-request-id` dari Go), `status`, `latency_ms`,
`tokens`, `cost_usd`, `llm_calls[].attempts`, dan `spans` (waterfall per tahap). Untuk menyimpan semua
record ke file, isi `ATLAS_TRACE_FILE=/tmp/atlas-trace.jsonl` di `.env`.

## 6. Troubleshooting

| Gejala | Penyebab umum | Perbaikan |
|---|---|---|
| Signup/login di FE: "Terjadi kesalahan", log Go `OPTIONS ... status=405` | `CORS_ALLOWED_ORIGINS` tidak ada di `.env` Go | Tambahkan `CORS_ALLOWED_ORIGINS=http://localhost:5173`, lalu restart Go |
| Log Go `UNAUTHENTICATED` | `AI_GRPC_TOKEN` (Go) ≠ `GRPC_AUTH_TOKEN` (AI) | Samakan token, lalu restart kedua service |
| Log Go `connection refused` / `no such host atlas-ai` | AI belum jalan atau tidak memakai overlay integrasi | Jalankan langkah 3 dengan `-f compose.integration.yaml` |
| `network atlas-backend-dev_default not found` saat start AI | Go belum dijalankan | Jalankan langkah 2 dulu |
| Status `FAILED_PRECONDITION` di log AI | `OPENAI_API_KEY` kosong/salah, nama model salah, atau akun tidak punya akses ke model | Cek `.env` AI; nama model harus persis |
| Semua feedback `UNAVAILABLE`, log AI `429 Too Many Requests` padahal volume kecil | Saldo kredit OpenAI habis (`insufficient_quota`) | Isi kredit di halaman billing OpenAI; job `PENDING` diproses ulang otomatis |
| Status `UNAVAILABLE` sesekali | OpenAI 5xx/429. 5xx sudah dicoba ulang 1x; feedback dicoba ulang Go sampai 8x | Tunggu, atau cek kuota/rate limit akun |
| `DEADLINE_EXCEEDED` di konsultasi | Model lambat; anggaran 45 dtk dari batas Go 60 dtk | Cek `latency_ms` per `llm_calls`; pertimbangkan `CONSULTATION_REASONING_EFFORT=low` |
| Feedback tetap `PENDING` | Worker Go belum polling, atau `DOCKER_AI_GRPC_ADDR` kosong | Cek log Go; isi `DOCKER_AI_GRPC_ADDR=atlas-ai:50051` |
| Port 3333 / 5173 / 5433 / 50051 bentrok | Ada aplikasi lain yang memakai port | Matikan aplikasi tersebut atau ubah mapping port |
| Dua container AI menjawab bergantian | Ada container lama lain dengan alias `atlas-ai` | `docker ps`, lalu hentikan container lama |

## 7. Menghentikan

```powershell
# di masing-masing folder repo
docker compose -f compose.dev.yaml down                          # Go / FE (tambah -v untuk hapus DB)
docker compose -f compose.yaml -f compose.integration.yaml down   # AI
```
