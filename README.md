# Day 7 — Agentic RAG

**Phạm Quang Tuấn · 2A202603824 · L3B**

PDF 276 trang → native text + mô tả bảng/hình → E5 Day 6 / Qwen3-Embedding-8B →
Qdrant local + BM25/RRF → model gọi retrieval thật → trả lời có nguồn hoặc từ chối.

[Báo cáo](report/individual/2A202603824_PhamQuangTuan.md) ·
[Nguồn và quyền](docs/DATA_ACCESS.md) ·
[Kết quả validation](results/submission_validation.json)

## Kiểm tra bản nộp

Python 3.12; chạy từ thư mục gốc bằng PowerShell:

```powershell
py -m pip install -r requirements.txt
py scripts/download_source.py
py -m src.indexing restore
py -m unittest discover -s tests -v
py scripts/validate_submission.py
py scripts/replay_metrics.py
```

`restore` dùng final vector NPZ, không gọi API hoặc tải model. PDF/crop/model weights,
cache/Qdrant và log debug không đưa vào Git. Manifest, chunk, final vectors, benchmark,
trace/nhãn/metrics và code được giữ để kiểm tra/tái lập. Nhãn AI không được giả là nhãn người chấm.

Qdrant local giữ file lock: không chạy `restore`, `search`, `answer` hoặc benchmark
ở hai tiến trình cùng lúc trên một đường dẫn index. Unit tests dùng store tạm riêng.

## Truy xuất và hỏi đáp

Lexical không cần GPU/API:

```powershell
py -m src.run search --method bm25 --question "Figure O.4 investment infusion innovation"
```

Dense E5 dùng model và venv có sẵn ở repo Day 6 bên cạnh; không tải lại.
Qwen query cần Colab session/model và bridge local theo [hướng dẫn](docs/COLAB_EMBEDDING.md).
Đặt `API_KEY` trong `.env` theo `.env.example` cho model trả lời:

```powershell
py -m src.run search --method dense_e5 --question "Chiến lược 3i gồm những gì?"
py -m src.run answer --question "Chiến lược 3i gồm những gì?" --output .runtime/answer.json
```

Model trả lời qua OpenProvider; gateway không hỗ trợ đầy đủ chuẩn OpenAI.
Adapter ghi việc chuyển tool result sang user-context và không giả kết quả khi lỗi.

## Tái chạy pipeline

```powershell
py -m src.ingest
py scripts/embed_e5.py
# Colab setup, upload corpus, batch Qwen và vision: xem docs/COLAB_EMBEDDING.md
py -m src.indexing finalize
py -m src.indexing build
py -m src.experiments dev
py -m src.experiments test
py scripts/make_report.py
py scripts/validate_submission.py
```

Không ghi đè run/selection đã khóa. Muốn làm thí nghiệm mới, dùng checkout/thư mục
kết quả mới và công bố version; kết quả sau khi xem test là replay, không phải holdout mới.

## Định dạng và bằng chứng

`data/chunks.json` giữ nguyên schema example: `id,text,source_id,source,page,metadata`.
`page` là trang PDF gốc 1-based; metadata giữ bbox, loại native/vision và model version.
Gold không vào index/agent. Multi-hop dùng `evidence_groups` (mọi nhóm bắt buộc).

- [Index manifest](data/index_manifest.json), [thống kê xử lý](data/processing_summary.json).
- [Benchmark cơ bản](data/benchmark/basic_questions.json), [nâng cao](data/benchmark/advanced_questions.json).
- `results/experiments/<config>/<suite>/<split>/`: tool trace, context, câu trả lời,
  citation, labels/reasons, metrics, usage và lỗi theo từng query.
- `py scripts/render_evidence.py` tạo gallery local từ PDF/bbox; không phát hành crop bên thứ ba.

Phần cơ bản tối đa 85 điểm; bonus nâng cao chỉ khi đủ yêu cầu và ≥90% test thực tế.
Không dùng kết quả mock hoặc bỏ câu lỗi để báo đạt. Đề gốc: [GUIDE](GUIDE.md),
[BENCHMARK](BENCHMARK.md), [ADVANCED](ADVANCED.md), [SUBMISSION](SUBMISSION.md).
