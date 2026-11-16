# Code Day 7

- `ingest.py`: trích PDF và metadata trang/vùng, schema tương thích example.
- `indexing.py`: ghép mô tả trực quan, kiểm tra hash/vector, build/restore Qdrant.
- `retrieval.py`: E5 Day 6, Qwen Colab, BM25 và hybrid RRF.
- `agent.py`: model thật chọn tool, quản lý context/citation/abstention và trace lỗi.
- `experiments.py`: dev, selection lock, test và hỗ trợ audit offline; gold chỉ vào evaluator.
- `colab_bridge.py`, `colab_embeddings.py`: API loopback, cache, adapter và budget guard.
- `pipeline.py`: contract evidence/multi-hop và mẫu số nâng cao.
- `run.py`: lệnh verify/search/answer/restore.

`starter/` giữ nguyên làm đối chiếu bài mẫu; kết quả mock/unit fixture không được
dùng làm benchmark agent thật.
