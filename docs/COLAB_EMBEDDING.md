# Embedding Colab đã chạy

**A100-SXM4 80 GB · Qwen3-Embedding-8B · BF16 · 4.096 chiều · L2.**
Revision và cấu hình cố định ở [colab_embedding.json](../configs/colab_embedding.json).
Model cho phép 32k context; triển khai này giới hạn **8.192 token**, batch GPU 4,
không cắt input âm thầm. E5 Day 6 giữ làm baseline; Qwen dùng index riêng.

API local: `http://127.0.0.1:8765/v1/embeddings`.
Bridge gọi kernel Colab qua kết nối đã xác thực, không public tunnel. Key API
local nằm trong `.runtime/bridge-key` đã gitignore; OAuth ở Windows Credential Manager.

## Dùng trong Python

```python
from src.colab_embeddings import ColabEmbedder

embedder = ColabEmbedder()
vectors = embedder.embed_documents(["Nội dung tài liệu..."])
query_vector = embedder.embed_query("Câu hỏi của bạn?")
```

API gần định dạng OpenAI embeddings; nếu gọi HTTP trực tiếp, truyền
`Authorization: Bearer <local-key>` và **`input_type: "query"`** cho câu hỏi
(`document` là mặc định). Không dùng E5 prefix `passage:`/`query:` cho Qwen.
Cache phân biệt loại input, model revision và cấu hình instruction.

## Khởi động lại sau khi GPU đã dừng

Từ PowerShell tại repo (tạo GPU sẽ dùng compute units):

```powershell
.\scripts\colab.ps1 sessions
# Chỉ tạo nếu day07-embedding chưa chạy:
.\scripts\colab.ps1 new -s day07-embedding --gpu A100 --high-mem
.\scripts\colab.ps1 exec -s day07-embedding -f /mnt/c/PROJECT/_VINstudy/K5-L3B-DAY07-2A202603824-PhamQuangTuan-RetrievalContext/scripts/colab_setup.py --timeout 600
.\scripts\start_colab_bridge.ps1
```

Nếu cần chạy script trên **kernel đã tồn tại**, dùng `scripts/colab_rpc.py` trong
venv CLI: adapter tạo frontend session mới, giữ nguyên kernel/model. Đây là cách
khắc phục việc CLI 0.8.0 bị kẹt khi dùng lại frontend session ID trên runtime này.
Không sửa thủ công file session/token của CLI.

## Chi phí và dừng

Lần chạy này Colab báo **8,88 CU/giờ**, không phải giá cố định cho mọi lần cấp GPU.
Bridge tự giải phóng GPU sau **30 phút không có request** nếu tiến trình bridge
còn chạy. Khi không cần nữa, dừng chủ động:

```powershell
.\scripts\stop_colab_bridge.ps1
```

Nếu runtime mất kết nối/bị Colab thu hồi, input chưa cache trả lỗi 503; không sinh
vector giả hay tự cấp GPU khác. Cache đã có vẫn dùng được. `/health` kiểm tra
bridge local; không phải cam kết runtime Colab luôn sống.

## Minh chứng

- [Phần cứng/model thực](../results/embedding_runtime.json).
- Chất lượng retrieval/agent được chấm bằng benchmark trong `results/experiments/`.
- Các smoke test kết nối và script cấu hình máy cá nhân chỉ giữ local, không nộp.

## Batch chuẩn bị corpus Day 7

Các script chạy trong cùng kernel sau `colab_setup.py`:

1. Upload `data/processed/chunks_native.json` thành `/content/day07_chunks.json`.
2. Chạy `scripts/colab_embed_dataset.py`; download `/content/day07_qwen_vectors.npz`
   và `/content/day07_qwen_manifest.json` về `data/embeddings/qwen_native.npz`
   và `data/embeddings/qwen_manifest.json`.
3. Chạy `scripts/colab_vision_setup.py`, rồi `scripts/colab_vision_batch.py`;
   download `/content/wdr_visual_descriptions.json` về
   `data/processed/visual_descriptions.json`.
4. `py -m src.indexing finalize`, rồi `py -m src.indexing build`.

Dùng `py scripts/run_colab_job.py scripts/colab_vision_batch.py --log .runtime/vision.log`
cho batch lâu. Bridge ghi nhận hoạt động của job thật để budget guard không dừng
GPU giữa job; không can thiệp chính sách liveness của Colab. Runtime/kernel vẫn
có thể bị Colab thu hồi, job giữ checkpoint theo trang/batch để resume.

Lỗi DNS WSL được xử lý trong venv CLI bằng fallback sang DNS Windows nếu DNS WSL
thất bại; không đổi DNS hệ thống, không tắt kiểm tra chứng chỉ TLS.
