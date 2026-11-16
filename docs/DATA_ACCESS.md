# Nguồn, quyền và cách tái tạo

Corpus: World Bank. 2024. *World Development Report 2024: The Middle-Income Trap*.
Washington, DC: World Bank. DOI `10.1596/978-1-4648-2078-6`.

- [Trang chính thức](https://www.worldbank.org/en/publication/wdr2024).
- [PDF chính thức](https://openknowledge.worldbank.org/bitstreams/8dca4aff-e0f5-4865-b245-ec9c4583aa60/download).
- [Mirror đã tải được](https://www.healthsystemsmatter.com/files/GeneralReports/World%20Development%20Report%202024.pdf).
- 276 trang PDF. SHA-256 trong `data/source_manifest.json`; `page` luôn là trang PDF gốc, bắt đầu từ 1. Sau front matter, số trang in = trang PDF − 32.

Trang PDF 6 ghi **CC BY 3.0 IGO**, yêu cầu attribution; các thành phần của bên thứ ba
có thể cần quyền riêng. PDF/crop không đưa vào Git; người chấm tải nguồn bằng
`py scripts/download_source.py` và kiểm tra hash. Crop tái tạo từ `bbox` bằng PyMuPDF;
toàn trang gốc là bằng chứng quyết định khi vùng tự động chưa chính xác.

This is an adaptation of an original work by The World Bank. Views and opinions
expressed in the adaptation are the sole responsibility of the author or authors
of the adaptation and are not endorsed by The World Bank.

Bản dịch/tóm tắt tiếng Việt và mô tả hình do AI hỗ trợ, không phải bản dịch chính thức
của World Bank. Nội dung bên thứ ba và các số liệu trích dẫn vẫn giữ nguồn trên trang gốc.

## Dữ liệu nộp

`data/chunks.json` là JSON array UTF-8 đúng example: `id,text,source_id,source,page,metadata`.
Metadata có `bbox` theo PDF point, gốc trên trái, loại native/vision, phiên bản và model
đã dùng. Mô tả vision là dữ liệu dẫn xuất, không tự chứng minh nội dung hình đúng.
`data/processed/page_coverage.json` ghi mọi trang, lỗi và phần chưa xử lý;
`data/processing_summary.json` tổng hợp giới hạn, gồm mô tả chạm output limit.

Benchmark cơ bản/nâng cao tách ở `data/benchmark/`; gold không nằm trong chunk/index.
`gold` là các evidence thay thế. Với multi-hop, mọi nhóm trong `evidence_groups`
đều bắt buộc, trong mỗi nhóm có thể có evidence thay thế. Visual evidence giữ bbox,
figure/table locator, hàng/cột/đơn vị và đáp án đã đối chiếu lúc soạn.

`data/embeddings/*_final.npz` chứa vector/ID để tái tạo Qdrant bằng
`py -m src.indexing restore`, không phải model weights. Mỗi model có kho riêng,
hash corpus/vector và namespace riêng trong `data/index_manifest.json`.

`.gitattributes` giữ nguyên byte JSON khi Git add/checkout để hash corpus,
benchmark và selection lock không bị đổi chỉ vì chuyển LF/CRLF trên Windows.

## Kết nối Day 6

Giữ nguyên model local `multilingual-e5-small` và environment Day 6. Worker chạy
trong venv gốc vì môi trường Python toàn cục có xung đột DLL PyArrow khi import
Sentence Transformers. Không tải lại model. Kiểm tra kho Qdrant cũ đã thực hiện
local và không ghi đè; `results/day6_connection.json` ghi số point và phép truy xuất.
Corpus Day 7 là WDR công khai; không đưa nguyên văn Harry Potter vào repo/API.

Qwen3-Embedding-8B chạy trên Colab, BF16, 4096 chiều; E5 384 chiều float32.
Sai số norm BF16 được validator giới hạn 0,005; Qdrant cosine tự chuẩn hóa lại.
Chi tiết runtime, bridge và điều kiện dừng GPU ở `docs/COLAB_EMBEDDING.md`.
