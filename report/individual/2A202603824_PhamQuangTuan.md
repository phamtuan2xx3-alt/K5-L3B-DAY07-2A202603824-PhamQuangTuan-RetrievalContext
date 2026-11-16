# Báo cáo Day 7 — Phạm Quang Tuấn · 2A202603824 · L3B

## Dữ liệu và benchmark

Corpus **World Development Report 2024**, 276 trang PDF; 1108 chunk (911 native + 197 mô tả hình). Đọc trực quan 111 trang, 0 lỗi. 13 trang trắng/front matter được thống kê riêng; mô tả trang [107, 177, 205] chạm output limit và không được coi là trích xuất hoàn hảo.

Nguồn/quyền **CC BY 3.0 IGO** và ngoại lệ bên thứ ba: [DATA_ACCESS](../../docs/DATA_ACCESS.md). Mỗi chunk giữ ID, source, trang PDF 1-based, bbox, loại nội dung và model provenance. Vision được đối chiếu nguồn khi soạn benchmark; mô tả AI có thể sai hoặc thiếu, PDF gốc là bằng chứng quyết định.

E5 local Day 6: `multilingual-e5-small`, 384 chiều L2 float32; Qwen3-Embedding-8B: 4096 chiều BF16 L2 trên A100 80GB. Hai namespace/index Qdrant tách biệt, cùng corpus/hash; version, vector hash và số point ở [index manifest](../../data/index_manifest.json). Đã truy vấn kho Day 6 thật bằng môi trường cũ, không ghi đè và không gửi nội dung sách lên API: [kết nối](../../results/day6_query_connection.json).

Benchmark cơ bản: dev **15 có đáp án + 2 thiếu**, test **5 + 2**. Nâng cao riêng: dev **15 + 3**, test **24 + 6**; 24 test có đáp án gồm 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text. Gold có quote gốc, bbox, locator hàng/cột/đơn vị hoặc các nhóm evidence bắt buộc; tách khỏi input agent. Nhãn được AI hỗ trợ và ghi provenance, không giả là đã được người dùng/giảng viên phê duyệt.

## Retrieval, agent và lựa chọn

So sánh dense E5, BM25, dense Qwen và hybrid Qwen+BM25 RRF(k=60). Hai thử nghiệm cải thiện là đổi embedding/candidate budget và hybrid+page expansion. Hit@1/Hit@3 dùng cùng định nghĩa strict quote+trang/source; multi-hop phải đủ mọi nhóm. Chỉ số này có thể đánh giá thấp evidence trực quan diễn giải đúng nhưng không chứa nguyên quote; không thay bằng điểm semantic mơ hồ.

| Cấu hình / bộ / split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Grounded /2 | Citation | Abstention | Lỗi | Giây/câu |
|---|---|---|---|---|---|---|---:|---:|
| baseline_e5/advanced/dev | 0/15 (0.0%) | 1/15 (6.7%) | 2/15 (13.3%) | 0.214 (N=14) | 3/3 (100.0%) | 3/3 (100.0%) | 1 | 55.11 |
| baseline_e5/basic/dev | 5/15 (33.3%) | 6/15 (40.0%) | 7/15 (46.7%) | 0.567 (N=15) | 13/14 (92.9%) | 2/2 (100.0%) | 0 | 43.56 |
| baseline_e5/basic/test | 0/5 (0.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 0.750 (N=4) | 3/3 (100.0%) | 2/2 (100.0%) | 1 | 38.41 |
| bm25/advanced/dev | 3/15 (20.0%) | 4/15 (26.7%) | 8/15 (53.3%) | 0.929 (N=14) | 19/19 (100.0%) | 3/3 (100.0%) | 1 | 31.22 |
| bm25/basic/dev | 2/15 (13.3%) | 7/15 (46.7%) | 6/15 (40.0%) | 1.000 (N=7) | 8/8 (100.0%) | 2/2 (100.0%) | 8 | 29.17 |
| dense_qwen/advanced/dev | 4/15 (26.7%) | 7/15 (46.7%) | 4/15 (26.7%) | 0.643 (N=14) | 12/12 (100.0%) | 3/3 (100.0%) | 1 | 29.13 |
| dense_qwen/basic/dev | 8/15 (53.3%) | 11/15 (73.3%) | 12/15 (80.0%) | 1.000 (N=15) | 24/24 (100.0%) | 2/2 (100.0%) | 0 | 50.58 |
| hybrid_qwen/advanced/dev | 4/15 (26.7%) | 6/15 (40.0%) | 6/15 (40.0%) | 0.875 (N=12) | 12/14 (85.7%) | 3/3 (100.0%) | 3 | 32.59 |
| hybrid_qwen/advanced/test | 9/24 (37.5%) | 10/24 (41.7%) | 12/24 (50.0%) | 0.864 (N=22) | 36/38 (94.7%) | 6/6 (100.0%) | 2 | 50.48 |
| hybrid_qwen/basic/dev | 5/15 (33.3%) | 10/15 (66.7%) | 11/15 (73.3%) | 1.000 (N=14) | 25/25 (100.0%) | 2/2 (100.0%) | 1 | 54.06 |
| hybrid_qwen/basic/test | 1/5 (20.0%) | 1/5 (20.0%) | 3/5 (60.0%) | 0.750 (N=4) | 4/4 (100.0%) | 2/2 (100.0%) | 1 | 42.79 |

Grounded /2 dùng các câu có đáp án chạy không lỗi, N hiển thị ngay trong ô; lỗi báo riêng theo benchmark cơ bản. Khi chọn cấu hình, lỗi tính 0 trên tổng câu đã công bố. Điểm đạt nâng cao luôn giữ toàn bộ lỗi trong mẫu số.

Agent dùng `gemini-3.8-flash-high` thật, tool search/read_page, số vòng và context budget trong [configs](../../configs/experiments.json). Nguồn là dữ liệu, không là chỉ dẫn; agent trả citation inline hoặc từ chối. Gateway dịch role tool thành role function không được upstream chấp nhận; adapter thực thi tool thật rồi gửi kết quả dưới user-context, trace ghi rõ. Lượt cuối bỏ tools hẳn; sửa JSON tối đa một lần, không sửa bằng gold.

**Chọn `hybrid_qwen` trên dev**, khóa lúc `2026-11-16T05:15:55.280812+00:00`. Tiêu chí: grounded answer (lỗi hệ thống tính 0 khi chọn), citation, rồi thời gian. Hash corpus/index, config, code agent/retrieval và benchmark được khóa trước test. [Selection lock](../../results/selection_lock.json). Không dùng test để chỉnh.

## Phân tích lỗi và giới hạn

1. Dev ban đầu có output bị cắt/sai JSON. Giữ trace chẩn đoán local; tăng output budget 1800→4096 và thêm một lượt yêu cầu đúng schema. Đánh giá cuối dùng cùng giao thức đã sửa cho mọi cấu hình; không ghi đè lỗi để làm đẹp kết quả.
2. Dev ban đầu citation trỏ tới context của lượt trước đã bị thay. Sửa quản lý lịch sử để chỉ gửi context cộng dồn hiện tại, giữ query history nhưng bỏ payload cũ. Citation cuối phải thuộc context thật. Các sai nội dung/evidence còn lại được giữ và phân tích trong trace, không coi ID hợp lệ là grounding.

Audit trực tiếp 17 câu dev cơ bản xác nhận lỗi bỏ lỡ 108 quốc gia, ngưỡng 11%, số 11 nền kinh tế trong FAT và mức bao phủ 88% phát thải. Một câu năng lượng có diễn đạt mơ hồ nên được chấm một phần; nhãn context của một câu thiếu bằng chứng được sửa về 0. [Source audit](../../results/source_audit_dev_baseline.json) giữ các trang đã đọc và nhãn ban đầu, không thay gold.

Model và tool có thể lỗi; lỗi được lưu cùng partial trace. Metrics cơ bản báo lỗi riêng theo BENCHMARK.md; nâng cao giữ lỗi trong mẫu số. Nhãn AI/LLM judge không thay thế việc người chấm đối chiếu hình gốc. Benchmark nhỏ, cùng tài liệu, một số hình dùng ở cả hai split với yêu cầu khác nhau; kết quả không suy rộng thành chất lượng tổng quát.

## Thử thách nâng cao

| Cấu hình / nhóm test | Đạt / tổng |
|---|---|
| hybrid_qwen / absent | 6/6 (100.0%) |
| hybrid_qwen / chart | 3/5 (60.0%) |
| hybrid_qwen / diagram_image | 4/5 (80.0%) |
| hybrid_qwen / multi_hop | 1/4 (25.0%) |
| hybrid_qwen / table | 6/6 (100.0%) |
| hybrid_qwen / text | 4/4 (100.0%) |
| **hybrid_qwen / toàn bộ** | **24/30 (80.0%)** |

**Chưa đạt ngưỡng bonus 90%; không nhận là đã đạt 15 điểm nâng cao.** Việc công nhận bonus còn cần đầy đủ nguồn, phạm vi index và đối chiếu từng trang/vùng; không làm tròn hay bỏ câu lỗi.

## Tái lập, thời gian và chi phí

Model calls đã lưu: **491**; usage được cộng nguyên từ gateway: `{"prompt_tokens": 1051197, "completion_tokens": 28420, "total_tokens": 1276776}`. Đây là usage của các response đã lưu, không khẳng định đầy đủ billing của mọi lần timeout/setup/chẩn đoán. Tên model là alias gateway, không là chứng nhận độc lập danh tính vendor. Gateway không công bố giá USD kiểm chứng được nên chi phí USD để `null`, không bịa giá. Colab A100 tại thời điểm chạy báo 8,88 CU/giờ; usage/balance snapshot lưu riêng, bao gồm cả giai đoạn setup/chẩn đoán.

Lệnh cài/chạy/restore và validation ở [README](../../README.md). Kết quả từng câu, câu trả lời, tool call, context, citation, nhãn và lý do ở `results/experiments/`; [bảng từng query](../query_results.csv). Có thể tái tính aggregates offline bằng `py scripts/replay_metrics.py`. Crop/source PDF, cache, model weights, runtime secrets và log debug được ignore; final vectors, manifests, benchmark, code, trace/nhãn/metrics được giữ.


Colab đã giải phóng GPU: **0 active assignments**, **0 CU/giờ**. Balance 404.78 → 387.47 CU; chênh lệch toàn tài khoản 17.31 CU, gồm setup/chẩn đoán/idle, không phải chi phí riêng mỗi câu. Snapshot: [colab_cost.json](../../results/colab_cost.json).
