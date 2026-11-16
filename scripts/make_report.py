"""Render the concise submission report from real saved experiment artifacts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
source = read(ROOT/'data/source_manifest.json')
processing = read(ROOT/'data/processing_summary.json')
index = read(ROOT/'data/index_manifest.json')
lock_path = ROOT/'results/selection_lock.json'
lock = read(lock_path) if lock_path.exists() else None


def ratio(value):
    if not value or value.get('N', 0) == 0:
        return '0/0 (N/A)'
    return f"{value['n']}/{value['N']} ({100*value['n']/value['N']:.1f}%)"


lines = [
    '# Báo cáo Day 7 — Phạm Quang Tuấn · 2A202603824 · L3B', '',
    '## Dữ liệu và benchmark', '',
    f"Corpus **World Development Report 2024**, {source['pdf_pages']} trang PDF; {processing['chunks']} chunk ({processing['chunk_kinds']['native_text']} native + {processing['chunk_kinds']['vision_description']} mô tả hình). "
    f"Đọc trực quan {processing['vision_pages']} trang, {processing['vision_errors']} lỗi. 13 trang trắng/front matter được thống kê riêng; "
    f"mô tả trang {processing['vision_output_limit_pages']} chạm output limit và không được coi là trích xuất hoàn hảo.", '',
    'Nguồn/quyền **CC BY 3.0 IGO** và ngoại lệ bên thứ ba: [DATA_ACCESS](../../docs/DATA_ACCESS.md). '
    'Mỗi chunk giữ ID, source, trang PDF 1-based, bbox, loại nội dung và model provenance. '
    'Vision được đối chiếu nguồn khi soạn benchmark; mô tả AI có thể sai hoặc thiếu, PDF gốc là bằng chứng quyết định.', '',
    'E5 local Day 6: `multilingual-e5-small`, 384 chiều L2 float32; Qwen3-Embedding-8B: 4096 chiều BF16 L2 trên A100 80GB. '
    'Hai namespace/index Qdrant tách biệt, cùng corpus/hash; version, vector hash và số point ở [index manifest](../../data/index_manifest.json). '
    'Đã truy vấn kho Day 6 thật bằng môi trường cũ, không ghi đè và không gửi nội dung sách lên API: [kết nối](../../results/day6_query_connection.json).', '',
    'Benchmark cơ bản: dev **15 có đáp án + 2 thiếu**, test **5 + 2**. Nâng cao riêng: dev **15 + 3**, test **24 + 6**; '
    '24 test có đáp án gồm 6 bảng, 5 ảnh/sơ đồ, 5 biểu đồ, 4 multi-hop, 4 text. '
    'Gold có quote gốc, bbox, locator hàng/cột/đơn vị hoặc các nhóm evidence bắt buộc; tách khỏi input agent. '
    'Nhãn được AI hỗ trợ và ghi provenance, không giả là đã được người dùng/giảng viên phê duyệt.', '',
    '## Retrieval, agent và lựa chọn', '',
    'So sánh dense E5, BM25, dense Qwen và hybrid Qwen+BM25 RRF(k=60). Hai thử nghiệm cải thiện là đổi embedding/candidate budget và hybrid+page expansion. '
    'Hit@1/Hit@3 dùng cùng định nghĩa strict quote+trang/source; multi-hop phải đủ mọi nhóm. '
    'Chỉ số này có thể đánh giá thấp evidence trực quan diễn giải đúng nhưng không chứa nguyên quote; không thay bằng điểm semantic mơ hồ.', '',
    '| Cấu hình / bộ / split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Grounded /2 | Citation | Abstention | Lỗi | Giây/câu |',
    '|---|---|---|---|---|---|---|---:|---:|'
]
flat_rows = []
metric_sets = []
for path in sorted((ROOT/'results/experiments').glob('*/*/*/agent_metrics.ai.json')):
    config, suite, split = path.parents[2].name, path.parents[1].name, path.parent.name
    metrics = read(path)
    retrieval = read(path.with_name('retrieval_metrics.json'))
    runtime_path = path.with_name('runtime_metrics.json')
    runtime = read(runtime_path) if runtime_path.exists() else None
    final_cer = runtime['final_context_evidence_rate'] if runtime else None
    mean = metrics['grounded_answer']['mean_normalized']
    graded_n = sum(metrics['grounded_answer']['counts'].values())
    lines.append(f"| {config}/{suite}/{split} | {ratio(retrieval['Hit@1'])} | {ratio(retrieval['Evidence Hit@3'])} | {ratio(final_cer)} | {mean:.3f} (N={graded_n}) | {ratio(metrics['citation_validity'])} | {ratio(metrics['abstention'])} | {metrics['errors']} | {metrics['mean_seconds']:.2f} |" if mean is not None else f'| {config}/{suite}/{split} | N/A | N/A | N/A | N/A | N/A | N/A | {metrics["errors"]} | {metrics["mean_seconds"]:.2f} |')
    metric_sets.append({'config': config, 'suite': suite, 'split': split, 'metrics': metrics, 'retrieval': retrieval})
    traces = read(path.with_name('agent_traces.json'))
    reviews = {r['query_id']: r for r in read(path.with_name('reviews.ai.json'))}
    for trace in traces:
        review = reviews[trace['query_id']]
        flat_rows.append({'config': config, 'suite': suite, 'split': split, 'query_id': trace['query_id'],
                          'status': trace['status'], 'answer': trace.get('answer', ''),
                          'citations': '|'.join(trace.get('citations', [])), 'pass': review['pass'],
                          'answer_grade': review.get('answer_grade'), 'context_grade': review.get('context_grade'),
                          'seconds': trace['seconds'], 'reason': review.get('reason', ''), 'error': trace.get('error', '')})
lines.extend(['', 'Grounded /2 dùng các câu có đáp án chạy không lỗi, N hiển thị ngay trong ô; lỗi báo riêng theo benchmark cơ bản. '
              'Khi chọn cấu hình, lỗi tính 0 trên tổng câu đã công bố. Điểm đạt nâng cao luôn giữ toàn bộ lỗi trong mẫu số.', '',
              'Agent dùng `gemini-3.8-flash-high` thật, tool search/read_page, số vòng và context budget trong [configs](../../configs/experiments.json). '
              'Nguồn là dữ liệu, không là chỉ dẫn; agent trả citation inline hoặc từ chối. Gateway dịch role tool thành role function không được upstream chấp nhận; '
              'adapter thực thi tool thật rồi gửi kết quả dưới user-context, trace ghi rõ. Lượt cuối bỏ tools hẳn; sửa JSON tối đa một lần, không sửa bằng gold.', ''])
if lock:
    lines.extend([f"**Chọn `{lock['selected']['id']}` trên dev**, khóa lúc `{lock['locked_at_utc']}`. "
                  'Tiêu chí: grounded answer (lỗi hệ thống tính 0 khi chọn), citation, rồi thời gian. '
                  'Hash corpus/index, config, code agent/retrieval và benchmark được khóa trước test. '
                  '[Selection lock](../../results/selection_lock.json). Không dùng test để chỉnh.', ''])
else:
    lines.extend(['**Chưa khóa dev/test; chưa có kết quả test cuối.**', ''])
lines.extend(['## Phân tích lỗi và giới hạn', '',
              '1. Dev ban đầu có output bị cắt/sai JSON. Giữ trace chẩn đoán local; tăng output budget 1800→4096 và thêm một lượt yêu cầu đúng schema. '
              'Đánh giá cuối dùng cùng giao thức đã sửa cho mọi cấu hình; không ghi đè lỗi để làm đẹp kết quả.',
              '2. Dev ban đầu citation trỏ tới context của lượt trước đã bị thay. Sửa quản lý lịch sử để chỉ gửi context cộng dồn hiện tại, giữ query history nhưng bỏ payload cũ. '
              'Citation cuối phải thuộc context thật. Các sai nội dung/evidence còn lại được giữ và phân tích trong trace, không coi ID hợp lệ là grounding.', '',
              'Audit trực tiếp 17 câu dev cơ bản xác nhận lỗi bỏ lỡ 108 quốc gia, ngưỡng 11%, số 11 nền kinh tế trong FAT và mức bao phủ 88% phát thải. '
              'Một câu năng lượng có diễn đạt mơ hồ nên được chấm một phần; nhãn context của một câu thiếu bằng chứng được sửa về 0. '
              '[Source audit](../../results/source_audit_dev_baseline.json) giữ các trang đã đọc và nhãn ban đầu, không thay gold.', '',
              'Model và tool có thể lỗi; lỗi được lưu cùng partial trace. Metrics cơ bản báo lỗi riêng theo BENCHMARK.md; nâng cao giữ lỗi trong mẫu số. '
              'Nhãn AI/LLM judge không thay thế việc người chấm đối chiếu hình gốc. Benchmark nhỏ, cùng tài liệu, một số hình dùng ở cả hai split với yêu cầu khác nhau; '
              'kết quả không suy rộng thành chất lượng tổng quát.', '',
              '## Thử thách nâng cao', '',
              '| Cấu hình / nhóm test | Đạt / tổng |', '|---|---|'])
for row in metric_sets:
    if row['suite'] == 'advanced' and row['split'] == 'test':
        for group, value in row['metrics']['groups'].items():
            lines.append(f"| {row['config']} / {group} | {ratio(value)} |")
        lines.append(f"| **{row['config']} / toàn bộ** | **{ratio(row['metrics']['advanced_pass'])}** |")
selected_test = next((r for r in metric_sets if lock and r['config'] == lock['selected']['id'] and r['suite'] == 'advanced' and r['split'] == 'test'), None)
if selected_test:
    score = selected_test['metrics']['advanced_pass']
    threshold = score['n']*10 >= score['N']*9
    lines.extend(['', ('Ngưỡng số học ≥90% đã đạt trên nhãn AI hỗ trợ.' if threshold else '**Chưa đạt ngưỡng bonus 90%; không nhận là đã đạt 15 điểm nâng cao.**')+
                  ' Việc công nhận bonus còn cần đầy đủ nguồn, phạm vi index và đối chiếu từng trang/vùng; không làm tròn hay bỏ câu lỗi.'])
else:
    lines.extend(['', 'Chưa có test nâng cao cuối; không công bố đạt bonus.'])
calls = [call for path in (ROOT/'results/experiments').glob('*/*/*/agent_traces.json') for trace in read(path) for call in trace.get('model_calls', [])]
usage = {}
for call in calls:
    for key, value in (call.get('usage') or {}).items():
        if isinstance(value, (int, float)):
            usage[key] = usage.get(key, 0)+value
lines.extend(['', '## Tái lập, thời gian và chi phí', '',
              f"Model calls đã lưu: **{len(calls)}**; usage được cộng nguyên từ gateway: `{json.dumps(usage)}`. "
              'Đây là usage của các response đã lưu, không khẳng định đầy đủ billing của mọi lần timeout/setup/chẩn đoán. '
              'Tên model là alias gateway, không là chứng nhận độc lập danh tính vendor. '
              'Gateway không công bố giá USD kiểm chứng được nên chi phí USD để `null`, không bịa giá. '
              'Colab A100 tại thời điểm chạy báo 8,88 CU/giờ; usage/balance snapshot lưu riêng, bao gồm cả giai đoạn setup/chẩn đoán.', '',
              'Lệnh cài/chạy/restore và validation ở [README](../../README.md). '
              'Kết quả từng câu, câu trả lời, tool call, context, citation, nhãn và lý do ở `results/experiments/`; '
              '[bảng từng query](../query_results.csv). '
              'Có thể tái tính aggregates offline bằng `py scripts/replay_metrics.py`. '
              'Crop/source PDF, cache, model weights, runtime secrets và log debug được ignore; final vectors, manifests, benchmark, code, trace/nhãn/metrics được giữ.', ''])
cost_path = ROOT/'results/colab_cost.json'
if cost_path.exists():
    cost = json.loads(cost_path.read_text(encoding='utf-8'))
    lines.extend(['', f"Colab đã giải phóng GPU: **{cost['active_assignments']} active assignments**, "
                  f"**{cost['final_usage_rate_cu_per_hour']} CU/giờ**. Balance "
                  f"{cost['initial_balance_cu']} → {cost['final_balance_cu']} CU; chênh lệch toàn tài khoản "
                  f"{cost['account_balance_delta_cu']} CU, gồm setup/chẩn đoán/idle, không phải chi phí riêng mỗi câu. "
                  'Snapshot: [colab_cost.json](../../results/colab_cost.json).', ''])

target = ROOT/'report/individual/2A202603824_PhamQuangTuan.md'
target.write_text('\n'.join(lines), encoding='utf-8')
if flat_rows:
    with (ROOT/'report/query_results.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
print('Report generated from',len(metric_sets),'completed metric sets;',len(flat_rows),'query rows')
