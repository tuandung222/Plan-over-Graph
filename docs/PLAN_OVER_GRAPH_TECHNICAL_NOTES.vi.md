# Plan-over-Graph: Technical Notes (Vietnamese)

Tài liệu này tổng hợp đầy đủ các nội dung đã trao đổi và xác minh trực tiếp từ mã nguồn + paper của repo:
- Repo: https://github.com/zsq259/Plan-over-Graph
- Paper: https://arxiv.org/abs/2502.14563

## 1) Repo này ứng với paper nào?

Paper tương ứng:
- **Plan-over-Graph: Towards Parallelable LLM Agent Schedule**
- arXiv ID: `2502.14563`

Đã tải local:
- PDF: `paper/2502.14563.pdf`
- HTML: `paper/2502.14563.html`

## 2) Paper dùng benchmark/training dataset nào?

### 2.1 Training dataset
Paper không dùng benchmark chuẩn bên ngoài kiểu GSM8K/WebShop để huấn luyện planning chính.
Họ tạo **synthetic task-graph dataset** riêng.

Theo Section 5.1 và Table 2:
- Tổng train: **12,000** mẫu.
- Node scale: `10`, `30`, `50`.
- Graph structure: `Random` + `Tree-based`.
- Mỗi cấu hình train có 2,000 mẫu (6 cấu hình = 12,000).

### 2.2 Benchmark/evaluation
Paper đánh giá trên 3 nhóm:
- **Baseline graph tests** (linear edge scaling): node `10/20/30/40/50`, mỗi node có `Random + Tree-based`, mỗi config `100` mẫu.
- **Edge-variation tests**: random graph ở node `10` và `30`, mỗi config `1000` mẫu.
- **Textual-query real-life set**: **200** task mô phỏng tình huống thực tế (paper nói xây bằng DeepSeek-R1).

## 3) Dataset có public không?

### 3.1 Trạng thái public
README của repo trỏ tới HF dataset:
- `hastin/plan-over-graph`

Kiểm tra API HF tại thời điểm phân tích cho thấy:
- `private=false`
- `gated=false`

=> Dataset repo đang ở trạng thái public.

### 3.2 Mức độ đầy đủ dữ liệu public
Trong `data.zip` public có:
- `dev/train/*`
- `dev/test/*`
- `result/*`

Tuy nhiên có một điểm quan trọng:
- `dev/test/10-3-1000-r.json`
- `dev/test/30-3-1000-r.json`

Hai file trên trong zip hiện là **Git LFS pointer** (không phải JSON thực), nên phần edge-variation test chưa public đầy đủ nội dung data thực qua gói này.

## 4) Giải pháp này có phù hợp LLM-based Agentic System không?

### 4.1 Khi phù hợp
Phù hợp tốt nếu hệ thống có:
- Bài toán tách thành subtask rõ dependency (DAG).
- Nhiều nhánh chạy song song được.
- Cần tối ưu time/cost/success.

### 4.2 Khi kém phù hợp
Ít phù hợp nếu:
- Nhiệm vụ quá mở, phụ thuộc quan sát động liên tục.
- Cần re-plan liên tục theo tín hiệu môi trường runtime.
- Tool output quá nhiễu, khó xác thực tự động.

## 5) Planning trong paper có phải chỉ là “chia task cho worker”?

Có chia nhỏ, nhưng không dừng ở đó.

Planning ở đây là:
1. Decompose task thành các subtask.
2. Gắn dependency giữa subtask.
3. Lập lịch thực thi song song khi dependency cho phép.
4. Tối ưu thời gian hoàn thành và chi phí.

Tức là **decompose + dependency-aware parallel scheduling**.

## 6) Ví dụ minh hoạ trực quan

Task lớn: ra mắt landing page sản phẩm.

Subtask:
- A: research
- B: messaging
- C: UI design
- D: copy writing
- E: frontend implementation
- F: QA/publish

Dependency:
- `A -> B`
- `A -> C`
- `B + C -> D`
- `C + D -> E`
- `E -> F`

Điểm cốt lõi plan-over-graph:
- `B` và `C` chạy song song sau `A`, giảm makespan so với làm tuần tự toàn bộ.

## 7) Format planning trong mã nguồn

### 7.1 Input cho planner (abstask)
JSON object dạng:
- `rules`: list rule `{id, source, target, time, cost}` (id có thể có/không tùy template).
- `initial_source`: list node ban đầu.
- `target`: node đích.

Ví dụ:
```json
{
  "rules": [
    {"id": 0, "source": ["N1"], "target": ["N2"], "time": 3, "cost": 1}
  ],
  "initial_source": ["N1"],
  "target": "N2"
}
```

### 7.2 Output plan mà model trả
Runtime chấp nhận:
- JSON list trực tiếp, hoặc
- JSON object có key `plan`.

Mỗi phần tử plan (subtask) có các trường chính:
- `name`
- `source` (hoặc có thể suy từ `perform_rule_indx`)
- `target` (hoặc có thể suy từ `perform_rule_indx`)
- `dependencies`

Ví dụ:
```json
[
  {
    "name": "Subtask1",
    "source": ["N1"],
    "target": ["N2"],
    "dependencies": []
  },
  {
    "name": "Subtask2",
    "source": ["N2"],
    "target": ["N3"],
    "dependencies": ["Subtask1"]
  }
]
```

Biến thể có `perform_rule_indx` (template `abstask_plan_ref`) cũng được hỗ trợ.

### 7.3 Dữ liệu train/eval dùng format gì?
Trong sample dataset:
- `question` chứa graph task (`rules`, `initial_source`, `target`).
- `answer` là list subtasks với format `name/source/target/dependencies`.
- `feasible` là phương án feasible khác dùng cho mix/DPO.

## 8) Quan hệ với tool-calling là gì?

Có liên hệ về mặt ý tưởng, nhưng không phải tool-calling runtime thực sự.

### 8.1 Có gì giống
- Mỗi subtask có thể xem như một “tool step” trừu tượng.
- Scheduler chạy theo DAG dependency.

### 8.2 Chưa có gì
- Không có function schema/arguments của tool thật.
- Không gọi external API/tool thật trong runner mặc định.
- `TTRunner` chỉ trả tên task, sau đó `TTEnv` commit để kiểm tra rule/time/cost.

=> Đây là **task-graph simulation framework**, chưa phải production tool-orchestration framework.

## 9) Prompt planning thực tế đang dùng

Prompt được render từ template theo:
- `instruction.format(example=example, task=...)`

Các mode chính:
- `abstask` + template `abstask_plan`.
- `specific_task` + template `specific_task_plan`.
- `specific_task` + `--extractor true` thì chạy thêm prompt `extract_rules` trước rồi planning.

Script chạy mẫu:
- `script/test.sh`
- `script/test_query.sh`
- `script/test_query_extract.sh`

## 10) Kết luận thực dụng

- Paper + repo có đóng góp rõ ràng ở tầng **parallel planning on dependency graph**.
- Điểm mạnh: biểu diễn task thành DAG + schedule song song + tối ưu thời gian/chi phí.
- Hạn chế khi áp dụng production agent system: chưa có tầng tool registry/runtime feedback đầy đủ.
- Nếu muốn đưa vào hệ thống agent thực, nên dùng như **planner/scheduler layer**, rồi nối với execution layer thật (tool catalog, schema, retries, fallback, replan).

## 11) Tài nguyên tham khảo nhanh

- Repo: https://github.com/zsq259/Plan-over-Graph
- Paper abs: https://arxiv.org/abs/2502.14563
- Paper pdf: https://arxiv.org/pdf/2502.14563
- Paper html: https://arxiv.org/html/2502.14563
- HF dataset: https://huggingface.co/datasets/hastin/plan-over-graph

