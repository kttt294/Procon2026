# Kế Hoạch Chuyển Dự Án Sang Bài Tập Lớn "Thuật Toán Ứng Dụng"

## 1. Mục tiêu

Mục tiêu của file này là định hướng lại dự án HEXA UDON thành một bài tập lớn phù hợp với học phần **Thuật toán ứng dụng**.

Repo hiện tại đã có nhiều thành phần nâng cao, bao gồm simulator, A*, heuristic, benchmark, visualizer và cả Reinforcement Learning. Tuy nhiên, để đạt điểm cao trong học phần thuật toán, phần báo cáo và demo cần đặt trọng tâm vào:

- Mô hình hóa bài toán thành đồ thị có trọng số.
- Thiết kế thuật toán tìm đường.
- Thiết kế chiến lược phân công và lập lịch.
- So sánh nhiều thuật toán bằng thực nghiệm.
- Phân tích độ phức tạp ước lượng.

Reinforcement Learning nên được giữ như **hướng mở rộng nâng cao**, không nên là trọng tâm chính của bài tập lớn.

## 2. Tên đề tài đề xuất

**Tối ưu hóa lập kế hoạch đa tác tử trên bản đồ lục giác có ràng buộc tài nguyên**

Tên này phù hợp với học phần Thuật toán ứng dụng vì nhấn mạnh:

- Tối ưu hóa.
- Lập kế hoạch.
- Đa tác tử.
- Đồ thị lục giác.
- Ràng buộc tài nguyên.

## 3. Định hướng trình bày trong báo cáo

### 3.1. Không trình bày như bot chơi game

Không nên mô tả dự án theo hướng:

> Xây dựng bot chơi game HEXA UDON.

Nên mô tả theo hướng:

> Xây dựng và đánh giá các thuật toán lập kế hoạch cho bài toán đa tác tử trên đồ thị lục giác có ràng buộc về bước đi, nhiên liệu, tồn kho và giao thông.

### 3.2. Trọng tâm của bài tập lớn

Trọng tâm nên gồm các nhóm thuật toán:

1. Biểu diễn đồ thị lục giác.
2. Tìm đường hợp lệ và gần tối ưu bằng Dijkstra/A*.
3. Greedy baseline cho việc chọn điểm tài nguyên.
4. Lookahead heuristic để cải thiện greedy.
5. Phân công xe cho điểm tài nguyên.
6. Lập lịch tiếp tế nhiên liệu.
7. Kiểm tra tính hợp lệ của lời giải.
8. Benchmark và phân tích thời gian chạy.

### 3.3. RL chỉ là phần mở rộng

Repo hiện tại đã có RL, nhưng trong bài tập lớn nên ghi:

> Ngoài các thuật toán heuristic, nhóm có thử nghiệm Reinforcement Learning như một hướng mở rộng để học chiến lược từ simulator. Phần này không thay thế phần thuật toán chính mà được dùng để so sánh và đánh giá thêm.

Nếu thời gian báo cáo có hạn, chỉ cần đưa RL vào mục "Hướng phát triển" hoặc "Thử nghiệm nâng cao".

## 4. Hiện trạng repo

### 4.1. Đã có

| Thành phần | File / thư mục | Trạng thái |
|---|---|---|
| Mô hình dữ liệu | `src/env/models.py` | Đã có |
| Bản đồ lục giác | `src/env/hex_grid.py` | Đã có |
| Simulator | `src/env/simulator.py` | Đã có |
| Tìm đường A* | `src/pathfinding/astar.py` | Đã có |
| Greedy baseline | `src/strategy/greedy.py` | Đã có |
| Lookahead heuristic | `src/strategy/lookahead.py` | Đã có |
| Chọn loại agent | `src/strategy/agent_selector.py` | Đã có |
| Validator | `src/env/validator.py` | Đã có |
| Scoring | `src/env/scoring.py` | Đã có |
| Random map generator | `src/env/map_generator.py` | Đã có |
| Benchmark | `src/benchmark.py` | Đã có |
| Replay | `replay/` | Đã có |
| Visualizer terminal | `visualizer/` | Đã có |
| RL MAPPO | `src/rl/` | Đã có |
| Unit tests | `tests/` | Đã có |

### 4.2. Cần sửa theo hướng bài tập lớn

| Việc cần làm | Lý do |
|---|---|
| Viết lại README theo format bài tập lớn | Để giảng viên nhìn thấy mục tiêu thuật toán ngay từ đầu |
| Thêm báo cáo thuật toán | Cần có mô hình hóa, thuật toán, độ phức tạp, thực nghiệm |
| Thêm bảng so sánh kết quả | Chứng minh Lookahead tốt hơn Greedy |
| Thêm phần phân tích độ phức tạp | Bắt buộc để đúng tính chất học phần |
| Thêm script chạy demo đơn giản | Để chấm bài nhanh |
| Giảm vai trò RL trong tài liệu chính | Tránh bị đánh giá là lệch sang học máy |

## 5. Cấu trúc tài liệu nên thêm

Nên thêm các file sau:

```text
procon2026/
├─ README.md
├─ BTL_implementationPlan.md
├─ BTL_report.md
├─ BTL_experimentResults.md
├─ BTL_complexityAnalysis.md
└─ docs/
   ├─ problem_model.md
   ├─ algorithms.md
   ├─ benchmark_protocol.md
   └─ rl_extension.md
```

Trong đó:

- `README.md`: giới thiệu ngắn, cách chạy demo, cách chạy test.
- `BTL_report.md`: báo cáo chính.
- `BTL_experimentResults.md`: bảng kết quả benchmark.
- `BTL_complexityAnalysis.md`: phân tích độ phức tạp.
- `docs/problem_model.md`: mô hình hóa bài toán.
- `docs/algorithms.md`: mô tả Greedy, A*, Lookahead.
- `docs/benchmark_protocol.md`: cách sinh map và đo kết quả.
- `docs/rl_extension.md`: mô tả RL như phần mở rộng.

## 6. Kế hoạch sửa dự án

### Bước 1: Viết README mới

README cần trả lời nhanh:

- Bài toán là gì?
- Repo giải bài toán bằng những thuật toán nào?
- Cách chạy test.
- Cách chạy benchmark.
- Cách chạy demo visualizer.
- RL nằm ở đâu và có vai trò gì.

Trong README, thứ tự nên là:

1. Mô tả bài toán.
2. Thuật toán chính.
3. Cấu trúc repo.
4. Hướng dẫn chạy.
5. Kết quả tóm tắt.
6. Hướng mở rộng RL.

### Bước 2: Viết báo cáo chính

File: `BTL_report.md`

Cấu trúc báo cáo nên gồm:

1. Giới thiệu bài toán.
2. Mô hình hóa bài toán.
3. Biểu diễn bản đồ lục giác.
4. Thuật toán tìm đường.
5. Chiến lược Greedy baseline.
6. Chiến lược Lookahead heuristic.
7. Xử lý nhiên liệu và tiếp tế.
8. Xử lý giao thông.
9. Validator kiểm tra lời giải.
10. Thực nghiệm và đánh giá.
11. Phân tích độ phức tạp.
12. Kết luận và hướng phát triển.

### Bước 3: Viết phần mô hình hóa

File: `docs/problem_model.md`

Cần trình bày:

- Mỗi ô lục giác là một đỉnh trong đồ thị.
- Mỗi di chuyển hợp lệ sang ô kề là một cạnh.
- Trọng số cạnh phụ thuộc vào địa hình của ô nguồn.
- Ao là đỉnh không thể đi vào.
- Traffic làm thay đổi trọng số cạnh của các ô đường.
- Mỗi ngày là một vòng lặp lập kế hoạch.
- Mỗi xe là một tác tử.
- Tổng step mỗi ngày là tài nguyên dùng chung.
- Nhiên liệu là tài nguyên riêng của xe tuần tra.
- Tồn kho spot là tài nguyên giới hạn.

### Bước 4: Viết phần thuật toán

File: `docs/algorithms.md`

Cần mô tả rõ các thuật toán:

#### 4.1. Hex-grid neighbor

- Dùng even-r offset coordinate.
- Mỗi cell có tối đa 6 láng giềng.
- Ô biên bản đồ có ít láng giềng hơn.

#### 4.2. A*

- Trạng thái: cell hiện tại.
- Cost: tổng step đã dùng.
- Constraint: fuel và step budget.
- Heuristic: khoảng cách hex đến đích.
- Trường hợp xấu: gần tương đương Dijkstra.

#### 4.3. Greedy baseline

- Mỗi xe tuần tra chọn spot gần nhất thuộc series chưa thu.
- Nếu không còn series mới thì chọn spot gần nhất còn hàng.
- Xe tiếp tế đi tới xe tuần tra thiếu fuel nhất.

#### 4.4. Lookahead heuristic

- Gán series theo mức độ ưu tiên.
- Tránh nhiều xe cùng lấy một series mới.
- Mỗi xe có thể đi qua nhiều spot trong một ngày.
- Chia step budget theo độ quan trọng của route.
- Xe tiếp tế dự đoán điểm gặp.
- Nếu tiếp tế không kịp, xe tuần tra chèn điểm gặp gần xe tiếp tế hơn.

#### 4.5. Agent selector

- Thử nhiều phương án số xe tuần tra / xe tiếp tế.
- Đánh giá vị trí bắt đầu của xe.
- Chọn phương án có kết quả mô phỏng tốt nhất.

#### 4.6. RL extension

- Trình bày ngắn gọn.
- Không đặt làm thuật toán chính.
- Ghi rõ đây là hướng thử nghiệm nâng cao.

### Bước 5: Thêm phần phân tích độ phức tạp

File: `BTL_complexityAnalysis.md`

Không cần chứng minh quá nặng, nhưng nên có bảng:

| Thành phần | Độ phức tạp ước lượng | Ghi chú |
|---|---:|---|
| Tạo láng giềng hex grid | O(V) | V = số cell |
| A* một cặp điểm | O(E log V) | Trường hợp xấu |
| Multi-waypoint path | O(K * E log V) | K = số waypoint |
| Greedy assignment | O(A * S * path_cost) | A = số xe, S = số spot |
| Lookahead assignment | O(A * S log S + A * K * path_cost) | Ước lượng |
| Simulator một ngày | O(T) | T = tổng số action |
| Benchmark N game | O(N * D * planner_cost) | D = số ngày |

Cần bổ sung thực nghiệm thời gian chạy trên:

- Map 8x8.
- Map 16x16.
- Map 32x32.

### Bước 6: Benchmark nghiêm túc

File: `BTL_experimentResults.md`

Cần chạy ít nhất:

```bash
python src/benchmark.py --games 100 --strategies greedy lookahead
```

Nếu có thời gian:

```bash
python src/benchmark.py --games 500 --strategies greedy lookahead
```

Cần ghi lại:

- Avg unique series.
- Avg daily series.
- Avg total udon.
- Avg runtime.
- Win rate của Lookahead so với Greedy.

Nên benchmark riêng theo kích thước:

- 8x8.
- 16x16.
- 32x32.

Nếu `benchmark.py` chưa hỗ trợ chia theo kích thước thì cần sửa thêm option:

```bash
python src/benchmark.py --games 100 --size 8
python src/benchmark.py --games 100 --size 16
python src/benchmark.py --games 100 --size 32
```

### Bước 7: Demo chấm bài

Cần có một lệnh demo đơn giản:

```bash
python src/main.py sim
```

Và một lệnh benchmark:

```bash
python src/benchmark.py --games 100
```

Nên thêm vào README:

```bash
python -m pytest tests/
python src/main.py sim
python src/benchmark.py --games 100
```

### Bước 8: Giữ RL nhưng tách thành phần mở rộng

Repo hiện tại đã có:

- `src/rl/actor_critic.py`
- `src/rl/mappo.py`
- `src/rl/curriculum.py`
- `src/rl/selfplay.py`

Trong tài liệu nên ghi:

> Phần RL được xây dựng như một hướng mở rộng. Do trọng tâm học phần là Thuật toán ứng dụng, kết quả chính của báo cáo tập trung vào Greedy, A*, Lookahead, Agent Selector và Benchmark.

Nếu train RL có kết quả tốt, đưa vào phần nâng cao.

Nếu train RL chưa tốt, vẫn có thể ghi nhận:

- RL cần simulator chuẩn.
- Training tốn nhiều tài nguyên.
- Chưa phải phương án phù hợp nhất trong thời gian ngắn.
- Heuristic có lợi thế về tính ổn định và khả năng giải thích.

## 7. Việc cần làm ngay

Thứ tự ưu tiên:

1. Viết `README.md` mới theo hướng bài tập lớn.
2. Viết `BTL_report.md` bản đầu.
3. Viết `docs/problem_model.md`.
4. Viết `docs/algorithms.md`.
5. Viết `BTL_complexityAnalysis.md`.
6. Chạy benchmark 100-500 games và viết `BTL_experimentResults.md`.
7. Sửa `benchmark.py` nếu cần để benchmark theo kích thước map.
8. Nếu còn thời gian, train RL và đưa vào phần mở rộng.

## 8. Tiêu chí để nhắm A+

Đề tài có khả năng đạt điểm cao nếu thể hiện được:

- Mô hình hóa bài toán rõ ràng.
- Có nhiều thuật toán để so sánh, không chỉ một cách làm.
- Có phân tích độ phức tạp.
- Có thực nghiệm trên nhiều kích thước input.
- Có validator và test để chứng minh lời giải đúng.
- Có benchmark để chứng minh cải tiến của Lookahead so với Greedy.
- Có visualizer/replay để giải thích hành vi thuật toán.
- Có hướng mở rộng RL nhưng không làm mờ trọng tâm của môn học.

## 9. Ranh giới phạm vi

Nên làm:

- Tập trung vào Greedy, A*, Lookahead, Agent Selector.
- Benchmark và phân tích kết quả nghiêm túc.
- Mô tả RL là phần mở rộng.

Không nên làm:

- Biến báo cáo thành báo cáo học máy.
- Nói quá nhiều về PPO/MAPPO nếu chưa có kết quả train tốt.
- Bỏ qua phần độ phức tạp.
- Chỉ demo bot chạy mà không có so sánh thuật toán.

## 10. Kết luận định hướng

Dự án hiện tại đã đủ nền tảng kỹ thuật. Việc cần làm tiếp theo không phải viết thêm quá nhiều code mới, mà là **đóng gói lại dự án như một bài tập lớn thuật toán**:

- Tài liệu hóa bài toán.
- Làm rõ thuật toán.
- Chạy benchmark.
- Phân tích độ phức tạp.
- Trình bày kết quả có bằng chứng.

Nếu làm đúng hướng này, đề tài phù hợp với học phần Thuật toán ứng dụng và có khả năng nhắm mức điểm A+.
