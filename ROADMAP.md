# HEXA UDON — Roadmap 3 tháng (mục tiêu: Top 1)

> Ký hiệu: ✅ Đã xong | ⬜ Chưa làm

---

## THÁNG 1 — Nền tảng vững + Heuristic hoàn chỉnh

> Mục tiêu cuối tháng: có baseline tin cậy để benchmark mọi thứ sau này.
> Nếu simulator còn bug → mọi thứ ở tháng 2-3 đều vô nghĩa.

### Simulator
- ✅ `src/env/hex_grid.py` — hex grid, adjacency, hex_distance
- ✅ `src/env/models.py` — MatchConfig, MapData, DayState, DayOrder, AgentState
- ✅ `src/env/simulator.py` — apply_day(), traffic model, fuel model, scoring
- ⬜ Unit test simulator — từng rule một:
  - di chuyển vào ao → bị chặn
  - hết fuel → dừng giữa chừng
  - steps_left shared → xe sau bị giới hạn bởi xe trước
  - traffic tính đúng từ 2 ngày trước (ngày 1 = clear, ngày 2 = chỉ dùng ngày 1)
  - udon chỉ lấy 1 lần / spot / ngày / xe (dù ghé nhiều lần)
  - inventory refill về max đầu mỗi ngày mới
  - tồn kho riêng từng đội (đối thủ lấy không ảnh hưởng)
- ✅ Random map generator (`src/env/map_generator.py`):
  - sinh map ngẫu nhiên trong range 8×8 → 32×32
  - vary: tỉ lệ terrain, số spot, số series, phân bố spot
  - seed-based để reproduce

### Pathfinding
- ✅ `src/pathfinding/astar.py` — Weighted A*, multi_waypoint_path

### Heuristic
- ✅ `src/strategy/greedy.py` — greedy baseline
- ✅ `src/strategy/lookahead.py` — look-ahead heuristic:
  - ✅ global series assignment (không 2 xe waste vào cùng uncollected series)
  - ✅ multi-spot routing mỗi ngày
  - ✅ budget allocation theo priority
  - ✅ supply intercept prediction (push model: supply tự di chuyển đến patrol)
  - ✅ hybrid fuel fallback: nếu supply dự báo không kịp đến → patrol chủ động chèn rendezvous waypoint trên đường đến supply
  - ✅ end-of-day repositioning
- ✅ `src/strategy/agent_selector.py` — **free ELO, thường bị bỏ qua**:
  - Input: map layout, vị trí start, series distribution
  - Output: n_patrol vs n_supply tối ưu + vị trí nào làm patrol/supply
  - Approach: score vị trí (patrol = gần nhiều series; supply = trung tâm các xe), simulate top-K combo với LookaheadPlanner → chọn tốt nhất theo avg unique_series
- ✅ Benchmark runner (`src/benchmark.py`):
  - chạy N game với random maps, thống kê avg unique_series / avg udon / win rate
  - dùng để so sánh Greedy vs Lookahead vs RL sau này
- ⬜ Parameter tuning cho Lookahead (random search, ~500 games):
  - `new_series_bonus`, `secondary_max`, `low_fuel_threshold`, `reposition_weight`

### Tests
- ✅ `tests/test_hex_grid.py`:
  - neighbor đúng 6 hướng, edge cell không out-of-bound
  - `hex_distance` vs expected cube coordinate values
- ✅ `tests/test_astar.py`:
  - path tìm được, cost đúng, lake bị chặn
  - fuel budget cắt đúng chỗ, step budget cắt đúng chỗ
- ✅ `tests/test_simulator.py`:
  - di chuyển vào lake bị block
  - hết fuel dừng giữa chừng
  - `steps_left` shared đúng thứ tự giữa agents
  - traffic dùng đúng 2 ngày trước (ngày 1 = clear, ngày 2 = chỉ dùng ngày 1)
  - udon chỉ lấy 1 lần / spot / ngày / xe
  - inventory refill về max đầu ngày mới
  - tồn kho riêng từng đội (đối thủ lấy không ảnh hưởng)
- ✅ `tests/test_validator.py` — action hợp lệ / không hợp lệ

### Validator & Scoring
- ✅ `src/env/validator.py` — kiểm tra `List[DayOrder]` trước khi POST:
  - cell đích kề ô hiện tại (direction hợp lệ)
  - không đi vào lake
  - tổng step của tất cả agents ≤ `steps_left`
  - patrol: tổng fuel cost ≤ `agent.fuel`
  - output: `(is_valid: bool, errors: List[str])`
  - **Dùng cho chiến thuật**: submit greedy ngay (~100ms), override bằng lookahead nếu validator pass và còn thời gian
- ✅ `src/env/scoring.py` — extract pure function từ simulator:
  - `compute_score(state) -> Score` với 3 tiêu chí: unique_series → daily_series_sum → total_udon
  - Dùng để optimizer so sánh 2 plan mà không cần reset simulator

### Visualizer & Replay
- ✅ `visualizer/terminal.py` — terminal ASCII, không cần GUI:
  - Grid: `P1`=patrol, `S1`=supply, `**`=spot còn hàng, `..`=spot hết, `##`=lake, `==`/`=B`/`=C`=road, `^^`=mountain
  - Thanh fuel `[████░░]` từng xe patrol
  - Series đã collect / còn lại
- ✅ `replay/` — ghi JSON mỗi ngày (day, state, orders, agent_positions, collected_series, traffic):
  - `ReplayRecorder.record(state, orders)` → `save(path)`
  - `ReplayPlayer.load(path)` → iterate frames, `agent_trace(id)`, `summary()`
  - Dùng để debug bot sau trận và trong training RL

### Traffic Predictor
- ✅ `src/env/traffic_predictor.py` — dự đoán traffic ngày mai:
  - Heuristic: giả định đối thủ dùng greedy đến spot gần nhất → A* path → đếm bước ở road cell
  - `predict_and_scale()`: scale theo số đội ẩn (chỉ thấy 1 phần vị trí đối thủ)
  - Cộng vào `TrafficModel` → biết ngày mai đường nào kẹt để lookahead tránh chủ động

### Input/Output & Client
- ✅ `spec_input_output.md` — dự đoán JSON format
- ✅ `src/client/http_client.py` — GET state, POST action, retry

### Checkpoint cuối tháng 1
```
✓ Simulator pass toàn bộ unit test
✓ Lookahead > Greedy trên benchmark 100 random games
✓ Random map generator chạy được, tái hiện được
```

---

## THÁNG 2 — RL + Curriculum + Self-play

> Mục tiêu cuối tháng: RL beat Lookahead trên map vừa (16×16).
> Nếu không đạt → debug reward/architecture trước khi tiếp tục.

### RL Infrastructure
- ✅ `src/rl/actor_critic.py` — CNN encoder + per-agent MLP + actor/critic
- ✅ `src/rl/mappo.py` — MAPPO, rollout buffer, GAE-Lambda
- ⬜ Reward shaping cải thiện:
  - potential-based: `phi(s) = -min_distance_to_nearest_uncollected_series_spot`
  - `r_shaped = r + gamma * phi(s') - phi(s)`
- ⬜ Logging: TensorBoard hoặc wandb — loss, reward, unique_series per episode
- ⬜ Curriculum learning engine (`src/rl/curriculum.py`):
  ```
  Level 1: map 8×8,  4 ngày,  3 agents, 3 series   ← bắt đầu đây
  Level 2: map 12×12, 5 ngày, 4 agents, 4 series
  Level 3: map 16×16, 6 ngày, 5 agents, 5 series
  Level 4: map 24×24, 8 ngày, 6 agents, 6 series
  Level 5: map 32×32, 10 ngày, 8 agents, 8 series
  Điều kiện advance: RL win rate > 60% vs Lookahead ở level hiện tại
  ```

### Self-play + Opponent Modeling
- ⬜ Self-play pool (`src/rl/selfplay.py`):
  - duy trì pool 5 checkpoint cũ làm "đối thủ"
  - mỗi episode: sample ngẫu nhiên 1 checkpoint từ pool
  - cập nhật pool mỗi 1000 episodes
- ⬜ Opponent features trong state:
  - vị trí xe đối thủ → RL tự học suy ra traffic ngày mai
  - (đã có `opponent_cells` trong DayState, chỉ cần encode vào map feature)
- ⬜ Simple opponent traffic estimator (heuristic, dùng song song):
  - giả sử đối thủ dùng greedy → estimate step counts của họ
  - cộng vào traffic model → dự đoán traffic ngày mai sớm hơn

### Population-based Training (PBT) — nếu có đủ compute
- ⬜ Train song song N agent (N=4–8) với hyperparams khác nhau
  - mỗi agent có learning rate, entropy coef, reward weights riêng
- ⬜ Exploit/explore mỗi K episodes:
  - agent yếu copy weights từ agent mạnh hơn
  - perturb nhẹ hyperparams sau khi copy
- ⬜ Kết quả: tìm được hyperparams tốt hơn grid search, đa dạng behavior trong pool self-play
- Có thể bỏ qua nếu không có multi-GPU; single-GPU thì làm tuần tự

### Checkpoint cuối tháng 2
```
✓ RL beat Greedy trên map 8×8 (Level 1)
✓ RL đang train được lên Level 3 (16×16)
✓ Self-play chạy ổn định, pool đang cập nhật
✓ RL beat Lookahead trên ít nhất Level 2
```

---

## THÁNG 3 — MCTS + Tích hợp + Contest Prep

> Mục tiêu: hệ thống hoàn chỉnh, tested, reliable ngày thi.

### MCTS trên value function học được (AlphaZero-style)
> Episode chỉ 4-10 ngày → cây MCTS nông → khả thi trong time_limit.

- ⬜ `src/strategy/mcts.py`:
  - Node = DayState sau khi apply assignment action
  - Expand = top-K assignments theo Lookahead score (beam, không brute-force)
  - Evaluate = critic network từ MAPPO (thay random rollout)
  - Select = UCB1: `score = Q + c * sqrt(ln(N_parent) / N_node)`
  - Time budget: chạy đến `time_limit_ms - 500ms` rồi dừng, trả best action
- ⬜ Benchmark: MCTS vs pure RL vs Lookahead trên 100 games map 24×24 và 32×32
- ⬜ Inference time profiling: đảm bảo MCTS < 3000ms cho worst case

### Khi BTC công bố thông số còn thiếu
- ⬜ Cập nhật `fuel_max` → retrain RL với giá trị thật (1-2 tuần)
- ⬜ Cập nhật `steps_per_day` → re-tune budget allocation trong Lookahead
- ⬜ Cập nhật `n_teams`, traffic thresholds → re-calibrate traffic model

### Contest infrastructure
- ⬜ Fallback chain hoàn chỉnh và tested dưới time pressure:
  ```
  MCTS (chạy đến timeout)
    ↓ exception hoặc timeout
  RL policy (deterministic, ~50ms)
    ↓ exception
  Lookahead heuristic (~5ms)
    ↓ exception
  Greedy (~1ms) ← không bao giờ crash
  ```
- ⬜ Pre-submit pattern: submit Greedy ngay khi nhận state (~100ms), override bằng bài tốt hơn nếu còn thời gian
- ⬜ Retry logic: nếu server báo invalid → fix và submit lại trong time_limit
- ⬜ Debug logger: lưu toàn bộ (state, orders, response) mỗi ngày → replay sau trận
- ⬜ Edge case testing:
  - map chỉ 1 series → chiến thuật hoàn toàn khác
  - fuel_max rất thấp → supply car critical
  - map toàn đường → traffic dominates
  - tất cả spot bị đội khác khai thác nhanh → phải linh hoạt

### Checkpoint cuối tháng 3
```
✓ Tournament: MCTS > RL > Lookahead > Greedy (hoặc ít nhất MCTS ≥ RL)
✓ Toàn bộ fallback chain tested, không crash dưới time pressure
✓ Agent selector chạy được trước trận
✓ Sẵn sàng ngày thi
```

---

## Điểm quyết định quan trọng

| Thời điểm | Câu hỏi | Nếu NO |
|---|---|---|
| Tuần 2 tháng 1 | Simulator pass hết unit test? | Dừng mọi thứ, fix trước |
| Cuối tháng 1 | Lookahead > Greedy trên benchmark? | Debug lookahead |
| Giữa tháng 2 | RL > Greedy trên map 8×8? | Debug reward shaping / architecture |
| Cuối tháng 2 | RL > Lookahead trên map 16×16? | Không làm MCTS, tập trung tune RL + Lookahead hybrid |
| Giữa tháng 3 | MCTS > RL? | Bỏ MCTS, dùng RL làm primary |

---

## Thứ tự ưu tiên tuyệt đối

```
1. Simulator đúng        → nền tảng của mọi thứ
2. Benchmark runner      → không có thước đo thì không biết đang đi đúng không
3. RL curriculum         → học từ dễ đến khó, không nhảy thẳng vào map lớn
4. Self-play             → bắt buộc để học traffic dynamics
5. Agent type selector   → free ELO, ít team làm
6. MCTS                  → squeeze thêm 5–10%, làm sau khi RL ổn
7. Contest infra         → đừng thua vì bug mạng hoặc timeout
```
