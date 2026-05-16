# HEXA UDON — RL Design Spec

---

## 1. Phân tích bài toán

### Thứ tự ưu tiên thắng
1. **Số loại udon (series) khác nhau** — ưu tiên tuyệt đối
2. Tổng loại tích lũy theo ngày (cumulative daily types)
3. Tổng số udon
4. Tổng thời gian submit

→ RL phải tối ưu **đa dạng hóa series** trước tiên, không phải số lượng thuần túy.

### Các thách thức chính

| Thách thức | Mô tả |
|---|---|
| Multi-agent | 3–8 xe, 2 loại (tuần tra / tiếp tế), cần phối hợp |
| Step budget dùng chung | Tổng step chia cho TẤT CẢ agent |
| Fuel không refill qua ngày | Tuần tra cần tiếp tế hoặc lên kế hoạch tiết kiệm |
| Traffic động & self-inflicted | Dùng nhiều 1 đường → hôm sau chậm hơn |
| Đối thủ ảnh hưởng traffic | Traffic tính theo TẤT CẢ đội → cần đoán behavior đối thủ |
| Sparse reward | Udon chỉ nhận khi đi qua spot |
| Time-limited inference | Phải submit trong time_limit_ms mỗi ngày |

---

## 2. Kiến trúc tổng quan — Hierarchical RL

```
┌──────────────────────────────────────────────────┐
│          Tầng 1: Strategic Planner (RL)           │
│  Đầu vào: full state ngày d                       │
│  Đầu ra: mục tiêu (spot) cho mỗi xe mỗi ngày     │
│  Granularity: 1 quyết định / ngày / xe            │
│  Thuật toán: MAPPO                                │
└──────────────────┬───────────────────────────────┘
                   │ waypoints (danh sách spot mục tiêu)
┌──────────────────▼───────────────────────────────┐
│          Tầng 2: Tactical Executor (A*)           │
│  Đầu vào: waypoints + terrain + traffic           │
│  Đầu ra: sequence of {cmd, dir} cho cả ngày       │
│  Thuật toán: Weighted A* trên hex grid            │
└──────────────────────────────────────────────────┘
```

**Lý do dùng Hierarchical**:
- Action space ngày-level nhỏ hơn nhiều so với step-level (tránh bùng nổ tổ hợp)
- Pathfinding đã có giải pháp tốt (A*) → không cần học lại
- Dễ debug, dễ fallback khi RL chưa tốt

---

## 3. State Space (đầu vào Tầng 1)

```python
State = {
    # --- Tĩnh (từ Phase 0, cache lại) ---
    terrain:           int[C]       # terrain mỗi ô (0,1,2,3)
    spot_series:       int[C]       # series_id của spot tại ô (-1 nếu không có)
    spot_max_inv:      int[C]       # max_inventory của spot (-1 nếu không có)
    hex_adjacency:     int[C][6]    # ID 6 ô lân cận (-1 nếu không tồn tại)

    # --- Động (cập nhật mỗi ngày) ---
    traffic:           int[C]       # 0/1/2 cho road cells, 0 cho các ô khác
    spot_inventory:    int[S]       # tồn kho còn lại mỗi spot
    agent_cell:        int[A]       # vị trí cell mỗi agent
    agent_fuel:        int[A]       # nhiên liệu còn (patrol), -1 (supply)
    opponent_cells:    int[O]       # vị trí xe đối thủ

    # --- Progress ---
    collected_mask:    bool[N_series]  # series đã thu >= 1 lần (cả game)
    daily_mask:        bool[N_series]  # series đã thu hôm nay

    # --- Time ---
    day:               int           # ngày hiện tại (1–D)
    steps_left:        int           # step còn lại hôm nay
    days_left:         int           # số ngày còn lại (= total_days - day)
}
```

**Encoding cho mạng neural**:
- Hex grid → CNN 2D (map cuộn về offset coordinates) hoặc GNN
- Agent states → MLP sau khi concat với spatial embedding của ô đó
- `collected_mask` → embedding vector (N_series dims)
- `day`, `steps_left`, `days_left` → normalized scalar

---

## 4. Action Space (đầu ra Tầng 1)

**Mỗi xe tuần tra**: chọn 1 spot mục tiêu để đi tới trong ngày
```
action_patrol_i ∈ {spot_0, spot_1, ..., spot_K-1, STAY}
```

**Mỗi xe tiếp tế**: chọn 1 xe tuần tra cần tiếp tế
```
action_supply_j ∈ {patrol_0, patrol_1, ..., patrol_P-1, IDLE}
```

**Kích thước action space** ≈ K^P × P^S (K spots, P patrol, S supply)
→ với K=20, P=5, S=2: ~20^5 × 5^2 ≈ 32M → quá lớn cho joint action

**Giải pháp**: factorized action — mỗi agent quyết định độc lập:
- Actor riêng cho mỗi loại xe (shared weights)
- Centralized critic đánh giá joint value

---

## 5. Reward Function

```python
def reward(state, action, next_state):
    r = 0.0

    # Ưu tiên 1: loại udon MỚI chưa từng thu (cả game)
    new_series = set(next_state.collected_mask) - set(state.collected_mask)
    r += 100.0 * len(new_series)

    # Ưu tiên 2: loại udon thu được trong ngày (tích lũy tie-break 2)
    r += 10.0 * len(set(next_state.daily_mask) - set(state.daily_mask))

    # Ưu tiên 3: số udon thực (tie-break 3)
    r += 1.0 * (next_state.total_udon - state.total_udon)

    # Phạt hiệu quả: bước vô ích
    r -= 0.05 * wasted_steps(state, next_state)

    # Phạt cứng: xe tuần tra hết xăng bị kẹt
    r -= 10.0 * fuel_depleted_count(next_state)

    return r
```

**Potential-based shaping** (tránh sparse reward):
```python
phi(s) = -min_distance_to_uncollected_series_spot(s)
shaping = gamma * phi(next_state) - phi(state)
r_shaped = r + shaping
```

**Reward cuối game** (terminal):
```python
r_terminal = (
    500.0 * len(total_unique_series)
  + 50.0  * cumulative_daily_series_score
  + 5.0   * total_udon
)
```

---

## 6. Thuật toán

### Strategic Planner — MAPPO

```
Algorithm: Multi-Agent PPO (MAPPO)
Training:  Centralized training (shared global state cho critic)
Execution: Decentralized (mỗi agent chỉ dùng local obs khi deploy)

Actor:   obs_i  → π(action_i | obs_i)        [per-agent]
Critic:  state  → V(state)                   [centralized, shared]
```

### Tactical Executor — Weighted A*

```
Cost function: f(n) = g(n) + h(n)
g(n): tổng step đã dùng từ nguồn đến n
      = step_cost(terrain_of_n, traffic_of_n)
h(n): heuristic = hex_distance(n, goal) × min_step_cost
      (admissible vì min_step_cost = 1 cho đường thông thoáng)

Constraints:
  - Không đi vào ao (terrain=2)
  - Không vượt quá fuel còn lại (xe tuần tra)
  - Không vượt quá steps_left (shared budget)
```

### Supply Coordination — Rule-based (Phase 1)

```
IF patrol_car.fuel < threshold AND supply_car.idle:
    supply_car.target = patrol_car cần tiếp tế gần nhất
```

→ Sau này có thể nâng lên RL nếu cần.

---

## 7. Network Architecture

```
              ┌─────────────┐
hex_map ──────│  CNN / GNN  │──── spatial_embedding[C]
              └──────┬──────┘
                     │
          agent_cell → lookup → agent_spatial_embed[A]
                     │
          concat(agent_spatial_embed, agent_fuel, agent_type)
                     │
              ┌──────▼──────┐
              │     MLP     │──── agent_feature[A]
              └──────┬──────┘
                     │
          concat(agent_feature, collected_mask_embed, day_info)
                     │
              ┌──────▼──────┐
              │ Actor head  │──── action_logits (softmax → target_spot)
              │ Critic head │──── V(state)
              └─────────────┘
```

**Hyperparameters (khởi điểm)**:

| Param | Giá trị |
|---|---|
| Hidden size | 256 |
| GNN layers | 3 |
| Actor LR | 3e-4 |
| Critic LR | 1e-3 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip epsilon | 0.2 |
| Entropy coef | 0.01 |
| Batch size | 32 episodes |

---

## 8. Môi trường Training (Simulator)

### Cần implement

```python
class HexaUdonEnv:
    def reset(config) -> state
    def step(actions) -> (next_state, reward, done, info)
    def render() -> visualization

# Các thành phần cần implement:
- HexGrid: tọa độ, adjacency, distance
- TrafficModel: tính traffic từ 2 ngày gần nhất
- FuelModel: tính fuel tiêu thụ khi di chuyển
- SpotModel: inventory, series, thu thập udon
- StepBudgetModel: shared step pool
- ScoringModel: unique series, cumulative, total udon
```

### Opponent modeling trong Simulator

- Phase 1: Opponent dùng heuristic (greedy spot gần nhất)
- Phase 2: Self-play — đối thủ là bản cũ của chính mình
- Phase 3: Population-based training

---

## 9. Lộ trình triển khai

```
Phase 1 — Simulator (tuần 1-2)
├── HexGrid cơ bản + A* pathfinding
├── TrafficModel
├── SpotModel + ScoringModel
└── FuelModel + StepBudget

Phase 2 — Baseline heuristic (tuần 3)
├── Greedy: mỗi xe tuần tra đi spot gần nhất thuộc series chưa thu
├── Đánh giá trên simulator → benchmark
└── Dùng làm "floor" để so sánh với RL

Phase 3 — Single-agent RL (tuần 4-5)
├── 1 xe tuần tra, không có tiếp tế
├── PPO cơ bản + potential shaping
└── So sánh với greedy baseline

Phase 4 — Multi-agent RL (tuần 6-8)
├── MAPPO cho nhiều xe tuần tra
├── Thêm xe tiếp tế (rule-based trước)
└── Self-play opponent

Phase 5 — Tích hợp + deploy (tuần 9-10)
├── HTTP client (GET state, POST action)
├── Timeout handling + fallback mechanism
├── Logging, replay, debug UI
└── Stress test trên LAN
```

---

## 10. Ràng buộc thời gian inference

```
time_limit_ms: 5000ms (ước tính)

Phân bổ:
  HTTP GET + parse JSON:  ~100ms
  RL forward pass (GPU):  ~50ms
  A* pathfinding (x6 xe): ~300ms  (worst: 32×32 map)
  Encode + HTTP POST:     ~100ms
  Buffer an toàn:         4450ms  ✓

Fallback strategy:
  t=0ms:    Chuẩn bị bài fallback (STAY tất cả)
  t+50ms:   Bắt đầu tính RL
  t_limit-500ms: Nếu chưa xong → dùng bài greedy nhanh
  t_limit-200ms: Submit bài tốt nhất có được
```

---

## 11. Điểm chiến thuật đặc biệt

### Khai thác traffic
- Dùng road nhiều → ngày sau chậm hơn → **chủ động giảm dùng road** nếu không cần thiết
- Ngược lại: nếu đối thủ dùng road nhiều → road đó bị kẹt → **lợi thế cho ta khi tránh road đó**

### Chọn loại agent trước trận
Đây là quyết định trước game, cần thuật toán riêng:
```
Input: map (số spot, phân bố series, địa hình)
Output: số xe tuần tra vs xe tiếp tế

Approach: Evolutionary search hoặc greedy heuristic
  - Nhiều spot xa, địa hình khó → cần nhiều xe tiếp tế hơn
  - Spot dày, gần nhau → ít xe tiếp tế, tập trung tuần tra
```

### Inventory strategy
- Đầu ngày spot refill về max → ưu tiên đi spot có max_inventory=1 (hàng ít hơn, cạnh tranh hơn)
- Nếu series đã thu rồi, bỏ qua spot đó → tiết kiệm step

---

## 12. Thông số chưa công bố — Fuel Max

**Vấn đề**: Tại thời điểm chuẩn bị, BTC **chưa công bố** giá trị `fuel_max` trong đề bài.
Khi thi thật, `fuel_max` sẽ được biết trước (nằm trong match config hoặc thông báo riêng).

**Những gì đã biết từ đề bài**:
- Trong cùng 1 trận, `fuel_max` **cố định** qua tất cả các ngày.
- Tất cả xe tuần tra trong cùng 1 trận có **cùng `fuel_max`**.
- Ngày 1, tất cả xe tuần tra bắt đầu với `fuel = fuel_max` (đầy bình).
- `fuel_max` có thể khác nhau giữa các trận (BTC quy định mỗi trận).

**Hệ quả khi training trước khi có thông số chính thức**:
- Không thể hardcode `fuel_max` → phải để model học robust trên nhiều giá trị.
- Khi BTC công bố giá trị chính thức → retrain tập trung vào đúng giá trị đó.

**Cách xử lý hiện tại (trước khi có thông số)**:

```python
# Training: sample fuel_max mỗi episode từ range hợp lý
FUEL_MAX_CANDIDATES = [10, 15, 20, 25, 30]
fuel_max = random.choice(FUEL_MAX_CANDIDATES)

# Normalize fuel: luôn dùng fuel_max thực tế, không hardcode
fuel_norm = agent.fuel / fuel_max
```

**TODO khi BTC công bố fuel_max**:
1. Cập nhật `FUEL_MAX_CANDIDATES = [<giá trị thật>]` trong `mappo.py`
2. Cập nhật `MatchConfig.fuel_max` trong `models.py`
3. Retrain model với đúng giá trị → kết quả tốt hơn nhiều so với robust training

---

## Câu hỏi thiết kế còn mở

| # | Câu hỏi | Ảnh hưởng |
|---|---|---|
| 1 | Số xe tiếp tế tối ưu? | Ảnh hưởng action space và strategy |
| 2 | Nên dùng CNN hay GNN cho hex grid? | Ảnh hưởng tốc độ training |
| 3 | Khi nào nên chuyển từ rule-based supply sang RL supply? | Complexity |
| 4 | Self-play hay fixed opponent trong training? | Sample efficiency |
| 5 | Nên encode hex grid bằng axial hay offset coordinates? | Implementation detail |
| 6 | Khoảng giá trị hợp lý của `fuel_max` để sample khi train? | Cần hỏi BTC hoặc đoán từ map size |
