# HEXA UDON — Spec Input / Output (nội bộ team)

> Đây là spec **dự đoán** dựa trên đề bài. Khi BTC công bố protocol chính thức, cần đối chiếu và cập nhật.

---

## Tổng quan luồng giao tiếp

```
[Server]  ---(GET /match-config)--→  [Bot]   (trước trận, 1 lần)
[Server]  ---(GET /state/day/{d})-→  [Bot]   (mỗi ngày, nhận full state)
[Bot]     ---(POST /action/day/{d})→ [Server] (mỗi ngày, trong time_limit)
```

**Lưu ý thời gian**: Mỗi ngày bot phải nhận state → tính toán → submit action trong `time_limit_ms`.
Nếu quá hạn, server lấy bài valid cuối cùng (hoặc coi như không nộp).
→ **Bot cần có fallback action sẵn trước khi tính toán**.

---

## PHASE 0 — Cấu hình trận (trước trận, 1 lần)

### GET /match-config → Response

```json
{
  "match_config": {
    "width": 10,
    "height": 8,
    "total_days": 6,
    "steps_per_day": [120, 120, 100, 100, 80, 80],
    "n_teams": 4,
    "traffic_threshold_busy": 3.0,
    "traffic_threshold_congested": 7.0
  },
  "map": {
    "cells": [
      { "id": 0,  "terrain": 0 },
      { "id": 5,  "terrain": 1 },
      { "id": 9,  "terrain": 3 },
      { "id": 22, "terrain": 2 }
    ],
    "spots": [
      { "cell_id": 12, "series_id": 2, "max_inventory": 3 },
      { "cell_id": 27, "series_id": 1, "max_inventory": 2 },
      { "cell_id": 55, "series_id": 2, "max_inventory": 1 }
    ]
  },
  "my_agents": [
    { "id": 1, "type": 0, "start_cell": 0,  "fuel_capacity": 20 },
    { "id": 2, "type": 0, "start_cell": 7,  "fuel_capacity": 20 },
    { "id": 3, "type": 1, "start_cell": 14 }
  ]
}
```

### Bảng giá trị

| Trường | Kiểu | Mô tả |
|---|---|---|
| `width`, `height` | int | Kích thước map (8–32) |
| `total_days` | int | Số ngày của trận (4–10) |
| `steps_per_day` | int[] | Steps budget mỗi ngày (chia cho tất cả agent) |
| `n_teams` | int | Số đội thi đấu (để tính traffic) |
| `traffic_threshold_busy` | float | Ngưỡng đông |
| `traffic_threshold_congested` | float | Ngưỡng kẹt xe |
| `cells[].id` | int | Số thứ tự ô (0 → width×height−1), đánh từ trái sang phải, trên xuống dưới |
| `cells[].terrain` | int | 0=đồng bằng, 1=núi, 2=ao, 3=đường |
| `spots[].cell_id` | int | Ô chứa spot (phải là terrain=0) |
| `spots[].series_id` | int | Loại udon của spot này |
| `spots[].max_inventory` | int | Tồn kho tối đa (1 đến n_agents) |
| `my_agents[].type` | int | 0=xe tuần tra, 1=xe tiếp tế |
| `my_agents[].fuel_capacity` | int | Dung tích bình xăng (chỉ tuần tra) |

**Lưu ý**: `map.cells` và `map.spots` chỉ cần gửi 1 lần ở phase 0 vì tĩnh suốt trận.

---

## PHASE 1 — State mỗi ngày (full-state)

### GET /state/day/{d} → Response

```json
{
  "day": 2,
  "steps_left": 100,
  "time_limit_ms": 5000,
  "traffic": [
    { "cell_id": 45, "status": 1 },
    { "cell_id": 67, "status": 2 }
  ],
  "spots": [
    { "cell_id": 12, "inventory": 2 },
    { "cell_id": 27, "inventory": 3 },
    { "cell_id": 55, "inventory": 1 }
  ],
  "my_agents": [
    { "id": 1, "type": 0, "cell": 12, "fuel": 5 },
    { "id": 2, "type": 0, "cell": 20, "fuel": 18 },
    { "id": 3, "type": 1, "cell": 13 }
  ],
  "opponents": [
    { "id": 101, "cell": 30 },
    { "id": 102, "cell": 35 },
    { "id": 201, "cell": 10 }
  ],
  "my_score": {
    "unique_series": [1, 2],
    "total_udon": 5,
    "daily_series_history": [[2], [1, 2]]
  }
}
```

### Bảng giá trị

| Trường | Kiểu | Mô tả |
|---|---|---|
| `day` | int | Ngày hiện tại (1–D) |
| `steps_left` | int | Tổng step còn lại cho ngày này (toàn đội) |
| `time_limit_ms` | int | Thời gian tối đa để submit (ms) |
| `traffic[].cell_id` | int | Ô đường có traffic (chỉ terrain=3) |
| `traffic[].status` | int | 0=thông thoáng, 1=đông, 2=kẹt xe |
| `spots[].inventory` | int | Tồn kho còn lại của đội mình tại spot đó |
| `my_agents[].cell` | int | Ô hiện tại của agent |
| `my_agents[].fuel` | int | Nhiên liệu còn lại (chỉ xe tuần tra) |
| `opponents[].cell` | int | Vị trí xe đối thủ (để tính traffic nội bộ) |
| `my_score.unique_series` | int[] | Danh sách series đã thu được ít nhất 1 lần |
| `my_score.daily_series_history` | int[][] | Series thu được mỗi ngày (để tính tie-break 2) |

**Lưu ý tối ưu**: `traffic` chỉ gửi các ô có status ≠ 0. Ô đường không có trong list → mặc định status=0.

---

## PHASE 1 — Action mỗi ngày

### POST /action/day/{d} — Request body

```json
{
  "orders": [
    {
      "agent_id": 1,
      "actions": [
        { "cmd": "move", "dir": 2 },
        { "cmd": "move", "dir": 2 },
        { "cmd": "move", "dir": 0 },
        { "cmd": "stay" }
      ]
    },
    {
      "agent_id": 2,
      "actions": [
        { "cmd": "move", "dir": 4 }
      ]
    },
    {
      "agent_id": 3,
      "actions": [
        { "cmd": "move", "dir": 2 },
        { "cmd": "stay" }
      ]
    }
  ]
}
```

### Bảng giá trị

| Trường | Kiểu | Mô tả |
|---|---|---|
| `agent_id` | int | ID xe thực hiện lệnh |
| `actions` | object[] | Chuỗi lệnh thực hiện trong ngày |
| `cmd` | string | `"move"` hoặc `"stay"` |
| `dir` | int (0–5) | Hướng di chuyển (chỉ khi cmd=move) |

### Hướng di chuyển trên hex grid

```
     0 (lên trên trái)   1 (lên trên phải)
  5 (trái) [CELL] 2 (phải)
     4 (xuống trái)   3 (xuống phải)
```

*(cần xác nhận lại với BTC khi có protocol chính thức)*

### Ràng buộc output

| Rule | Mô tả |
|---|---|
| Tổng step tiêu thụ ≤ `steps_left` | Mỗi lệnh move tốn step theo terrain đích |
| Nhiên liệu xe tuần tra ≥ 0 | Mỗi move tốn fuel theo terrain nguồn |
| Không di chuyển vào ao (terrain=2) | Server báo invalid |
| Agent không cần order | Mặc định STAY cả ngày nếu không có trong `orders` |
| Actions list rỗng | Hợp lệ → agent STAY cả ngày |

---

## Chi phí di chuyển

| Terrain đích | Step tiêu thụ | Fuel tiêu thụ (xe tuần tra) |
|---|---|---|
| Đồng bằng (0) | 2 | 1 |
| Núi (1) | 3 | 2 |
| Ao (2) | **Không vào được** | — |
| Đường thông thoáng (3, status=0) | 1 | 2 |
| Đường đông (3, status=1) | 2 | 2 |
| Đường kẹt xe (3, status=2) | 4 | 2 |

**Lưu ý**: Step tính theo terrain **đích** (ô đang bước vào), fuel tính theo terrain **nguồn** (ô đang đứng trước khi di chuyển).

---

## Server Response sau khi submit

```json
{
  "status": "valid",
  "message": ""
}
```

```json
{
  "status": "invalid",
  "message": "Agent 1: insufficient fuel at step 3"
}
```

Bot nên retry nếu nhận `invalid` và còn trong `time_limit_ms`.

---

## Câu hỏi chưa rõ (cần hỏi BTC)

| # | Câu hỏi |
|---|---|
| 1 | Server gửi full state hay chỉ delta mỗi ngày? |
| 2 | Cơ chế tiếp tế: xe tiếp tế và tuần tra phải cùng ô không? Cần lệnh REFUEL hay tự động? |
| 3 | Xe tiếp tế tiếp bao nhiêu fuel mỗi lần? Giới hạn bao nhiêu? |
| 4 | Nếu xe tuần tra hết xăng giữa chừng, actions tiếp theo bị bỏ qua hay server báo lỗi? |
| 5 | Direction encoding: 0–5 theo chiều nào? Cần sơ đồ chính xác |
| 6 | Phase 0 (config) là 1 request riêng hay gộp vào GET state ngày 1? |
| 7 | Đối thủ: server trả vị trí tất cả xe đối thủ hay chỉ xe gần? |
| 8 | `opponents` có bao gồm xe tiếp tế của đối thủ không? |
| 9 | Behavior khi actions list quá dài (tốn nhiều step hơn steps_left)? |
| 10 | Có endpoint xem lại replay / history sau trận không? |
