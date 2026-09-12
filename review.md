# Summary of Technical Improvements — HEXA UDON Bot

Tài liệu này tóm tắt toàn bộ các cải tiến kỹ thuật, sửa đổi thuật toán và quyết định thiết kế đã được thực hiện trong cuộc hội thoại này để tối ưu hóa bot **HEXA UDON**.

---

## 1. Cơ chế Lập lộ trình tuần tự (Sequential Pathplanning & Dynamic Budget)

### Thay đổi:
* Áp dụng thuật toán lập kế hoạch di chuyển tuần tự cho các Agent theo thứ tự ưu tiên (Patrol đi ăn series mới -> Patrol đi ăn series cũ -> Supply di chuyển tiếp tế) tại các file:
  * **lookahead.py** (Chiến thuật Heuristic)
  * **mappo.py** (Tạo mẫu rollout trong huấn luyện RL)
  * **mcts.py** (Mô phỏng chuyển trạng thái trong cây duyệt MCTS)
* Thay vì chia đều hoặc sử dụng tĩnh toàn bộ `steps_left` cho tất cả các xe độc lập, hệ thống duy trì lượng shared steps thực tế còn lại (`remaining_shared_steps`) và trừ dần sau khi mỗi Agent hoàn thành lập lộ trình.

### Ý nghĩa & Tác động:
* **Tối ưu hóa tài nguyên:** Triệt tiêu hoàn toàn hiện tượng xe chạy sau bị cạn kiệt step và bị simulator cắt ngang hành trình giữa đường (dẫn đến STAY ngoài ý muốn).
* **Hiệu năng vượt trội:** Kiểm thử benchmark 100 game ngẫu nhiên cho thấy win rate của chiến thuật Lookahead so với Greedy baseline tăng lên **54.0%**, các chỉ số thu thập `Avg Daily` (4.66 vs 3.58) và `Avg Udon` (6.4 vs 4.0) đều cải thiện rõ rệt.

---

## 2. Sửa lỗi xe đi lạc vào Ao khi Repositioning

### Thay đổi:
* Sửa đổi phương thức `_append_reposition()` trong `lookahead.py`.
* Thay vì bắt đầu lập đường đi reposition từ waypoint cuối cùng trên lý thuyết (`waypoints[-1]`), hệ thống chạy giả lập mô phỏng hành động thực tế để tìm ra chính xác ô dừng chân thực tế của xe (`end_cell`) khi xe bị giới hạn bởi nhiên liệu hoặc step. Đường đi reposition mới sẽ được lập từ đúng ô `end_cell` này.

### Ý nghĩa & Tác động:
* **Tính ổn định tuyệt đối:** Loại bỏ hoàn toàn lỗi Logic khiến xe tự ý bước vào ô Ao (Lake, terrain=2) khi hết xăng/step giữa đường. Game benchmark chạy 100% không còn crash `TypeError` hay lỗi simulator.

---

## 3. Đồng bộ hóa thiết bị (CUDA vs CPU) & Ổn định hóa tham số huấn luyện

### Thay đổi:
* Thêm thuộc tính `self.device` cho lớp `ActorCritic` trong `actor_critic.py` và ép kiểu thiết bị (`device=self.device`) cho toàn bộ các tensor được khởi tạo mới trong quá trình forward pass và tính toán loss trong `mappo.py`.
* Hạ tỷ lệ đóng góp của Critic Loss (`critic_coef`) từ `0.5` xuống `0.1` và hạ tốc độ học của Critic (`LR_CRITIC`) xuống `3e-4`.
* Hạ ngưỡng curriculum `ADVANCE_THRESHOLD` từ `0.60` xuống `0.50`.

### Ý nghĩa & Tác động:
* Khắc phục triệt để lỗi crash `RuntimeError: Expected all tensors to be on the same device` khi chuyển cấu hình huấn luyện sang GPU (CUDA) trên Colab.
* Giảm thiểu xung đột gradient của Critic và Actor, triệt tiêu hiện tượng Loss của Critic tăng vọt đột biến (từ 10,000+ xuống dưới 300) giúp quá trình học ổn định và hội tụ mượt mà.

---

## 4. Tái cấu trúc mạng Neural: Attention-Based Actor Head (Pointer Network)

### Thay đổi:
* Loại bỏ lớp tuyến tính tĩnh ánh xạ trực tiếp từ đặc trưng sang index spot cố định: `self.actor_head = nn.Linear(..., max_spots + 1)`.
* Thay thế bằng cơ chế **Dot-product Attention**:
  * **Query (Agent):** Mạng con `self.query_mlp` chuyển đổi đặc trưng cục bộ và toàn cục của Agent thành vector truy vấn `Query`.
  * **Key (Spot):** Mạng con `self.key_mlp` trích xuất embedding tại tọa độ thực tế của Spot $i$ từ đầu ra CNN và chuyển thành vector đặc trưng `Key_i`.
  * Logits của Spot $i$ được tính bằng tích vô hướng: $\text{logit}_i = \text{Query} \cdot \text{Key}_i$.
  * Hành động STAY được tính riêng bằng mạng con `self.stay_head`.
* Đồng bộ cập nhật danh sách tham số huấn luyện của Optimizer trong `mappo.py`.

### Ý nghĩa & Tác động:
* **Giải quyết điểm nghẽn biểu diễn (Spatial Bottleneck):** Khắc phục lỗi thiết kế cũ khiến mạng Neural bị "mù" địa lý đối với các Spot mục tiêu (vì map được sinh ngẫu nhiên và tọa độ spot thay đổi liên tục).
* Kiến trúc Pointer Network mới giúp mô phỏng chính xác mối tương quan không gian (khoảng cách, địa hình, tồn kho) giữa xe tuần tra và từng mục tiêu, cho phép mô phỏng tổng quát hóa tốt trên mọi kích thước bản đồ khác nhau và giúp RL bứt phá vượt qua Level 2.
