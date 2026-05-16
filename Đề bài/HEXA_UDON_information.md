PROCON - HẠNG MỤC THI ĐẤU "HEXA UDON"

**Mục tiêu**: đi qua các spot hiệu quả nhất, thu thập được nhiều udon nhất

**Đơn vị thời gian**: đơn vị thời gian nhỏ nhất dùng để ra lệnh là "step". Trận đấu được chia thành nhiều "ngày". Mỗi ngày có số step riêng và số step mỗi ngày có thể khác nhau. Một trận có thể kéo dài từ 4 đến 10 ngày

**Cấu trúc MAP**: toàn bộ bàn chơi gọi là "map", mỗi ô lục giác gọi là "cell", mỗi cell có tối đa 6 ô lân cận. Kích thước map tối thiểu và 8x8, tối đa là 32x32. Các cell được đánh số từ 0 đến hết (như trong Đề bài/map.png)

**Loại địa hình**: mỗi cell có 1 loại địa hình: đồng bằng, núi, ao, đường. Agent có thể di chuyển vào đồng bằng, núi, đường và không thể đi vào ao. Chi phí di chuyển:

| Địa hình             | Step di chuyển | Nhiên liệu |

| -------------------- | -------------- | ---------- |

| Đồng bằng            | 2              | 1          |

| Núi                  | 3              | 2          |

| Ao                   | Không vào được | -          |

| Đường (thông thoáng) | 1              | 2          |

| Đường (đông)         | 2              | 2          |

| Đường (kẹt xe)       | 4              | 2          |

**Trạng thái giao thông của đường** gồm thông thoáng, đông, kẹt xe. Cách xác định là dựa trên lượng giao thông của 02 ngày trước đó. Nếu là ngày đầu của trận đấu thì tất cả đường đều ở trạng thái thông thoáng, ngày 2 thì chỉ dựa vào ngày đầu để tính trạng thái giao thông của dường. Traffic của 1 cell đường = Tổng số step mà tất cả agent của tất cả các đội đã dừng tại ô đó trong 2 ngày gần nhất chia cho số đội

**Ngưỡng**: có 2 loại là ngưỡng đông và ngưỡng kẹt xe

Traffic < ngưỡng đông → thông thoáng

Ngưỡng đông ≤ traffic < ngưỡng kẹt xe → đông

Traffic ≥ ngưỡng kẹt xe → kẹt xe

Lưu ý: trạng thái đường chỉ cập nhất đầu mỗi ngày, trong cùng 1 ngày sẽ không đổi. Tất cả các đội dùng chung trạng thái giao thông

SPOT: chỉ một số ô đồng bằng có spot. Khi xe tuân tra tới spot (đi qua spot chứ không cần dừng lại) thì tự động nhận 1 phần udon. Trong cùng 1 ngày, mỗi xe tuần tra chỉ lấy tối đa 1 udon từ một spot, dù ghé nhiều lần cũng không thể nhận thêm. Tổng số udon mà xe có thể mang là không giới hạn.

**Series và loại Udon**: các spot được chia thành nhiều series, mỗi series tương ứng với 1 loại udon khác nhau. 01 spot chỉ thuộc đúng 01 series. Số lượng series nhỏ hơn hoặc bằng số lượng spot trên map. Vị trí spot và series của nó không đổi trong trận đấu.

**Tồn kho**: mỗi spot có mức tồn kho tối đa riêng, đầu mỗi ngày spot được refill về mức tối đa. Giới hạn tồn kho là từ 1 đến số lượng agent của 1 đội (bằng số xe tuần tra + số xe tiếp tế do BTC quy định). Khi lấy udon thì tồn kho giảm 1, nếu tồn kho = 0 thì không thể lấy thêm. Tồn kho là của riêng từng đội, đội khác lấy không ảnh hưởng tới đội mình

**AGENT**: mỗi đội có từ 3 đến 8 agent. vị trí ban đầu được chỉ định sẵn và luôn là ô đồng bằng không có spot. Có 2 loại agent là:

* Xe tuần tra: chức năng là đi tới spot và thu thập udon, xe tuần tra không thể tiếp nhiên liệu
* Xe tiếp tế: chức năng là tiếp tế nhiên liệu cho xe tuần tra, xe tiếp tế không thể lấy udon

Người chơi được chọn loại cho từng agent trước trận (có lẽ ta nên phát triển thêm 1 thuật toán riêng cho phần chọn loại agent này nhỉ?) và không được đổi trong suốt trận (tức là trong mọi ngày của trận đó đều ko thể đổi)

**Luật di chuyển**: Agent có thể di chuyển sang 6 ô lân cận, có thể đi vào ô đang có agent khác đứng, có thể ở trạng thái "chờ" nếu hết nhiên liệu hoặc hết quỹ số bước trong ngày. Agent không thể di chuyển tới ô không liền kề hoặc đi vào ao. Sau khi nhận lệnh, agent sẽ mất số step tương ứng với địa hình hiện tại trước khi tới ô tiếp theo. Xe tuần tra thì sẽ tốn nhiên liệu dựa trên địa hình tại thời điểm bắt đầu di chuyển. Về phần số bước, BTC sẽ quy định 1 số bước n là tổng số bước cho tất cả các agent, xe tiếp tế di chuyển cũng tốn số step ngang xe tuần tra.

**TIẾN TRÌNH TRẬN ĐẤU**:

\- trước trận: server cung cấp cấu trúc map, người chơi chọn loại cho từng agent

\- trong trận: server gửi thông tin map và agent, người chơi nộp hành động ngày 1. kết thúc ngày: server cập nhật vị trí agent và trạng thái giao thông. người chơi tiếp tục cho các ngày tiếp theo, lặp lại đến hết số ngày quy định

- Lưu ý:

> “Mỗi trận sẽ quy định:
>
> * số ngày
> * số step mỗi ngày
> * thời gian trả lời mỗi ngày”

=> chương trình của bạn chỉ có một khoảng thời gian hữu hạn để:

1. nhận state từ server
2. tính toán
3. submit action

---

Nếu hết thời gian mà chưa submit:

* rất có thể server sẽ:
* lấy bài valid cuối cùng trước đó
* hoặc coi như không có bài cho ngày đó

**THÔNG SỐ ĐẦU NGÀY**:

\- ngày 1:

&#x09;agent ở vị trí khởi tạo

&#x09;xe tuần tra đầy nhiên liệu

&#x09;đường đều thông thoáng

&#x09;spot đầy hàng

\- từ ngày 2 trở đi:

&#x09;agent bắt đầu tại vị trí cuối ngày trước

&#x09;nhiên liệu giữ nguyên từ ngày trước đó

&#x09;trạng thái đường tính từ traffic

&#x09;spot refill đầy tồn kho

**XÁC ĐỊNH THẮNG THUA:**

\- đội có nhiều loại udon khác nhau nhất sẽ thắng

\- nếu hòa thì xét tổng cộng dồn số loại odon thu được theo từng ngày

\- nếu vẫn hòa thì xét tổng số udon

\- nếu vẫn hòa thì đội nào có tổng thời gian submit ít hơn sẽ thắng

\- nếu vẫn hòa: random hoặc hòa

**SUBMIT:**

\- server báo valid hoặc invalid (format error) -> có thể submit lại trong thời gian cho phép của ngày đó

\- spam submit quá nhiều có thể bị xử thua

\- server chỉ lấy bài valid cuối cùng

**FORMAT INPUT/OUTPUT**: hiện tại BTC chưa cung cấp nên tạm thời ta sẽ thử thiết kế, tôi đã thiết kế thử trong file Đề bài/input\_output.png

**Giao tiếp mạng**: các đội kết nối PC qua LAN có dây, dùng HTTP POST/GET. Nghiên cấm sử dụng wifi

**Contest này là dạng viết bot tự động chứ không phải thao tác tay trên web.** BTC sẽ cung cấp: protocol giao tiếp ; sample client ; source code mẫu

Bot của team sẽ: GET dữ liệu map/trạng thái từ server ; Tự tính chiến thuật ; POST action lên server

Game có nhiều agent, nhiều ngày, traffic động nên chơi tay gần như không khả thi. Web UI nếu có thì chủ yếu để xem visualizer/debug/replay.

**LƯU Ý**

\- khai thác phần này "Người chơi càng dùng nhiều một tuyến đường thì hôm sau tuyến đó càng dễ bị “đông” hoặc “kẹt xe”." xem có giúp được gì ko
