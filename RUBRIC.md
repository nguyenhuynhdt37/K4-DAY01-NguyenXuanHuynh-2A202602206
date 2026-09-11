# Rubric formative – Báo cáo Ngày 1

> **Áp dụng cho bản Lab #1 này:** formative để học viên tự kiểm tra và GV phản hồi; không quy đổi sang
> điểm chính thức, không đặt điểm đạt/điểm liệt, quality floor hoặc quyết định GO/NO-GO. Nếu học phần cần
> ghi nhận hoàn thành, GV dùng Complete/Incomplete dựa trên evidence tối thiểu bên dưới.

| Năng lực quan sát | Đủ bằng chứng | Cần bổ sung | Chưa chứng minh |
| --- | --- | --- | --- |
| Phân loại ảnh | Dẫn đúng record; giải thích nhãn cấp ảnh, taxonomy, rank và model score | Có record nhưng thiếu một phần diễn giải hoặc nhầm phạm vi | Không có evidence hoặc coi prediction là ground truth |
| Phát hiện vật thể | Đọc đúng `sample_id`, lớp, score, `bbox_xyxy`, kích thước và ảnh hưởng của threshold | Evidence đúng nhưng diễn giải tọa độ/threshold chưa đủ | Đọc sai hệ tọa độ hoặc dùng threshold như quy tắc ground truth |
| Instance segmentation | Đọc đúng instance, polygon và khác biệt với box | Có evidence nhưng chưa giải thích instance/biên | Đồng nhất polygon với box hoặc `instance_id` với class/track ID |
| Guideline và QC | Đề xuất quy tắc có thể áp dụng; tách rule khỏi trường hợp cần escalation | Quy tắc còn mơ hồ hoặc chưa gắn với quan sát | Chép prediction thành rule hoặc tự quyết ca không đủ thông tin |
| Vòng đời và an toàn | Nối đúng evidence qua guideline, ground truth, model, QC/rework; không có dữ liệu nhạy cảm | Thiếu một mắt xích hoặc quy tắc bảo vệ dữ liệu | Công khai dữ liệu không được phép hoặc không phân biệt các artifact |
| Tái lập | Ghi package, checkpoint, sample/checksum, threshold, device và mọi thay đổi | Thiếu một vài trường có thể bổ sung | Không xác định được cấu hình tạo output |

## Evidence tối thiểu để nhận phản hồi

- `REPORT.md` có trích dẫn record từ cả ba JSON.
- Ba JSON và ba PNG vượt qua ô validation cuối notebook.
- Có ít nhất một lỗi hoặc điểm mơ hồ được đối chiếu bằng mắt.
- Có hành động riêng cho annotator và reviewer/QC.
- Không có ảnh riêng, dữ liệu khách hàng/VinFast hoặc dữ liệu cá nhân không cần thiết trong bài công khai.

Mô hình dự đoán sai không làm bài kém đi. Ghi nhận và phân tích lỗi đúng cách là evidence có giá trị.
