# Hướng dẫn học viên – Ngày 1

## Trước khi chạy

Đây là bài thực hành quan sát bằng Google Colab. Không cần cài CVAT hoặc Python trên máy cá nhân. Lần đầu
chạy sẽ cài đúng phiên bản Ultralytics, tải ba checkpoint YOLO11n và ba ảnh mẫu; hãy chờ ô mã hoàn tất.
Luồng chuẩn giống nhau trên Windows, macOS và Ubuntu vì mọi thao tác bắt buộc đều chạy trong trình duyệt;
Git/terminal chỉ là lựa chọn thêm cho học viên đã quen.

Preflight:

- [ ] Đã mở notebook từ repository nguồn bằng Colab.
- [ ] Biết cách mở và sửa `REPORT.md` mà notebook tạo sẵn trong panel Files của Colab.
- [ ] Không đưa ảnh hoặc dữ liệu nhạy cảm vào Colab; tên repository chỉ chứa họ tên và MSSV bắt buộc.
- [ ] Có quyền lưu tệp vào Google Drive dùng để nộp bài.
- [ ] Đã tạo repository bài làm từ template và biết thư mục nộp bài là `report/`.
- [ ] Biết cách giải nén ZIP bằng File Explorer (Windows), Finder (macOS) hoặc Archive Manager (Ubuntu).

## Quy trình 60 phút

Mốc thời gian dưới đây là hướng dẫn vận hành, chưa phải ngưỡng chấm chính thức:

| Phút | Việc cần làm | Evidence |
| ---: | --- | --- |
| 0–5 | Mở notebook, đọc scope và chạy setup | phiên bản + device |
| 5–15 | Chạy classification, đọc top-5 | JSON + biểu đồ |
| 15–30 | Chạy detection, đọc một box và so sánh threshold | JSON + ảnh box |
| 30–45 | Chạy instance segmentation, đọc một polygon | JSON + ảnh mask |
| 45–55 | Mở `REPORT.md` trong Colab và hoàn thành bằng dẫn chứng | báo cáo |
| 55–60 | Tạo ZIP, giải nén vào `report/`, commit, push và nộp link trên VLearn | ZIP trên Drive + link repo |

## Cách đọc đầu ra

| Tệp | Một record là gì? | Trường cần đọc |
| --- | --- | --- |
| `classification_predictions.json` | một lớp được xếp hạng cho toàn ảnh | `class_id`, `class_name`, `rank`, `score`, `taxonomy_name` |
| `detection_predictions.json` | một vật thể được model phát hiện | `class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height` |
| `segmentation_predictions.json` | một instance và đa giác của nó | `instance_id`, `class_name`, `score`, `polygon_xy` |

`classification_top5.png` minh họa sample `traffic`; hai hình detection/segmentation minh họa sample
`kitchen`. Khi trích JSON, luôn ghi `sample_id` để record khớp đúng hình.

### Phân loại ảnh

- Checkpoint `yolo11n-cls.pt` xếp hạng các lớp ImageNet-1K cho toàn ảnh.
- `rank=1` là lớp có model score cao nhất, không phải ground truth đã được con người xác nhận.
- “Một lớp cho mỗi ảnh” chỉ là ví dụ của bài toán single-label classification đang quan sát; một dự án khác
  có thể quy định multi-label hoặc cách xử lý ảnh nhiều chủ thể khác.

### Phát hiện vật thể

- `bbox_xyxy = [x_min, y_min, x_max, y_max]` dùng pixel, gốc tọa độ ở góc trên trái.
- Hạ threshold thường giữ thêm prediction; nâng threshold thường loại thêm prediction.
- Threshold là cấu hình lọc prediction, không phải quy tắc bỏ qua object khi tạo ground truth.

### Phân đoạn theo từng đối tượng

- `polygon_xy` là danh sách các điểm `[x, y]` theo pixel quanh biên instance.
- Hai object cùng lớp vẫn cần hai instance riêng.
- `instance_id` có dạng `<sample_id>-NNN`; đây là ID trong output lab, không phải class ID hoặc tracking ID.

## Câu hỏi báo cáo

### A. Phân loại ảnh

1. Record hạng 1 mô tả toàn ảnh hay một object riêng?
2. Vì sao cần giữ cả `class_id`, `class_name` và tên taxonomy?
3. Nếu ảnh có nhiều chủ thể, guideline cần nói rõ điều gì?
4. Vì sao model score không được chép thành ground truth?

### B. Phát hiện vật thể

1. Chọn một record và diễn giải lớp, score, vị trí, chiều rộng và chiều cao.
2. Nếu ảnh có ba object cùng lớp, prediction/ground-truth unit được tổ chức ra sao?
3. So sánh ít nhất hai threshold. Khi giữ thêm prediction, reviewer có thêm việc gì?
4. Đề xuất một quy tắc box chặt và một câu hỏi cần escalation khi object bị che khuất/cắt mép.

### C. Phân đoạn theo từng đối tượng

1. `polygon_xy` bổ sung chi tiết gì so với box?
2. `instance_id` phân biệt hai object cùng lớp như thế nào?
3. Việc gán nhãn và QC khó hơn ra sao khi cần bám biên?
4. Đề xuất một quy tắc biên mask và một câu hỏi escalation cho vùng mơ hồ.

### D. Tổng hợp và an toàn dữ liệu

Mô tả chuỗi `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`. Với mỗi tác vụ,
nêu định dạng nhãn, một lỗi/điểm mơ hồ, hành động của annotator và điều reviewer cần xem. Kết thúc bằng một
quy tắc bảo vệ dữ liệu.

## Xử lý lỗi thường gặp

- **Cài package/tải weight lâu:** chờ ô hiện dấu hoàn tất; không bấm chạy lặp nhiều lần.
- **Colab không cấp GPU:** tiếp tục bằng CPU. Nếu có nguy cơ trễ timeline, báo mentor; không tự đổi model.
- **Tải ảnh lỗi hoặc checksum sai:** chạy lại đúng ô tải ảnh một lần. Nếu vẫn lỗi, chụp thông báo và báo
  mentor; không thay bằng ảnh cá nhân.
- **Không thấy output:** kiểm tra đã chạy đủ các ô trước đó và mở panel Files của Colab.
- **Không kết nối được Drive:** cho phép Colab truy cập đúng tài khoản Drive dùng để nộp bài; nếu tài
  khoản trường chặn quyền, chụp thông báo và báo mentor.
- **Ô đóng gói báo khóa chưa hợp lệ:** thay `KHOA` bằng mã khóa do chương trình cung cấp, chỉ dùng chữ
  cái và số, không dùng dấu cách.
- **Ô đóng gói báo báo cáo còn trống:** lưu thay đổi trong `REPORT.md`, sau đó chạy lại riêng ô cuối.
- **Giải nén xong có thêm một thư mục bọc:** bạn đang dùng ZIP cũ. ZIP đúng phải mở ra thấy ngay
  `REPORT.md` và `day1_lab_outputs/`; báo mentor nếu chạy notebook mới mà vẫn gặp cấu trúc cũ.
- **`git status` không thấy bài làm:** kiểm tra hai mục đã nằm trực tiếp dưới `report/`, sau đó chạy
  `git add report`.
- **Số prediction khác bạn bên cạnh:** đối chiếu package, checkpoint, threshold, sample checksum và device
  trước khi kết luận.

## Tự kiểm tra

- [ ] Có evidence từ cả ba JSON và đúng `sample_id`.
- [ ] Có đúng ba PNG trong `day1_lab_outputs/visuals/`.
- [ ] Phân biệt `prediction`, `ground truth`, `class`, `model score`, `box`, `polygon` và `instance`.
- [ ] Không coi confidence/model score là điểm chất lượng nhãn.
- [ ] Nêu ít nhất một lỗi hoặc điểm mơ hồ thực sự quan sát được.
- [ ] Ghi lại mọi thay đổi về code, checkpoint, threshold và môi trường.
- [ ] Có `IMAGE_ATTRIBUTION.md` trong output; họ tên/MSSV chỉ nằm ở tên repository, không nằm trong output.
- [ ] ZIP trong `MyDrive/AI20K-Day1/` mở ra thấy trực tiếp `REPORT.md` và `day1_lab_outputs/`.
- [ ] Hai mục đó nằm trong `report/`; đã commit, push và nộp link repository trên VLearn.
