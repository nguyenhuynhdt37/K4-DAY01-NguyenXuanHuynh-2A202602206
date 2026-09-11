# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.11.16 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): class_id=468, class_name="cab", rank=1, score=0.5109, taxonomy_name="ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? Model gán một nhãn duy nhất cho cả ảnh, không phân biệt từng vật thể. Score 0.51 nghĩa là model đoán toàn ảnh thuộc lớp "cab" với xác suất 51%.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Tập dữ liệu huấn luyện — ở đây là ImageNet-1K với 1000 lớp. Model chỉ có thể trả về những lớp đó, không tự nghĩ ra lớp mới.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? ID để ghép khóa khi xử lý, tên để người đọc hiểu, taxonomy để biết đang dùng bảng lớp nào. Thiếu taxonomy thì ID 468 ở checkpoint khác có thể là lớp hoàn toàn khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Quy định rõ chủ thể nào được chọn làm nhãn (vd: chiếm diện tích lớn nhất, hoặc nằm trung tâm), tránh mỗi annotator chọn một kiểu.
- Vì sao model score không phải ground truth? Score là xác suất model ước tính, model vẫn có thể sai. Ground truth phải do người xác nhận theo guideline, không thể dùng score thay thế.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name="person", score=0.9126, bbox_xyxy=[385.33, 69.24, 498.92, 348.92], bbox_width=113.58, bbox_height=279.68
- Diễn giải vị trí box bằng lời: Người đứng ở khoảng 60–78% chiều rộng ảnh, từ gần đỉnh xuống khoảng 82% chiều cao — lệch phải, chiếm phần lớn nửa trên ảnh bếp.
- So sánh số prediction ở hai threshold: threshold 0.35 → 11 records; nâng lên 0.50 → còn 6 records (mất 5 box có score thấp gồm 2 cup, 2 bowl nhỏ).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp giữ thêm object, ít bỏ sót nhưng reviewer phải lọc nhiều box sai hơn. Threshold cao ngược lại, review nhanh nhưng dễ bỏ sót object nhỏ/mờ.
- Đề xuất một quy tắc box chặt: Box phải khít với phần nhìn thấy của object, không chừa khoảng trắng quá 3px ở bất kỳ cạnh nào, và không cắt cụt phần nào còn thấy rõ.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định ngưỡng tối thiểu để vẫn gán nhãn (vd: thấy ≥ 30% object). Nếu không xác định được lớp vì bị che quá nhiều thì escalation để reviewer quyết định bỏ qua hay gán nhãn với flag occluded.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id="kitchen-001", class_name="person", score=0.8993, 348 điểm, polygon bắt đầu tại [446.0, 70.0], [445.0, 71.0], ...
- Polygon bổ sung chi tiết gì so với box? Polygon bám theo đúng đường viền object, loại bỏ phần nền nằm trong box. Với object hình phức tạp như người, phần diện tích thực khác xa so với diện tích box.
- `instance_id` dùng để làm gì và không phải loại ID nào? Dùng để phân biệt từng instance riêng trong output (hai người cùng lớp sẽ là kitchen-001 và kitchen-002). Không phải class_id, cũng không phải tracking ID theo video.
- Đề xuất một quy tắc biên mask: Biên phải bám viền thực, sai số không quá 2px. Với chi tiết lồi lõm nhỏ hơn 4px (kẽ ngón tay, nếp quần áo mảnh) thì được phép bo lại cho đơn giản.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline phải quy định cách vẽ biên khi hai object chạm nhau (biên ảo hay chồng lấp). Vùng bị che khuất không xác định được biên thực → escalation để reviewer quyết định có gán hay đánh dấu difficult.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth                            | Lỗi hoặc điểm mơ hồ quan sát được                                                                               | Annotator làm gì?                                                  | Reviewer xem gì?                                                                           |
| --------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Phân loại ảnh         | Một nhãn lớp (class_id + class_name) cho toàn ảnh        | Ảnh kitchen bị model đoán thành "gong" — sai lớp hoàn toàn; ảnh nhiều vật không rõ chủ thể chính                | Chọn nhãn đúng theo guideline, không chép score của model          | Nhãn có khớp chủ thể chính không, có nhầm lớp gần nghĩa không                              |
| Phát hiện vật thể     | Bounding box xyxy + class_id cho từng object             | Person bị cắt mép (x_min ≈ 0) — không rõ có nên gán nhãn không; bowl nhỏ score 0.38 khó xác định thật hay nhiễu | Vẽ box khít từng object, gán đúng lớp, flag object bị cắt mép      | Box có bao đúng không, có bỏ sót object không, truncated/occluded xử lý đúng chưa          |
| Instance segmentation | Polygon pixel + class_id + instance_id cho từng instance | Polygon 348 điểm khó kiểm tra thủ công; cup nhỏ ở góc tối dễ bị bỏ sót                                          | Vẽ polygon bám biên, tạo instance riêng cho mỗi object dù cùng lớp | Polygon có bám đúng viền không, có trộn lẫn hai instance không, có bỏ sót object nhỏ không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng ảnh công khai đã xác nhận checksum; không upload ảnh chứa khuôn mặt, biển số xe hoặc dữ liệu cá nhân lên Colab hay GitHub công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach / Giảng viên phụ trách qua kênh hỗ trợ chính thức của lớp, không tự xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
