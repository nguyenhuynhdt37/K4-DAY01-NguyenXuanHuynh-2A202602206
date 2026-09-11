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

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `class_id`: 468, `class_name`: "cab", `rank`: 1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
  - Đây là prediction cấp **toàn ảnh** (image-level): model gán **một nhãn duy nhất** cho toàn bộ bức ảnh `traffic`, không phân biệt từng vật thể riêng lẻ. Score 0.51 nghĩa là model tin tưởng 51% rằng ảnh này thuộc lớp "cab" (taxi) trong số 1 000 lớp ImageNet.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - **Tập dữ liệu huấn luyện (ImageNet-1K)** định nghĩa class list, không phải model tự tạo ra. Khi checkpoint `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K, nó chỉ có thể trả về một trong 1 000 lớp đó. Bất kỳ dự án gán nhãn thực tế nào cũng phải dùng taxonomy của riêng mình, không phụ thuộc vào class list của checkpoint.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` (số nguyên) để tra cứu nhanh và ghép khóa; `class_name` để người đọc hiểu nội dung; `taxonomy_name` để biết bảng lớp nào đang được dùng. Nếu chỉ lưu `class_id` mà không lưu taxonomy, con số 468 sẽ không có nghĩa khi đổi checkpoint hoặc dùng taxonomy khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline phải quy định **chủ thể chính** (dominant subject) là gì; ví dụ: vật thể chiếm diện tích lớn nhất, hoặc vật thể nằm trung tâm ảnh. Cần quy định rõ để tránh annotator chọn nhãn khác nhau cho cùng một ảnh, và nên bổ sung quy tắc escalation khi nhiều vật thể cùng "quan trọng".
- Vì sao model score không phải ground truth?
  - Score là **xác suất model ước tính** dựa trên các đặc trưng học được từ dữ liệu huấn luyện. Model có thể sai (ví dụ score cao cho "cab" nhưng ảnh thực sự là xe buýt). Ground truth phải do **con người xác nhận theo guideline**; dùng score để bỏ qua bước xác nhận đó là sai quy trình gán nhãn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - `class_name`: "person", `score`: 0.912624, `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92], `bbox_width`: 113.58 px, `bbox_height`: 279.68 px
- Diễn giải vị trí box bằng lời:
  - Hộp giới hạn bắt đầu từ tọa độ **(x_min=385, y_min=69)** ở góc trên-phải ảnh và kéo dài đến **(x_max=499, y_max=349)** ở phần giữa-dưới. Hộp có chiều rộng ~114 px và chiều cao ~280 px, cho thấy đây là người đứng chiếm khoảng 1/5 chiều rộng ảnh (640 px) và 65% chiều cao ảnh (427 px), nằm lệch phải.
- So sánh số prediction ở hai threshold:
  - Ở threshold 0.35 (hiện tại): **11 predictions** cho sample `kitchen` gồm 2 person, 5 bowl, 2 oven, 2 cup. Nếu nâng threshold lên 0.50: chỉ còn **6 predictions** (person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61), loại bỏ 5 record có score < 0.50.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi hạ threshold (giữ thêm prediction): độ bao phủ tăng, ít bỏ sót object, nhưng reviewer phải kiểm tra nhiều box hơn (gồm cả các box sai/nhầm). Khi nâng threshold: reviewer xem ít box hơn nhưng có nguy cơ bỏ sót object thật. Đây là đánh đổi giữa **recall** và **precision** trong quy trình QC.
- Đề xuất một quy tắc box chặt:
  - Box phải bao sát object theo hull convex nhỏ nhất có thể, không chừa khoảng trống quá 5% chiều rộng hoặc chiều cao object ở bất kỳ cạnh nào. Box không được cắt cụt bất kỳ phần nhìn thấy rõ nào của object.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định ngưỡng che khuất tối thiểu để gán nhãn (ví dụ: chỉ gán nhãn nếu ≥ 30% object hiện diện). Trường hợp che khuất > 70% hoặc object bị cắt mép không xác định được lớp → escalation lên reviewer cấp cao để quyết định bỏ qua hay vẫn gán nhãn với thuộc tính `occluded=true`.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: "kitchen-001", `class_name`: "person", `score`: 0.899318, `polygon_point_count`: 348 điểm. Ba điểm đầu tiên: [446.0, 70.0], [445.0, 71.0], [444.0, 71.0].
- Polygon bổ sung chi tiết gì so với box?
  - Polygon theo sát **đường viền thực của object** (hình dạng bất kỳ), trong khi bounding box chỉ là hình chữ nhật bao ngoài. Với object có hình dạng phức tạp như người (tay, chân), polygon giúp loại bỏ vùng nền nằm trong box nhưng không thuộc object, từ đó phép tính diện tích và phân tích pixel chính xác hơn nhiều.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` (ví dụ "kitchen-001") dùng để **phân biệt từng instance riêng** trong output của bài lab, đặc biệt khi có nhiều object cùng lớp (ví dụ hai người sẽ là "kitchen-001" và "kitchen-002"). Đây **không phải** `class_id` (mã lớp), cũng **không phải** tracking ID xuyên suốt nhiều frame video; nó chỉ tồn tại trong phạm vi một lần chạy inference.
- Đề xuất một quy tắc biên mask:
  - Biên polygon phải bám theo đường viền **nhìn thấy thực sự** của object, sai số không quá 2 pixel so với biên thực. Với các vùng lồi lõm nhỏ (ví dụ kẽ ngón tay), annotator được phép dùng đường thẳng nối hai điểm cách nhau ≤ 4 px để tránh over-segmentation không cần thiết.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline phải quy định cách vẽ biên khi hai object tiếp xúc nhau (ví dụ bowl đặt sát nhau): vẽ biên ảo phân tách hay chấp nhận chồng lấp. Với vùng mờ (motion blur) hoặc che khuất bởi object khác mà không xác định được đường biên thực → escalation để reviewer cấp cao quyết định có gán nhãn hay đánh dấu `crowd`/`difficult`.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | ----------------------------- | --------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         | Một nhãn lớp (class_id + class_name) cho toàn ảnh theo taxonomy dự án | Ảnh `kitchen` được model gán lớp "gong" (score 0.42) trong khi ảnh là cảnh nhà bếp — model dự đoán sai lớp; ảnh nhiều chủ thể không có quy tắc chọn nhãn rõ ràng | Xem ảnh, đối chiếu guideline taxonomy, chọn nhãn đúng theo hướng dẫn về "chủ thể chính", ghi lại trong công cụ gán nhãn | Kiểm tra nhãn khớp với chủ thể chính trong ảnh, phát hiện nhãn sai lớp hoặc nhãn không rõ ràng |
| Phát hiện vật thể     | Bounding box xyxy + class_id cho từng object theo taxonomy COCO-80 | Box `person` ở mép trái (x_min=0.12) bị cắt mép — không rõ có nên gán nhãn không; hai bowl nhỏ score ~0.38–0.46 có box rất nhỏ, khó phân biệt bowl thật hay nhiễu | Vẽ bounding box bao sát từng object theo quy tắc box chặt trong guideline, gán đúng class_id, đánh dấu object bị cắt mép/che khuất | Kiểm tra box có bao đúng object, class_id đúng taxonomy, không bỏ sót object, xử lý đúng trường hợp occluded/truncated |
| Instance segmentation | Polygon (polygon_xy) + class_id + instance_id cho từng instance riêng biệt | Instance "kitchen-001" (person) có 348 điểm polygon — biên phức tạp, khó kiểm tra thủ công; các cup nhỏ ở vùng tối ảnh bếp có thể bị bỏ sót | Vẽ polygon bám biên từng instance, tạo instance riêng cho mỗi object kể cả cùng lớp, đánh dấu vùng mơ hồ nếu có | Kiểm tra polygon bám biên đúng, không trộn lẫn hai instance, không bỏ sót object trong ảnh |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Tất cả ảnh dùng trong bài lab phải là ảnh công khai đã được xác nhận checksum; không tải lên Colab hay GitHub công khai bất kỳ ảnh nào chứa khuôn mặt người, biển số xe, thông tin cá nhân hoặc dữ liệu nội bộ.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach / Giảng viên phụ trách của lớp ngay lập tức qua kênh hỗ trợ chính thức, không tự xử lý hoặc tải thêm dữ liệu.

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
