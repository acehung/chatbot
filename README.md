# FinBot - Trợ lý Tài chính & Quản trị Thanh khoản AI

Ứng dụng trợ lý tài chính doanh nghiệp chạy cục bộ (Tiếng Việt), xây dựng với **Streamlit** và **SQLite**. Hỗ trợ cả chế độ **AI Text-to-SQL** (qua OpenAI Responses API kèm Few-shot learning từ bộ ví dụ đã duyệt) và **Bộ định tuyến thông minh cục bộ (Offline)** với hơn 11 chủ đề phân tích tài chính chuyên sâu.

## Cấu trúc dự án

```text
e:\chatbot
├── app.py                 # Giao diện chính Streamlit (6 phân hệ chức năng)
├── finbot/                # Package xử lý nghiệp vụ cốt lõi
│   ├── __init__.py
│   ├── db.py              # Kết nối SQLite, khởi tạo schema/demo, bảo mật Read-only Authorizer
│   ├── chatbot.py         # Định tuyến Offline, AI Text-to-SQL, sinh Insights tự động & Auto-Eval
│   ├── analytics.py       # Tính toán KPI thanh khoản, AR Aging, Pareto 80/20, xu hướng dòng tiền
│   └── ocr.py             # Bóc tách văn bản hóa đơn PDF/OCR & chuẩn hóa dữ liệu
├── data/
│   ├── finbot.db          # Cơ sở dữ liệu SQLite cục bộ
│   ├── du_lieu_demo_hoa_don.csv # Nguồn hóa đơn và doanh thu demo
│   └── evaluation_examples.jsonl
└── requirements.txt
```

## Chạy ứng dụng

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
streamlit run app.py
```

Để bật chế độ AI Text-to-SQL, đặt biến môi trường `OPENAI_API_KEY` (và tùy chọn `OPENAI_MODEL`), hoặc nhập trực tiếp trong mục **⚙️ Cấu hình & Công cụ** trên thanh bên (Sidebar) của ứng dụng.

Để OCR PDF scan, cần cài Tesseract OCR cùng gói ngôn ngữ tiếng Việt và tiếng Anh (`vie`, `eng`). Ứng dụng dùng PyMuPDF, pytesseract và Pillow; PDF có lớp văn bản được trích xuất trực tiếp. Nếu OCR chưa khả dụng, có thể dán nội dung nhận dạng vào ô văn bản.

Dữ liệu demo hóa đơn và doanh thu được nạp từ `data/du_lieu_demo_hoa_don.csv`. Dùng nút **Nạp lại dữ liệu Demo chuẩn** để đồng bộ lại hai bảng từ CSV. CSV không có chi phí và số dư tiền, nên bảng Cashflow và dự báo dòng tiền để trống thay vì hiển thị số liệu demo tự tạo.

Có thể nạp CSV khác trong **⚙️ Cấu hình & Công cụ → Thêm dữ liệu từ file CSV**. Chọn **Gộp và cập nhật** để thêm hóa đơn và cập nhật các số hóa đơn trùng; chọn **Thay toàn bộ** để thay dữ liệu hóa đơn/doanh thu hiện tại. Tải file CSV mẫu ngay tại expander để xem đúng cấu trúc cột. Ứng dụng kiểm tra định dạng ngày, số tiền và hai phép đối soát trước khi ghi.

## Các tính năng chính

1. **💬 Chat tài chính thông minh:**
   - Hỗ trợ nút **Câu hỏi gợi ý nhanh** theo 8 chủ đề phổ biến.
   - Nhận diện hơn 11 ý định tài chính ở chế độ Offline: Công nợ quá hạn, Phân tích tuổi nợ (AR Aging), Thống kê trạng thái hóa đơn, Top khách hàng công nợ, Tra cứu theo tên khách hàng cụ thể, Doanh thu Pareto 80/20, So sánh kênh bán (Online vs Trực tiếp), Xu hướng doanh thu 7/30 ngày, Dòng tiền & Runway, và Báo cáo tổng quan.
   - **Tự động sinh nhận xét & cảnh báo (Financial Insights):** Phân tích điểm nóng quá hạn, tỷ trọng nhóm Pareto, tốc độ đốt tiền (Burn rate) và khuyến nghị hành động thay vì chỉ hiển thị bảng thô.
   - Cho phép xem câu lệnh SQL thực thi và tải kết quả ra file `.csv`.

2. **📊 Bảng điều khiển Sức khỏe tài chính:**
   - Bộ lọc thời gian linh hoạt (`7 ngày`, `30 ngày`, `60 ngày`, `Toàn bộ thời gian`).
   - 6 chỉ số KPI: **DSO ước tính**, **Quick Ratio (proxy)**, **Runway**, **Tỷ lệ nợ quá hạn (%)**, **Tỷ lệ thu hồi hóa đơn (%)**, và **Dòng tiền ròng kỳ gần nhất**.
   - 3 tab trực quan hóa chuyên sâu: **Phân tích Tuổi nợ (AR Aging)**, **Doanh thu & Quy tắc Pareto 80/20**, và **Xu hướng Dòng tiền (Cashflow)**; kèm dự báo số dư tiền 3 tháng theo xu hướng tuyến tính của tối đa 6 kỳ gần nhất.

3. **🧾 Bóc tách chứng từ & Đối soát số học:**
   - Trích xuất văn bản từ PDF có lớp chữ; OCR PDF scan bằng Tesseract (`vie+eng`) hoặc dán nội dung OCR (có sẵn nút nạp mẫu hóa đơn thử nghiệm).
   - Tự động chuẩn hóa ngày tháng (`DD/MM/YYYY` → `YYYY-MM-DD`), tự động suy luận số tiền còn thiếu và kiểm tra đẳng thức kế toán (`Tiền trước thuế + VAT = Tổng thanh toán`) trước khi cho phép ghi vào sổ cái.

4. **⚡ Tác vụ Thu hồi nợ & Thanh toán:**
   - **Tạo mã VietQR:** Chọn nhanh hóa đơn chưa thu để tự động điền số tiền và nội dung chuyển khoản.
   - **Soạn email nhắc nợ:** Hỗ trợ 3 giọng văn (*Nhắc nhở lịch sự*, *Đôn đốc khẩn cấp*, *Đề xuất lịch thanh toán*) và tải file `.eml`.

5. **🛡️ Kiểm toán & 🗳️ Đánh giá AI (Auto-Eval):**
   - Toàn bộ truy vấn dữ liệu từ Chatbot bị khóa ở chế độ chỉ đọc (`SELECT` / `WITH`) bằng cả bộ lọc cú pháp và `sqlite3.set_authorizer`.
   - Duyệt phản hồi 👍/👎 của người dùng, ẩn danh hóa dữ liệu nhạy cảm, xuất file `evaluation_examples.jsonl` và chạy **Auto-Eval Suite** trực tiếp trên giao diện để đo độ chính xác của Chatbot.

## Kịch bản demo nhanh

1. Trong tab **Chat tài chính**, hỏi “Những hóa đơn nào đang quá hạn?” và mở SQL để xem câu SELECT. Sau đó thử gửi `DELETE FROM Invoices`; ứng dụng phải từ chối truy vấn và ghi trạng thái `BLOCKED` trong tab **Kiểm toán**.
2. Chọn câu hỏi gợi ý về doanh thu theo danh mục hoặc dòng tiền để xem bảng và biểu đồ.
3. Trong tab **Chứng từ**, nạp hóa đơn mẫu hoặc tải PDF/ảnh, kiểm tra tổng tiền rồi ghi vào sổ cái; sau đó tạo VietQR hoặc email nhắc nợ trong tab **Tác vụ**.

VietQR được phục vụ qua VietQR Quick Link nên cần internet để tải ảnh mã QR.
