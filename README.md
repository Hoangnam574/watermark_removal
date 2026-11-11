# Watermark Removal

Dự án ứng dụng Deep Learning (Image Inpainting) để tự động phát hiện và xóa watermark trên hình ảnh, tái tạo lại vùng ảnh bị che khuất một cách tự nhiên và liền mạch. Dự án cung cấp cả giao diện Web trực quan (Flask) lẫn công cụ dòng lệnh (CLI), cùng pipeline huấn luyện mô hình.

---

## 🌟 Tính Năng Chính

- **Giao diện Web trực quan (Flask)**:
  - Hỗ trợ kéo & thả hoặc chọn nhiều ảnh cùng lúc.
  - Xem trước và so sánh trực tiếp kết quả trước / sau khi xóa watermark (side-by-side comparison).
  - Tải về ảnh đã xử lý nhanh chóng.
- **Công cụ dòng lệnh (CLI)**:
  - Xử lý từng ảnh đơn lẻ với độ chính xác cao (`main.py`).
  - Hỗ trợ kiểm thử hàng loạt (`batch_test.py` và `guided_batch_test.py`).
- **Huấn luyện mô hình (Training)**:
  - Kịch bản huấn luyện mô hình Image Inpainting với Generative Inpainting (`train.py`).
- **Kiến trúc mô hình**:
  - Dựa trên cơ chế **Contextual Attention** (CVPR 2018) và **Gated Convolution** (ICCV 2019) giúp tái tạo chi tiết và kết cấu hình ảnh chân thực.

---

## 📁 Cấu Trúc Thư Mục

```text
watermark-removal/
├── app.py                  # Ứng dụng Web Flask
├── main.py                 # CLI xử lý xóa watermark cho 1 ảnh
├── batch_test.py           # Kịch bản kiểm thử hàng loạt ảnh
├── guided_batch_test.py    # Kịch bản kiểm thử batch có hướng dẫn
├── train.py                # Pipeline huấn luyện mô hình
├── inpaint_model.py        # Định nghĩa kiến trúc mô hình Inpaint Contextual Attention
├── inpaint_ops.py          # Các phép toán tùy chỉnh TensorFlow cho inpainting
├── preprocess_image.py     # Tiền xử lý hình ảnh và tạo mặt nạ (mask)
├── inpaint.yml             # File cấu hình mô hình và huấn luyện
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── templates/
│   └── index.html          # Giao diện người dùng Web
├── static/                 # Tài nguyên web (uploads, outputs)
├── utils/                  # Mặt nạ mẫu (watermark masks)
└── model/                  # Thư mục chứa trọng số mô hình đã huấn luyện (checkpoint)
```

---

## 🛠️ Cài Đặt

### 1. Yêu cầu hệ thống
- **Python**: khuyến nghị **Python 3.7** (để tương thích tốt nhất với TensorFlow 1.15).
- Khuyến nghị sử dụng môi trường ảo (`venv` hoặc `conda`).

### 2. Clone mã nguồn
```bash
git clone https://github.com/Hoangnam574/watermark_removal.git
cd watermark_removal
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu cần cài đặt thủ công `neuralgym`:
> ```bash
> pip install git+https://github.com/JiahuiYu/neuralgym
> ```

### 4. Tải trọng số mô hình (Model Checkpoints)
Tải mô hình đã được huấn luyện sẵn từ liên kết sau:
👉 [Google Drive Download Checkpoints](https://drive.google.com/drive/folders/1xRV4EdjJuAfsX9pQme6XeoFznKXG0ptJ?usp=sharing)

Sau khi tải về, đặt các file vào thư mục `model/`:
- `model/checkpoint` *(Lưu ý: Nếu Google Drive đổi tên thành checkpoint.txt, hãy đổi lại thành checkpoint)*
- `model/snap-0.data-00000-of-00001`
- `model/snap-0.index`
- `model/snap-0.meta`

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Khởi chạy Giao diện Web (Flask)
```bash
python app.py
```
Mở trình duyệt web và truy cập địa chỉ:
```
http://127.0.0.1:5000
```
Tại giao diện web:
- Kéo thả hoặc bấm chọn ảnh cần xóa watermark.
- Nhấn **Xóa Watermark**.
- Xem kết quả so sánh và nhấn **Tải ảnh đã xử lý**.

---

### 2. Sử dụng dòng lệnh (CLI)

#### Xử lý một ảnh đơn:
```bash
python main.py --image input/sample.jpg --output output/result.png --checkpoint_dir model/ --watermark_type istock
```

Các tham số chính:
- `--image`: Đường dẫn đến ảnh đầu vào chứa watermark.
- `--output`: Đường dẫn lưu ảnh kết quả sau khi xóa watermark.
- `--checkpoint_dir`: Thư mục chứa trọng số mô hình (mặc định: `model/`).
- `--watermark_type`: Kiểu watermark cần xóa (ví dụ: `istock`).

#### Chạy kiểm thử hàng loạt (Batch Test):
```bash
python batch_test.py --flist flist.txt --image_height 512 --image_width 512 --checkpoint_dir model/
```

---

### 3. Huấn luyện mô hình (Training)
Để huấn luyện mô hình với tập dữ liệu mới:
1. Cấu hình danh sách dữ liệu trong file `inpaint.yml`.
2. Chạy kịch bản:
```bash
python train.py
```

---

## 📚 Tham Khảo (Citations)

Mô hình inpainting được phát triển dựa trên các nghiên cứu:

```bibtex
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```

---

## 📄 Bản Quyền & Giấy Phép
Dự án được phân phối cho mục đích học tập và nghiên cứu (Non-Commercial / Educational Use).
