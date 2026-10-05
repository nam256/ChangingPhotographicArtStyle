# Neural Style Transfer - Cuối Kỳ

## 📋 Mô tả
Dự án Neural Style Transfer sử dụng VGG19 để chuyển đổi phong cách nghệ thuật cho hình ảnh.

## 🎯 Yêu cầu
- Python 3.7+
- PyTorch 2.0+
- Google Colab (khuyến nghị) hoặc GPU với CUDA

## 📁 Cấu trúc thư mục
```
cuoiky/
├── data/
│   ├── content/          # Ảnh content
│   └── style/            # Ảnh style
├── models/
│   └── vgg_model.py      # VGG model wrapper
├── utils/
│   ├── preprocessing.py  # Image preprocessing
│   ├── losses.py         # Loss functions
│   └── visualization.py  # Visualization utilities
├── main.py               # Main training script
├── requirements.txt
└── README.md
```

## 🚀 Cách sử dụng

### Option 1: Test Local (Chỉ test components)

```bash
# Cài đặt packages
pip install -r requirements.txt

# Chạy test
python test_local.py
```

**Lưu ý:** Test local chỉ kiểm tra components, không train full. Để train full cần GPU mạnh hoặc dùng Colab.

### Option 2: Setup trên Google Colab (BẮT BUỘC cho training)
Xem file **COLAB_GUIDE.md** để có hướng dẫn chi tiết.

Tóm tắt:
```python
# Cài đặt dependencies
!pip install torch torchvision pillow numpy matplotlib tqdm

# Upload code và ảnh lên Colab
# Xem COLAB_GUIDE.md hoặc dùng colab_notebook.ipynb
```

### 3. Test Components (Khuyến nghị)
```python
from test_components import *

# Test từng component trước khi train
test_model_loading()
test_preprocessing()
test_loss_functions()
test_full_pipeline()
```

### 3. Upload ảnh
- Upload ảnh content vào `data/content/`
- Upload ảnh style vào `data/style/`

### 4. Chạy training
```python
from main import train_neural_style_transfer

train_neural_style_transfer(
    content_path="data/content/your_content.jpg",
    style_path="data/style/your_style.jpg",
    output_path="output.jpg",
    num_iterations=200,
    image_size=(256, 256),
    content_weight=1e5,
    style_weight=1e6,
    tv_weight=1e-4,
    learning_rate=0.01,
    device='cuda'
)
```

## ⚙️ Hyperparameters

### Baseline (Cân bằng)
- `content_weight = 1e5`
- `style_weight = 1e6`
- `tv_weight = 1e-4`
- `learning_rate = 0.01`

### Nếu style quá mạnh (mất content)
- Tăng `content_weight` lên `1e6`

### Nếu content quá mạnh (không thấy style)
- Giảm `content_weight` xuống `1e4`

## 📝 Lưu ý
- **BẮT BUỘC**: Dùng Google Colab để chạy VGG19
- **BẮT BUỘC**: Gradient clipping để tránh NaN
- Image size: 256x256 hoặc 512x512 trên Colab
- Iterations: 200-300 trên Colab

## 🐛 Troubleshooting
Xem `plan.md` và `code.md` để biết cách xử lý các lỗi thường gặp.

