# 🚀 Hướng Dẫn Chạy Trên Google Colab

## Bước 1: Setup Colab

1. Vào https://colab.research.google.com
2. Tạo notebook mới
3. Chọn **Runtime → Change runtime type → GPU (T4)**

## Bước 2: Cài đặt Dependencies

```python
# Cài đặt packages
!pip install torch torchvision pillow numpy matplotlib tqdm

# Kiểm tra GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Bước 3: Upload Code

### Cách 1: Zip và Upload từ local

**Trên máy local (Windows):**
1. Vào thư mục `C:\Users\nam\Desktop\ckpy\`
2. Zip thư mục `cuoiky` thành file `cuoiky.zip`
   - Click chuột phải vào thư mục `cuoiky` → Send to → Compressed (zipped) folder
   - Hoặc dùng WinRAR/7-Zip để zip
3. File zip sẽ có tên `cuoiky.zip` tại `C:\Users\nam\Desktop\ckpy\cuoiky.zip`

**Trên Google Colab:**
```python
from google.colab import files
import zipfile
import os

# Upload file zip cuoiky.zip
print("📤 Upload file cuoiky.zip:")
uploaded = files.upload()

# Tìm file zip vừa upload
zip_filename = None
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        zip_filename = filename
        break

if zip_filename:
    # Giải nén
    print(f"📦 Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('/content')
    print(f"✅ Extracted {zip_filename}")
    
# Add to path
import sys
import os

# Thêm thư mục cuoiky vào Python path
cuoiky_path = '/content/cuoiky'
if os.path.exists(cuoiky_path):
    sys.path.insert(0, cuoiky_path)
    print(f"✅ Added {cuoiky_path} to Python path")
else:
    print(f"⚠️ Path {cuoiky_path} not found!")

# Kiểm tra file structure
print("\n📁 File structure:")
if os.path.exists(cuoiky_path):
    !ls -la /content/cuoiky/
    print("\n📁 Checking key files:")
    key_files = ['main.py', 'test_components.py', 'models/vgg_model.py', 'utils/losses.py']
    for file in key_files:
        file_path = f'/content/cuoiky/{file}'
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND!")
else:
    print("❌ Không tìm thấy file zip!")
```

### Cách 2: Upload trực tiếp từ Google Drive (Nếu có)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy từ Drive (nếu đã upload cuoiky.zip lên Drive)
!cp /content/drive/MyDrive/cuoiky.zip /content/
!unzip /content/cuoiky.zip -d /content/

# Add to path
import sys
sys.path.append('/content/cuoiky')
```

### Cách 3: Clone từ GitHub (nếu có)
```python
!git clone <your-repo-url>
!cd cuoiky
import sys
sys.path.append('/content/cuoiky')
```

## Bước 4: Upload Ảnh

```python
from google.colab import files
import os

# Tạo thư mục với đường dẫn đầy đủ
!mkdir -p /content/cuoiky/data/content /content/cuoiky/data/style

# Upload content image
print("📤 Upload content image:")
content_uploaded = files.upload()
content_filename = list(content_uploaded.keys())[0]
!mv {content_filename} /content/cuoiky/data/content/content.jpg
print(f"✅ Content image saved: /content/cuoiky/data/content/content.jpg")

# Kiểm tra file đã upload
if os.path.exists('/content/cuoiky/data/content/content.jpg'):
    print(f"   File size: {os.path.getsize('/content/cuoiky/data/content/content.jpg')} bytes")
else:
    print("   ⚠️ File not found!")

# Upload style image
print("\n📤 Upload style image:")
style_uploaded = files.upload()
style_filename = list(style_uploaded.keys())[0]
!mv {style_filename} /content/cuoiky/data/style/style.jpg
print(f"✅ Style image saved: /content/cuoiky/data/style/style.jpg")

# Kiểm tra file đã upload
if os.path.exists('/content/cuoiky/data/style/style.jpg'):
    print(f"   File size: {os.path.getsize('/content/cuoiky/data/style/style.jpg')} bytes")
else:
    print("   ⚠️ File not found!")
```

## Bước 5: Test Components (Optional)

**⚠️ QUAN TRỌNG: Sau khi upload code mới, cần restart runtime!**
- Runtime → Restart runtime (hoặc Ctrl+M .)
- Hoặc chạy lại cell import

**📝 Lưu ý về Import:**
- Nếu IDE (VS Code, PyCharm) báo lỗi "Import test_components could not be resolved" → **Đây chỉ là warning của IDE, KHÔNG ảnh hưởng khi chạy trên Colab**
- Code sẽ chạy bình thường trên Colab vì đã thêm `/content/cuoiky` vào `sys.path`
- Để IDE không báo lỗi, bạn có thể ignore warning này hoặc thêm `# type: ignore` sau import

```python
# ⚠️ QUAN TRỌNG: Kiểm tra file có đúng code mới không trước!

import sys
import os

# Bước 0: KIỂM TRA FILE CÓ ĐÚNG CODE MỚI KHÔNG
file_path = '/content/cuoiky/test_components.py'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Testing load_image' in content or 'load_image(' in content.split('def test_preprocessing')[1].split('def test_loss_functions')[0] if 'def test_preprocessing' in content else '':
        print("❌ File test_components.py vẫn là code CŨ!")
        print("   → Cần upload lại cuoiky.zip mới hoặc xóa và upload lại")
        print("   → Hoặc restart runtime và upload lại code")
        raise RuntimeError("File test_components.py is old version. Please upload new code.")
    else:
        print("✅ File test_components.py là code MỚI")
else:
    print(f"❌ File không tồn tại: {file_path}")
    raise FileNotFoundError(f"File not found: {file_path}")

# Bước 1: Xóa cache Python của project
!rm -rf /content/cuoiky/__pycache__ /content/cuoiky/*/__pycache__ 2>/dev/null || true

# Bước 2: Đảm bảo cuoiky trong path (ưu tiên đầu tiên)
cuoiky_path = '/content/cuoiky'
if cuoiky_path not in sys.path:
    sys.path.insert(0, cuoiky_path)
    print(f"✅ Added {cuoiky_path} to Python path")
else:
    # Nếu đã có, xóa và thêm lại ở đầu để ưu tiên
    sys.path.remove(cuoiky_path)
    sys.path.insert(0, cuoiky_path)
    print(f"✅ Moved {cuoiky_path} to front of Python path")

# Bước 3: Xóa CHỈ modules của project (KHÔNG xóa PyTorch/system modules!)
modules_to_remove = []
project_modules = ['test_components', 'preprocessing', 'losses', 'visualization', 'vgg_model', 'main']
for module_name in list(sys.modules.keys()):
    if any(keyword in module_name for keyword in project_modules):
        if not any(skip in module_name for skip in ['torch', 'numpy', 'PIL', 'matplotlib', 'tqdm', 'google.colab']):
            modules_to_remove.append(module_name)

for m in modules_to_remove:
    try:
        del sys.modules[m]
    except KeyError:
        pass
print(f"✅ Removed {len(modules_to_remove)} old project modules")

# Bước 4: Import lại từ đầu
try:
    from test_components import (
        test_model_loading,
        test_preprocessing,
        test_loss_functions,
        test_full_pipeline
    )
    print("✅ Imported test functions successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Current sys.path: {sys.path[:3]}...")
    print(f"   File exists: {os.path.exists('/content/cuoiky/test_components.py')}")
    raise

# Test từng component
print("\n🧪 Testing components...\n")

# Test 1: Model Loading
test_model_loading()

# Test 2: Preprocessing (test resize và tensor conversion, KHÔNG test load_image)
test_preprocessing()

# Test 3: Loss Functions
test_loss_functions()

# Test 4: Full Pipeline với dummy data
test_full_pipeline()

print("\n✅ All tests completed!")
print("Note: test_preprocessing() tests tensor conversion, not file loading.")
print("      File loading will be tested when you upload actual images.")
```

**⚠️ Nếu test trên local chạy ngon nhưng Colab bị lỗi "Testing load_image...":**

Đây là do file `test_components.py` trên Colab vẫn là code cũ. Làm theo các bước sau:

**🔴 GIẢI PHÁP 1: Kiểm tra và sửa file trực tiếp trên Colab**

Chạy code này để kiểm tra và sửa file:

```python
# Kiểm tra file test_components.py có đúng code mới không
import os

file_path = '/content/cuoiky/test_components.py'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Kiểm tra xem có dòng "Testing load_image" không (code cũ)
    if 'Testing load_image' in content:
        print("❌ File test_components.py vẫn là code CŨ!")
        print("   Cần upload lại file mới hoặc sửa trực tiếp")
        print("\n📝 Sửa file trực tiếp trên Colab:")
        print("   - Mở file /content/cuoiky/test_components.py")
        print("   - Tìm và xóa dòng 'Testing load_image...'")
        print("   - Hoặc upload lại cuoiky.zip mới")
    else:
        print("✅ File test_components.py là code MỚI")
        print("   Vấn đề có thể do cache. Restart runtime và thử lại.")
else:
    print(f"❌ File không tồn tại: {file_path}")
```

**🔴 GIẢI PHÁP 2: Upload lại code mới (KHUYẾN NGHỊ)**

1. **Xóa thư mục cũ:**
```python
!rm -rf /content/cuoiky
```

2. **Restart runtime:**
   - Runtime → Restart runtime

3. **Upload lại cuoiky.zip mới** (Bước 3)

4. **Chạy test lại** (Bước 5)

2. **Nếu vẫn lỗi RuntimeError về PyTorch:**
   - Đây là do code cố reload PyTorch modules
   - **Giải pháp:** Restart runtime và KHÔNG chạy lại cell xóa modules
   - Chỉ cần restart và import lại bình thường:
   ```python
   # Sau khi restart runtime, chỉ cần:
   import sys
   sys.path.insert(0, '/content/cuoiky')
   from test_components import (
       test_model_loading,
       test_preprocessing,
       test_loss_functions,
       test_full_pipeline
   )
   ```

**🔵 GIẢI PHÁP NÂNG CAO (Nếu không muốn restart):**

2. **Chạy lại cell upload code (Bước 3)** để đảm bảo code mới được extract

3. **Chạy cell test với code reload đầy đủ (xem code ở trên)**

4. **Nếu vẫn lỗi, kiểm tra:**
```python
# Kiểm tra file có tồn tại không
import os
print(f"test_components.py exists: {os.path.exists('/content/cuoiky/test_components.py')}")
print(f"Current directory: {os.getcwd()}")
print(f"sys.path: {sys.path[:5]}")

# Kiểm tra nội dung file
!head -20 /content/cuoiky/test_components.py
```

5. **Nếu vẫn không được, upload lại code:**
   - Xóa thư mục cũ: `!rm -rf /content/cuoiky`
   - Upload lại `cuoiky.zip`
   - Extract lại
   - Restart runtime
   - Chạy lại test

**Lưu ý:** 
- `test_preprocessing()` test tensor conversion, không test file loading
- File loading sẽ được test khi bạn upload ảnh thật ở Bước 4
- Nếu có lỗi, kiểm tra lại import và file structure

## Bước 6: Train Neural Style Transfer

**⚠️ Trước khi train, đảm bảo đã reload modules đúng cách:**

```python
# Reload main module nếu cần
import importlib
import sys

# Xóa cache của project (KHÔNG xóa cache của PyTorch/system)
!rm -rf /content/cuoiky/__pycache__ /content/cuoiky/*/__pycache__ 2>/dev/null || true

# Xóa CHỈ modules của project (KHÔNG xóa PyTorch/system modules)
project_modules_to_remove = ['main', 'models.vgg_model', 'utils.preprocessing', 'utils.losses', 'utils.visualization']
for m in project_modules_to_remove:
    if m in sys.modules:
        try:
            del sys.modules[m]
        except KeyError:
            pass

# Đảm bảo path đúng
if '/content/cuoiky' not in sys.path:
    sys.path.insert(0, '/content/cuoiky')

# Import lại
from main import train_neural_style_transfer

# Train với baseline hyperparameters
generated_image, losses = train_neural_style_transfer(
    content_path="/content/cuoiky/data/content/content.jpg",  # Đường dẫn đầy đủ
    style_path="/content/cuoiky/data/style/style.jpg",        # Đường dẫn đầy đủ
    output_path="/content/cuoiky/output.jpg",                 # Đường dẫn đầy đủ
    num_iterations=200,  # Colab: 200-300
    image_size=(256, 256),  # Colab: (256, 256) hoặc (512, 512)
    content_weight=1e5,
    style_weight=1e6,
    tv_weight=1e-4,
    learning_rate=0.01,
    save_every=50,
    device='cuda'
)
```

## Bước 7: Download Kết Quả

```python
from google.colab import files

# Download kết quả
files.download('output.jpg')
files.download('output_losses.png')

# Download checkpoints nếu có
import glob
for checkpoint in glob.glob('*_iter_*.jpg'):
    files.download(checkpoint)
```

## ⚙️ Tune Hyperparameters

### Nếu style quá mạnh (mất content):
```python
train_neural_style_transfer(
    ...,
    content_weight=1e6,  # Tăng từ 1e5
    style_weight=1e6,
    ...
)
```

### Nếu content quá mạnh (không thấy style):
```python
train_neural_style_transfer(
    ...,
    content_weight=1e4,  # Giảm từ 1e5
    style_weight=1e6,
    ...
)
```

### Nếu ảnh quá tối:
```python
train_neural_style_transfer(
    ...,
    learning_rate=0.02,  # Tăng từ 0.01
    num_iterations=300,  # Tăng từ 200
    ...
)
```

### Nếu có artifacts/noise:
```python
train_neural_style_transfer(
    ...,
    tv_weight=1e-3,  # Tăng từ 1e-4
    num_iterations=200,  # Giảm nếu cần
    ...
)
```

## 🐛 Troubleshooting

### GPU không available:
- Kiểm tra Runtime → Change runtime type → GPU
- Restart runtime nếu cần

### Out of memory:
- Giảm image_size xuống (256, 256)
- Giảm num_iterations

### Loss thành NaN:
- Đã có gradient clipping trong code
- Nếu vẫn NaN, giảm learning_rate xuống 0.001

### Import error:
- Kiểm tra sys.path.append('/content/cuoiky')
- Kiểm tra file structure

## 📝 Lưu Ý

- **BẮT BUỘC**: Dùng GPU runtime trên Colab
- **BẮT BUỘC**: VGG19 (đúng yêu cầu đề bài)
- Thời gian training: ~30-60 phút cho 200 iterations
- Save checkpoint mỗi 50 iterations để không mất progress

