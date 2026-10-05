"""
Script để test Neural Style Transfer trên máy local
Chỉ dùng để test components, không train full (cần GPU mạnh)
"""

import torch
import sys
import os

# Kiểm tra GPU
print("=" * 60)
print("KIỂM TRA HỆ THỐNG")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️ GPU not available - sẽ chạy trên CPU (rất chậm)")
print()

# Import test functions
print("=" * 60)
print("IMPORT MODULES")
print("=" * 60)
try:
    from test_components import (
        test_model_loading,
        test_preprocessing,
        test_loss_functions,
        test_full_pipeline
    )
    print("✅ Import thành công!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Đảm bảo bạn đang chạy từ thư mục cuoiky/")
    sys.exit(1)
print()

# Chạy tests
print("=" * 60)
print("CHẠY TESTS")
print("=" * 60)
print()

results = []

# Test 1: Model Loading
print("\n" + "=" * 60)
print("TEST 1: Model Loading")
print("=" * 60)
try:
    result = test_model_loading()
    results.append(("Model Loading", result))
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(("Model Loading", False))

# Test 2: Preprocessing
print("\n" + "=" * 60)
print("TEST 2: Preprocessing")
print("=" * 60)
try:
    result = test_preprocessing()
    results.append(("Preprocessing", result))
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(("Preprocessing", False))

# Test 3: Loss Functions
print("\n" + "=" * 60)
print("TEST 3: Loss Functions")
print("=" * 60)
try:
    result = test_loss_functions()
    results.append(("Loss Functions", result))
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(("Loss Functions", False))

# Test 4: Full Pipeline (chỉ với dummy data, không train thật)
print("\n" + "=" * 60)
print("TEST 4: Full Pipeline (Dummy Data)")
print("=" * 60)
try:
    result = test_full_pipeline()
    results.append(("Full Pipeline", result))
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(("Full Pipeline", False))

# Summary
print("\n" + "=" * 60)
print("KẾT QUẢ TỔNG HỢP")
print("=" * 60)
for test_name, passed in results:
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{test_name:20s}: {status}")

all_passed = all(result[1] for result in results)
print()
if all_passed:
    print("🎉 Tất cả tests đều PASSED!")
    print()
    print("📝 Lưu ý:")
    print("   - Để train full, bạn cần GPU mạnh hoặc dùng Google Colab")
    print("   - Xem COLAB_GUIDE.md để biết cách chạy trên Colab")
else:
    print("⚠️ Một số tests FAILED. Kiểm tra lại code và dependencies.")
    print()
    print("💡 Gợi ý:")
    print("   - Kiểm tra đã cài đủ packages: pip install -r requirements.txt")
    print("   - Kiểm tra GPU driver nếu dùng GPU")
    print("   - Xem lỗi chi tiết ở trên")

