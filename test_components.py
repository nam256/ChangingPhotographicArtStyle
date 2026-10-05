"""
Test script để kiểm tra các components
Chạy script này để test từng phần trước khi train full
"""

import torch
from PIL import Image

from models.vgg_model import load_vgg_model
from utils.preprocessing import image_to_tensor, tensor_to_image
# Note: load_image() requires file path, not PIL Image, so we test it separately
from utils.losses import (
    content_loss,
    style_loss,
    gram_matrix,
    total_variation_loss,
    compute_total_loss,
)


def test_model_loading():
    """Test 1: Load VGG19 model"""
    print("=" * 50)
    print("TEST 1: Model Loading")
    print("=" * 50)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model = load_vgg_model(model_name='vgg19', device=device)
        print("✅ VGG19 loaded successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        features = model.forward(dummy_input)
        print(f"✅ Forward pass successful")
        print(f"   Number of feature layers: {len(features)}")
        
        # Test content features
        content_feat = model.get_content_features(dummy_input)
        print(f"✅ Content features shape: {content_feat.shape}")
        
        # Test style features
        style_feat_dict = model.get_style_features(dummy_input)
        print(f"✅ Style features from {len(style_feat_dict)} layers")
        for layer, feat in style_feat_dict.items():
            print(f"   Layer {layer}: {feat.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_preprocessing():
    """
    Test 2: Image preprocessing
    Test tensor conversion, KHÔNG test load_image() (cần file path thật)
    """
    print("\n" + "=" * 50)
    print("TEST 2: Image Preprocessing")
    print("=" * 50)
    print("Note: This test does NOT use load_image() (requires file path)")
    print("      It tests tensor conversion functions only.\n")
    
    try:
        # Tạo dummy image (PIL Image object)
        dummy_img = Image.new('RGB', (500, 400), color='red')
        print(f"✅ Created dummy PIL image, size: {dummy_img.size}")
        
        # Test resize image trực tiếp (KHÔNG dùng load_image)
        print("\n1. Testing image resize (thumbnail)...")
        dummy_img_resized = dummy_img.copy()
        dummy_img_resized.thumbnail((256, 256), Image.Resampling.LANCZOS)
        print(f"   ✅ Image resized, new size: {dummy_img_resized.size}")
        
        # Test image_to_tensor
        print("\n2. Testing image_to_tensor (PIL → Tensor)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        tensor = image_to_tensor(dummy_img_resized, size=(256, 256), device=device)
        print(f"   ✅ Tensor shape: {tensor.shape}")
        print(f"   ✅ Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"   ✅ Tensor device: {tensor.device}")
        
        # Test tensor_to_image
        print("\n3. Testing tensor_to_image (Tensor → PIL)...")
        img_back = tensor_to_image(tensor)
        print(f"   ✅ Converted back to PIL image, size: {img_back.size}")
        
        # Verify round-trip
        print("\n4. Verifying round-trip conversion...")
        tensor2 = image_to_tensor(img_back, size=None, device=device)
        diff = (tensor - tensor2).abs().max().item()
        print(f"   ✅ Max difference: {diff:.6f} (should be very small)")
        
        print("\n" + "=" * 50)
        print("✅ All preprocessing tests passed!")
        print("Note: load_image() requires actual file path, will be tested")
        print("      when you upload real images in Step 4.")
        print("=" * 50)
        
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test 3: Loss functions"""
    print("\n" + "=" * 50)
    print("TEST 3: Loss Functions")
    print("=" * 50)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create dummy features
        batch, channels, height, width = 1, 512, 28, 28
        feat1 = torch.randn(batch, channels, height, width).to(device)
        feat2 = torch.randn(batch, channels, height, width).to(device)
        
        # Test content loss
        print("Testing content_loss...")
        c_loss = content_loss(feat1, feat2)
        print(f"✅ Content loss: {c_loss.item():.4f}")
        assert not torch.isnan(c_loss), "Content loss is NaN!"
        
        # Test Gram matrix
        print("Testing gram_matrix...")
        gram = gram_matrix(feat1)
        print(f"✅ Gram matrix shape: {gram.shape}")
        assert gram.shape == (batch, channels, channels), "Wrong Gram matrix shape!"
        
        # Test style loss
        print("Testing style_loss...")
        s_loss = style_loss(feat1, feat2)
        print(f"✅ Style loss: {s_loss.item():.4f}")
        assert not torch.isnan(s_loss), "Style loss is NaN!"
        
        # Test TV loss
        print("Testing total_variation_loss...")
        dummy_img = torch.randn(1, 3, 256, 256).to(device)
        tv_loss = total_variation_loss(dummy_img)
        print(f"✅ TV loss: {tv_loss.item():.6f}")
        assert not torch.isnan(tv_loss), "TV loss is NaN!"
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 4: Full pipeline với dummy data"""
    print("\n" + "=" * 50)
    print("TEST 4: Full Pipeline (Dummy Data)")
    print("=" * 50)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model
        model = load_vgg_model(model_name='vgg19', device=device)
        
        # Create dummy images
        content_img = torch.randn(1, 3, 256, 256).to(device)
        style_img = torch.randn(1, 3, 256, 256).to(device)
        generated_img = content_img.clone().requires_grad_(True)
        
        # Test compute_total_loss
        print("Testing compute_total_loss...")
        total_loss, c_loss, s_loss, tv_loss = compute_total_loss(
            generated_img,
            model,
            content_img,
            style_img,
            content_weight=1e5,
            style_weight=1e6,
            tv_weight=1e-4,
            device=device
        )
        
        print(f"✅ Total loss: {total_loss.item():.4f}")
        print(f"   Content loss: {c_loss.item():.4f}")
        print(f"   Style loss: {s_loss.item():.4f}")
        print(f"   TV loss: {tv_loss.item():.6f}")
        
        # Check NaN
        assert not torch.isnan(total_loss), "Total loss is NaN!"
        assert not torch.isnan(c_loss), "Content loss is NaN!"
        assert not torch.isnan(s_loss), "Style loss is NaN!"
        assert not torch.isnan(tv_loss), "TV loss is NaN!"
        
        # Test backward
        print("Testing backward pass...")
        total_loss.backward()
        print(f"✅ Backward pass successful")
        print(f"   Gradient norm: {generated_img.grad.norm().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing Neural Style Transfer Components\n")
    
    results = []
    
    # Run tests
    results.append(("Model Loading", test_model_loading()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Loss Functions", test_loss_functions()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
    else:
        print("\n⚠️ Some tests failed. Please fix errors before training.")

