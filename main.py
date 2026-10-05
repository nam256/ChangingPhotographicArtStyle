"""
Main script cho Neural Style Transfer
Sử dụng VGG19 trên Google Colab (đúng yêu cầu đề bài)
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from models.vgg_model import load_vgg_model
from utils.preprocessing import preprocess_images, tensor_to_image
from utils.losses import compute_total_loss
from utils.visualization import show_images, plot_losses


def initialize_generated_image(init_mode, content_tensor, style_tensor):
    """
    Khởi tạo ảnh sinh ra theo chiến lược mong muốn để tăng độ "style"
    """
    if init_mode == 'style':
        init = style_tensor.clone()
    elif init_mode == 'hybrid':
        init = 0.4 * content_tensor + 0.6 * style_tensor
    elif init_mode == 'noise':
        noise = torch.randn_like(content_tensor)
        init = 0.3 * content_tensor + 0.7 * noise
    else:  # mặc định dùng content
        init = content_tensor.clone()
    
    return init.requires_grad_(True)


STYLE_LAYER_FOCUS_VANGOGH = {
    # Ưu tiên texture ở các layer thấp, giảm layer sâu để tạo phong cách mạnh hơn
    0: 2.0,   # conv1_1
    5: 1.5,   # conv2_1
    10: 1.2,  # conv3_1
    21: 0.9,  # conv4_1
    30: 0.7   # conv5_1
}

STYLE_PRESETS = {
    "baseline": dict(
        num_iterations=250,
        image_size=(768, 768),
        content_weight=2e4,
        content_weight_end=2e4,
        style_weight=1e6,
        tv_weight=1e-5,
        learning_rate=0.001,
        save_every=50,
        style_layer_weights=None,
        init_mode='content'
    ),
    "vangogh_strong": dict(
        num_iterations=400,
        image_size=(512, 512),
        content_weight=1.2e4,
        content_weight_end=4e3,
        style_weight=5e6,
        tv_weight=4e-5,
        learning_rate=0.006,
        save_every=50,
        style_layer_weights=STYLE_LAYER_FOCUS_VANGOGH,
        init_mode='hybrid'
    ),
    "vangogh_extreme": dict(
        num_iterations=600,
        image_size=(512, 512),
        content_weight=1e3,
        content_weight_end=3e2,
        style_weight=1e7,
        tv_weight=2e-5,
        learning_rate=0.005,
        save_every=50,
        style_layer_weights={k: v * 1.2 for k, v in STYLE_LAYER_FOCUS_VANGOGH.items()},
        init_mode='style'
    )
}


def train_neural_style_transfer(
    content_path,
    style_path,
    output_path='output.jpg',
    num_iterations=200,
    image_size=(256, 256),
    content_weight=1e5,
    content_weight_end=None,
    style_weight=1e6,
    tv_weight=1e-4,
    learning_rate=0.01,
    save_every=50,
    style_layer_weights=None,
    init_mode='content',
    device='cuda'
):
    """
    Train Neural Style Transfer
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Path to save output image
        num_iterations: Số iterations (Colab: 200-300)
        image_size: Image size (Colab: (256, 256) hoặc (512, 512))
        content_weight: Weight ban đầu cho content loss
        content_weight_end: Weight cuối (nếu muốn giảm dần, None = cố định)
        style_weight: Weight cho style loss
        tv_weight: Weight cho TV loss
        learning_rate: Learning rate cho optimizer
        save_every: Save checkpoint mỗi N iterations
        style_layer_weights: Dict layer index -> weight (ưu tiên layer style mạnh hơn)
        init_mode: 'content' | 'style' | 'hybrid' | 'noise'
        device: 'cuda' hoặc 'cpu'
    """
    # Kiểm tra GPU
    if device == 'cuda':
        assert torch.cuda.is_available(), "GPU not available! Use device='cpu' or use Colab."
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # **QUAN TRỌNG**: Kiểm tra file ảnh tồn tại trước khi load
    print("🔍 Checking image files...")
    if not os.path.exists(content_path):
        raise FileNotFoundError(
            f"Content image not found: {content_path}\n"
            f"Please upload your content image first (Bước 3 - Upload Ảnh)"
        )
    if not os.path.exists(style_path):
        raise FileNotFoundError(
            f"Style image not found: {style_path}\n"
            f"Please upload your style image first (Bước 3 - Upload Ảnh)"
        )
    print(f"✅ Content image found: {content_path}")
    print(f"✅ Style image found: {style_path}")
    
    # Load và preprocess images
    print("\n📦 Loading images...")
    content_tensor, style_tensor = preprocess_images(
        content_path, style_path, size=image_size, device=device
    )
    print(f"✅ Images loaded: {image_size}")
    
    # Load VGG19 model
    print("📦 Loading VGG19 model...")
    model = load_vgg_model(model_name='vgg19', device=device)
    print("✅ VGG19 loaded successfully")
    
    # Initialize generated image theo init_mode
    print("📦 Initializing generated image...")
    generated_image = initialize_generated_image(init_mode, content_tensor, style_tensor)
    print("✅ Generated image initialized")
    
    # Setup optimizer
    optimizer = optim.Adam([generated_image], lr=learning_rate)
    
    # Loss history
    losses_history = {
        'content': [],
        'style': [],
        'tv': [],
        'total': []
    }
    
    print(f"\n🚀 Starting training for {num_iterations} iterations...")
    print(f"   Content weight start: {content_weight}")
    if content_weight_end is not None:
        print(f"   Content weight end:   {content_weight_end}")
    print(f"   Style weight:          {style_weight}")
    print(f"   TV weight:             {tv_weight}")
    print(f"   Learning rate:         {learning_rate}")
    print(f"   Init mode:             {init_mode}\n")
    
    if content_weight_end is None:
        content_weight_end = content_weight
    
    # Training loop
    for iteration in tqdm(range(num_iterations), desc="Training"):
        optimizer.zero_grad()
        progress = iteration / max(num_iterations - 1, 1)
        current_content_weight = (
            content_weight * (1 - progress) + content_weight_end * progress
        )
        
        # Forward pass: Calculate losses
        total_loss, content_loss_val, style_loss_val, tv_loss_val = compute_total_loss(
            generated_image,
            model,
            content_tensor,
            style_tensor,
            content_weight=current_content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            style_layer_weights=style_layer_weights,
            device=device
        )
        
        # **BẮT BUỘC**: Check NaN (tránh Case 6)
        if torch.isnan(total_loss):
            print(f"\n⚠️ NaN detected at iteration {iteration}! Stopping...")
            break
        
        # Backward pass
        total_loss.backward()
        
        # **BẮT BUỘC**: Gradient clipping (tránh Case 6)
        torch.nn.utils.clip_grad_norm_(generated_image, max_norm=1.0)
        
        # Update image
        optimizer.step()
        
        # Clamp values để tránh giá trị ngoài phạm vi
        # Note: Image đã được normalize, nên không cần clamp [0,1]
        # Nhưng có thể clamp để tránh giá trị quá lớn
        with torch.no_grad():
            # Clamp để tránh giá trị quá lớn (sau khi denormalize)
            # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
            # Với normalization, giá trị hợp lý nằm trong khoảng [-2, 2]
            generated_image.data.clamp_(-2, 2)
        
        # Record losses
        losses_history['content'].append(content_loss_val.item())
        losses_history['style'].append(style_loss_val.item())
        losses_history['tv'].append(tv_loss_val.item())
        losses_history['total'].append(total_loss.item())
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"\nIteration {iteration + 1}/{num_iterations}:")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  Content Loss ({current_content_weight:.2e}): {content_loss_val.item():.4f}")
            print(f"  Style Loss: {style_loss_val.item():.4f}")
            print(f"  TV Loss: {tv_loss_val.item():.6f}")
            if style_layer_weights:
                print(f"  Style Layer Weights: {style_layer_weights}")
        
        # Save checkpoint
        if (iteration + 1) % save_every == 0:
            checkpoint_path = output_path.replace('.jpg', f'_iter_{iteration+1}.jpg')
            save_image(generated_image, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
            
            # Visualize
            show_images(content_tensor, style_tensor, generated_image)
    
    # Save final result
    print(f"\n💾 Saving final result to {output_path}...")
    save_image(generated_image, output_path)
    
    # Plot loss curves
    loss_plot_path = output_path.replace('.jpg', '_losses.png')
    print(f"📊 Plotting loss curves to {loss_plot_path} ...")
    plot_losses(losses_history, save_path=loss_plot_path)
    
    print("\n✅ Training completed!")
    
    return generated_image, losses_history


def save_image(tensor, path):
    """Save tensor as image"""
    image = tensor_to_image(tensor)
    image.save(path)
    print(f"✅ Image saved: {path}")


if __name__ == "__main__":
    # Example usage
    # Trên Colab, upload ảnh vào thư mục và chỉnh đường dẫn
    
    content_path = "data/content/your_content.jpg"
    style_path = "data/style/your_style.jpg"
    output_path = "output.jpg"
    
    # Check if files exist
    if not os.path.exists(content_path):
        print(f"⚠️ Content image not found: {content_path}")
        print("   Please upload your content image to this path.")
    
    if not os.path.exists(style_path):
        print(f"⚠️ Style image not found: {style_path}")
        print("   Please upload your style image to this path.")
    
    if os.path.exists(content_path) and os.path.exists(style_path):
        preset_name = "baseline"  # 'baseline' | 'vangogh_strong' | 'vangogh_extreme'
        preset_params = STYLE_PRESETS[preset_name]
        print(f"\n🎚 Using preset: {preset_name}")
        for key, value in preset_params.items():
            if key != "style_layer_weights":
                print(f"   {key}: {value}")
        if preset_params.get("style_layer_weights"):
            print(f"   style_layer_weights: {preset_params['style_layer_weights']}")
        
        # Train với preset được chọn
        generated_image, losses = train_neural_style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            device='cuda',
            **preset_params
        )
    else:
        print("\n📝 Please upload your images first!")
        print("   Content image: data/content/your_content.jpg")
        print("   Style image: data/style/your_style.jpg")

