"""
Image Preprocessing cho Neural Style Transfer
Tránh Case 9: Size Mismatch
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def load_image(image_path, size=None, keep_aspect_ratio=True):
    """
    Load và preprocess image
    
    Args:
        image_path: Path to image file
        size: Target size (width, height). Nếu None, giữ nguyên size
        keep_aspect_ratio: Nếu True, resize giữ aspect ratio (tránh stretch)
    
    Returns:
        PIL Image
    """
    import os
    # Kiểm tra file tồn tại trước
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}\n"
            f"Please upload your image first (Bước 3 - Upload Ảnh in notebook)"
        )
    
    try:
        img = Image.open(image_path).convert('RGB')
        
        if size is not None:
            if keep_aspect_ratio:
                # Resize giữ aspect ratio (tránh Case 9: Size Mismatch)
                img.thumbnail(size, Image.Resampling.LANCZOS)
            else:
                img = img.resize(size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")


def image_to_tensor(image, size=None, device='cuda'):
    """
    Convert PIL Image to PyTorch tensor với normalization
    
    Args:
        image: PIL Image
        size: Target size (width, height). Nếu None, giữ nguyên
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        Tensor [1, 3, H, W] normalized theo ImageNet
    """
    # ImageNet normalization (tránh Case 14: Màu sắc lệch)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Transform pipeline
    if size is not None:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Convert to tensor và add batch dimension
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor về PIL Image
    
    Args:
        tensor: Tensor [1, 3, H, W] đã normalized
    
    Returns:
        PIL Image
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    tensor = tensor.cpu().clone()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.squeeze(0))
    
    return image


def preprocess_images(content_path, style_path, size=(256, 256), device='cuda'):
    """
    Preprocess cả content và style images
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        size: Target size (width, height). 
              Colab: (256, 256) hoặc (512, 512)
              Local: (224, 224)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        content_tensor, style_tensor: Both [1, 3, H, W]
    """
    # Load images
    content_img = load_image(content_path, size=size)
    style_img = load_image(style_path, size=size)
    
    # Convert to tensors
    content_tensor = image_to_tensor(content_img, size=size, device=device)
    style_tensor = image_to_tensor(style_img, size=size, device=device)
    
    return content_tensor, style_tensor

