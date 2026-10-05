"""
Loss Functions cho Neural Style Transfer
- Content Loss: Giữ nội dung ảnh gốc
- Style Loss: Áp dụng phong cách nghệ thuật (Gram Matrix)
- Total Variation Loss: Giảm noise và artifacts
"""

import torch
import torch.nn.functional as F


def content_loss(content_features, generated_features):
    """
    Content Loss: MSE giữa content features và generated features
    
    Args:
        content_features: Features từ content image [B, C, H, W]
        generated_features: Features từ generated image [B, C, H, W]
    
    Returns:
        Content loss (scalar)
    """
    return F.mse_loss(content_features, generated_features)


def gram_matrix(features):
    """
    Tính Gram matrix cho style representation
    
    Gram matrix capture correlations giữa các feature maps,
    đại diện cho style của ảnh
    
    Args:
        features: Feature tensor [B, C, H, W]
    
    Returns:
        Gram matrix [B, C, C]
    """
    batch, channels, height, width = features.size()
    
    # Reshape: [B, C, H*W]
    features = features.view(batch, channels, height * width)
    
    # Gram matrix = features @ features^T
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by number of elements
    gram = gram / (channels * height * width)
    
    return gram


def style_loss(style_features, generated_features):
    """
    Style Loss: MSE giữa Gram matrices của style và generated features
    
    Args:
        style_features: Features từ style image [B, C, H, W]
        generated_features: Features từ generated image [B, C, H, W]
    
    Returns:
        Style loss (scalar)
    """
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)
    
    return F.mse_loss(style_gram, generated_gram)


def total_variation_loss(image):
    """
    Total Variation Loss: Giảm noise và artifacts (tránh Case 13)
    
    Penalize sự thay đổi lớn giữa các pixel lân cận
    
    Args:
        image: Image tensor [B, C, H, W]
    
    Returns:
        TV loss (scalar)
    """
    batch, channels, height, width = image.size()
    
    # Horizontal variation
    tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).sum()
    
    # Vertical variation
    tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).sum()
    
    # Normalize
    tv_loss = (tv_h + tv_w) / (batch * channels * height * width)
    
    return tv_loss


def compute_total_loss(
    generated_image,
    model,
    content_image,
    style_image,
    content_weight=1e5,
    style_weight=1e6,
    tv_weight=1e-4,
    style_layer_weights=None,
                      device='cuda'
):
    """
    Tính tổng loss cho Neural Style Transfer
    
    Args:
        generated_image: Generated image tensor [1, 3, H, W]
        model: VGGFeatureExtractor model
        content_image: Content image tensor [1, 3, H, W]
        style_image: Style image tensor [1, 3, H, W]
        content_weight: Weight cho content loss (default: 1e5)
        style_weight: Weight cho style loss (default: 1e6)
        tv_weight: Weight cho TV loss (default: 1e-4)
        style_layer_weights: Dict layer_index -> weight (ưu tiên layer nào thể hiện style nhiều hơn)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        total_loss, content_loss_val, style_loss_val, tv_loss_val
    """
    # Extract content features
    content_features = model.get_content_features(content_image)
    generated_content_features = model.get_content_features(generated_image)
    
    # Calculate content loss
    content_loss_val = content_loss(content_features, generated_content_features)
    
    # Extract style features từ tất cả style layers
    style_features_dict = model.get_style_features(style_image)
    generated_style_features_dict = model.get_style_features(generated_image)
    
    # Calculate style loss từ tất cả layers
    style_loss_val = 0.0
    total_style_weight = 0.0
    
    for layer in style_features_dict.keys():
        layer_weight = 1.0
        if style_layer_weights and layer in style_layer_weights:
            layer_weight = style_layer_weights[layer]
        
        layer_loss = style_loss(
            style_features_dict[layer],
            generated_style_features_dict[layer]
        )
        style_loss_val += layer_weight * layer_loss
        total_style_weight += layer_weight
    
    if total_style_weight > 0:
        style_loss_val /= total_style_weight
    else:
        style_loss_val /= len(style_features_dict)  # Average
    
    # Calculate TV loss
    tv_loss_val = total_variation_loss(generated_image)
    
    # Total loss
    total_loss = (
        content_weight * content_loss_val +
        style_weight * style_loss_val +
        tv_weight * tv_loss_val
    )
    
    return total_loss, content_loss_val, style_loss_val, tv_loss_val

