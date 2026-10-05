"""
VGG Model Wrapper for Neural Style Transfer
Sử dụng VGG19 pre-trained để extract features
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """
    Extract features từ VGG19 cho Neural Style Transfer
    
    VGG19 layers:
    - Content layer: layer 21 (conv_4)
    - Style layers: layers [0, 5, 10, 21, 30] (conv_1 đến conv_5)
    """
    
    def __init__(self, model_name='vgg19'):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19
        if model_name == 'vgg19':
            # Fix warning: dùng weights thay vì pretrained
            try:
                vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            except AttributeError:
                # Fallback cho version cũ
                vgg = models.vgg19(pretrained=True)
            # VGG19 layer indices cho style và content
            self.content_layers = [21]  # conv_4
            self.style_layers = [0, 5, 10, 21, 30]  # conv_1 đến conv_5
        elif model_name == 'vgg16':
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            except AttributeError:
                # Fallback cho version cũ
                vgg = models.vgg16(pretrained=True)
            # VGG16 layer indices
            self.content_layers = [15]  # conv_4
            self.style_layers = [0, 5, 10, 15, 19]  # conv_1 đến conv_5
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'vgg19' or 'vgg16'")
        
        # Chỉ lấy phần features (bỏ classifier)
        self.features = vgg.features
        
        # Freeze tất cả weights (không train)
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Set to eval mode (giảm memory)
        self.features.eval()
        
        self.model_name = model_name
    
    def forward(self, x, layers=None):
        """
        Extract features từ các layers chỉ định
        
        Args:
            x: Input image tensor [B, C, H, W]
            layers: List các layer indices cần extract. 
                   Nếu None, extract tất cả style và content layers
        
        Returns:
            Dictionary với key là layer index, value là feature tensor
        """
        if layers is None:
            layers = list(set(self.content_layers + self.style_layers))
        
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in layers:
                features[i] = x
        
        return features
    
    def get_content_features(self, x):
        """Extract content features từ content layer"""
        features = self.forward(x, layers=self.content_layers)
        return features[self.content_layers[0]]
    
    def get_style_features(self, x):
        """Extract style features từ tất cả style layers"""
        features = self.forward(x, layers=self.style_layers)
        return {layer: features[layer] for layer in self.style_layers}


def load_vgg_model(model_name='vgg19', device='cuda'):
    """
    Load VGG model và move to device
    
    Args:
        model_name: 'vgg19' (đúng yêu cầu) hoặc 'vgg16' (backup)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        VGGFeatureExtractor model
    """
    model = VGGFeatureExtractor(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model

