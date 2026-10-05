"""
Visualization utilities cho Neural Style Transfer
"""

import matplotlib.pyplot as plt
import torch
from .preprocessing import tensor_to_image


def show_images(content_img, style_img, generated_img, save_path=None):
    """
    Hiển thị content, style và generated images
    
    Args:
        content_img: Content image tensor [1, 3, H, W]
        style_img: Style image tensor [1, 3, H, W]
        generated_img: Generated image tensor [1, 3, H, W]
        save_path: Path để save figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to images
    content_pil = tensor_to_image(content_img)
    style_pil = tensor_to_image(style_img)
    generated_pil = tensor_to_image(generated_img)
    
    axes[0].imshow(content_pil)
    axes[0].set_title('Content Image')
    axes[0].axis('off')
    
    axes[1].imshow(style_pil)
    axes[1].set_title('Style Image')
    axes[1].axis('off')
    
    axes[2].imshow(generated_pil)
    axes[2].set_title('Generated Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_losses(losses_history, save_path=None):
    """
    Plot loss curves
    
    Args:
        losses_history: Dictionary với keys: 'content', 'style', 'tv', 'total'
        save_path: Path để save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = range(len(losses_history['total']))
    
    axes[0, 0].plot(iterations, losses_history['content'], label='Content Loss')
    axes[0, 0].set_title('Content Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(iterations, losses_history['style'], label='Style Loss', color='orange')
    axes[0, 1].set_title('Style Loss')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(iterations, losses_history['tv'], label='TV Loss', color='green')
    axes[1, 0].set_title('Total Variation Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(iterations, losses_history['total'], label='Total Loss', color='red')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

