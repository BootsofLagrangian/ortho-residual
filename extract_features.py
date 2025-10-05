#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from tqdm import tqdm

from base import Classifier, PRESET_VIT
from preactresnet import PRESET_PREACT_RESNET
# from connect import set_connect
from data_utils import get_dataset
from ortho_models import OrthoBlock


def load_checkpoint(ckpt_path):
    """Load checkpoint and extract args and model state dict."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt['args'], ckpt['model']


def create_model_for_features(args, state_dict):
    """Create model and modify it for feature extraction."""
    
    
    # Get dataset info
    dataset_name = args.dataset.split(':')[0]
    in_chans = 1 if dataset_name in ["mnist"] else 3
    num_classes = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet1k": 1000,
    }[dataset_name]
    
    # Create model based on model type
    if args.model == "vit":
        # Get ViT configuration
        model_size = getattr(args, 'model_size', getattr(args, 'preset', None))
        assert model_size in PRESET_VIT, f"Unknown ViT preset: {model_size}"
        preset = PRESET_VIT[model_size]
        embed_dim = preset['embed_dim']
        depth = preset['depth'] 
        num_heads = preset['num_heads']
        patch_size = 16  # Default patch size for ViT
        res_conn = "linear" if not getattr(args, 'orthogonal_residual', False) else "orthogonal"
        model = Classifier(
            img_size=getattr(args, 'image_size', 224),
            dim=embed_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=depth,
            in_chans=in_chans,
            num_classes=num_classes,  # Keep original num_classes for proper loading
            class_token=True,
            reg_tokens=0,
            pos_embed="learn",
            block_class=OrthoBlock,
            residual_connection=res_conn,
            orthogonal_method=getattr(args, 'orthogonal_method', "feature"),
            residual_eps=getattr(args, 'orthogonal_eps', 1e-6),
            residual_perturbation=getattr(args, 'orthogonal_perturbation', None),
        )
    elif args.model == "preactresnet":
        # Handle PreActResNet models
        model_size = getattr(args, 'model_size', getattr(args, 'preset', None))
        model_name = args.model + '-' + model_size
        assert model_name in PRESET_PREACT_RESNET, f"Unknown PreActResNet preset: {model_name}"
        depths = PRESET_PREACT_RESNET[model_name]
        
        from preactresnet import PreActResNet
        model = PreActResNet(
            depths=depths,
            input_shape=(in_chans, getattr(args, 'image_size', 224), getattr(args, 'image_size', 224)),
            num_classes=num_classes,
            drop_path_rate=getattr(args, 'drop_path', 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    print(model.blocks[0].__class__.__name__, "loaded with state dict")
    print(getattr(model.blocks[0], "residual_kwargs", {}).get("method", "linear"), "residual connection method")
    
    # Replace classifier head with Identity for feature extraction
    if args.model == "vit":
        # For ViT, replace the head
        model.classifier = nn.Identity()
    elif args.model == "preactresnet":
        # For PreActResNet, replace the linear layer
        model.linear = nn.Identity()
    print(model)
    return model


def extract_features(model, dataloader, device):
    """Extract features from validation dataset."""
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            
            # Forward pass - this will return features before the classifier
            features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels


def main():
    parser = argparse.ArgumentParser(description='Extract features from trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./features',
                        help='Directory to save extracted features')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for extraction')
    
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt_args, state_dict = load_checkpoint(args.checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if "eps" in k:
            new_state_dict[k] = v.reshape(1,)
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    # Print checkpoint info
    model_size = getattr(ckpt_args, 'model_size', getattr(ckpt_args, 'preset', 'unknown'))
    print(f"Model: {ckpt_args.model}-{model_size}")
    print(f"Dataset: {ckpt_args.dataset}")
    if "imagenet1k" in ckpt_args.dataset:
        dataset_name = "timm/imagenet-1k-wds"
    else:
        dataset_name = ckpt_args.dataset
    image_size = getattr(ckpt_args, 'image_size', 224)  # Default to 224 for ImageNet
    print(ckpt_args)
    print(f"Image size: {image_size}")
    print(f"Orthogonal residual: {getattr(ckpt_args, 'orthogonal_residual', False)}")
    if hasattr(ckpt_args, 'orthogonal_method'):
        print(f"Orthogonal method: {ckpt_args.orthogonal_method}")
    
    # Get validation dataset
    # Create a minimal args object for get_dataset
    dataset_args = argparse.Namespace(
        dataset=ckpt_args.dataset,
        random_erase=0.0,  # No random erasing for validation
        randaugment_N=0,
        randaugment_M=0,
        model="vit" if ckpt_args.model == "vit" else "preactresnet",
    )
    dataset, _, collate_eval, _, _ = get_dataset(dataset_args)
    if "test" in dataset:
        test_dataset = dataset["test"]
    elif "val" in dataset:
        test_dataset = dataset["val"]
    elif "valid" in dataset:
        test_dataset = dataset["valid"]
    elif "validation" in dataset:
        test_dataset = dataset["validation"]
    else:
        raise ValueError("No test/validation set found in the dataset.")
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_eval
    )
    
    # Create model with feature extraction setup
    model = create_model_for_features(ckpt_args, state_dict)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Extract features
    features, labels = extract_features(model, val_loader, args.device)
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save features
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename based on checkpoint info
    ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
    connection_type = 'ortho' if getattr(ckpt_args, 'orthogonal_residual', False) else 'linear'
    model_info = f"{ckpt_args.model}-{model_size}"
    dataset_name = ckpt_args.dataset.split(':')[0]
    save_path = os.path.join(args.output_dir, f'{dataset_name}_{model_info}_{connection_type}_{ckpt_name}_features.npz')
    
    np.savez(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")
    
    # Print some statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")


if __name__ == '__main__':
    main()