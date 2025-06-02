# evaluate_hf_ortho_vit.py
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import argparse
from typing import Tuple, List

def accuracy_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> List[int]:
    """
    Given model outputs and targets, return a list of correct-counts
    for each k in topk.
    """
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res

def evaluate_model(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load model from Hugging Face Hub
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Image transformations for ImageNet-1k ViT evaluation
    # Based on user's data_utils.py for ViT on imagenet1k
    img_size = args.img_size # Should match model's expected input size, e.g., 224 for ViT-B
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_eval = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load ImageNet-1k validation dataset
    # Using standard 'imagenet-1k' dataset from Hugging Face datasets
    # Ensure you have access or it will download (can be large)
    # The image key is 'jpg' and label key is 'cls' for this dataset.
    print("Loading ImageNet-1k validation dataset...")
    try:
        val_dataset = load_dataset(
            "timm/imagenet-1k-wds",
            split="validation",
            # cache_dir=args.dataset_cache_dir,
        )
    except Exception as e:
        print(f"Failed to load 'imagenet-1k'. Error: {e}")
        print("Please ensure you have access to the ImageNet-1k dataset on Hugging Face Hub,")
        print("or specify a local path if you have it downloaded.")
        print("You might need to authenticate with `huggingface-cli login` and accept dataset terms.")
        return

    def collate_fn(batch):
        # Filter out None items if any image failed to load/transform
        # batch = [item for item in batch if item['image'] is not None] # some datasets might have None
        # if not batch:
        #     return None, None
        images = torch.stack([transform_eval(item['jpg']) for item in batch])
        labels = torch.tensor([item['cls'] for item in batch])
        return images, labels

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"Dataset loaded. Number of validation samples: {len(val_dataset)}")

    # Evaluation loop
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            if images is None or labels is None: # Skip batch if collate_fn returned None
                continue
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            counts = accuracy_counts(logits, labels, topk=(1, 5))
            correct_top1 += counts[0]
            correct_top5 += counts[1]
            total_samples += images.size(0)

    if total_samples == 0:
        print("No samples were processed. Check dataset loading and paths.")
        return

    top1_accuracy = (correct_top1 / total_samples) * 100
    top5_accuracy = (correct_top5 / total_samples) * 100

    print("\n--- Evaluation Results ---")
    print(f"Model: {args.model_name_or_path}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OrthoViT model on ImageNet-1k")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="BootsofLagrangian/ortho-vit-b-imagenet1k-hf",
        help="Name of the model on Hugging Face Hub or local path.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Directory to cache the Hugging Face dataset.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for evaluation (must match model's expected input).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32, # Adjust based on your GPU memory
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,  # Adjust based on your system
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use CPU even if CUDA is available.",
    )
    args = parser.parse_args()

    evaluate_model(args)