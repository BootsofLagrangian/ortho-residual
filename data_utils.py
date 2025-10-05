import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any


class _TransformedDataset(Dataset):
    """Lightweight view over a dataset that applies transforms in __getitem__."""

    def __init__(self, base: Any, image_key: str, label_key: str, transform):
        self._base = base
        self._image_key = image_key
        self._label_key = label_key
        self._transform = transform

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        sample = self._base[idx]
        image = sample[self._image_key]
        if self._transform is not None:
            image = self._transform(image)
        label = sample[self._label_key]
        return image, label


def get_dataset(args):
    name = args.dataset.lower()

    def random_erasing():
        return transforms.RandomErasing(
            p=args.random_erase,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value="random",
        )

    if name == "mnist":
        dataset = load_dataset("mnist")
        in_chans = 1
        image_key = "image"
        label_key = "label"
        num_classes = 10
        transform_train = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
        transform_eval = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ]
        )
    elif name == "cifar10":
        dataset = load_dataset("uoft-cs/cifar10")
        in_chans = 3
        image_key = "img"
        label_key = "label"
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        if "resnet" in args.model:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(
                        num_ops=args.randaugment_N, magnitude=args.randaugment_M
                    ),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.ToTensor(),
                    random_erasing(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    elif name == "cifar100":
        dataset = load_dataset("uoft-cs/cifar100")
        in_chans = 3
        image_key = "img"
        label_key = "fine_label"
        num_classes = 100
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        if "resnet" in args.model:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(
                        num_ops=args.randaugment_N, magnitude=args.randaugment_M
                    ),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    transforms.ToTensor(),
                    random_erasing(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    elif name == "tiny-imagenet":
        dataset = load_dataset("zh-plus/tiny-imagenet")
        in_chans = 3
        image_key = "image"
        label_key = "label"
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if "resnet" in args.model:
            transform_train = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.RandomResizedCrop(
                        64, scale=(0.7, 1.0), ratio=(0.75, 1.33)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            train_tfms = [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=args.randaugment_N, magnitude=args.randaugment_M
                ),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                random_erasing(),
                transforms.Normalize(mean, std),
            ]
            eval_tfms = [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            transform_train = transforms.Compose(train_tfms)
            transform_eval = transforms.Compose(eval_tfms)

    elif name == "imagenet1k":
        dataset = load_dataset("timm/imagenet-1k-wds")
        in_chans = 3
        image_key = "jpg"  # PIL.Image 객체
        label_key = "cls"
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if "resnet" in args.model:
            transform_train = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.RandomResizedCrop(
                        224, scale=(0.7, 1.0), ratio=(0.75, 1.33)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.RandomResizedCrop(
                        224, scale=(0.8, 1.0), ratio=(0.75, 1.33)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandAugment(
                        num_ops=args.randaugment_N, magnitude=args.randaugment_M
                    ),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=args.random_erase, value="random"),
                    transforms.Normalize(mean, std),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    dataset = {
        split: _TransformedDataset(
            split_dataset,
            image_key=image_key,
            label_key=label_key,
            transform=transform_train if split == "train" else transform_eval,
        )
        for split, split_dataset in dataset.items()
    }

    def collate_train(batch):
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), torch.as_tensor(labels)

    def collate_eval(batch):
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), torch.as_tensor(labels)

    return dataset, collate_train, collate_eval, in_chans, num_classes


def load_state_dict_ckpt(model, sd):
    before_loading = {k: v.clone() for k, v in model.state_dict().items()}

    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    after_loading = {k: v.clone() for k, v in model.state_dict().items()}

    for k in before_loading:
        if not torch.allclose(before_loading[k], after_loading[k]):
            print(f"Parameter {k} changed after loading")
            print(f"Before: {before_loading[k]}")
            print(f"After: {after_loading[k]}")
