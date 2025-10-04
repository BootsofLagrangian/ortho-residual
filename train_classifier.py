#!/usr/bin/env python
import torch
# Enable TF32 on supported hardware
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from glob import glob
import logging
import time
import random
import argparse
import os
import math
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from typing import Tuple, List, Dict, Optional

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from preactresnet import PRESET_PREACT_RESNET
from base import Classifier, PRESET_VIT
from ortho_models import OrthoBlock
from connect import _METRICS, ConnLoggerMixin, set_connect
from data_utils import get_dataset

def create_logger(logging_dir, debug: bool = False):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_state_dict_for_save(m):
    # 1. unwrap DDP
    if isinstance(m, torch.nn.parallel.DistributedDataParallel):
        m = m.module
    # 2. unwrap torch.compile
    m = getattr(m, "_orig_mod", m)
    return m.state_dict()

def load_state_dict_ckpt(model, sd):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model = getattr(model, "_orig_mod", model)
    model.load_state_dict(sd)
def log_global_grad_norm(model, step, max_log_steps=50):
    tot, tot_sq = 0.0, 0.0
    for p in model.parameters():
        if p.grad is None: continue
        g2 = p.grad.detach().pow(2).sum()
        tot_sq += g2
    return {
        "grad/global_norm": torch.sqrt(tot_sq).item()
    }


def accuracy_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,5),
) -> List[int]:
    """
    Given model outputs and targets, return a list of correct‐counts
    for each k in topk.
    """
    maxk = max(topk)

    # (batch_size, maxk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    # (maxk, batch_size)
    pred = pred.t()
    # compare to (1, batch_size) broadcasted
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # return list of sums over first k rows
    return [correct[:k].reshape(-1).sum().item() for k in topk]


def evaluate_and_log(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    topk: Tuple[int,int] = (1,5),
) -> Dict[str, float]:
    """
    Runs model over loader, does cross‐entropy loss (summed),
    computes top‐k accuracies, does all_reduce for each metric,
    and returns a dict { 'val_loss':…, 'val_acc@1':…, 'val_acc@5':… }.
    """
    model.eval()

    sum_loss = 0.0
    sum_correct = [0 for _ in topk]
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # sum batch loss so we can average accurately later
            sum_loss += F.cross_entropy(logits, y, reduction='sum').item()

            # accumulate correct counts
            counts = accuracy_counts(logits, y, topk=topk)
            for i, c in enumerate(counts):
                sum_correct[i] += c

            total_samples += y.size(0)

    # wrap as tensors for distributed reduction
    t_loss    = torch.tensor(sum_loss,       device=device)
    t_total   = torch.tensor(total_samples,  device=device)
    dist.all_reduce(t_loss,  op=dist.ReduceOp.SUM)
    dist.all_reduce(t_total, op=dist.ReduceOp.SUM)

    # prepare return dict
    results: Dict[str, float] = {}
    val_loss = t_loss.item() / t_total.item()
    results["val_loss"] = val_loss

    # reduce each correct‐count and compute accuracy
    for k, c in zip(topk, sum_correct):
        t_c = torch.tensor(c, device=device, dtype=torch.float32)
        dist.all_reduce(t_c, op=dist.ReduceOp.SUM)
        results[f"val_acc@{k}"] = t_c.item() / t_total.item()

    return results


def main(args):
    # Dataset-specific parameters
    if args.dataset == "mnist":
        img_size = 28
        patch_size = 7
        num_classes = 10
    elif args.dataset == "cifar100":
        img_size = 32
        patch_size = 4
        num_classes = 100
    elif args.dataset == "cifar10":
        img_size = 32
        patch_size = 4
        num_classes = 10
    elif args.dataset == "tiny-imagenet":
        img_size = 64
        patch_size = 8
        num_classes = 200
    elif args.dataset == "imagenet1k":
        img_size = 224
        patch_size = 16
        num_classes = 1000
        
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob, switch_prob=args.cutmix_switch_prob,
            label_smoothing=args.label_smoothing, num_classes=num_classes
        )
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = lambda x, y: (x, y)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


    # Initialize distributed process group with NCCL backend
    dist.init_process_group(backend="nccl")
    # Get local rank from environment variables (torchrun sets LOCAL_RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = args.seed * dist.get_world_size() + rank
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}. local seed={seed}. global batch size={args.global_batch_size}.")
    res_conn = "orthogonal" if args.orthogonal_residual else "linear"
    ortho_method = args.orthogonal_method if args.orthogonal_residual else "linear"
    if args.orthogonal_method == "negative":
        ortho_method = "negative"
    run_name = f"{args.model}-{args.preset}/{patch_size}_{res_conn}_{ortho_method}_{args.dataset}_seed{args.seed}"
    print(f"Run name: {run_name}")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = run_name.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, args.debug)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(project=args.project, config=vars(args), name=f"{run_name}-{experiment_index:03d}")
    else:
        logger = create_logger(None)

    # Set the seed for reproducibility (unique for each rank)
    torch.manual_seed(args.seed + rank)

    # Calculate local batch size per GPU given the global batch size
    local_batch_size = args.global_batch_size // world_size
    local_batch_size = local_batch_size // args.grad_accumulate_steps
    assert (local_batch_size * world_size * args.grad_accumulate_steps) == args.global_batch_size

    # Model presets: each preset defines embed_dim, depth, and num_heads
    presets = PRESET_VIT
    preset_params = presets[args.preset]
    embed_dim = preset_params["embed_dim"]
    depth = preset_params["depth"]
    num_heads = preset_params["num_heads"]

    if rank == 0:
        logger.info(f"Training {run_name} on device: {device}")

    # Load dataset and determine the train and test splits
    dataset, collate_train, collate_eval, in_chans, num_classes = get_dataset(args)
    train_dataset = dataset["train"]
    
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

    # Create Distributed Samplers for training and testing
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank, shuffle=True,
        seed=args.seed
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size,
        num_workers=16,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler, collate_fn=collate_train
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=local_batch_size,
        sampler=test_sampler, collate_fn=collate_eval
    )

    # Build the model based on the chosen option
    if "resnet" in args.model:
        residual_kwargs = dict(
            residual_connection=res_conn,
            orthogonal_method=args.orthogonal_method,
            residual_eps=args.orthogonal_eps,
            residual_perturbation=args.orthogonal_perturbation,
            log_interval=args.log_interval,
            log_activations=(rank == 0 and args.log_activations),
            num_classes=num_classes,
            is_layernorm_classifier=args.is_layernorm_classifier,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        resnet = PRESET_PREACT_RESNET[args.model]
        base_model = resnet(**residual_kwargs)
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
    elif args.model == "vit":
        base_model = Classifier(
            img_size=img_size,
            dim=embed_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=depth,
            in_chans=in_chans,
            num_classes=num_classes,
            class_token=True,    # Use a CLS token
            reg_tokens=0,
            pos_embed="learn",
            block_class=OrthoBlock,
            drop_path=args.drop_path,
            residual_connection=res_conn,
            orthogonal_method=args.orthogonal_method,
            residual_eps=args.orthogonal_eps,
            residual_perturbation=args.orthogonal_perturbation,
            modulate=False,
            mlp_dropout=args.mlp_dropout,
            log_interval=args.log_interval,
            log_activations=(rank == 0 and args.log_activations),
            gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if args.orthogonal_pattern is not None:
        pattern = args.orthogonal_pattern.split(",")
        pattern = [int(p) for p in pattern]
        if max(pattern) > len(base_model.blocks):
            raise ValueError(f"Invalid pattern: {args.orthogonal_pattern}. Must be less than {len(base_model.blocks)}.")
        i = 0
        set_connect(base_model, pattern=pattern, logger=logger)

    logger.info("Number of parameters: %d", sum(p.numel() for p in base_model.parameters()))

    base_model.to(device)
    
    if args.torch_compile:
        # Compile the model with torch.compile
        base_model = torch.compile(
            base_model, 
            mode=args.torch_compile_mode, 
            backend=args.torch_compile_backend,
            fullgraph=False,
        )
        logger.info("Model compiled with torch.compile.")
    
    # Wrap the model with Distributed Data Parallel
    model = DDP(base_model, device_ids=[local_rank])
    if rank == 0:
        logger.info(model.module)
    for module in model.module.modules():
        if isinstance(module, ConnLoggerMixin):
            module.set_step_fn(lambda: train_steps)

    if args.optimizer == "adam":
        betas = (args.adam_beta1, args.adam_beta2)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay,
            fused=True,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            fused=True,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer}")

    steps_per_epoch = min(len(train_loader) // args.grad_accumulate_steps,  args.max_train_steps)
    warmup_iters    = steps_per_epoch * args.warmup_epoch
    min_lr_factor: Optional[float] = None
    if args.min_lr is not None:
        if args.lr <= 0:
            raise ValueError("min_lr requires positive base learning rate")
        if args.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        ratio = args.min_lr / args.lr
        min_lr_factor = min(ratio, 1.0)

    def apply_min_lr(scale: float) -> float:
        if min_lr_factor is None:
            return scale
        return min_lr_factor + (1.0 - min_lr_factor) * scale

    if args.lr_scheduler_type == "cos":
        # cosine decay after warm‑up
        def lr_lambda(step):
            if warmup_iters > 0 and step < warmup_iters:
                return step / warmup_iters
            pct = (step - warmup_iters) / max(1, args.epochs*steps_per_epoch - warmup_iters)
            scale = 0.5 * (1 + math.cos(math.pi * pct))
            scale = max(0.0, scale)
            return apply_min_lr(scale)
    elif args.lr_scheduler_type == "lin":  # linear decay
        def lr_lambda(step):
            if warmup_iters > 0 and step < warmup_iters:
                return step / warmup_iters
            pct = (step - warmup_iters) / max(1, args.epochs*steps_per_epoch - warmup_iters)
            scale = max(0.0, 1.0 - pct)
            return apply_min_lr(scale)
    elif args.lr_scheduler_type == "const":
        def lr_lambda(step):
            if warmup_iters > 0 and step < warmup_iters:
                return step / warmup_iters
            return apply_min_lr(1.0)
    elif args.lr_scheduler_type == "step":
        # Parse decay epochs and convert to steps
        assert args.lr_scheduler_decay_epoch is not None, "lr_scheduler_decay_epoch must be specified for step scheduler."
        decay_epochs = [int(e) for e in args.lr_scheduler_decay_epoch.split(",")]
        decay_steps = sorted([e * steps_per_epoch for e in decay_epochs])

        def lr_lambda(step):
            # Warmup phase
            if warmup_iters > 0 and step < warmup_iters:
                if args.warmup_type == "linear":
                    return float(step) / float(max(1, warmup_iters))
                elif args.warmup_type == "const":  
                    return args.warmup_const
                else:
                    raise ValueError(f"Unknown warmup type: {args.warmup_type}")

            # After warmup, determine the decay factor
            factor = 1.0
            for decay_step in decay_steps:
                if step >= decay_step:
                    factor *= args.lr_scheduler_decay_ratio
                else:
                    # Since decay_steps is sorted, we can break early
                    break
            factor = max(0.0, factor)
            return apply_min_lr(factor)
    else:
        raise ValueError(f"Unknown learning rate scheduler type: {args.lr_scheduler_type}")
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_fp16 = args.use_amp and (args.amp_dtype == "float16")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.GradScaler(enabled=use_fp16)

    train_steps, log_steps, val_loss = 0, 0, 0.0
    # Training loop
    opt_steps    = 0                     # real optimizer steps (= logging step)
    accum_steps  = args.grad_accumulate_steps
    assert accum_steps >= 1

    for epoch in range(args.epochs):
        if train_steps > args.max_train_steps:
            if rank == 0:
                print(f"Reached max training steps: {train_steps}. Stopping training.")
            break
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss, log_steps = 0.0, 0
        current_accum = 0  # Track accumulated steps within the current group
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for i, (x, y) in enumerate(train_loader):
            if train_steps > args.max_train_steps:
                break
            
            # Control orthogonal connections if needed
            if args.orthogonal_prob is not None:
                set_connect(model.module, prob=args.orthogonal_prob, logger=None)

            x = x.to(device)
            y = y.to(device)
            x, y = mixup_fn(x, y)

            # Forward pass with proper AMP handling
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.use_amp):
                logits = model(x)
            
            # Calculate loss outside AMP context to use full FP32 precision
            loss = criterion(logits, y)
            # Scale loss for gradient accumulation (will be adjusted at the end if needed)
            loss = loss / accum_steps
            
            # Backward pass with proper scaling
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update tracking variables
            current_accum += 1
            running_loss += loss.item() * accum_steps
            
            # Handle gradient accumulation - process when we've accumulated enough steps or at the last batch
            is_last_batch = (i == len(train_loader) - 1)
            perform_step = (current_accum == accum_steps) or is_last_batch
            
            if perform_step:
                # Scale the gradients properly if we're at the last batch with incomplete accumulation
                scale_factor = 1.0
                if is_last_batch and current_accum < accum_steps:
                    # Rescale the gradients to match what they would have been with full accumulation
                    scale_factor = accum_steps / current_accum
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(scale_factor)
                
                # Only log steps that correspond to actual parameter updates
                log_steps += 1
                
                # Handle AMP scaling and gradient clipping
                if use_fp16:
                    scaler.unscale_(optimizer)
                    
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Log gradient norms if needed
                grad_log = {}
                if train_steps % args.log_interval == 0 and rank == 0:
                    grad_log = log_global_grad_norm(model, train_steps)

                # Optimizer step
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Learning rate scheduler step
                if isinstance(lr_scheduler,
                            (torch.optim.lr_scheduler.CosineAnnealingLR,
                            torch.optim.lr_scheduler.LambdaLR)):
                    lr_scheduler.step()

                # Update step counters - only increment after actual optimizer steps
                opt_steps += 1
                train_steps += 1
                
                # Reset accumulation counter
                current_accum = 0
                
                # Logging at regular intervals
                if train_steps % args.log_interval == 0:
                    torch.cuda.synchronize()
                    end_time = time.time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Calculate samples per second (accounting for batch size and world size)
                    samples_per_sec = log_steps * local_batch_size * dist.get_world_size() / (end_time - start_time)

                    # Calculate average loss across all processes (per optimizer step)
                    optimizer_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(optimizer_loss, op=dist.ReduceOp.SUM)
                    optimizer_loss = optimizer_loss.item() / dist.get_world_size()
                    avg_train_loss = optimizer_loss / accum_steps

                    # Collect model stats if available
                    stats = None
                    buf_log = {}

                    if rank == 0:
                        if hasattr(model.module, "pop_stats"):
                            stats = model.module.pop_stats()
                            logger.debug(f"stats={stats}")
                            for stat in stats:
                                module_name = stat.module_name
                                block_id = stat.block_id
                                for k in _METRICS:
                                    buf_log[f"{module_name}/block{block_id:02d}/{k}"] = getattr(stat, k)

                        logger.info(
                            f"(step={train_steps:07d}) " \
                            f"Train Loss: {avg_train_loss:.4f} " \
                            f"Train Steps/Sec: {steps_per_sec:.2f} " \
                            f"Train Samples/Sec: {samples_per_sec:.1f}"
                        )

                        # Log training metrics to a separate table in wandb
                        wandb.log(
                            {
                                "train/loss": avg_train_loss,
                                "train/step": train_steps,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "train/samples_per_sec": samples_per_sec,
                                **{f"train/{k}": v for k, v in grad_log.items()},
                                **{f"activation/{k}": v for k, v in buf_log.items()},
                            }, step=train_steps)
                    
                    # Reset tracking variables for the next logging interval
                    running_loss, log_steps = 0.0, 0
                    start_time = time.time()
                
                # Save model checkpoint at regular intervals
                if rank == 0 and train_steps % args.save_every_steps == 0:
                    checkpoint = {
                        "model": get_state_dict_for_save(model),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "opt_steps": opt_steps,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Synchronize across processes
            dist.barrier()
            
        # End of epoch handling
        if rank == 0:
            if (epoch + 1) % args.save_every_epochs == 0:
                checkpoint = {
                    "model": get_state_dict_for_save(model),
                    "opt": optimizer.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                    "opt_steps": opt_steps,
                    "epoch": epoch
                }
                checkpoint_path = f"{checkpoint_dir}/epoch{epoch + 1:03d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()

        # Validation after each epoch
        model.eval()
        logs = evaluate_and_log(model, test_loader, device)
        val_loss = logs["val_loss"]
        
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
    
        if rank == 0:
            # Log validation metrics to a separate table in wandb
            wandb.log({
                "val/loss": logs["val_loss"],
                "val/acc@1": logs["val_acc@1"],
                "val/acc@5": logs["val_acc@5"],
                "epoch": epoch,
                "val/step": train_steps,  # Use the same step counter for alignment
            }, step=train_steps)
            
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} "
                f"Val Loss: {logs['val_loss']:.4f}, "
                f"Val Acc@1: {logs['val_acc@1']:.4f}, "
                f"Val Acc@5: {logs['val_acc@5']:.4f}"
            )
        # end looop

    logger.info("Done!")
    dist.barrier()
    if rank == 0:
        wandb.finish()
    # Clean up the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Parse arguments, including global batch size
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="ortho-classifier",)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--results_dir", type=str, default="results-classifier",
                        help="Directory to save results.")
    
    parser.add_argument("--dataset", choices=[
        "mnist", "cifar10", "cifar100", "imagenet1k", "imagenet1k-latents", "tiny-imagenet", "tiny-imagenet-randaug"], default="mnist") 
    
    # model parameters
    parser.add_argument("--model", choices=["vit", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], default="vit",
                        help="Select baseline ViT ('vit') or ResNet34.")
    parser.add_argument("--preset", choices=["S", "B", "L", "H"], default="B",
                        help="Preset for model size: S (small), M (medium), L (large).")
    parser.add_argument("--orthogonal_residual", action="store_true",
                    help="Enable orthogonal residual connections if specified.")
    parser.add_argument("--orthogonal_prob", type=float, default=None,
                        help="Rate of orthogonal residual connections.")
    parser.add_argument("--orthogonal_method", type=str, choices=["negative", "feature", "global"],
                        default="feature",
                        help="Method for orthogonal residual connections.")
    parser.add_argument("--orthogonal_pattern", type=str, default=None,
                        help="Pattern for orthogonal residual connections.")
    parser.add_argument("--orthogonal_eps", type=float, default=1e-6,
                        help="Epsilon for orthogonal residual connections.")
    parser.add_argument("--orthogonal_perturbation", type=float, default=None,
                        help="Perturbation for orthogonal residual connections.")
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--is_layernorm_classifier", action="store_true",
                        help="Use layernorm classifier instead of linear classifier.")
    
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epoch", type=int, default=5,
                        help="Number of warmup epochs.")
    parser.add_argument("--warmup_type", type=str, default="linear",
                        choices=["linear", "const"],
                        help="Warmup type for learning rate scheduler.")
    parser.add_argument("--warmup_const", type=float, default=0.1,
                        help="Constant warmup value.")
    parser.add_argument("--max_train_steps", type=int, default=600000)
    parser.add_argument("--global_batch_size", type=int, default=128,
                        help="Global batch size across all GPUs.")
    parser.add_argument("--grad_accumulate_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", type=str, default="none",
                        choices=["none", "torch", "unsloth"],
                        help="Enable gradient checkpointing for memory efficiency.")
    
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile for model compilation.")
    parser.add_argument("--torch_compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Torch compile mode.")
    parser.add_argument("--torch_compile_backend", type=str, default="inductor",
                        choices=["inductor", "aot_eager", "aot_ts"],
                        help="Torch compile backend.")
    
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable automatic mixed precision (AMP) for training.")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Data type for AMP.")

    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam",)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for SGD optimizer.")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Gradient clipping value.")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay for optimizer.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum learning rate value to reach at the end of scheduling.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cos",
                        choices=["cos", "lin", "const", "step"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--lr_scheduler_decay_epoch", type=str, default=None,
                        help="Learning rate scheduler step type.")
    parser.add_argument("--lr_scheduler_decay_ratio", type=float, default=0.1,
                        help="Learning rate scheduler ratio.")

    parser.add_argument("--log_activations", action="store_true",
                        help="Enable logging of activation statistics.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Interval for logging training progress.")
    parser.add_argument("--save_every_steps", type=int, default=5000,
                        help="Interval for saving checkpoints.")
    parser.add_argument("--save_every_epochs", type=int, default=10,
                        help="Interval for saving checkpoints.")
    
    # Mixup and Label Smoothing
    parser.add_argument("--use_mixup", action="store_true",
                        help="Enable mixup augmentation.")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="Alpha for mixup augmentation.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor.")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                        help="Alpha for cutmix augmentation.")
    parser.add_argument("--cutmix_switch_prob", type=float, default=0.5,
                        help="Probability of switching between mixup and cutmix.")
    parser.add_argument("--mixup_prob", type=float, default=0.5,
                        help="Probability of applying mixup augmentation.")
    parser.add_argument("--random_erase", type=float, default=0.0,
                        help="Probability of applying random erase augmentation.")
    parser.add_argument("--randaugment_N", type=int, default=3)
    parser.add_argument("--randaugment_M", type=int, default=7)
    
    
    parser.add_argument("--debug", action="store_true",
                    help="Print verbose debug logs.")

    
    args = parser.parse_args()
    
    if args.config_file:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: {key} not found in args.")
    main(args)
