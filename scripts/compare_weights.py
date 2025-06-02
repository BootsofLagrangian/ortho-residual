# compare_weights.py
import torch
from collections import OrderedDict
from transformers import AutoModelForImageClassification
import argparse

try:
    from ortho_residual.vit.modeling_ortho_vit import OrthoViTForImageClassification, OrthoViTModel
    from ortho_residual.vit.configuration_ortho_vit import OrthoViTConfig
except ModuleNotFoundError:
    print("Make sure modeling_ortho_vit.py and configuration_ortho_vit.py are in your PYTHONPATH or current directory.")
    raise

def get_state_dict_from_checkpoint(ckpt_path):
    """Loads the state_dict from the '.pt' file."""
    # weights_only=False is important if the checkpoint contains more than just weights,
    # like optimizer states, but for model state_dict, it might not strictly be necessary
    # if the 'model' field only contains tensors. However, to be safe with pickled data:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False) # Removed weights_only for safety with older torch versions or complex pickles
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint

def convert_ortho_block_keys_for_comparison(original_block_sd, hf_block_idx, config_dict):
    """
    Converts keys from a single original OrthoBlock to match Hugging Face ViTLayer keys.
    This is a simplified version for comparison, assuming config_dict has necessary fields.
    """
    new_sd = OrderedDict()
    hf_prefix = f"vit.encoder.layer.{hf_block_idx}."
    hidden_size = config_dict.get("hidden_size")
    elementwise_affine_ln = config_dict.get("elementwise_affine_ln", False) # Default from your config

    # LayerNorm before attention
    if elementwise_affine_ln and "norm1.weight" in original_block_sd:
        new_sd[hf_prefix + "layernorm_before.weight"] = original_block_sd["norm1.weight"]
        if "norm1.bias" in original_block_sd: # Should not exist if affine is False
             new_sd[hf_prefix + "layernorm_before.bias"] = original_block_sd["norm1.bias"]

    # Attention block
    if "attn.qkv.weight" in original_block_sd:
        original_qkv_weight = original_block_sd["attn.qkv.weight"]
        q_weight, k_weight, v_weight = torch.split(original_qkv_weight, hidden_size, dim=0)
        new_sd[hf_prefix + "attention.attention.query.weight"] = q_weight
        new_sd[hf_prefix + "attention.attention.key.weight"] = k_weight
        new_sd[hf_prefix + "attention.attention.value.weight"] = v_weight

    if original_block_sd.get("attn.qkv.bias") is not None:
        original_qkv_bias = original_block_sd["attn.qkv.bias"]
        q_bias, k_bias, v_bias = torch.split(original_qkv_bias, hidden_size, dim=0)
        new_sd[hf_prefix + "attention.attention.query.bias"] = q_bias
        new_sd[hf_prefix + "attention.attention.key.bias"] = k_bias
        new_sd[hf_prefix + "attention.attention.value.bias"] = v_bias

    if "attn.proj.weight" in original_block_sd:
        new_sd[hf_prefix + "attention.output.dense.weight"] = original_block_sd["attn.proj.weight"]
    if original_block_sd.get("attn.proj.bias") is not None:
        new_sd[hf_prefix + "attention.output.dense.bias"] = original_block_sd["attn.proj.bias"]

    # LayerNorm after attention
    if elementwise_affine_ln and "norm2.weight" in original_block_sd:
        new_sd[hf_prefix + "layernorm_after.weight"] = original_block_sd["norm2.weight"]
        if "norm2.bias" in original_block_sd: # Should not exist if affine is False
            new_sd[hf_prefix + "layernorm_after.bias"] = original_block_sd["norm2.bias"]

    # MLP block
    if "mlp.fc1.weight" in original_block_sd:
        new_sd[hf_prefix + "intermediate.dense.weight"] = original_block_sd["mlp.fc1.weight"]
    if "mlp.fc1.bias" in original_block_sd:
        new_sd[hf_prefix + "intermediate.dense.bias"] = original_block_sd["mlp.fc1.bias"]
    if "mlp.fc2.weight" in original_block_sd:
        new_sd[hf_prefix + "output.dense.weight"] = original_block_sd["mlp.fc2.weight"]
    if "mlp.fc2.bias" in original_block_sd:
        new_sd[hf_prefix + "output.dense.bias"] = original_block_sd["mlp.fc2.bias"]

    return new_sd

def remap_original_state_dict_to_hf(original_sd, hf_config_dict):
    """
    Remaps the entire original state dictionary to Hugging Face naming conventions.
    hf_config_dict is a dictionary representation of OrthoViTConfig attributes.
    """
    remapped_sd = OrderedDict()
    num_hidden_layers = hf_config_dict.get("num_hidden_layers")

    # 1. CLS token
    if "cls_token" in original_sd:
        remapped_sd["vit.embeddings.cls_token"] = original_sd["cls_token"]

    # 2. Position Embeddings - This is tricky because HF includes CLS token position.
    # The conversion script handles this by initializing from HF and copying patch part.
    # For comparison, we'll just map the patch part if it exists.
    # A more robust comparison might need to load an HF model just to get its pos embed shape.
    if "pos_embed" in original_sd:
        # This key is directly mapped in convert_original_classifier_to_hf by modifying
        # the initialized HF model's position_embeddings. For direct comparison, we note
        # that hf_model.vit.embeddings.position_embeddings[:, 1:, :] should match original_sd["pos_embed"]
        # We'll map it directly for now, and the comparison logic will handle the slicing if needed.
        remapped_sd["vit.embeddings.position_embeddings_patches_only"] = original_sd["pos_embed"]


    # 3. Patch Embeddings
    if "patchify.weight" in original_sd:
      remapped_sd["vit.embeddings.patch_embeddings.projection.weight"] = original_sd["patchify.weight"]
    if "patchify.bias" in original_sd:
      remapped_sd["vit.embeddings.patch_embeddings.projection.bias"] = original_sd["patchify.bias"]

    # 4. Transformer Blocks
    for i in range(num_hidden_layers):
        original_block_prefix = f"blocks.{i}."
        current_block_sd = OrderedDict()
        has_block_keys = False
        for k, v in original_sd.items():
            if k.startswith(original_block_prefix):
                current_block_sd[k[len(original_block_prefix):]] = v
                has_block_keys = True
        if has_block_keys:
            converted_block_sd = convert_ortho_block_keys_for_comparison(current_block_sd, i, hf_config_dict)
            remapped_sd.update(converted_block_sd)

    # 5. Final LayerNorm
    # Assuming OrthoViTModel.layernorm is affine=True and corresponds to original classifier.0
    if "classifier.0.weight" in original_sd:
        remapped_sd["vit.layernorm.weight"] = original_sd["classifier.0.weight"]
    if "classifier.0.bias" in original_sd:
        remapped_sd["vit.layernorm.bias"] = original_sd["classifier.0.bias"]

    # 6. Classifier head
    if "classifier.1.weight" in original_sd:
        remapped_sd["classifier.weight"] = original_sd["classifier.1.weight"]
    if "classifier.1.bias" in original_sd:
        remapped_sd["classifier.bias"] = original_sd["classifier.1.bias"]
        
    return remapped_sd

def compare_model_weights(args):
    # Load local original checkpoint
    print(f"Loading original local checkpoint from: {args.original_ckpt_path}")
    original_sd = get_state_dict_from_checkpoint(args.original_ckpt_path)
    if not original_sd:
        print("Failed to load original state dict.")
        return

    # Load HF model (this also fetches its config)
    print(f"Loading Hugging Face model from: {args.hf_model_name}")
    hf_model = AutoModelForImageClassification.from_pretrained(
        args.hf_model_name,
        trust_remote_code=True
    )
    hf_sd = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict() # Get config as a dictionary
    print("Hugging Face model and config loaded.")

    # Remap original state_dict keys to match HF naming
    print("Remapping original state_dict keys...")
    remapped_original_sd = remap_original_state_dict_to_hf(original_sd, hf_config_dict)
    print("Remapping complete.")

    # Compare weights
    max_diff_overall = 0.0
    mismatched_keys = []
    not_found_in_original = []

    print("\n--- Weight Comparison ---")
    for hf_key, hf_tensor in hf_sd.items():
        if hf_key == "vit.embeddings.position_embeddings" and "vit.embeddings.position_embeddings_patches_only" in remapped_original_sd:
            # Special handling for position embeddings: compare only the patch part
            original_tensor_patches = remapped_original_sd["vit.embeddings.position_embeddings_patches_only"]
            hf_tensor_patches = hf_tensor[:, 1:, :] # Skip CLS token position
            hf_tensor = hf_tensor[:, 1:, :]

            if hf_tensor_patches.shape != original_tensor_patches.shape:
                print(f"Shape mismatch for {hf_key} (patches vs patches): HF is {hf_tensor_patches.shape}, Original remapped is {original_tensor_patches.shape}")
                mismatched_keys.append(f"{hf_key} (shape_mismatch_patches)")
                continue
            
            diff = torch.abs(hf_tensor_patches - original_tensor_patches).max()
            original_tensor_to_compare = original_tensor_patches
            tensor_description = f"{hf_key} (patch part)"

        elif hf_key in remapped_original_sd:
            original_tensor_to_compare = remapped_original_sd[hf_key]
            if hf_tensor.shape != original_tensor_to_compare.shape:
                print(f"Shape mismatch for {hf_key}: HF is {hf_tensor.shape}, Original remapped is {original_tensor_to_compare.shape}")
                mismatched_keys.append(f"{hf_key} (shape_mismatch)")
                continue
            
            diff = torch.abs(hf_tensor - original_tensor_to_compare).max()
            tensor_description = hf_key
        else:
            # Skip keys that are expected to be in HF model but not in original (e.g., CLS part of pos_embed if not specially handled)
            # or if a layer was non-affine in original and now affine in HF (though it shouldn't have weights then in remapped)
            if "num_batches_tracked" in hf_key: # Skip batchnorm tracking stats
                continue
            if hf_key == "vit.embeddings.position_embeddings" and "vit.embeddings.position_embeddings_patches_only" not in remapped_original_sd:
                 print(f"Key {hf_key} (CLS part or full) not found directly in remapped original for comparison, original has no pos_embed.")
                 not_found_in_original.append(hf_key)
                 continue

            print(f"Key {hf_key} not found in remapped original state_dict for comparison.")
            not_found_in_original.append(hf_key)
            continue
        try:
            is_close = torch.allclose(hf_tensor, original_tensor_to_compare, atol=args.atol)
            max_local_diff = diff.item()
            max_diff_overall = max(max_diff_overall, max_local_diff)
        except Exception as e:
            print(f"Error comparing HF {hf_tensor.shape} and original: {original_tensor_to_compare.shape}, key={hf_key}\nerror={e}")
            return 
        
        if not is_close:
            print(f"MISMATCH in {tensor_description}: Max Diff = {max_local_diff:.6e}")
            mismatched_keys.append(hf_key)
        elif args.verbose:
            print(f"Match in {tensor_description}: Max Diff = {max_local_diff:.6e}")

    print("\n--- Comparison Summary ---")
    if not mismatched_keys and not not_found_in_original:
        print(f"All comparable weights are close (atol={args.atol}). Max overall difference: {max_diff_overall:.6e}")
    else:
        if mismatched_keys:
            print(f"Found {len(mismatched_keys)} mismatched tensors (max_diff > atol or shape mismatch):")
            for key in mismatched_keys:
                print(f"  - {key}")
        if not_found_in_original:
            print(f"Found {len(not_found_in_original)} HF keys not directly comparable or missing in remapped original:")
            for key in not_found_in_original:
                print(f"  - {key}")
        print(f"Max overall difference observed in checked parameters: {max_diff_overall:.6e}")

    # Check for keys in remapped_original_sd that were not in hf_sd (should be rare if mapping is correct)
    unmatched_original_keys = [k for k in remapped_original_sd.keys() if k not in hf_sd and k != "vit.embeddings.position_embeddings_patches_only"]
    if unmatched_original_keys:
        print(f"\nWarning: These remapped original keys were not found in the HF state_dict:")
        for key in unmatched_original_keys:
            print(f"  - {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare weights of original and HF ViT models.")
    parser.add_argument(
        "original_ckpt_path",
        type=str,
        help="Path to the original .pt checkpoint file."
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="BootsofLagrangian/ortho-vit-b-imagenet1k-hf",
        help="Name of the Hugging Face model on the Hub."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5, # Default tolerance for torch.allclose
        help="Absolute tolerance for comparing floating point tensors."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diff for matching keys as well."
    )
    args = parser.parse_args()

    compare_model_weights(args)