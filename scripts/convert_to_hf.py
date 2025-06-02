import torch
from collections import OrderedDict
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
# Make sure these modules can be imported
# Adjust paths if they are in a different directory structure
from ortho_residual.vit.modeling_ortho_vit import OrthoViTForImageClassification, OrthoViTModel
from ortho_residual.vit.configuration_ortho_vit import OrthoViTConfig

def get_state_dict_from_checkpoint(ckpt_path):
    """Loads the state_dict from the '.pt' file."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        return checkpoint["model"]
    # Fallback if the 'model' field doesn't exist, assume checkpoint IS the state_dict
    # This might happen if the model was saved directly using torch.save(model.state_dict(), path)
    # However, user's provided save_code uses a dictionary with 'model' key.
    return checkpoint 

def convert_ortho_block_keys(original_block_sd, hf_block_idx, config):
    """
    Converts keys from a single OrthoBlock to a Hugging Face ViTLayer.
    original_block_sd: state_dict of one user's OrthoBlock
    hf_block_idx: index of the block in the HF model
    config: OrthoViTConfig
    """
    new_sd = OrderedDict()
    hf_prefix = f"vit.encoder.layer.{hf_block_idx}."

    # LayerNorm before attention
    # Original OrthoBlock.norm1 has elementwise_affine=False
    # HF ViTLayer.layernorm_before uses config.elementwise_affine_ln
    if config.elementwise_affine_ln:
        new_sd[hf_prefix + "layernorm_before.weight"] = original_block_sd["norm1.weight"]
        # new_sd[hf_prefix + "layernorm_before.bias"] = original_block_sd["norm1.bias"] # Bias does not exist if affine is False
    # If not affine, these keys won't be in original_block_sd and shouldn't be mapped. HF model will use its non-affine LN.


    # Attention block: original uses timm.Attention with combined QKV
    # HF uses separate Q, K, V linear layers.
    original_qkv_weight = original_block_sd["attn.qkv.weight"]
    original_qkv_bias = original_block_sd.get("attn.qkv.bias", None) # qkv_bias can be False

    # Assuming dim is the first dimension for weights (out_features, in_features)
    # and Q, K, V weights are concatenated along the output dimension (dim=0)
    # Each part is config.hidden_size
    hidden_size = config.hidden_size
    
    # Split QKV weights
    q_weight, k_weight, v_weight = torch.split(original_qkv_weight, hidden_size, dim=0)
    new_sd[hf_prefix + "attention.attention.query.weight"] = q_weight
    new_sd[hf_prefix + "attention.attention.key.weight"] = k_weight
    new_sd[hf_prefix + "attention.attention.value.weight"] = v_weight

    if original_qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.split(original_qkv_bias, hidden_size, dim=0)
        new_sd[hf_prefix + "attention.attention.query.bias"] = q_bias
        new_sd[hf_prefix + "attention.attention.key.bias"] = k_bias
        new_sd[hf_prefix + "attention.attention.value.bias"] = v_bias
    
    # Attention projection
    new_sd[hf_prefix + "attention.output.dense.weight"] = original_block_sd["attn.proj.weight"]
    if "attn.proj.bias" in original_block_sd: # proj_bias can be False
         new_sd[hf_prefix + "attention.output.dense.bias"] = original_block_sd["attn.proj.bias"]

    # LayerNorm after attention
    # Original OrthoBlock.norm2 has elementwise_affine=False
    if config.elementwise_affine_ln:
        new_sd[hf_prefix + "layernorm_after.weight"] = original_block_sd["norm2.weight"]
        # new_sd[hf_prefix + "layernorm_after.bias"] = original_block_sd["norm2.bias"]

    # MLP block
    # HF: intermediate.dense (fc1), output.dense (fc2)
    new_sd[hf_prefix + "intermediate.dense.weight"] = original_block_sd["mlp.fc1.weight"]
    new_sd[hf_prefix + "intermediate.dense.bias"] = original_block_sd["mlp.fc1.bias"]
    new_sd[hf_prefix + "output.dense.weight"] = original_block_sd["mlp.fc2.weight"]
    new_sd[hf_prefix + "output.dense.bias"] = original_block_sd["mlp.fc2.bias"]
    
    # residual_eps is a non-persistent buffer, not in state_dict

    return new_sd

def convert_original_classifier_to_hf(original_vit_checkpoint_path, config_params, hf_save_path):
    """
    Converts an original ViT checkpoint to the Hugging Face OrthoViTForImageClassification format.
    """
    print(f"Loading original checkpoint from: {original_vit_checkpoint_path}")
    original_sd = get_state_dict_from_checkpoint(original_vit_checkpoint_path)
    
    # Create HF config
    # These should match the parameters of the trained model
    hf_config = OrthoViTConfig(**config_params)

    # Instantiate HF model
    hf_model = OrthoViTForImageClassification(hf_config)
    hf_model.eval()
    
    new_hf_sd = OrderedDict()
    
    # 1. CLS token
    if "cls_token" in original_sd: # Original model might not have cls_token if class_token=False
        new_hf_sd["vit.embeddings.cls_token"] = original_sd["cls_token"]
    
    # 2. Position Embeddings
    # Original: (1, num_patches, dim)
    # HF: (1, num_patches + 1, dim) where +1 is for CLS token
    # The original pos_embed is for patches only. HF prepends a pos_embed for CLS.
    # We need to construct HF's position_embeddings.
    # Typically, the CLS token position embedding is learnable and different.
    # If original_sd["pos_embed"] is (1, num_patches, dim),
    # HF model's `vit.embeddings.position_embeddings` will be initialized.
    # We can copy the patch position embeddings part.
    if "pos_embed" in original_sd:
        num_patches_original = original_sd["pos_embed"].shape[1]
        # Initialize hf_pos_embed with the model's init, then overwrite patch part
        hf_pos_embed_init = hf_model.state_dict()["vit.embeddings.position_embeddings"]
        if hf_pos_embed_init.shape[1] == num_patches_original +1: # Expected
             hf_pos_embed_init[:, 1:, :] = original_sd["pos_embed"]
             new_hf_sd["vit.embeddings.position_embeddings"] = hf_pos_embed_init
        else: # shape mismatch, fallback to just copying, may require model adjustment or specific init for CLS pos embed
            print(f"Warning: Positional embedding shape mismatch. Original: {original_sd['pos_embed'].shape}, HF expected for patches: {hf_pos_embed_init[:, 1:, :].shape}")
            new_hf_sd["vit.embeddings.position_embeddings"] = original_sd["pos_embed"] # This might be incorrect for HF structure


    # 3. Patch Embeddings (Conv2d projection)
    new_hf_sd["vit.embeddings.patch_embeddings.projection.weight"] = original_sd["patchify.weight"]
    new_hf_sd["vit.embeddings.patch_embeddings.projection.bias"] = original_sd["patchify.bias"]
    
    # 4. Transformer Blocks
    num_layers = hf_config.num_hidden_layers
    for i in range(num_layers):
        original_block_prefix = f"blocks.{i}."
        # Extract state_dict for this specific block from the original flat state_dict
        current_block_sd = OrderedDict()
        for k, v in original_sd.items():
            if k.startswith(original_block_prefix):
                current_block_sd[k[len(original_block_prefix):]] = v
        
        converted_block_sd = convert_ortho_block_keys(current_block_sd, i, hf_config)
        new_hf_sd.update(converted_block_sd)
        
    # 5. Final LayerNorm (before classifier head)
    # User's original Classifier: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
    # HF OrthoViTModel has self.layernorm before outputting sequence_output.
    # This self.layernorm in OrthoViTModel should correspond to user's classifier.0 (LayerNorm)
    # if hf_config.elementwise_affine_ln is True for this final layernorm (it is from OrthoViTConfig default)
    # The user's `Classifier` class's classifier[0] (LayerNorm) is nn.LayerNorm(dim), which has elementwise_affine=True by default.
    # The `OrthoViTConfig` has `elementwise_affine_ln` which applies to encoder layers and final layernorm.
    # We assume here that `hf_config.elementwise_affine_ln` for `vit.layernorm` matches `classifier.0` implicitly.
    # If `OrthoViTModel.layernorm` is configured with `elementwise_affine=True` via `hf_config.elementwise_affine_ln`.
    if "classifier.0.weight" in original_sd: # This is the LayerNorm from original classifier
        new_hf_sd["vit.layernorm.weight"] = original_sd["classifier.0.weight"]
        new_hf_sd["vit.layernorm.bias"] = original_sd["classifier.0.bias"]
    
    # 6. Classifier head (Linear layer)
    # User's original Classifier: classifier.1 (Linear)
    # HF OrthoViTForImageClassification: self.classifier (Linear)
    if "classifier.1.weight" in original_sd: # This is the Linear layer from original classifier
        new_hf_sd["classifier.weight"] = original_sd["classifier.1.weight"]
        new_hf_sd["classifier.bias"] = original_sd["classifier.1.bias"]

    # Load the new state dict into the HF model
    missing_keys, unexpected_keys = hf_model.load_state_dict(new_hf_sd, strict=False)
    print(f"HF Model state_dict loaded.")
    if missing_keys:
        print(f"Missing keys in HF model: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in HF model (not loaded): {unexpected_keys}")

    # Save the HF model
    print(f"Saving Hugging Face model to: {hf_save_path}")
    OrthoViTConfig.register_for_auto_class()
    OrthoViTModel.register_for_auto_class("AutoModel")
    OrthoViTForImageClassification.register_for_auto_class("AutoModelForImageClassification")

    hf_model.save_pretrained(hf_save_path)
    hf_model.push_to_hub(f"BootsofLagrangian/{config_params['residual_connection']}-vit-b-imagenet1k-hf", private=True)

    print("Conversion complete.")


if __name__ == "__main__":
    # --- Example Usage ---
    import argparse, json
    parser = argparse.ArgumentParser(description="Convert original ViT checkpoint to Hugging Face OrthoViTForImageClassification format.")
    parser.add_argument("original_checkpoint_path", type=str, help="Path to the original trained .pt file.")
    parser.add_argument("hf_model_save_path", type=str, default=".", help="Directory to save the Hugging Face model.")
    parser.add_argument("--config_args", type=json.loads, default=None, help="Configuration parameters for the ViT model.")
    args = parser.parse_args()
    config_args = { # ViT-B configuration example
        # These parameters should match the original model's training configuration
        "image_size": 224,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_fast",
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-6,
        "num_labels": 1000, # e.g., for ImageNet-1k
        "qkv_bias": True,
        "encoder_stride": 16, # This is typically the stride of the patch embeddings
        "residual_connection"   : "orthogonal", # or "linear"
        "orthogonal_method": "channel",    # or "global"
        "residual_connection_dim": -1, # Default for ViT, means last dimension
        "residual_eps": 1e-6,
        "residual_perturbation": None, # or the float value used
        "elementwise_affine_ln": False, # Important: Set this based on OrthoBlock's norm1/norm2
    }
    if args.config_args is None:
        args.config_args = config_args
    else:
        config_args.update(args.config_args) # Merge with defaults if provided
    args.config_args = config_args
    print(f"Using configuration: {args.config_args}")

    # Path to your original trained .pt file
    original_checkpoint_path = args.original_checkpoint_path # MODIFY THIS

    # Directory where the Hugging Face model will be saved
    hf_model_save_path = args.hf_model_save_path # MODIFY THIS

    # Configuration parameters for the ViT model that was trained.
    # These should match your `base.py` Classifier and `OrthoBlock` init args
    # and the args used in `train_classifier.py` for the specific checkpoint.
    
    # Example for a ViT-S like model (refer to PRESET_VIT in your base.py and recipes)
    # This is a placeholder, you MUST fill this with the correct config for your checkpoint.
    config_args = args.config_args
    # Call the conversion function
    convert_original_classifier_to_hf(original_checkpoint_path, config_args, hf_model_save_path)
    print(f"Converted model saved to {hf_model_save_path}")

    test_model = AutoModelForImageClassification.from_pretrained("BootsofLagrangian/ortho-vit-b-imagenet1k-hf", trust_remote_code=True)
    print("Test model loaded successfully.")
    print("Test Model:")
    print(test_model)
    # print(f"Test model config: {test_model.config}")
    # test_model.push_to_hub("BootsofLagrangian/ortho-vit-b-imagenet1k-hf", private=True)
