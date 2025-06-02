import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ortho_residual
from ortho_residual import OrthoLlamaConfig, OrthoLlamaForCausalLM
import copy # Import copy module
from torch.optim import AdamW  # Import AdamW optimizer
smollm_135_config_base = OrthoLlamaConfig(
    bos_token_id=0,
    eos_token_id=0,
    hidden_act="silu",
    hidden_size=576,
    initializer_range=0.041666666666666664,
    intermediate_size=1536,
    max_position_embeddings=2048,
    num_attention_heads=9,
    num_hidden_layers=30,
    num_key_value_heads=3,
    pad_token_id=None,
    rms_norm_eps=1.0e-05,
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=49152,
    # residual_connection specific parameters will be set later
    torch_dtype=torch.bfloat16,
    # residual connection specific parameters
    orthogonal_method="channel",
    residual_dim=-1,  # Default to last dimension
    residual_perturbation=None,
    residual_eps=1e-6,
)

smollm_360_config_base = OrthoLlamaConfig(
    bos_token_id=0,
    eos_token_id=0,
    hidden_act="silu",
    hidden_size=960,
    initializer_range=0.02,
    intermediate_size=2560,
    max_position_embeddings=2048,
    num_attention_heads=15,
    num_hidden_layers=32,
    num_key_value_heads=5,
    pad_token_id=None,
    rms_norm_eps=1.0e-05,
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=49152,
    # residual_connection specific parameters will be set later
    torch_dtype=torch.bfloat16,
    # residual connection specific parameters
    orthogonal_method="channel",
    residual_dim=-1,  # Default to last dimension
    residual_perturbation=None,
    residual_eps=1e-6,
)

smollm_1b_config_base = OrthoLlamaConfig(
    bos_token_id=0,
    eos_token_id=0,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=8192,
    max_position_embeddings=2048,
    num_attention_heads=32,
    num_hidden_layers=24,
    num_key_value_heads=32,
    pad_token_id=None,
    rms_norm_eps=1.0e-05,
    rope_interleaved=True,  # Note: This is True for 1B model
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=49152,
    # residual_connection specific parameters will be set later
    torch_dtype=torch.bfloat16,
    # residual connection specific parameters
    orthogonal_method="channel",
    residual_dim=-1,  # Default to last dimension
    residual_perturbation=None,
    residual_eps=1e-6,

)

# Store base configurations in a dictionary
base_model_presets = {
    "smollm_135m": smollm_135_config_base,
    "smollm_360m": smollm_360_config_base,
    "smollm_1b": smollm_1b_config_base,
}

# Define the base directory where presets will be saved
save_base_directory = "./preset/"

# Ensure the base directory exists
os.makedirs(save_base_directory, exist_ok=True)

# Training parameters
num_train_steps = 4  # Number of backpropagation steps
batch_size = 2  # Small batch size for training
seq_length = 12  # Small sequence length for training
learning_rate = 1e-4  # Learning rate for AdamW optimizer

# Set device for training and inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Save Tokenizer ---
tokenizer_name = "HuggingFaceTB/SmolLM2-360M"
# tokenizer_save_path = os.path.join(save_base_directory, "cosmo2-tokenizer") # Specific path for tokenizer

print(f"\n--- Processing Tokenizer: {tokenizer_name} ---")
tokenizer = None # Initialize tokenizer variable
try:
    print(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Tokenizer loaded successfully.")

<<<<<<< HEAD
    # Set pad_token to eos_token if eos_token exists
    # if tokenizer.eos_token:
    #     # print(f"Setting pad_token to eos_token ('{tokenizer.eos_token}', id: {tokenizer.eos_token_id})")
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     # tokenizer.pad_token_id = tokenizer.eos_token_id # Explicitly set pad_token_id
    # else:
    #     print("Warning: eos_token not found in tokenizer. pad_token not set.")

=======
>>>>>>> ed2ffd4 (fix typo)
    # Ensure bos_token is set
    if not tokenizer.bos_token:
        print("Warning: bos_token not found or set in tokenizer.")
    else:
        print(f"bos_token is set to '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")

<<<<<<< HEAD
    # Ensure the specific tokenizer save directory exists
    # os.makedirs(tokenizer_save_path, exist_ok=True)

    # Save the modified tokenizer
    # print(f"Saving tokenizer to '{tokenizer_save_path}'...")
    # tokenizer.save_pretrained(tokenizer_save_path)
    # print(f"Tokenizer saved successfully to '{tokenizer_save_path}'.")
=======
>>>>>>> ed2ffd4 (fix typo)

    # Get tokenizer info to update configs
    # tokenizer_pad_token_id = tokenizer.pad_token_id
    tokenizer_bos_token_id = tokenizer.bos_token_id
    tokenizer_eos_token_id = tokenizer.eos_token_id
    tokenizer_vocab_size = len(tokenizer) # Use actual tokenizer vocab size

    print(f"Tokenizer Info: vocab_size={tokenizer_vocab_size}, bos_token_id={tokenizer_bos_token_id}, eos_token_id={tokenizer_eos_token_id}")

except Exception as e:
    print(f"Error processing tokenizer '{tokenizer_name}': {e}")
    exit() # Exit if tokenizer processing fails

# Define residual connection types and their corresponding postfixes/settings
residual_options = [
    ('orthogonal', 'ortho', {"residual_connection": "orthogonal", "orthogonal_method": "channel", "residual_eps": 1e-6, "residual_perturbation": None}),
    ('linear', 'base', {"residual_connection": "linear"}) # 'identity' might not need extra params, or use defaults
]

# Iterate through the presets, residual types, save config, load model, etc.
for preset_name_base, config_base in base_model_presets.items():
    for res_type, postfix, res_params in residual_options:

        # Create a deep copy of the base config to modify
        config = copy.deepcopy(config_base)
        config.residual_connection = res_type  # Set residual connection type

        # Update config with tokenizer info
        # config.pad_token_id = tokenizer_pad_token_id # DO NOT set pad_token_id same as eos_token_id or bos_token_id
        config.bos_token_id = tokenizer_bos_token_id
        config.eos_token_id = tokenizer_eos_token_id
        config.vocab_size = tokenizer_vocab_size

        # Define the final preset name and save path including the postfix
        current_preset_name = f"{preset_name_base}-{postfix}"
        save_path = os.path.join(save_base_directory, current_preset_name)

        print(f"\n--- Processing preset: {current_preset_name} (residual: {res_type}) ---")

        # --- 1. Save Config ---
        print(f"Saving config for '{current_preset_name}' to '{save_path}'...")
        # Ensure the specific model directory exists
        os.makedirs(save_path, exist_ok=True)
        config.save_pretrained(save_path)
        print(f"Config for '{current_preset_name}' saved successfully.")

        # --- 2. Load Model from Config ---
        print(f"Loading model '{current_preset_name}' from config...")
        try:
            # Load model with random weights based on the config
            model = AutoModelForCausalLM.from_config(config)
            model.to(device)
            
            # Create AdamW optimizer
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {num_params:,}")
            
            # --- 3. Training with Backpropagation ---
            print(f"Performing {num_train_steps} backpropagation steps for '{current_preset_name}'...")
            
            # Set model to training mode for backpropagation
            model.train()
            
            for step in range(num_train_steps):
                # Create dummy input data
                dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
                dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
                dummy_labels = dummy_input_ids.clone()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                    labels=dummy_labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                print(f"Step {step+1}/{num_train_steps}, Training loss: {loss.item():.4f}")
            
            # --- 4. Set back to evaluation mode and perform inference ---
            model.eval()
            
            print(f"Performing inference after training for '{current_preset_name}'...")
            with torch.no_grad(): # No need to track gradients for inference
                dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
                dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
                dummy_labels = dummy_input_ids.clone()
                
                outputs = model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                    labels=dummy_labels
                )
                
                eval_loss = outputs.loss
                print(f"Inference loss after training: {eval_loss.item():.4f}")

            del model, outputs, loss  # Clean up model and outputs to free memory
            gc.collect()  # Run garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory if available
            print(f"Model '{current_preset_name}' processed successfully.")
            # --- 5. Save the trained model ---
            print(f"Saving vanilla model '{current_preset_name}' to '{save_path}'...")
            model = AutoModelForCausalLM.from_config(config)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path) # Save tokenizer as well
            print(f"Model '{current_preset_name}' saved successfully.")

        except Exception as e:
            print(f"Error processing preset '{current_preset_name}': {e}")
            # Clean up even if error occurs
            if 'model' in locals() and model is not None:
                del model
            if 'optimizer' in locals() and 'optimizer' in vars():
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print(f"\nAll model presets and residual types processed. Configs and models saved under '{save_base_directory}'.")
