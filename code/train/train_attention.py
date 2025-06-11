import torch
import torch.optim as optim
import numpy as np
import os

# Assuming PPM_attention.py and AtariEnv.py are in the same directory
# or accessible in the Python path.
from model.PPM_attention import PredictiveRepModel
from env.AtariEnv_random import AtariEnvManager

# --- Configuration ---
# Environment parameters
NUM_GAMES_TO_SELECT_FROM = 3  # Number of unique games to potentially use
NUM_ENVS = 128                 # Number of parallel environments (and batch size for model)
IMG_H, IMG_W, IMG_C = 210, 160, 3 # Standard Atari dimensions

# Model parameters
# Action dimension will be fetched from the environment
LATENT_DIM = 256
ENCODER_LAYERS = [2, 2]       # Example layer configuration for encoder
DECODER_LAYERS = [2, 2, 1]    # Example layer configuration for decoder

# Training parameters
LEARNING_RATE = 1e-4
NUM_TRAINING_STEPS = 20000
SEQ_LEN = 1                   # Sequence length (model expects B, L, C, H, W)
PRINT_INTERVAL = 100
SAVE_INTERVAL = 1000
MODEL_SAVE_DIR = "trained_models_attention"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_observations(obs_list, device):
    """
    Converts a list of observations (H, W, C) uint8 to a tensor (B, C, H, W) float.
    """
    processed_obs = []
    for obs_hwc in obs_list:
        obs_chw = np.transpose(obs_hwc, (2, 0, 1)) # HWC to CHW
        processed_obs.append(obs_chw)
    
    # Stack, normalize to [0, 1], convert to tensor
    obs_tensor_bchw = torch.tensor(np.array(processed_obs), dtype=torch.float32, device=device) #/ 255.0
    return obs_tensor_bchw

def main():
    print(f"Using device: {DEVICE}")

    # --- Initialize Environment Manager ---
    print("Initializing Atari Environment Manager...")
    try:
        env_manager = AtariEnvManager(num_games=NUM_GAMES_TO_SELECT_FROM, num_envs=NUM_ENVS, render_mode=None)
    except RuntimeError as e:
        print(f"Error initializing AtariEnvManager: {e}")
        print("Please ensure you have Atari ROMs installed (e.g., pip install ale-py && ale-import-roms <path_to_roms>)")
        return

    # Assuming all environments have the same action space
    action_dim = env_manager.envs[0].action_space.n
    print(f"Action dimension from environment: {action_dim}")

    # --- Initialize Model ---
    print("Initializing PredictiveRepModel...")
    model = PredictiveRepModel(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers_config=DECODER_LAYERS,
        target_img_h=IMG_H,
        target_img_w=IMG_W,
        action_dim=action_dim,
        latent_dim=LATENT_DIM
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Model checkpoints will be saved in: {MODEL_SAVE_DIR}")

    # --- Training Loop ---
    print("Starting training...")
    total_loss = 0
    
    for step in range(1, NUM_TRAINING_STEPS + 1):
        # 1. Sample actions for each environment
        actions_list = env_manager.sample_actions() # List of integer actions

        # 2. Step environments
        # experiences is a list of (obs1, action, obs2, reward, done) tuples
        experiences = env_manager.step(actions_list)

        # 3. Prepare batch for the model
        obs1_list = [exp[0] for exp in experiences] # List of (H,W,C) np.uint8 arrays
        obs2_list = [exp[2] for exp in experiences] # List of (H,W,C) np.uint8 arrays
        
        # Preprocess observations
        obs1_batch_bchw = preprocess_observations(obs1_list, DEVICE)
        obs2_target_batch_bchw = preprocess_observations(obs2_list, DEVICE)

        # Reshape for model input (B, L, C, H, W)
        obs1_input_blchw = obs1_batch_bchw.unsqueeze(1)
        obs2_target_blchw = obs2_target_batch_bchw.unsqueeze(1)
        
        # Actions to tensor (B, L)
        actions_tensor_bl = torch.tensor(actions_list, dtype=torch.long, device=DEVICE).unsqueeze(1)

        # 4. Train on batch
        loss = model.train_on_batch(obs1_input_blchw, actions_tensor_bl, obs2_target_blchw, optimizer)
        total_loss += loss

        if step % PRINT_INTERVAL == 0:
            avg_loss = total_loss / PRINT_INTERVAL
            log_message = f"Step: {step}/{NUM_TRAINING_STEPS}, Average Loss: {avg_loss:.6f}"
            print(log_message)
            
            # Check and delete existing files at first save
            if step == PRINT_INTERVAL:
                if os.path.exists('loss.csv'):
                    os.remove('loss.csv')
                if os.path.exists('training.log'):
                    os.remove('training.log')

            # Save to CSV file
            with open('loss.csv', 'a') as f:
                if step == PRINT_INTERVAL:  # Write header if first entry
                    f.write('step\taverage_loss\n')
                f.write(f'{step}\t{avg_loss}\n')
            
            # Save to log file
            with open('training.log', 'a') as f:
                f.write(log_message + '\n')
                
                total_loss = 0  # Reset sum for next interval

        if step % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'ppm_attention_step_{step}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            log_message = f"Saved model checkpoint to {checkpoint_path}"
            print(log_message)
            # Also log checkpoint saves
            with open('training.log', 'a') as f:
                f.write(log_message + '\n')

    # --- Cleanup ---
    print("Training finished.")
    env_manager.close()
    print("Environments closed.")

if __name__ == '__main__':
    main()