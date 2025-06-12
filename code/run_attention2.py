import torch
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from model.PPM_attention2 import EnhancedPredictiveRepModel
from env.AtariEnv_random import AtariEnvManager

# --- Enhanced Configuration ---
# Environment parameters
NUM_GAMES_TO_SELECT_FROM = 3  # Number of unique games to potentially use
NUM_ENVS = 3  # Increased batch size for better training
IMG_H, IMG_W, IMG_C = 210, 160, 3  # Standard Atari dimensions

# Model parameters
# Action dimension will be fetched from the environment
LATENT_DIM = 256
BASE_CHANNELS = 64   # 基础通道数 - 特征提取的"宽度" ,控制每层可以学习多少种不同的特征
NUM_ATTENTION_HEADS = 8
ENCODER_LAYERS = [2, 3, 4, 3]  # Deeper encoder: [layer1, layer2, layer3, layer4]
DECODER_LAYERS = [2, 2, 2, 1]  # Decoder configuration
USE_SKIP_CONNECTIONS = True  #可以跳过某些层直接传递

# Enhanced Training parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5     #正则化参数，用来防止模型过拟合
NUM_TRAINING_STEPS = 200    #20000
SEQ_LEN = 1  # Sequence length (model expects B, L, C, H, W)
PRINT_INTERVAL = 100
SAVE_INTERVAL = 1000
MODEL_SAVE_DIR = "/Users/feisong/Desktop/self-experience/code/Output/checkpoint/enhanced_trained_models_attention"
LOG_DIR = "/Users/feisong/Desktop/self-experience/code/Output/log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Advanced training settings
GRADIENT_CLIP_NORM = 1.0   #梯度裁剪
WARMUP_STEPS = 1000  #1-1000 lr从0慢慢增加到target_lr
USE_MIXED_PRECISION = True  # 混合精度训练
ACCUMULATION_STEPS = 2  # 梯度累积  模拟更大的批次?


def preprocess_observations(obs_list, device, normalize=True):
    """
    Converts a list of observations (H, W, C) uint8 to a tensor (B, C, H, W) float.
    Args:
        obs_list: List of observations
        device: Target device
        normalize: Whether to normalize to [-1, 1] range for Tanh output
    """
    processed_obs = []
    for obs_hwc in obs_list:
        obs_chw = np.transpose(obs_hwc, (2, 0, 1))  # HWC to CHW
        processed_obs.append(obs_chw)

    # Stack and convert to tensor
    obs_tensor_bchw = torch.tensor(np.array(processed_obs), dtype=torch.float32, device=device)

    #归一化
    if normalize:
        # Normalize to [-1, 1] range to match Tanh output
        obs_tensor_bchw = (obs_tensor_bchw / 255.0) * 2.0 - 1.0
    else:
        # Normalize to [0, 1] range
        obs_tensor_bchw = obs_tensor_bchw / 255.0

    return obs_tensor_bchw


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Creates a learning rate scheduler with warmup and cosine annealing."""
    # 先升后降的学习率
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, save_dir, is_best=False):
    """Save model checkpoint with all training state."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'model_config': {
            'img_in_channels': IMG_C,
            'img_out_channels': IMG_C,
            'encoder_layers': ENCODER_LAYERS,
            'decoder_layers': DECODER_LAYERS,
            'target_img_h': IMG_H,
            'target_img_w': IMG_W,
            'action_dim': 18,  # Will be updated with actual value
            'latent_dim': LATENT_DIM,
            'base_channels': BASE_CHANNELS,
            'num_attention_heads': NUM_ATTENTION_HEADS,
            'use_skip_connections': USE_SKIP_CONNECTIONS
        }
    }

    filename = f'enhanced_ppm_attention_step_{step}.pth'
    if is_best:
        filename = 'enhanced_ppm_attention_best.pth'

    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):
    """Load model checkpoint and restore training state."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['step'], checkpoint['loss']


def main():
    print(f" Using device: {DEVICE}")
    print(f" Mixed precision: {USE_MIXED_PRECISION}")
    print(f" Model features: Multi-head attention ({NUM_ATTENTION_HEADS} heads), Skip connections, SE blocks")

    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)

    # --- Initialize Environment Manager ---
    print("\n Initializing Atari Environment Manager...")
    try:
        env_manager = AtariEnvManager(num_games=NUM_GAMES_TO_SELECT_FROM, num_envs=NUM_ENVS, render_mode=None)
    except RuntimeError as e:
        print(f" Error initializing AtariEnvManager: {e}")
        print(
            "Please ensure you have Atari ROMs installed (e.g., pip install ale-py && ale-import-roms <path_to_roms>)")
        return

    # Get action dimension from environment
    action_dim = env_manager.envs[0].action_space.n
    print(f" Action dimension from environment: {action_dim}")

    # --- Initialize Enhanced Model ---
    print("\n Initializing Enhanced PredictiveRepModel...")
    model = EnhancedPredictiveRepModel(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        target_img_h=IMG_H,
        target_img_w=IMG_W,
        action_dim=action_dim,
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        use_skip_connections=USE_SKIP_CONNECTIONS
    ).to(DEVICE)

    # Calculate and print model parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f" Total parameters: {total_params:,}")
    # print(f" Trainable parameters: {trainable_params:,}")

    # --- Initialize Optimizer and Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_TRAINING_STEPS
    )

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and DEVICE.type == 'cuda' else None

    # --- Training Loop ---
    print("\n Starting enhanced training...")

    best_loss = float('inf')
    total_loss = 0
    step_times = []

    # Training state
    start_step = 1

    # Optional: Load from checkpoint
    # checkpoint_path = "path/to/checkpoint.pth"
    # if os.path.exists(checkpoint_path):
    #     start_step, _ = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path)
    #     start_step += 1

    for step in range(start_step, NUM_TRAINING_STEPS + 1):
        step_start_time = time.time()

        # Accumulate gradients over multiple steps for larger effective batch size
        accumulated_loss = 0

        for acc_step in range(ACCUMULATION_STEPS):
            # 1. Sample actions for each environment
            actions_list = env_manager.sample_actions()

            # 2. Step environments
            experiences = env_manager.step(actions_list)

            # 3. Prepare batch for the model
            obs1_list = [exp[0] for exp in experiences]
            obs2_list = [exp[2] for exp in experiences]

            # Preprocess observations (normalize to [-1, 1] for Tanh output)
            obs1_batch_bchw = preprocess_observations(obs1_list, DEVICE, normalize=True)
            obs2_target_batch_bchw = preprocess_observations(obs2_list, DEVICE, normalize=True)

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

            # if USE_MIXED_PRECISION and scaler:
            #     with torch.cuda.amp.autocast():
            #         _, predicted_obs2_raw = model(obs1_input_blchw, actions_tensor_bl)

            #         # Compute loss
            #         predicted_obs2 = predicted_obs2_raw.squeeze(1)
            #         obs2_target = obs2_target_blchw.squeeze(1)

            #         # Resize if necessary
            #         if predicted_obs2.shape[2:] != obs2_target.shape[2:]:
            #             predicted_obs2 = torch.nn.functional.interpolate(
            #                 predicted_obs2, size=obs2_target.shape[2:],
            #                 mode='bilinear', align_corners=False
            #             )

            #         loss = torch.nn.functional.mse_loss(predicted_obs2, obs2_target)
            #         loss = loss / ACCUMULATION_STEPS  # Scale loss for accumulation

            #     # Backward pass with scaling
            #     scaler.scale(loss).backward()
            #     accumulated_loss += loss.item()

            # else:
            #     # Standard precision training
            #     _, predicted_obs2_raw = model(obs1_input_blchw, actions_tensor_bl)

            #     predicted_obs2 = predicted_obs2_raw.squeeze(1)
            #     obs2_target = obs2_target_blchw.squeeze(1)

            #     if predicted_obs2.shape[2:] != obs2_target.shape[2:]:
            #         predicted_obs2 = torch.nn.functional.interpolate(
            #             predicted_obs2, size=obs2_target.shape[2:],
            #             mode='bilinear', align_corners=False
            #         )

            #     loss = torch.nn.functional.mse_loss(predicted_obs2, obs2_target)
            #     loss = loss / ACCUMULATION_STEPS

            #     loss.backward()
            #     accumulated_loss += loss.item()

        # # 5. Optimizer step after accumulation
        # if USE_MIXED_PRECISION and scaler:
        #     # Gradient clipping
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        #     optimizer.step()

        # optimizer.zero_grad()
        # scheduler.step()

        # Record metrics
        total_loss += accumulated_loss
        step_time = time.time() - step_start_time
        step_times.append(step_time)

        # Logging
        if step % PRINT_INTERVAL == 0:
            avg_loss = total_loss / PRINT_INTERVAL
            avg_step_time = np.mean(step_times[-PRINT_INTERVAL:])
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Step: {step:5d}/{NUM_TRAINING_STEPS} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {avg_step_time:.3f}s/step")

            # TensorBoard logging
            writer.add_scalar('Loss/Train', avg_loss, step)
            writer.add_scalar('Learning_Rate', current_lr, step)
            writer.add_scalar('Time/Step', avg_step_time, step)

            # Log to files
            with open(os.path.join(LOG_DIR, 'training.log'), 'a') as f:
                f.write(f"Step: {step}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}, Time: {avg_step_time:.3f}s\n")

            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, scheduler, scaler, step, avg_loss, MODEL_SAVE_DIR, is_best=True)
                print(f"🎉 New best loss: {best_loss:.6f}")

            total_loss = 0

        # Save checkpoint
        if step % SAVE_INTERVAL == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, scaler, step, accumulated_loss,
                                              MODEL_SAVE_DIR)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    # --- Final Save ---
    final_checkpoint_path = save_checkpoint(model, optimizer, scheduler, scaler, NUM_TRAINING_STEPS, accumulated_loss,
                                            MODEL_SAVE_DIR)
    print(f"✅ Training completed! Final checkpoint: {final_checkpoint_path}")

    # --- Cleanup ---
    writer.close()
    env_manager.close()

    print("\n=== Enhanced Training Summary ===")
    print(f" Total training steps: {NUM_TRAINING_STEPS:,}")
    print(f" Best loss achieved: {best_loss:.6f}")
    print(f" Average step time: {np.mean(step_times):.3f}s")
    # print(f" Total parameters: {total_params:,}")
    print(f" Architecture: Multi-head attention, Skip connections, SE blocks")
    print(" Enhanced model training completed successfully!")


if __name__ == '__main__':
    main()
