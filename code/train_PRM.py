# train_PRM.py
import torch.nn.functional as F
import argparse
import logging
import os
import json
import torch
from tqdm import tqdm
import datetime
import csv
import colorama
from colorama import Fore, Back, Style
import sys

# DDP Imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import from other files
from env.AtariDataLoder import create_atari_dataloaders
from model.PRM import PRM, create_optimized_optimizer, create_lr_scheduler

logger = logging.getLogger(__name__)

def print_loss_info(loss_dict, global_step, epoch, batch_idx, lr=None, loss_type="TRAIN"):
    """
    用彩色打印损失信息
    """
    color = Fore.RED if loss_type == "TRAIN" else Fore.BLUE
    
    print(f"\n{color}{'='*60}")
    print(f"{color}🔥 {loss_type} LOSS INFO - Step {global_step}")
    print(f"{color}{'='*60}")
    print(f"{Fore.YELLOW}Epoch: {epoch+1:3d} | Batch: {batch_idx+1:4d} | Step: {global_step:6d}")
    if lr is not None:
        print(f"{Fore.CYAN}Learning Rate: {lr:.2e}")
    
    print(f"{color}┌─ LOSSES ─────────────────────────────────────┐")
    print(f"{color}│ Total Loss:    {loss_dict['total_loss']:10.6f}              │")
    print(f"{color}│ Image Loss:    {loss_dict['loss_img']:10.6f}              │")
    print(f"{color}│ Done Loss:     {loss_dict['loss_done']:10.6f}              │")
    print(f"{color}│ Vector Loss:   {loss_dict['loss_vector']:10.6f}              │")
    if 'cosine_similarity' in loss_dict:
        print(f"{color}│ Cosine Sim:    {loss_dict['cosine_similarity']:10.6f}              │")
    if 'done_accuracy' in loss_dict:
        print(f"{color}│ Done Accuracy: {loss_dict['done_accuracy']:10.6f}              │")
    if 'psnr' in loss_dict:
        print(f"{color}│ PSNR:          {loss_dict['psnr']:10.2f} dB            │")
    print(f"{color}└──────────────────────────────────────────────┘")
    print(f"{color}{'='*60}{Style.RESET_ALL}")


def setup_ddp():
    """Initializes the DDP process group."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return rank, world_size


def cleanup_ddp():
    """Cleans up the DDP process group."""
    dist.destroy_process_group()


def add_all_training_args(parser: argparse.ArgumentParser):
    """添加所有训练相关的参数"""
    
    # === 数据路径和加载配置 ===
    data_group = parser.add_argument_group('Data Loading Configuration')
    data_group.add_argument('--data-dir', type=str, default='./Data/replay_data',
                           help='Directory containing PKL files')
    data_group.add_argument('--num-workers', type=int, default=4,
                           help='Number of data loading worker processes')
    
    # === 数据预处理配置 ===
    preprocess_group = parser.add_argument_group('Data Preprocessing Configuration')
    preprocess_group.add_argument('--img-height', type=int, default=210,
                                 help='Target image height')
    preprocess_group.add_argument('--img-width', type=int, default=160,
                                 help='Target image width')
    
    # === 序列和批次配置 ===
    sequence_group = parser.add_argument_group('Sequence and Batch Configuration')
    sequence_group.add_argument('--sequence-length', type=int, default=32,
                               help='Length of each training sequence (model will see sequence_length+1 frames)')
    sequence_group.add_argument('--batch-size', type=int, default=8,
                               help='Number of sequences per batch')
    sequence_group.add_argument('--overlap-steps', type=int, default=16,
                               help='Number of steps to move forward when creating overlapping sequences')
    
    # === 数据集划分配置 ===
    split_group = parser.add_argument_group('Dataset Split Configuration')
    split_group.add_argument('--train-split', type=float, default=0.8,
                            help='Proportion of episodes for training')
    split_group.add_argument('--val-split', type=float, default=0.1,
                            help='Proportion of episodes for validation')
    
    # === 训练配置 ===
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--epochs', type=int, default=50, 
                            help='Number of training epochs')
    train_group.add_argument('--output-dir', type=str, default='./Output', 
                            help='Base directory to save all training outputs')
    train_group.add_argument('--run-name', type=str, default='episode_100', 
                            help='A specific name for this training run')
    train_group.add_argument('--log-interval', type=int, default=50, 
                            help='How often (in batches) to log to CSV')

    # === PRM 模型配置 ===
    model_group = parser.add_argument_group('PRM Model Configuration')
    model_group.add_argument('--latent-dim', type=int, default=256, 
                            help='Dimension of the latent space')
    model_group.add_argument('--base-channels', type=int, default=64, 
                            help='Base number of channels for convolutional layers')
    model_group.add_argument('--transformer-layers', type=int, default=3, 
                            help='Number of layers in the Transformer predictor')
    model_group.add_argument('--num-attention-heads', type=int, default=8, 
                            help='Number of attention heads in the Transformer')
    model_group.add_argument('--loss-weights', type=float, nargs=3, default=[1.0, 1.0, 1.0], 
                            help='Weights for [image, done, vector] losses')
    model_group.add_argument('--no-skip-connections', action='store_true', 
                            help='Disable skip connections in the ImageDecoder')
    
    # === 优化器和调度器配置 ===
    optim_group = parser.add_argument_group('Optimizer and Scheduler Configuration')
    optim_group.add_argument('--lr', type=float, default=1e-4, 
                            help='Base learning rate')
    optim_group.add_argument('--weight-decay', type=float, default=1e-2, 
                            help='Weight decay for the AdamW optimizer')
    optim_group.add_argument('--warmup-steps', type=int, default=1000, 
                            help='Number of warmup steps for the learning rate scheduler')
    optim_group.add_argument('--grad-clip-norm', type=float, default=1.0, 
                            help='Maximum norm for gradient clipping')
    
    # === 调试和日志配置 ===
    debug_group = parser.add_argument_group('Debug and Logging Configuration')
    debug_group.add_argument('--log-level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging level')
    debug_group.add_argument('--max-episodes', type=int, default=None,
                            help='Maximum number of episodes to use (for debugging)')

# def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, csv_writer, csv_file, rank):
#     """
#     修改后的训练函数：添加每64个batch的评估和彩色日志
#     """
#     model.train()
#     progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False, disable=(rank != 0))

#     batch_count = 0
#     last_batch_data = None  # 保存最后一个batch用于评估
    
#     for batch_idx, batch in enumerate(progress_bar):
#         global_step = epoch * len(dataloader) + batch_idx
#         batch_count += 1
        
#         optimizer.zero_grad()
#         obs = batch['observations'].to(device, non_blocking=True)
#         actions = batch['actions'].to(device, non_blocking=True)
#         done = batch['done'].to(device, non_blocking=True)
        
#         B, T_plus_1, C, H, W = obs.shape
#         T = T_plus_1 - 1
        
#         input_images = obs[:, :-1, ...].reshape(B*T, C, H, W)
#         input_actions = actions[:, :-1].reshape(B*T)
#         target_images = obs[:, 1:, ...].reshape(B*T, C, H, W)
#         target_done = done[:, 1:].reshape(B*T)

#         _, predicted_features, predicted_images, predicted_done = model(input_images, input_actions)
        
#         with torch.no_grad():
#             target_features, _ = model.module.encoder(target_images)
            
#         loss, loss_dict = model.module.compute_loss(
#             predicted_images, target_images, predicted_done,
#             target_done, predicted_features, target_features
#         )
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
#         optimizer.step()
#         scheduler.step()

#         # 保存当前batch数据用于评估
#         last_batch_data = batch

#         if rank == 0:
#             current_lr = optimizer.param_groups[0]['lr']
            
#             # 更新进度条
#             progress_bar.set_postfix({
#                 'loss': f"{loss.item():.4f}",
#                 'img': f"{loss_dict['loss_img']:.4f}",
#                 'done': f"{loss_dict['loss_done']:.4f}",
#                 'vec': f"{loss_dict['loss_vector']:.4f}",
#                 'lr': f"{current_lr:.2e}"
#             })
            
#             # 🔥 每64个batch进行评估和打印
#             if batch_count % 64 == 0:
#                 # 训练损失信息（红色）
#                 print_loss_info(loss_dict, global_step + 1, epoch, batch_idx, current_lr, "TRAIN")
                
#                 # 在当前batch上进行评估（蓝色）
#                 eval_loss_dict = evaluate_on_current_batch(model, last_batch_data, device)
#                 print_loss_info(eval_loss_dict, global_step + 1, epoch, batch_idx, current_lr, "EVAL")
                
#                 # 写入CSV - 训练和评估损失都记录
#                 if csv_writer is not None and csv_file is not None:
#                     try:
#                         # 训练损失行
#                         train_log_data = [
#                             global_step + 1,
#                             "train",
#                             loss_dict['total_loss'],
#                             loss_dict['loss_img'],
#                             loss_dict['loss_done'],
#                             loss_dict['loss_vector'],
#                             loss_dict.get('cosine_similarity', 0.0),
#                             0.0,  # done_accuracy (训练时不计算)
#                             0.0,  # psnr (训练时不计算)
#                             current_lr
#                         ]
#                         csv_writer.writerow(train_log_data)
                        
#                         # 评估损失行
#                         eval_log_data = [
#                             global_step + 1,
#                             "eval",
#                             eval_loss_dict['total_loss'],
#                             eval_loss_dict['loss_img'],
#                             eval_loss_dict['loss_done'],
#                             eval_loss_dict['loss_vector'],
#                             eval_loss_dict.get('cosine_similarity', 0.0),
#                             eval_loss_dict.get('done_accuracy', 0.0),
#                             eval_loss_dict.get('psnr', 0.0),
#                             current_lr
#                         ]
#                         csv_writer.writerow(eval_log_data)
                        
#                         # 强制刷新文件
#                         csv_file.flush()
                        
#                     except Exception as e:
#                         print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")
            
#             # 常规日志记录（每个batch）
#             elif (batch_idx + 1) % args.log_interval == 0 and csv_writer is not None and csv_file is not None:
#                 try:
#                     regular_log_data = [
#                         global_step + 1,
#                         "train",
#                         loss_dict['total_loss'],
#                         loss_dict['loss_img'],
#                         loss_dict['loss_done'],
#                         loss_dict['loss_vector'],
#                         loss_dict.get('cosine_similarity', 0.0),
#                         0.0,  # done_accuracy
#                         0.0,  # psnr
#                         current_lr
#                     ]
#                     csv_writer.writerow(regular_log_data)
#                     csv_file.flush()
#                 except Exception as e:
#                     print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")


# 动态调整评估频率：根据总batch数自动调整
# 确保每个epoch都有记录：在最后一个batch强制记录
# 更智能的日志间隔：自适应调整到合理的值
def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, csv_writer, csv_file, rank):
    """
    修改后的训练函数：添加每64个batch的评估和彩色日志
    """
    model.train()
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False, disable=(rank != 0))

    batch_count = 0
    last_batch_data = None  # 保存最后一个batch用于评估
    
    #  修复：动态调整评估频率
    total_batches = len(dataloader)
    eval_interval = min(64, max(10, total_batches // 4))  # 评估间隔：最小10，最大64，或总batch数的1/4
    log_interval = min(args.log_interval, max(5, total_batches // 8))  # 日志间隔：调整为更合理的值
    
    if rank == 0:
        print(f"{Fore.CYAN}📊 Training with {total_batches} batches per epoch")
        print(f"{Fore.CYAN}📊 Evaluation every {eval_interval} batches")
        print(f"{Fore.CYAN}📊 Regular logging every {log_interval} batches{Style.RESET_ALL}")
    
    for batch_idx, batch in enumerate(progress_bar):
        global_step = epoch * len(dataloader) + batch_idx
        batch_count += 1
        
        optimizer.zero_grad()
        obs = batch['observations'].to(device, non_blocking=True)
        actions = batch['actions'].to(device, non_blocking=True)
        done = batch['done'].to(device, non_blocking=True)
        
        B, T_plus_1, C, H, W = obs.shape
        T = T_plus_1 - 1
        
        input_images = obs[:, :-1, ...].reshape(B*T, C, H, W)
        input_actions = actions[:, :-1].reshape(B*T)
        target_images = obs[:, 1:, ...].reshape(B*T, C, H, W)
        target_done = done[:, 1:].reshape(B*T)

        _, predicted_features, predicted_images, predicted_done = model(input_images, input_actions)
        
        with torch.no_grad():
            target_features, _ = model.module.encoder(target_images)
            
        loss, loss_dict = model.module.compute_loss(
            predicted_images, target_images, predicted_done,
            target_done, predicted_features, target_features
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        # 保存当前batch数据用于评估
        last_batch_data = batch

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'img': f"{loss_dict['loss_img']:.4f}",
                'done': f"{loss_dict['loss_done']:.4f}",
                'vec': f"{loss_dict['loss_vector']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # 🔥 修复：使用动态调整的评估频率
            if batch_count % eval_interval == 0:
                # 训练损失信息（红色）
                print_loss_info(loss_dict, global_step + 1, epoch, batch_idx, current_lr, "TRAIN")
                
                # 在当前batch上进行评估（蓝色）
                eval_loss_dict = evaluate_on_current_batch(model, last_batch_data, device)
                print_loss_info(eval_loss_dict, global_step + 1, epoch, batch_idx, current_lr, "EVAL")
                
                # 写入CSV - 训练和评估损失都记录
                if csv_writer is not None and csv_file is not None:
                    try:
                        # 训练损失行
                        train_log_data = [
                            global_step + 1,
                            "train",
                            loss_dict['total_loss'],
                            loss_dict['loss_img'],
                            loss_dict['loss_done'],
                            loss_dict['loss_vector'],
                            loss_dict.get('cosine_similarity', 0.0),
                            0.0,  # done_accuracy (训练时不计算)
                            0.0,  # psnr (训练时不计算)
                            current_lr
                        ]
                        csv_writer.writerow(train_log_data)
                        
                        # 评估损失行
                        eval_log_data = [
                            global_step + 1,
                            "eval",
                            eval_loss_dict['total_loss'],
                            eval_loss_dict['loss_img'],
                            eval_loss_dict['loss_done'],
                            eval_loss_dict['loss_vector'],
                            eval_loss_dict.get('cosine_similarity', 0.0),
                            eval_loss_dict.get('done_accuracy', 0.0),
                            eval_loss_dict.get('psnr', 0.0),
                            current_lr
                        ]
                        csv_writer.writerow(eval_log_data)
                        
                        # 强制刷新文件
                        csv_file.flush()
                        
                    except Exception as e:
                        print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")
            
            # 🔥 修复：使用动态调整的日志频率
            elif (batch_idx + 1) % log_interval == 0 and csv_writer is not None and csv_file is not None:
                try:
                    regular_log_data = [
                        global_step + 1,
                        "train",
                        loss_dict['total_loss'],
                        loss_dict['loss_img'],
                        loss_dict['loss_done'],
                        loss_dict['loss_vector'],
                        loss_dict.get('cosine_similarity', 0.0),
                        0.0,  # done_accuracy
                        0.0,  # psnr
                        current_lr
                    ]
                    csv_writer.writerow(regular_log_data)
                    csv_file.flush()
                except Exception as e:
                    print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")
            
            # 🔥 新增：确保每个epoch至少记录一次（最后一个batch）
            elif batch_idx == len(dataloader) - 1 and csv_writer is not None and csv_file is not None:
                try:
                    final_log_data = [
                        global_step + 1,
                        "train_final",
                        loss_dict['total_loss'],
                        loss_dict['loss_img'],
                        loss_dict['loss_done'],
                        loss_dict['loss_vector'],
                        loss_dict.get('cosine_similarity', 0.0),
                        0.0,  # done_accuracy
                        0.0,  # psnr
                        current_lr
                    ]
                    csv_writer.writerow(final_log_data)
                    csv_file.flush()
                    print(f"{Fore.GREEN}📝 Epoch {epoch+1} final batch logged{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}CSV写入错误: {e}{Style.RESET_ALL}")

def evaluate_on_current_batch(model, batch, device):
    """
    在当前batch上进行评估（不需要额外的验证数据）
    """
    model.eval()
    
    with torch.no_grad():
        obs = batch['observations'].to(device, non_blocking=True)
        actions = batch['actions'].to(device, non_blocking=True)
        done = batch['done'].to(device, non_blocking=True)
        
        B, T_plus_1, C, H, W = obs.shape
        T = T_plus_1 - 1
        
        input_images = obs[:, :-1, ...].reshape(B*T, C, H, W)
        input_actions = actions[:, :-1].reshape(B*T)
        target_images = obs[:, 1:, ...].reshape(B*T, C, H, W)
        target_done = done[:, 1:].reshape(B*T)

        _, predicted_features, predicted_images, predicted_done = model(input_images, input_actions)
        
        target_features, _ = model.module.encoder(target_images)
            
        loss, loss_dict = model.module.compute_loss(
            predicted_images, target_images, predicted_done,
            target_done, predicted_features, target_features
        )
        
        # 计算额外的评估指标
        # Done预测准确率
        done_pred_binary = (predicted_done > 0.5).float()
        done_accuracy = (done_pred_binary.view(-1) == target_done.float()).float().mean()
        
        # 图像重建PSNR
        mse_img = F.mse_loss(predicted_images, target_images, reduction='mean')
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_img + 1e-8))
        
        # 添加额外评估指标到loss_dict
        loss_dict.update({
            'done_accuracy': done_accuracy.item(),
            'psnr': psnr.item(),
            'mse_img': mse_img.item()
        })
    
    model.train()  # 切回训练模式
    return loss_dict

def validate_epoch(model, dataloader, device):
    """执行一个验证epoch"""
    model.eval()
    total_val_loss = 0
    all_loss_dicts = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in progress_bar:
            obs = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            done = batch['done'].to(device)

            B, T_plus_1, C, H, W = obs.shape
            T = T_plus_1 - 1

            input_images = obs[:, :-1, ...].reshape(B*T, C, H, W)
            input_actions = actions[:, :-1].reshape(B*T)
            target_images = obs[:, 1:, ...].reshape(B*T, C, H, W)
            target_done = done[:, 1:].reshape(B*T)

            _, predicted_features, predicted_images, predicted_done = model(input_images, input_actions)
            target_features, _ = model.encoder(target_images)
            
            _, loss_dict = model.compute_loss(
                predicted_images, target_images, predicted_done,
                target_done, predicted_features, target_features
            )
            
            # 计算额外的验证指标
            done_pred_binary = (predicted_done > 0.5).float()
            done_accuracy = (done_pred_binary.view(-1) == target_done.float()).float().mean()
            mse_img = torch.nn.functional.mse_loss(predicted_images, target_images)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_img + 1e-8))

            loss_dict['done_accuracy'] = done_accuracy.item()
            loss_dict['psnr'] = psnr.item()
            all_loss_dicts.append(loss_dict)
            total_val_loss += loss_dict['total_loss']
    
    # 计算所有验证批次的平均指标
    avg_metrics = {k: sum(d[k] for d in all_loss_dicts) / len(all_loss_dicts) for k in all_loss_dicts[0]}
    return total_val_loss / len(dataloader), avg_metrics

def main():
    """
    修改后的主函数
    """
    parser = argparse.ArgumentParser(description="Train a PRM with DDP and unified logging.")
    add_all_training_args(parser)
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    csv_writer, csv_file, run_path = None, None, None

    if rank == 0:
        run_path = os.path.join(args.output_dir, args.run_name)
        os.makedirs(run_path, exist_ok=True)
        
        log_file_path = os.path.join(run_path, 'training.log')
        logging.basicConfig(
            level=getattr(logging, args.log_level), 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
        )
        
        # 🔥 修复CSV文件创建 - 添加更多列
        csv_file_path = os.path.join(run_path, 'loss_log.csv')
        csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        
        # 包含训练和评估信息的CSV表头
        csv_writer.writerow([
            'global_step', 'mode', 'total_loss', 'loss_img', 'loss_done', 
            'loss_vector', 'cosine_similarity', 'done_accuracy', 'psnr', 'learning_rate'
        ])
        csv_file.flush()  # 立即写入表头

        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}🚀 Starting DDP Training with Evaluation")
        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.CYAN}Run name: {args.run_name}")
        print(f"{Fore.CYAN}Output dir: {run_path}")
        print(f"{Fore.CYAN}GPUs: {world_size}")
        print(f"{Fore.CYAN}Batch size per GPU: {args.batch_size}")
        print(f"{Fore.CYAN}Sequence length: {args.sequence_length}")
        print(f"{Fore.CYAN}Effective batch size: {args.batch_size * args.sequence_length * world_size}")
        print(f"{Fore.YELLOW}Evaluation: Every 64 batches on current batch")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        with open(os.path.join(run_path, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    # 使用原有的数据加载器（不修改）
    train_loader, val_loader = create_atari_dataloaders(args)
    
    if train_loader is None:
        logger.error("Failed to create train loader. Exiting.")
        cleanup_ddp()
        return
    
    # 为 DDP 创建分布式采样器
    train_dataset = train_loader.dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True
    )
    
    if rank == 0:
        if val_loader is not None:
            print(f"{Fore.GREEN}✅ Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}📝 Note: Using current batch evaluation instead of separate validation{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}✅ Created dataloaders - Train: {len(train_loader)} (no validation){Style.RESET_ALL}")

    # 模型、优化器、调度器（保持不变）
    model = PRM(
        use_skip_connections=(not args.no_skip_connections), 
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        transformer_layers=args.transformer_layers,
        num_attention_heads=args.num_attention_heads,
        loss_weights=args.loss_weights
    ).to(device)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = create_optimized_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
    max_steps = args.epochs * len(train_loader)
    scheduler = create_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=max_steps)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{Fore.MAGENTA}{'='*60}")
            print(f"{Fore.MAGENTA}🎯 EPOCH {epoch+1}/{args.epochs}")
            print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        
        # 🔥 使用修改后的训练函数
        train_epoch(model, train_loader, optimizer, scheduler, device, args, epoch, csv_writer, csv_file, rank)
        
        if rank == 0:
            # 由于我们每64个batch进行评估，这里只保存模型
            checkpoint_path = os.path.join(run_path, f'model_epoch_{epoch+1}.pth')
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"{Fore.GREEN}💾 Epoch {epoch+1} model saved!{Style.RESET_ALL}")

    if rank == 0:
        final_checkpoint_path = os.path.join(run_path, 'final_model.pth')
        torch.save(model.module.state_dict(), final_checkpoint_path)
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}🎉 TRAINING COMPLETED!")
        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}Final model saved: {final_checkpoint_path}")
        print(f"{Fore.GREEN}Loss log saved: {csv_file.name}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        
        if csv_file:
            csv_file.close()
    
    cleanup_ddp()


if __name__ == '__main__':
    main()
