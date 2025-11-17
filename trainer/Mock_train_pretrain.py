"""
MiniMind 预训练脚本

该脚本用于训练 MiniMind 语言模型的基础预训练任务。
支持分布式训练、混合精度、梯度累积、检查点保存/恢复等功能。

主要功能：
1. 支持单卡/多卡分布式训练
2. 混合精度训练（bfloat16/float16）
3. 梯度累积和梯度裁剪
4. 自动检查点保存和恢复
5. Wandb/SwanLab 实验跟踪
6. MoE架构支持
"""

import os
import sys

# 设置包名，确保相对导入正常工作
__package__ = "trainer"

# 将项目根目录添加到Python路径，以便导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import time  # 时间计算
import warnings  # 警告过滤
import torch  # PyTorch核心库
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器（用于CPU模式）
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # DDP并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import PretrainDataset  # 预训练数据集
from trainer.trainer_utils import (
    get_lr,  # 学习率调度
    Logger,  # 日志工具
    is_main_process,  # 判断是否为主进程
    lm_checkpoint,  # 检查点管理
    init_distributed_mode,  # 初始化分布式训练
    setup_seed,  # 设置随机种子
    init_model,  # 初始化模型
    SkipBatchSampler  # 跳过批次的采样器（用于续训）
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个epoch的核心函数
    
    该函数执行一个完整的训练epoch，包括：
    1. 前向传播和损失计算
    2. 反向传播和梯度累积
    3. 梯度裁剪和参数更新
    4. 日志记录和检查点保存
    
    Args:
        epoch: 当前epoch编号（从0开始）
        loader: 数据加载器
        iters: 总迭代次数（用于学习率调度和日志）
        start_step: 起始步数（用于续训时跳过已训练的步数）
        wandb: Wandb/SwanLab对象，用于实验跟踪（可选）
    
    注意：
    - 使用全局变量 args, model, optimizer, scaler, autocast_ctx, lm_config
    - 支持梯度累积，每accumulation_steps步更新一次参数
    - 支持MoE架构的辅助损失（aux_loss）
    """
    # 定义损失函数：交叉熵损失，reduction='none'表示不自动求平均
    # 这样可以对每个token位置单独计算损失，然后通过loss_mask控制哪些位置参与训练
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # 记录epoch开始时间，用于计算训练速度和剩余时间
    start_time = time.time()
    
    # 遍历数据加载器，enumerate从start_step+1开始计数（用于续训）
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # X: 输入token序列，形状 (batch_size, seq_len)
        # Y: 目标token序列（通常是X右移1位），形状 (batch_size, seq_len)
        # loss_mask: 损失掩码，形状 (batch_size, seq_len)，1表示计算损失，0表示忽略
        
        # 将数据移动到指定设备（GPU/CPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 计算当前步的学习率（使用学习率调度器）
        # epoch * iters + step: 当前总步数
        # args.epochs * iters: 总训练步数
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度上下文管理器（autocast）
        # 在GPU上使用bfloat16/float16加速训练，在CPU上使用nullcontext（无效果）
        with autocast_ctx:
            # 前向传播：模型预测
            # res.logits: 形状 (batch_size, seq_len, vocab_size)，每个位置的词汇表概率分布
            res = model(X)
            
            # 计算损失
            # 将logits和Y展平为2D: (batch_size * seq_len, vocab_size) 和 (batch_size * seq_len,)
            # 然后计算每个位置的交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 展平logits
                Y.view(-1)  # 展平目标
            ).view(Y.size())  # 恢复原始形状 (batch_size, seq_len)

            # 应用损失掩码，只计算有效位置的损失
            # loss * loss_mask: 将无效位置的损失置为0
            # .sum() / loss_mask.sum(): 计算有效位置的平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 添加MoE架构的辅助损失（负载均衡损失）
            # 如果使用MoE，aux_loss会鼓励专家负载均衡
            # 如果不使用MoE，aux_loss为0
            loss += res.aux_loss
            
            # 除以梯度累积步数，使得累积后的梯度等于单步的梯度
            # 这样梯度累积的效果等价于使用更大的batch size
            loss = loss / args.accumulation_steps

        # 使用GradScaler进行梯度缩放（用于混合精度训练）
        # scale(loss): 将损失乘以缩放因子，防止梯度下溢
        # backward(): 反向传播，计算梯度
        scaler.scale(loss).backward()

        # 每accumulation_steps步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备进行梯度裁剪和参数更新
            scaler.unscale_(optimizer)
            
            # 梯度裁剪：防止梯度爆炸
            # 将所有参数的梯度裁剪到最大范数为grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 使用缩放后的梯度更新参数
            scaler.step(optimizer)
            
            # 更新缩放因子（根据梯度是否溢出动态调整）
            scaler.update()

            # 清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)
            
            # 清空CUDA缓存，释放未使用的显存
            torch.cuda.empty_cache()

        # 定期打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已用时间
            spend_time = time.time() - start_time
            
            # 恢复真实的损失值（之前除以了accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            
            # 估算剩余时间（分钟）
            # spend_time / (step + 1): 平均每步用时
            # * iters: 总时间
            # // 60: 转换为分钟
            # - spend_time // 60: 减去已用时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印训练日志
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            # 如果使用wandb，记录指标
            if wandb: 
                wandb.log({
                    "loss": current_loss, 
                    "lr": current_lr, 
                    "epoch_Time": eta_min
                })

        # 定期保存检查点（只在主进程保存，避免多进程冲突）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式，确保BatchNorm/Dropout等层行为一致
            model.eval()
            
            # 根据是否使用MoE添加后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            
            # 构建检查点文件路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典
            # 如果是DDP模型，需要通过.module访问原始模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 转换为半精度（float16）以节省存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            
            # 保存模型权重
            torch.save(state_dict, ckp)
            
            # 保存完整检查点（包括优化器、scaler、epoch、step等）
            # 用于后续恢复训练状态
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            
            # 切换回训练模式
            model.train()


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    
    # 设备相关参数
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    
    # 训练策略参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数（有效batch size = batch_size * accumulation_steps）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（步数）")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Transformer层数）")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径（JSONL格式）")
    
    # 权重加载参数
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/SwanLab进行实验跟踪")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    
    # 解析命令行参数
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（如果使用多GPU）
    # 返回当前进程的本地rank（GPU编号）
    local_rank = init_distributed_mode()
    
    # 如果使用分布式训练，将设备设置为对应的GPU
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子，确保实验可复现
    # 每个进程使用不同的种子（基于rank），避免所有进程生成相同的数据
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建保存目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建模型配置对象
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 如果启用自动续训，尝试加载检查点
    # ckp_data包含：模型权重、优化器状态、scaler状态、epoch、step、wandb_id等
    ckp_data = lm_checkpoint(
        lm_config, 
        weight=args.save_weight, 
        save_dir='../checkpoints'
    ) if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 判断设备类型（CUDA或CPU）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 选择数据类型：bfloat16或float16
    # bfloat16: 更好的数值稳定性，推荐用于训练
    # float16: 更小的内存占用，但需要GradScaler防止梯度下溢
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 创建混合精度上下文管理器
    # CPU模式使用nullcontext（无效果），GPU模式使用autocast加速训练
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb/SwanLab ==========
    wandb = None
    # 只在主进程初始化wandb（避免多进程重复初始化）
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        
        # 如果从检查点恢复，尝试恢复wandb运行ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        
        # 如果有wandb_id，则恢复运行；否则创建新运行
        resume = 'must' if wandb_id else None
        
        # 构建运行名称（包含关键超参数）
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        
        # 初始化wandb
        wandb.init(
            project=args.wandb_project, 
            name=wandb_run_name, 
            id=wandb_id, 
            resume=resume
        )
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和tokenizer
    # 如果from_weight不为'none'，会加载预训练权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 创建预训练数据集
    train_ds = PretrainDataset(
        args.data_path, 
        tokenizer, 
        max_length=args.max_seq_len
    )
    
    # 如果使用分布式训练，创建分布式采样器
    # 分布式采样器确保每个进程看到不同的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建GradScaler（用于混合精度训练）
    # 只在float16模式下启用，bfloat16不需要scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建优化器（AdamW）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        model.load_state_dict(ckp_data['model'])
        
        # 恢复优化器状态（包括动量等）
        optimizer.load_state_dict(ckp_data['optimizer'])
        
        # 恢复scaler状态（包括缩放因子）
        scaler.load_state_dict(ckp_data['scaler'])
        
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包装模型 ==========
    # 如果使用分布式训练，用DDP包装模型
    if dist.is_initialized():
        # 设置DDP忽略的参数（这些参数不需要同步）
        # freqs_cos和freqs_sin是位置编码的缓存，每个进程可以独立计算
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        
        # 用DDP包装模型，实现数据并行
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 从start_epoch开始训练（如果从检查点恢复，会跳过已训练的epoch）
    for epoch in range(start_epoch, args.epochs):
        # 设置采样器的epoch（确保每个epoch的数据顺序不同）
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # 如果是第一个epoch且存在检查点，需要跳过已训练的步数
        if epoch == start_epoch and start_step > 0:
            # 创建跳过批次的采样器
            # 跳过前start_step个batch，从start_step+1开始
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1
            )
            
            # 创建数据加载器
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True  # 使用pin_memory加速GPU传输
            )
            
            # 打印续训信息
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            
            # 开始训练，传入总迭代次数和起始步数
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:
            # 默认情况：从头开始训练
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),  # 单卡时shuffle，多卡时由sampler控制
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            
            # 开始训练
            train_epoch(epoch, loader, len(loader), 0, wandb)
