"""
MiniMind 知识蒸馏（Knowledge Distillation）训练脚本

该脚本用于将大型教师模型的知识蒸馏到小型学生模型中。
通过让学生模型学习教师模型的软标签（soft labels），可以在保持性能的同时大幅减小模型规模。

主要特点：
1. 双模型架构：教师模型（大模型）和学生模型（小模型）
2. 蒸馏损失：使用KL散度让学生模型学习教师模型的概率分布
3. 混合损失：结合CE损失（ground truth）和蒸馏损失（teacher soft labels）
4. 温度缩放：使用温度参数软化概率分布，传递更多信息
5. 支持分布式训练、混合精度、梯度累积
6. 自动检查点保存和恢复
7. Wandb/SwanLab 实验跟踪

知识蒸馏原理：
- 教师模型提供软标签（softmax概率分布），包含比硬标签更多的信息
- 学生模型学习匹配教师模型的概率分布（KL散度）
- 温度T > 1时，概率分布更平滑，传递更多信息
- 总损失 = α * CE_loss + (1-α) * Distill_loss
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
import torch.nn.functional as F  # 神经网络函数（softmax, kl_div等）
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器（用于CPU模式）
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # DDP并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import SFTDataset  # SFT数据集（指令-回答格式）
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


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏损失（KL散度）
    
    知识蒸馏的核心思想：让学生模型学习教师模型的软标签（soft labels），
    而不是只学习硬标签（hard labels）。软标签包含更多信息，比如：
    - 硬标签：[0, 0, 1, 0] 只告诉我们第3个类别是正确的
    - 软标签：[0.1, 0.2, 0.6, 0.1] 告诉我们第3个类别最可能，但第2个也有一定概率
    
    温度缩放（Temperature Scaling）：
    - temperature = 1: 标准softmax，分布尖锐
    - temperature > 1: 分布更平滑，传递更多信息（推荐1.5-2.0）
    - temperature < 1: 分布更尖锐，传递信息更少
    
    Args:
        student_logits: 学生模型的logits，形状 (batch_size, vocab_size)
        teacher_logits: 教师模型的logits，形状 (batch_size, vocab_size)
        temperature: 温度参数，用于软化概率分布（推荐1.0-2.0）
        reduction: KL散度的归约方式，'batchmean'表示对batch求平均
        
    Returns:
        蒸馏损失（已乘以temperature^2进行缩放）
        
    数学公式：
        teacher_probs = softmax(teacher_logits / T)
        student_log_probs = log_softmax(student_logits / T)
        KL = KL(student_log_probs || teacher_probs)
        loss = T^2 * KL
        
    为什么要乘以T^2？
    - 因为logits被除以了T，梯度也被缩放了1/T
    - 乘以T^2可以恢复原始的梯度尺度，保持训练稳定性
    """
    # 计算教师模型的软标签（概率分布）
    # 使用no_grad()确保不计算梯度（教师模型不更新）
    with torch.no_grad():
        # softmax(teacher_logits / temperature): 使用温度缩放软化概率分布
        # detach(): 确保完全断开梯度连接
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 计算学生模型的log概率分布
    # log_softmax用于数值稳定性，避免先softmax再log可能的下溢
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算KL散度：KL(student || teacher)
    # KL散度衡量两个概率分布的差异
    # reduction='batchmean': 对batch维度求平均
    kl = F.kl_div(
        student_log_probs,  # 学生模型的log概率（第一个参数）
        teacher_probs,  # 教师模型的概率（第二个参数）
        reduction=reduction
    )
    
    # 乘以temperature^2恢复梯度尺度
    # 这是因为logits被除以了temperature，梯度也被缩放了
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """
    训练一个epoch的核心函数（知识蒸馏版本）
    
    该函数执行一个完整的训练epoch，包括：
    1. 学生模型前向传播
    2. 教师模型前向传播（不计算梯度）
    3. 计算CE损失（ground truth）和蒸馏损失（teacher soft labels）
    4. 混合损失：alpha * CE + (1-alpha) * Distill
    5. 反向传播和梯度累积
    6. 梯度裁剪和参数更新
    7. 日志记录和检查点保存
    
    Args:
        epoch: 当前epoch编号（从0开始）
        loader: 数据加载器（SFTDataset）
        iters: 总迭代次数（用于学习率调度和日志）
        teacher_model: 教师模型（大模型），用于提供软标签
        lm_config_student: 学生模型配置
        start_step: 起始步数（用于续训时跳过已训练的步数）
        wandb: Wandb/SwanLab对象，用于实验跟踪（可选）
        alpha: CE损失的权重，总损失 = alpha * CE + (1-alpha) * Distill
               alpha=0: 只使用蒸馏损失
               alpha=0.5: 平衡CE和蒸馏损失（推荐）
               alpha=1: 只使用CE损失（等价于普通训练）
        temperature: 蒸馏温度，用于软化概率分布（推荐1.5-2.0）
    
    注意：
    - 使用全局变量 args, model, optimizer, scaler, autocast_ctx
    - 教师模型始终在eval模式，不计算梯度
    - 只对学生模型进行梯度更新
    - 支持MoE架构的辅助损失（aux_loss）
    """
    # 记录epoch开始时间，用于计算训练速度和剩余时间
    start_time = time.time()
    
    # 确保教师模型在eval模式且不计算梯度
    # 教师模型只用于推理，不参与训练
    if teacher_model is not None:
        teacher_model.eval()  # 设置为评估模式（关闭dropout等）
        teacher_model.requires_grad_(False)  # 禁用梯度计算，节省内存和计算

    # 遍历数据加载器，enumerate从start_step+1开始计数（用于续训）
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # X: 输入token序列（指令+回答），形状 (batch_size, seq_len)
        # Y: 目标token序列（通常是X右移1位），形状 (batch_size, seq_len)
        # loss_mask: 损失掩码，形状 (batch_size, seq_len)
        #   在SFT中，通常指令部分为0，回答部分为1，只对回答部分计算损失
        
        # 将数据移动到指定设备（GPU/CPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 计算当前步的学习率（使用学习率调度器）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 学生模型前向传播 ==========
        # 使用混合精度上下文管理器（autocast）
        with autocast_ctx:
            # 学生模型前向传播
            # res.logits: 形状 (batch_size, seq_len, vocab_size)
            res = model(X)
            student_logits = res.logits

        # ========== 教师模型前向传播 ==========
        # 教师模型只在推理模式下运行，不计算梯度
        if teacher_model is not None:
            with torch.no_grad():  # 禁用梯度计算
                # 教师模型前向传播
                teacher_logits = teacher_model(X).logits
                
                # 处理词汇表大小不匹配的情况
                # 如果教师模型的词汇表大于学生模型，需要截断
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 展平损失掩码，便于后续索引操作
        loss_mask_flat = loss_mask.view(-1)
        
        # 1) Ground-Truth CE Loss（交叉熵损失）
        # 让学生模型学习正确的答案（硬标签）
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # 展平logits
            Y.view(-1),  # 展平目标
            ignore_index=0,  # 忽略padding位置（index=0）
            reduction='none'  # 不自动求平均，便于应用mask
        )
        
        # 应用损失掩码，只计算回答部分的损失
        # 指令部分的损失被忽略（loss_mask为0）
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        
        # 如果使用MoE，添加辅助损失（负载均衡损失）
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) Distillation Loss（蒸馏损失）
        # 让学生模型学习教师模型的软标签（概率分布）
        if teacher_model is not None:
            # 只对有效位置（loss_mask==1）计算蒸馏损失
            # 这样避免对padding位置计算不必要的损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 如果没有教师模型，蒸馏损失为0
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        # alpha控制两种损失的权重：
        # - alpha=0: 纯蒸馏，完全依赖教师模型
        # - alpha=0.5: 平衡，同时学习ground truth和teacher soft labels（推荐）
        # - alpha=1: 纯CE，等价于普通训练
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 使用GradScaler进行梯度缩放（用于混合精度训练）
        scaler.scale(loss).backward()

        # 每accumulation_steps步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备进行梯度裁剪和参数更新
            scaler.unscale_(optimizer)
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 使用缩放后的梯度更新参数（只更新学生模型）
            scaler.step(optimizer)
            
            # 更新缩放因子
            scaler.update()

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()

        # 定期打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印训练日志，包含总损失、CE损失、蒸馏损失
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} ce:{ce_loss.item():.4f} distill:{distill_loss.item():.4f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            # 如果使用wandb，记录指标
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": current_lr,
                    "epoch_Time": eta_min
                })

        # 定期保存检查点（只在主进程保存）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典（只保存学生模型）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 转换为半精度以节省存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            
            # 保存模型权重
            torch.save(state_dict, ckp)
            
            # 保存完整检查点
            lm_checkpoint(
                lm_config_student, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    # 知识蒸馏通常需要更多轮次，因为需要学习教师模型的复杂知识
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数（蒸馏通常需要更多轮次）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    
    # 蒸馏学习率通常比SFT稍大，但仍比预训练小
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率（蒸馏通常5e-6到1e-5）")
    
    # 设备相关参数
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    
    # 训练策略参数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数（有效batch size = batch_size * accumulation_steps）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（步数）")
    
    # 数据相关参数
    parser.add_argument("--max_seq_len", type=int, default=512, help="训练的最大截断长度")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径（JSONL格式，包含instruction和output）")
    
    # 学生模型架构参数（小模型）
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度（通常小于教师模型）")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量（通常少于教师模型）")
    
    # 教师模型架构参数（大模型）
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度（通常大于学生模型）")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量（通常多于学生模型）")
    
    # MoE架构参数
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 权重加载参数
    # 学生模型和教师模型可以基于不同的权重
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重训练（通常基于SFT权重）")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重（通常基于训练好的大模型）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 蒸馏超参数
    # alpha: CE损失和蒸馏损失的权重平衡
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL（推荐0.3-0.7）")
    
    # temperature: 蒸馏温度，用于软化概率分布
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0，1.5较常用）")
    
    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/SwanLab进行实验跟踪")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    
    # 解析命令行参数
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（如果使用多GPU）
    local_rank = init_distributed_mode()
    
    # 如果使用分布式训练，将设备设置为对应的GPU
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子，确保实验可复现
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建学生模型配置（小模型）
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size, 
        num_hidden_layers=args.student_num_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 创建教师模型配置（大模型）
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size, 
        num_hidden_layers=args.teacher_num_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # 如果启用自动续训，尝试加载检查点（只加载学生模型的检查点）
    ckp_data = lm_checkpoint(
        lm_config_student, 
        weight=args.save_weight, 
        save_dir='../checkpoints'
    ) if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb/SwanLab ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 运行名称包含学生和教师模型的规模信息
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生和教师模型 ==========
    # 初始化学生模型（小模型，需要训练）
    # 通常基于SFT权重开始训练
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化教师模型（大模型，只用于推理）
    # 通常基于训练好的大模型权重
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    
    # 教师模型设置为eval模式且不计算梯度
    # 教师模型只用于提供软标签，不参与训练
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    
    # 创建SFT数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 如果使用分布式训练，创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建GradScaler（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建优化器（只优化学生模型的参数）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复学生模型权重
        model.load_state_dict(ckp_data['model'])
        
        # 恢复优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        
        # 恢复scaler状态
        scaler.load_state_dict(ckp_data['scaler'])
        
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包装模型 ==========
    # 如果使用分布式训练，用DDP包装学生模型
    # 注意：教师模型不需要DDP包装，因为它不参与训练
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置采样器的epoch
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # 如果是第一个epoch且存在检查点，需要跳过已训练的步数
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(
                epoch, loader, len(loader) + start_step + 1, 
                teacher_model, lm_config_student, start_step, 
                wandb, args.alpha, args.temperature
            )
        else:
            # 默认情况：从头开始训练
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            train_epoch(
                epoch, loader, len(loader), 
                teacher_model, lm_config_student, 0, 
                wandb, args.alpha, args.temperature
            )
