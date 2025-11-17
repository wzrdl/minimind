"""
MiniMind DPO (Direct Preference Optimization) 训练脚本

DPO是一种直接偏好优化方法，用于训练语言模型以对齐人类偏好。
相比RLHF（Reinforcement Learning from Human Feedback），DPO不需要显式的奖励模型，
而是通过对比chosen（被选择的回答）和rejected（被拒绝的回答）来优化策略模型。

核心思想：
- 使用参考模型（ref_model）作为基准，计算参考策略的对数概率
- 使用策略模型（policy model）计算当前策略的对数概率
- 通过最大化chosen和rejected之间的对数概率差异来优化策略
- 损失函数：L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
  其中：y_w是chosen回答，y_l是rejected回答，β是温度参数

参考论文：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
"""

import os
import sys

# 设置包路径，确保可以正确导入项目模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的logits转换为对应标签的对数概率
    
    该函数用于计算模型对每个token的预测概率，这是DPO损失计算的基础。
    
    Args:
        logits: 模型输出的原始logits，形状为 (batch_size, seq_len, vocab_size)
        labels: 目标token的索引，形状为 (batch_size, seq_len)
    
    Returns:
        log_probs_per_token: 每个token的对数概率，形状为 (batch_size, seq_len)
                            表示模型对每个位置正确token的预测概率
    """
    # 对logits应用log_softmax，得到对数概率分布
    # dim=2表示在词汇表维度上计算softmax
    log_probs = F.log_softmax(logits, dim=2)
    
    # 使用gather从log_probs中提取对应labels位置的对数概率
    # labels.unsqueeze(2)将labels从(batch_size, seq_len)扩展为(batch_size, seq_len, 1)
    # 然后squeeze(-1)移除最后一个维度，得到(batch_size, seq_len)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算DPO损失函数
    
    DPO损失的核心是最大化chosen回答相对于rejected回答的对数概率差异，
    同时相对于参考模型进行正则化，防止策略偏离太远。
    
    损失公式：
    L_DPO = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    其中：
    - π_θ: 当前策略模型
    - π_ref: 参考模型（冻结）
    - y_w: chosen回答（被选择的）
    - y_l: rejected回答（被拒绝的）
    - β: 温度参数，控制优化强度
    - σ: sigmoid函数
    
    Args:
        ref_log_probs: 参考模型的对数概率，形状为 (batch_size, seq_len)
                       batch的前半部分是chosen，后半部分是rejected
        policy_log_probs: 策略模型的对数概率，形状为 (batch_size, seq_len)
                          batch的前半部分是chosen，后半部分是rejected
        mask: 注意力掩码，形状为 (batch_size, seq_len)，1表示有效token，0表示padding
        beta: DPO温度参数，控制优化强度。较大的beta会增强偏好信号，但可能导致过拟合
    
    Returns:
        loss: 平均DPO损失值（标量）
    
    Reference:
        https://github.com/jingyaogong/minimind/issues/298
    """
    # 计算每个序列的有效长度（非padding的token数量）
    # clamp_min(1e-8)防止零长度序列导致除零错误
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    # 对每个序列的对数概率进行mask并求平均
    # 这样可以只考虑有效token，忽略padding部分
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将batch分为chosen和rejected两部分
    # DPO数据集通常将chosen和rejected配对，所以batch_size是偶数
    # 前半部分是chosen，后半部分是rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]      # chosen的参考模型对数概率
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]      # rejected的参考模型对数概率
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]  # chosen的策略模型对数概率
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]  # rejected的策略模型对数概率

    # 计算对数比率（log ratio）
    # pi_logratios: 策略模型对chosen和rejected的对数概率差异
    # ref_logratios: 参考模型对chosen和rejected的对数概率差异
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # 计算DPO的核心项：策略模型的对数比率减去参考模型的对数比率
    # 这个差值表示策略模型相对于参考模型的改进程度
    logits = pi_logratios - ref_logratios
    
    # 计算DPO损失：-log(σ(β * logits))
    # logsigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    # 所以 -logsigmoid(β * logits) = log(1 + exp(-β * logits))
    # 当logits > 0时（即策略模型更偏好chosen），损失较小
    loss = -F.logsigmoid(beta * logits)
    
    # 返回平均损失
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练一个epoch的主循环函数
    
    该函数执行DPO训练的核心流程：
    1. 加载chosen和rejected数据对
    2. 使用参考模型计算参考对数概率（冻结，不更新梯度）
    3. 使用策略模型计算当前对数概率（需要更新梯度）
    4. 计算DPO损失并反向传播
    5. 定期记录日志和保存检查点
    
    Args:
        epoch: 当前epoch编号（从0开始）
        loader: DataLoader对象，用于加载训练数据
        iters: 该epoch的总迭代次数（用于计算学习率和ETA）
        ref_model: 参考模型（冻结，用于计算参考对数概率）
        lm_config: 模型配置对象
        start_step: 起始步数（用于续训场景）
        wandb: wandb对象，用于记录训练指标（可选）
        beta: DPO温度参数
    """
    start_time = time.time()
    
    # 遍历训练数据批次
    for step, batch in enumerate(loader, start=start_step + 1):
        # ========== 1. 数据准备 ==========
        # 将chosen和rejected数据移动到指定设备（GPU/CPU）
        x_chosen = batch['x_chosen'].to(args.device)      # chosen的输入序列
        x_rejected = batch['x_rejected'].to(args.device)  # rejected的输入序列
        y_chosen = batch['y_chosen'].to(args.device)      # chosen的目标序列（用于计算对数概率）
        y_rejected = batch['y_rejected'].to(args.device)  # rejected的目标序列
        mask_chosen = batch['mask_chosen'].to(args.device)    # chosen的注意力掩码
        mask_rejected = batch['mask_rejected'].to(args.device)  # rejected的注意力掩码
        
        # 将chosen和rejected数据拼接在一起，形成一个大的batch
        # 这样可以在一次前向传播中同时处理两种数据，提高效率
        # 拼接后的batch：前半部分是chosen，后半部分是rejected
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ========== 2. 学习率调度 ==========
        # 计算当前步的学习率（通常使用余弦退火或线性衰减）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== 3. 前向传播和损失计算 ==========
        # 使用混合精度训练上下文（autocast）加速训练并节省显存
        with autocast_ctx:
            # 使用参考模型计算参考对数概率（不需要梯度）
            # 参考模型在DPO训练中保持冻结，作为正则化基准
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 将参考模型的logits转换为对数概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 使用策略模型计算当前对数概率（需要梯度）
            # 策略模型是我们要优化的模型
            outputs = model(x)
            logits = outputs.logits
            # 将策略模型的logits转换为对数概率
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 计算DPO损失
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 梯度累积：将损失除以累积步数，这样最终的梯度等于累积多个batch的平均梯度
            loss = loss / args.accumulation_steps

        # ========== 4. 反向传播 ==========
        # 使用混合精度scaler进行反向传播
        # scale操作可以防止梯度下溢（在float16训练中很重要）
        scaler.scale(loss).backward()

        # ========== 5. 梯度更新 ==========
        # 当累积了足够的梯度时，执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 在更新前取消scale，准备梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪：防止梯度爆炸，提高训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行优化器步骤（更新参数）
            scaler.step(optimizer)
            # 更新scaler的scale因子（用于下一次迭代）
            scaler.update()
            # 清零梯度，为下一次迭代做准备
            # set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)
            # 清空CUDA缓存，释放未使用的显存
            torch.cuda.empty_cache()

        # ========== 6. 日志记录 ==========
        # 定期打印训练进度和指标
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 恢复真实的损失值（之前除以了accumulation_steps）
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间（分钟）
            # 公式：总时间 = 已用时间 / 已完成步数 * 总步数
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印训练日志
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            # 如果启用了wandb，记录训练指标
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ========== 7. 保存检查点 ==========
        # 定期保存模型检查点（只在主进程保存，避免多GPU重复保存）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式（关闭dropout等）
            model.eval()
            # 根据是否使用MoE架构添加后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构建检查点文件路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取模型状态字典
            # 如果使用DDP，需要从model.module中获取
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 将模型权重转换为半精度（float16）以节省存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            # 保存模型权重
            torch.save(state_dict, ckp)
            
            # 保存完整的检查点（包括优化器状态、scaler状态等，用于续训）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            
            # 切换回训练模式
            model.train()


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    
    # 模型保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size（每个GPU的batch size）")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘，DPO通常使用很小的学习率）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    
    # 数据加载和训练配置
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数（用于模拟更大的batch size）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔（每N步打印一次）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（每N步保存一次）")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Transformer层数）")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（超过此长度的序列会被截断）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和模型加载参数
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径（JSONL格式，包含chosen和rejected对）")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练（通常是SFT后的模型）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # DPO特定参数
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数（温度参数，控制优化强度，通常0.1-0.5）")
    
    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（如果使用多GPU）
    # 返回当前进程的local_rank（本地GPU编号）
    local_rank = init_distributed_mode()
    
    # 如果启用了分布式训练，将设备设置为对应的GPU
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子，确保实验可复现
    # 不同进程使用不同的种子（基于rank），避免所有进程生成相同的数据
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建模型配置对象
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, 
                               num_hidden_layers=args.num_hidden_layers, 
                               use_moe=bool(args.use_moe))
    
    # 如果启用续训，尝试加载检查点数据
    # 检查点包含模型权重、优化器状态、训练进度等信息
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度训练 ==========
    # 判断设备类型（CUDA或CPU）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 选择数据类型（bfloat16或float16）
    # bfloat16通常更稳定，float16在某些硬件上可能更快
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 创建混合精度上下文
    # CPU不支持autocast，所以使用nullcontext（空上下文）
    # CUDA使用autocast来加速训练并节省显存
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb（实验跟踪） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 只在主进程初始化wandb，避免多进程重复记录
        import swanlab as wandb
        
        # 如果从检查点恢复，尝试恢复wandb运行ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果有wandb_id，则必须恢复（resume='must'）
        resume = 'must' if wandb_id else None
        
        # 构建运行名称（包含关键超参数）
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        
        # 初始化wandb运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    # 初始化策略模型（要优化的模型）
    # 通常从SFT后的模型开始训练
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化参考模型（冻结，不更新）
    # 参考模型与策略模型初始权重相同，但在训练过程中保持不变
    # 参考模型用于计算参考策略的对数概率，作为正则化基准
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()  # 设置为评估模式
    ref_model.requires_grad_(False)  # 冻结所有参数，不计算梯度
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # ========== 6. 准备数据集和数据加载器 ==========
    # 创建DPO数据集
    # DPODataset会加载JSONL格式的数据，包含chosen和rejected对
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 如果使用分布式训练，创建分布式采样器
    # 分布式采样器确保每个进程看到不同的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建梯度scaler（用于混合精度训练）
    # 只有在使用float16时才启用scaler（bfloat16通常不需要）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建优化器（AdamW）
    # DPO通常使用很小的学习率，避免过度偏离参考模型
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 7. 从检查点恢复训练状态 ==========
    # 初始化训练起始位置
    start_epoch, start_step = 0, 0
    
    # 如果存在检查点数据，恢复模型、优化器和scaler的状态
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'从检查点恢复：epoch {start_epoch}, step {start_step}')
    
    # ========== 8. 分布式数据并行（DDP）包装模型 ==========
    # 如果使用多GPU训练，用DDP包装模型
    if dist.is_initialized():
        # 忽略某些参数（如位置编码的sin/cos），这些参数不需要同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用DDP包装模型，实现多GPU并行训练
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 如果使用分布式采样器，设置epoch（确保每个epoch的数据顺序不同）
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 处理续训场景：如果是从检查点恢复的第一个epoch，需要跳过已训练的步数
        if epoch == start_epoch and start_step > 0:
            # 创建跳过批次采样器，跳过前start_step个batch
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 训练时传入start_step，确保日志和检查点编号正确
            train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            # 正常训练：从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None),  # 单GPU时shuffle，多GPU时由sampler控制
                              sampler=train_sampler, 
                              num_workers=args.num_workers, 
                              pin_memory=True)  # pin_memory=True可以加速GPU数据传输
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
