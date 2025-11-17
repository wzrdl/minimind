"""
MiniMind GRPO (Group Relative Policy Optimization) 训练脚本

GRPO是一种强化学习算法，用于训练语言模型以对齐人类偏好。
相比PPO，GRPO不需要Critic模型（价值函数），而是通过组内相对排名来计算优势函数。

核心思想：
- 对每个prompt生成多个样本（num_generations个）
- 计算每个样本的奖励
- 在组内（同一prompt的多个生成）计算相对优势：advantage = (reward - mean) / std
- 使用per-token级别的损失函数，结合优势函数和KL散度惩罚
- 损失函数：L = -exp(logp - logp.detach()) * advantage + β * KL(π_θ || π_ref)

GRPO的优势：
1. 不需要训练Critic模型，简化了架构
2. 组内相对排名更稳定，不受奖励模型绝对尺度影响
3. Per-token级别的优化更精细

核心组件：
1. Policy模型：当前策略模型，用于生成回答
2. Reference模型：参考策略，用于KL散度正则化
3. Reward模型：奖励模型，评估生成回答的质量

参考论文：Group Relative Policy Optimization (GRPO)
"""

import os
import sys

# 设置包路径，确保可以正确导入项目模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励
    
    该函数计算GRPO训练中的奖励信号，包括：
    1. 格式奖励（仅用于推理模型）：检查回答是否符合特定格式
    2. 标记奖励（仅用于推理模型）：检查是否包含必要的XML标记
    3. 奖励模型分数：使用预训练的奖励模型评估回答质量
    
    注意：在GRPO中，responses的长度是 prompts 长度的 num_generations 倍，
    因为每个prompt会生成多个样本。
    
    Args:
        prompts: 输入提示列表，每个元素是一个字符串，长度为 B
        responses: 模型生成的回答列表，长度为 B * num_generations
                   responses[i*num_generations:(i+1)*num_generations] 对应 prompts[i]
        reward_model: 奖励模型，用于评估回答质量
        reward_tokenizer: 奖励模型的tokenizer
    
    Returns:
        rewards: 每个回答的总奖励，形状为 (B * num_generations,)，dtype=torch.float32
    """
    def reasoning_model_reward(rewards):
        """
        计算推理模型的格式和标记奖励
        
        推理模型需要生成特定格式的回答：
        <think>
        ...推理过程...
        </think>
        <answer>
        ...最终答案...
        </answer>
        
        Args:
            rewards: 当前奖励张量，形状为 (B * num_generations,)
        
        Returns:
            rewards: 添加格式和标记奖励后的奖励张量
        """
        # 检查回答是否符合标准格式（带或不带空行）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 如果匹配成功，给予0.5的格式奖励
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 检查是否包含必要的XML标记（即使格式不完全正确）
        def mark_num(text):
            """
            计算标记奖励
            
            检查文本中是否包含必要的XML标记，每个标记正确出现一次给予0.25奖励。
            
            Args:
                text: 要检查的文本
            
            Returns:
                reward: 标记奖励值（0.0-1.0）
            """
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励为零
    rewards = torch.zeros(len(responses), device=args.device)
    
    # ========== 格式和标记奖励（仅用于推理模型） ==========
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # ========== 奖励模型分数 ==========
    # 使用预训练的奖励模型评估每个回答的质量
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0  # 奖励分数的裁剪范围

        # 遍历每个prompt和对应的多个生成样本
        for i in range(batch_size):
            for j in range(args.num_generations):
                # 计算response在responses列表中的索引
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析prompt中的消息格式（ChatML格式）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 构建完整的对话（prompt + response）
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                # 使用奖励模型计算分数
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # 将分数裁剪到[-scale, scale]范围内，防止异常值
                score = max(min(score, scale), -scale)

                # ========== 推理模型的额外奖励计算 ==========
                # 对于推理模型，除了整体回答的奖励，还单独计算<answer>部分的奖励
                if args.reasoning == 1:
                    # 提取<answer>标签中的内容
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # 对answer内容单独计算奖励
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 加权组合：整体回答40%，答案部分60%
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # 将奖励模型分数转换为张量并添加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    GRPO训练一个epoch的主循环函数
    
    该函数执行GRPO训练的核心流程：
    1. 对每个prompt生成多个样本（num_generations个）
    2. 计算每个样本的奖励
    3. 在组内计算相对优势（组内标准化）
    4. 计算per-token级别的对数概率
    5. 计算GRPO损失（结合优势函数和KL散度惩罚）
    6. 反向传播并更新参数
    7. 记录日志和保存检查点
    
    GRPO的关键特点：
    - 不需要Critic模型，使用组内相对排名计算优势
    - Per-token级别的损失计算，更精细的优化
    - 组内标准化使优势函数更稳定
    
    Args:
        epoch: 当前epoch编号（从0开始）
        loader: DataLoader对象，用于加载训练数据
        iters: 该epoch的总迭代次数
        ref_model: 参考模型（冻结），用于KL散度正则化
        reward_model: 奖励模型（冻结），用于计算奖励
        reward_tokenizer: 奖励模型的tokenizer
        start_step: 起始步数（用于续训场景）
        wandb: wandb对象，用于记录训练指标（可选）
    """
    model.train()
    
    # 遍历训练数据批次
    for step, batch in enumerate(loader, start=start_step + 1):
        # ========== 1. 数据准备和编码 ==========
        prompts = batch['prompt']  # list[str], length B
        
        # 对prompts进行tokenize和padding
        # padding_side='left': GRPO需要左侧padding（生成时prompt在左侧）
        # add_special_tokens=False: 不添加特殊token（如BOS/EOS），由模型自己处理
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # 如果设置了max_seq_len，截断prompt（保留最后max_seq_len个token）
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ========== 2. 使用Policy模型生成多个样本 ==========
        with torch.no_grad():
            # DDP模型需要使用.module访问底层的generate方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # 对每个prompt生成num_generations个样本
            # num_return_sequences: 每个输入生成多个序列
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]
            # outputs包含prompt和生成的response：形状为 [B*num_gen, P+R]

        # 提取生成的completion部分（去掉prompt）
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        # ========== 3. 计算per-token对数概率 ==========
        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            计算每个token的对数概率
            
            该函数用于计算模型对生成序列中每个token的预测概率。
            这是GRPO损失计算的基础，因为GRPO使用per-token级别的优化。
            
            Args:
                mdl: 模型（Policy或Reference）
                input_ids: 完整的输入序列（prompt + completion），形状为 [B*num_gen, P+R]
                n_keep: 要计算对数概率的token数量（即completion的长度）
            
            Returns:
                per_token_logps: 每个token的对数概率，形状为 [B*num_gen, R]
            """
            # 处理inference tensor（如果存在）
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # 获取logits（只保留最后n_keep+1个位置的logits，因为需要shift-by-one）
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            # 对每个序列计算对数概率
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 使用gather提取对应token位置的对数概率
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        # 计算Policy模型的对数概率（需要梯度）
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        
        # 计算Reference模型的对数概率（不需要梯度，用于KL散度）
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # ========== 4. 解码回答并计算奖励 ==========
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # ========== 5. 计算组内相对优势 ==========
        # GRPO的核心：在组内（同一prompt的多个生成）计算相对优势
        # 将rewards重新组织为组：每个prompt的num_generations个样本为一组
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        
        # 计算每组内的均值和标准差
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        
        # 计算组内标准化的优势函数：advantage = (reward - mean) / std
        # 这样优势函数表示相对于组内平均水平的偏差
        # clamp到[-10, 10]防止异常值
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        # 全局标准化：使优势函数的均值为0，标准差为1
        # 这有助于稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # ========== 6. 创建completion掩码 ==========
        # 创建掩码，标记哪些位置是有效的completion（不包括padding和EOS之后的部分）
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        # 找到每个序列中EOS token的位置（如果没有EOS，则使用序列长度）
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 创建掩码：标记从开始到EOS（或序列结束）的所有位置
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]

        # ========== 7. 计算GRPO损失 ==========
        # KL散度：当前策略与参考策略的差异
        kl_div = ref_per_token_logps - per_token_logps
        # Per-token KL散度：exp(kl_div) - kl_div - 1
        # 这是KL散度的二阶近似，更稳定
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        
        # GRPO损失的核心计算
        # per_token_logps - per_token_logps.detach(): 只对优势项计算梯度
        # exp(...) * advantages: 重要性采样项，结合优势函数
        # -args.beta * per_token_kl: KL散度惩罚项
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # [B*num_gen, R]
        
        # 对每个序列求平均（只考虑有效token），然后对所有序列求平均
        # 除以accumulation_steps用于梯度累积
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # scalar
        
        # 反向传播
        loss.backward()

        # ========== 8. 梯度更新 ==========
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪：防止梯度爆炸
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 清零梯度
            optimizer.zero_grad()
            # 清空CUDA缓存
            torch.cuda.empty_cache()

        # ========== 9. 日志记录 ==========
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps  # 恢复真实损失值
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            # 打印训练日志
            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            # 记录到wandb
            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # ========== 10. 保存检查点 ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取模型状态字典
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            # 保存模型权重（半精度）
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点（包括优化器、调度器等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()

        # ========== 11. 清理显存 ==========
        # 删除不需要的变量，释放显存
        # GRPO生成多个样本，显存占用较大，需要及时清理
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()  # 强制垃圾回收


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    
    # 模型保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（每个GPU的batch size）")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率（通常很小，避免策略变化过快）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    
    # 数据加载和训练配置
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数（用于模拟更大的batch size）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸，0表示不裁剪）")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔（每N步打印一次）")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔（每N步保存一次）")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Transformer层数）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度（response的最大token数）")
    
    # 数据和模型加载参数
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径（JSONL格式）")
    parser.add_argument('--reasoning', type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # GRPO特定参数
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数（GRPO的核心参数，通常4-16）")
    parser.add_argument("--beta", type=float, default=0.02, help="KL散度惩罚系数（防止策略偏离参考模型太远）")
    
    # 奖励模型参数
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径（用于计算奖励）")
    
    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    
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
    # max_seq_len设置为prompt和生成的总长度
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, 
                               num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, 
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        
        # 初始化wandb运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # 根据是否训练推理模型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Policy模型（当前策略模型，要优化的模型）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Reference模型（参考策略，用于KL散度正则化）
    # Reference模型在整个训练过程中保持不变，作为正则化基准
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward模型（奖励模型，用于计算奖励）
    # Reward模型是预训练的，在整个训练过程中保持冻结
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # ========== 6. 准备数据集和优化器 ==========
    # 创建RLAIF数据集
    # RLAIF (Reinforcement Learning from AI Feedback) 数据集包含prompt和偏好信息
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    
    # 如果使用分布式训练，创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算总迭代次数（用于学习率调度器）
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 计算总的优化器步数（考虑梯度累积）
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    
    # 创建学习率调度器（余弦退火）
    # CosineAnnealingLR: 学习率从初始值逐渐降低到eta_min
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 7. 从检查点恢复训练状态 ==========
    # 初始化训练起始位置
    start_epoch, start_step = 0, 0
    
    # 如果存在检查点数据，恢复模型、优化器和调度器的状态
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
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
            grpo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            # 正常训练：从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False,  # 不丢弃最后一个不完整的batch
                              shuffle=(train_sampler is None),  # 单GPU时shuffle，多GPU时由sampler控制
                              num_workers=args.num_workers, 
                              sampler=train_sampler)
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
