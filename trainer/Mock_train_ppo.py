"""
MiniMind PPO (Proximal Policy Optimization) 训练脚本

PPO是一种强化学习算法，用于训练语言模型以对齐人类偏好。
相比DPO，PPO使用显式的奖励模型（Reward Model）来计算奖励信号，
并通过Actor-Critic架构来优化策略。

核心组件：
1. Actor模型：当前策略模型，用于生成回答
2. Critic模型：价值函数，估计状态价值
3. Old Actor模型：旧策略，用于重要性采样（importance sampling）
4. Reference模型：参考策略，用于KL散度正则化，防止策略偏离太远
5. Reward模型：奖励模型，评估生成回答的质量

PPO算法流程：
1. 使用Actor模型生成回答
2. 使用Reward模型计算奖励
3. 使用Critic模型估计价值
4. 计算优势函数：advantage = reward - value
5. 计算重要性采样比率：ratio = π_θ(a|s) / π_old(a|s)
6. 计算裁剪后的策略损失：L_clip = -min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
7. 计算价值损失：L_value = MSE(value, reward)
8. 计算KL散度惩罚：L_kl = KL(π_θ || π_ref)
9. 总损失：L = L_clip + vf_coef * L_value + kl_coef * L_kl

参考论文：Proximal Policy Optimization Algorithms (Schulman et al., 2017)
"""

import os
import sys

# 设置包路径，确保可以正确导入项目模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    """
    Critic模型（价值函数模型）
    
    Critic模型用于估计状态价值V(s)，即给定当前状态（prompt+部分生成），
    估计未来能获得的期望奖励。Critic模型基于Actor模型的架构，
    但将语言模型头（lm_head）替换为价值头（value_head），输出单一标量值。
    
    在PPO中，Critic用于：
    1. 计算优势函数：advantage = reward - V(s)
    2. 通过价值损失训练：L_value = MSE(V(s), reward)
    
    Args:
        params: MiniMindConfig配置对象
    """
    def __init__(self, params):
        super().__init__(params)
        # 将语言模型头替换为价值头
        # 价值头输出单一标量值，表示当前状态的价值估计
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，计算每个位置的状态价值
        
        Args:
            input_ids: 输入token IDs，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            **kwargs: 其他参数
        
        Returns:
            values: 每个位置的价值估计，形状为 (batch_size, seq_len)
        """
        # 使用基础Transformer模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 应用层归一化
        hidden_states = self.model.norm(outputs[0])
        # 通过价值头获取每个位置的价值估计
        # squeeze(-1)移除最后一个维度，得到 (batch_size, seq_len)
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励
    
    该函数计算PPO训练中的奖励信号，包括：
    1. 格式奖励（仅用于推理模型）：检查回答是否符合特定格式
    2. 标记奖励（仅用于推理模型）：检查是否包含必要的XML标记
    3. 奖励模型分数：使用预训练的奖励模型评估回答质量
    
    Args:
        prompts: 输入提示列表，每个元素是一个字符串
        responses: 模型生成的回答列表，每个元素是一个字符串
        reward_model: 奖励模型，用于评估回答质量
        reward_tokenizer: 奖励模型的tokenizer
    
    Returns:
        rewards: 每个回答的总奖励，形状为 (batch_size,)，dtype=torch.float32
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
        
        该函数通过正则表达式检查格式是否正确，并给予格式奖励和标记奖励。
        格式奖励和标记奖励的目的是：
        1. 引导模型学习正确的输出格式
        2. 防止奖励信号过于稀疏（如果只依赖奖励模型，格式错误时奖励可能为0）
        
        Args:
            rewards: 当前奖励张量，形状为 (batch_size,)
        
        Returns:
            rewards: 添加格式和标记奖励后的奖励张量
        """
        # ========== 1. 格式奖励 ==========
        # 检查回答是否符合标准格式（带或不带空行）
        # pattern1: <think>\n...\n</think>\n<answer>\n...\n</answer>
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # pattern2: <think>\n...\n</think>\n\n<answer>\n...\n</answer>（多一个空行）
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        # 使用正则表达式匹配每个回答
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 如果匹配成功，给予0.5的格式奖励
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # ========== 2. 标记奖励 ==========
        # 检查是否包含必要的XML标记（即使格式不完全正确）
        # 这有助于防止奖励信号过于稀疏，即使格式不完全正确也能获得部分奖励
        def mark_num(text):
            """
            计算标记奖励
            
            检查文本中是否包含必要的XML标记，每个标记正确出现一次给予0.25奖励。
            这样可以鼓励模型学习使用这些标记，即使格式不完全正确。
            
            Args:
                text: 要检查的文本
            
            Returns:
                reward: 标记奖励值（0.0-1.0）
            """
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
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
    # 奖励模型通常是在人类偏好数据上训练的，能够评估回答的有用性、无害性等
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 解析prompt中的消息格式（ChatML格式）
            # 格式：<|im_start|>role\ncontent<|im_end|>
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整的对话（prompt + response）
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            # 使用奖励模型计算分数
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 将分数裁剪到[-scale, scale]范围内，防止异常值
            scale = 3.0
            score = max(min(score, scale), -scale)

            # ========== 推理模型的额外奖励计算 ==========
            # 对于推理模型，除了整体回答的奖励，还单独计算<answer>部分的奖励
            # 这样可以更关注最终答案的质量，而不仅仅是推理过程
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
                    # 这样更强调最终答案的质量
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        # 将奖励模型分数转换为张量并添加到总奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    PPO训练一个epoch的主循环函数
    
    该函数执行PPO训练的核心流程：
    1. 使用Actor模型生成回答
    2. 使用Reward模型计算奖励
    3. 使用Critic模型估计价值
    4. 计算优势函数和重要性采样比率
    5. 计算PPO裁剪损失、价值损失和KL散度惩罚
    6. 反向传播并更新参数
    7. 定期更新old_actor_model（用于重要性采样）
    8. 记录日志和保存检查点
    
    Args:
        epoch: 当前epoch编号（从0开始）
        loader: DataLoader对象，用于加载训练数据
        iters: 该epoch的总迭代次数
        old_actor_model: 旧策略模型（冻结），用于重要性采样
        ref_model: 参考模型（冻结），用于KL散度正则化
        actor_scheduler: Actor模型的学习率调度器
        critic_scheduler: Critic模型的学习率调度器
        reward_model: 奖励模型（冻结），用于计算奖励
        reward_tokenizer: 奖励模型的tokenizer
        start_step: 起始步数（用于续训场景）
        wandb: wandb对象，用于记录训练指标（可选）
    """
    actor_model.train()
    critic_model.train()

    # 遍历训练数据批次
    for step, batch in enumerate(loader, start=start_step + 1):
        # ========== 1. 数据准备和编码 ==========
        prompts = batch["prompt"]  # list[str], length B
        # 对prompts进行tokenize和padding
        # padding='left'是因为PPO需要左侧padding（在generate时使用）
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        # 计算每个prompt的实际长度（不包括padding）
        prompt_lengths = enc.attention_mask.sum(dim=1)  # [B]

        # ========== 2. 使用Actor模型生成回答 ==========
        with torch.no_grad():
            # DDP模型需要使用.module访问底层的generate方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            # 使用采样生成回答
            # do_sample=True: 使用采样而非贪婪解码
            # temperature=0.8: 控制采样的随机性（较低的温度使分布更尖锐）
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R]
            # gen_out包含prompt和生成的response：形状为 [B, P+R]

        # ========== 3. 解码回答并计算奖励 ==========
        # 提取生成的response部分（去掉prompt）
        responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) 
                         for i in range(len(prompts))]
        # 使用奖励模型计算每个回答的奖励
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # ========== 4. 使用Critic模型估计价值 ==========
        # 创建完整的注意力掩码（包括prompt和response）
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        # Critic模型估计每个位置的状态价值
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        # 获取每个序列最后一个有效token的位置（即response的最后一个token）
        last_indices = full_mask.sum(dim=1) - 1  # [B]
        # 提取每个序列最后一个位置的价值（这是response结束时的状态价值）
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        # 计算优势函数：advantage = reward - value
        # 优势函数表示实际奖励与预期价值之间的差异
        # 正值表示表现好于预期，负值表示表现差于预期
        advantages = rewards - values.detach()  # [B]

        # ========== 5. 计算Actor模型的对数概率 ==========
        # 获取Actor模型对生成序列的logits
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        # 创建标签（用于计算对数概率）
        # 使用shift-by-one：logits[i]对应labels[i]（下一个token）
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        # 计算每个token的对数概率
        # logits[:, :-1]对应labels（去掉最后一个logit）
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        # 创建掩码：只计算response部分的对数概率（不包括prompt和padding）
        seq_len = gen_out.size(1) - 1
        # resp_mask: 标记哪些位置属于response（而非prompt）
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
        # final_mask: response部分且非padding的位置
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        # 计算每个序列的总对数概率（只对response部分求和）
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== 6. 计算Old Actor和Reference模型的对数概率 ==========
        with torch.no_grad():
            # Old Actor模型的对数概率（用于重要性采样）
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # Reference模型的对数概率（用于KL散度正则化）
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== 7. 计算PPO损失 ==========
        # KL散度：当前策略与old策略的差异（用于监控）
        kl = (actor_logp - old_logp).mean()  # scalar
        # KL散度：当前策略与参考策略的差异（用于正则化）
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        
        # 重要性采样比率：ratio = π_θ(a|s) / π_old(a|s)
        # 这个比率用于修正使用旧策略数据训练新策略时的偏差
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO裁剪损失的核心计算
        # surr1: 未裁剪的策略梯度项
        surr1 = ratio * advantages  # [B]
        # surr2: 裁剪后的策略梯度项
        # clip_epsilon通常为0.1或0.2，限制策略更新的幅度
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        # PPO策略损失：取surr1和surr2的最小值（保守更新）
        # 负号是因为要最大化优势，但优化器是最小化损失
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        
        # 价值损失：Critic模型应该准确预测奖励
        value_loss = F.mse_loss(values, rewards)  # scalar
        
        # 总损失：策略损失 + 价值损失 + KL散度惩罚
        # vf_coef: 价值函数系数，平衡策略损失和价值损失
        # kl_coef: KL散度系数，防止策略偏离参考模型太远
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        
        # 反向传播
        loss.backward()

        # ========== 8. 梯度更新 ==========
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪：防止梯度爆炸
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            # 更新参数
            actor_optimizer.step()
            critic_optimizer.step()
            # 更新学习率
            actor_scheduler.step()
            critic_scheduler.step()
            # 清零梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            # 清空CUDA缓存
            torch.cuda.empty_cache()

        # ========== 9. 日志记录 ==========
        if is_main_process():
            # 计算平均回答长度（用于监控）
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            # 提取训练指标
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            # 记录到wandb
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            # 打印训练日志
            Logger(f"Epoch: {epoch+1}, Step: {step}/{iters}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        # ========== 10. 更新Old Actor模型 ==========
        # 定期更新old_actor_model，用于重要性采样
        # 这是PPO算法的关键：使用旧策略的数据训练新策略，需要重要性采样修正
        if (step + 1) % args.update_old_actor_freq == 0:
            # 获取当前Actor模型的状态字典
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            # 更新old_actor_model（detach并移到CPU以节省显存）
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ========== 11. 保存检查点 ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取Actor模型状态字典
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            # 保存Actor模型权重（半精度）
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # 使用lm_checkpoint保存完整状态（包括critic、优化器、调度器等）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    
    # 模型保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（每个GPU的batch size）")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率（通常很小，避免策略变化过快）")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16或float16）")
    
    # 数据加载和训练配置
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数（用于模拟更大的batch size）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值（防止梯度爆炸）")
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
    
    # PPO特定参数
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数ε（通常0.1-0.2，限制策略更新幅度）")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数（平衡策略损失和价值损失）")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数（防止策略偏离参考模型太远）")
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率（每N步更新一次）")
    
    # 奖励模型参数
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径（用于计算奖励）")
    
    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        
        # 初始化wandb运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # 根据是否训练推理模型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Actor模型（当前策略模型，要优化的模型）
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # PPO需要左侧padding，因为生成时prompt在左侧
    tokenizer.padding_side = 'left'
    
    # Old Actor模型（旧策略，用于重要性采样）
    # Old Actor与Actor初始权重相同，但会定期更新以保存旧策略的快照
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # Reference模型（参考策略，用于KL散度正则化）
    # Reference模型在整个训练过程中保持不变，作为正则化基准
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Critic模型（价值函数模型）
    # Critic模型基于Actor模型的架构，但将lm_head替换为value_head
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    # 从检查点加载权重（strict=False允许value_head不存在）
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    
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
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    
    # 如果使用分布式训练，创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总迭代次数（用于学习率调度器）
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 计算总的优化器步数（考虑梯度累积）
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    
    # 创建学习率调度器（余弦退火）
    # CosineAnnealingLR: 学习率从初始值逐渐降低到eta_min
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 7. 从检查点恢复训练状态 ==========
    # 初始化训练起始位置
    start_epoch, start_step = 0, 0
    
    # 如果存在检查点数据，恢复所有模型和优化器的状态
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'从检查点恢复：epoch {start_epoch}, step {start_step}')
    
    # ========== 8. 分布式数据并行（DDP）包装模型 ==========
    # 如果使用多GPU训练，用DDP包装Actor和Critic模型
    if dist.is_initialized():
        # 忽略某些参数（如位置编码的sin/cos），这些参数不需要同步
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用DDP包装模型，实现多GPU并行训练
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        # Old Actor模型不需要DDP包装（只在生成时使用，不参与训练）
        old_actor_model.to(args.device)
    
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
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            # 正常训练：从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None),  # 单GPU时shuffle，多GPU时由sampler控制
                              sampler=train_sampler, 
                              num_workers=args.num_workers, 
                              pin_memory=True)  # pin_memory=True可以加速GPU数据传输
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
