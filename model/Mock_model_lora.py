import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 低秩适应层
    
    LoRA的核心思想：不直接微调原始权重矩阵W，而是学习一个低秩分解的增量矩阵ΔW
    原始前向传播: y = Wx
    LoRA前向传播: y = Wx + ΔWx = Wx + BAx
    其中 ΔW = BA，B ∈ R^(out_features × rank)，A ∈ R^(rank × in_features)
    且 rank << min(in_features, out_features)，大大减少了可训练参数
    
    优势：
    1. 参数效率：只需要训练 rank × (in_features + out_features) 个参数
    2. 模块化：可以轻松添加/移除，不影响原始模型
    3. 可组合：可以为不同任务训练不同的LoRA权重
    """
    def __init__(self, in_features, out_features, rank):
        """
        初始化LoRA层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: LoRA的秩（rank），控制低秩矩阵的大小
                  较小的rank意味着更少的参数，但可能表达能力不足
                  较大的rank意味着更多参数，但更接近全量微调
                  通常rank取4、8、16等较小值
        """
        super().__init__()
        # LoRA的秩（rank），控制低秩矩阵的大小
        # rank越小，参数越少，但表达能力可能不足
        # rank越大，参数越多，但更接近全量微调的效果
        self.rank = rank
        
        # 低秩矩阵A: (in_features, rank)
        # 将输入从in_features维度降维到rank维度
        # bias=False: LoRA通常不使用偏置，保持参数效率
        self.A = nn.Linear(in_features, rank, bias=False)
        
        # 低秩矩阵B: (rank, out_features)
        # 将rank维度的特征升维到out_features维度
        # 最终 ΔW = BA，形状为 (out_features, in_features)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 矩阵A使用高斯初始化
        # mean=0.0, std=0.02: 小方差初始化，确保初始时ΔW接近0
        # 这样初始时 y = Wx + BAx ≈ Wx，不会破坏预训练模型的性能
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        
        # 矩阵B全0初始化
        # 这是LoRA的关键：初始时B=0，所以BA=0，ΔW=0
        # 确保初始时LoRA层不产生任何影响，完全依赖原始权重W
        # 训练时只需要更新B，A可以固定或一起更新
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        前向传播：计算低秩适应增量
        
        Args:
            x: 输入张量，形状为 (..., in_features)
            
        Returns:
            输出张量，形状为 (..., out_features)
            计算过程：B(A(x))，等价于 (BA)x = ΔWx
        """
        # 先通过矩阵A降维: x -> A(x)，形状 (..., in_features) -> (..., rank)
        # 再通过矩阵B升维: A(x) -> B(A(x))，形状 (..., rank) -> (..., out_features)
        # 等价于计算 ΔWx，其中 ΔW = BA
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    将LoRA适配器应用到模型的线性层
    
    该函数会遍历模型中的所有模块，找到符合条件的线性层（通常是方阵），
    为每个线性层添加一个LoRA适配器，并修改其前向传播函数以包含LoRA增量。
    
    工作原理：
    1. 找到所有满足条件的Linear层（通常是注意力层的Q、K、V投影等）
    2. 为每个层创建LoRA适配器
    3. 修改前向传播：y = Wx + BAx = original_forward(x) + lora(x)
    
    Args:
        model: 要应用LoRA的PyTorch模型
        rank: LoRA的秩，默认为8
              较小的rank（4-16）通常就足够，可以大大减少可训练参数
        
    注意：
    - 只对方阵线性层（weight.shape[0] == weight.shape[1]）应用LoRA
    - 这通常包括注意力层的Q、K、V投影矩阵
    - 其他线性层（如FFN的gate/up/down投影）可能需要不同的处理策略
    """
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 检查是否为Linear层且是方阵（输入输出维度相同）
        # 这通常对应注意力机制中的Q、K、V投影层
        # 例如：hidden_size -> hidden_size 的投影
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 获取原始线性层的输入输出维度
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            
            # 创建LoRA适配器，并移动到模型所在的设备（CPU/GPU）
            lora = LoRA(in_features, out_features, rank=rank).to(model.device)
            
            # 将LoRA适配器作为模块的属性保存
            # 这样可以通过 module.lora 访问，便于后续保存/加载
            setattr(module, "lora", lora)
            
            # 保存原始的前向传播函数
            original_forward = module.forward

            # 定义新的前向传播函数，包含LoRA增量
            # 使用默认参数显式绑定，避免闭包问题
            # layer1=original_forward: 原始线性层的前向传播
            # layer2=lora: LoRA适配器的前向传播
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 原始输出 + LoRA增量 = Wx + BAx
                # 这样既保留了预训练权重W，又添加了可训练的增量ΔW=BA
                return layer1(x) + layer2(x)

            # 替换模块的前向传播函数
            # 现在调用 module(x) 时会自动包含LoRA增量
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从文件加载LoRA权重到模型中
    
    该函数会从保存的文件中读取LoRA权重，并将它们加载到模型中对应的LoRA适配器。
    只加载LoRA权重，不影响原始模型的权重。
    
    Args:
        model: 已应用LoRA的PyTorch模型（需要先调用apply_lora）
        path: LoRA权重文件的路径
        
    工作流程：
    1. 加载保存的state_dict
    2. 遍历模型中的所有模块
    3. 找到有LoRA适配器的模块
    4. 从state_dict中提取对应的LoRA权重
    5. 加载到模块的LoRA适配器中
    
    注意：
    - 模型必须先调用apply_lora添加LoRA适配器
    - 只加载LoRA权重（A和B矩阵），不修改原始模型权重
    - 可以加载不同rank的LoRA权重，但需要匹配模型结构
    """
    # 从文件加载state_dict，并映射到模型所在的设备（CPU/GPU）
    state_dict = torch.load(path, map_location=model.device)
    
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 检查模块是否有LoRA适配器
        if hasattr(module, 'lora'):
            # 从state_dict中提取当前模块的LoRA权重
            # state_dict中的key格式为: "module_name.lora.A.weight", "module_name.lora.B.weight"
            # 需要提取出 "A.weight", "B.weight" 这样的key，以便加载到LoRA适配器
            lora_state = {
                k.replace(f'{name}.lora.', ''): v 
                for k, v in state_dict.items() 
                if f'{name}.lora.' in k
            }
            
            # 将提取的权重加载到模块的LoRA适配器中
            # 这会更新LoRA适配器的A和B矩阵的权重
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    保存模型中的LoRA权重到文件
    
    该函数会收集模型中所有LoRA适配器的权重，并将它们保存到文件中。
    只保存LoRA权重（A和B矩阵），不保存原始模型权重，大大减小文件大小。
    
    Args:
        model: 已应用LoRA的PyTorch模型
        path: 保存LoRA权重文件的路径
        
    工作流程：
    1. 创建一个空的state_dict
    2. 遍历模型中的所有模块
    3. 找到有LoRA适配器的模块
    4. 获取LoRA适配器的state_dict（包含A和B的权重）
    5. 添加模块名称前缀，保存到state_dict中
    6. 将所有LoRA权重保存到文件
    
    优势：
    - 文件大小小：只保存LoRA权重，通常只有几MB到几十MB
    - 模块化：可以为不同任务保存不同的LoRA权重
    - 可组合：可以加载多个LoRA权重进行组合
    
    注意：
    - 保存的权重包含模块的完整路径名
    - 加载时需要模型结构与保存时一致
    """
    # 创建空的state_dict用于存储所有LoRA权重
    state_dict = {}
    
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 检查模块是否有LoRA适配器
        if hasattr(module, 'lora'):
            # 获取LoRA适配器的state_dict（包含A.weight和B.weight）
            lora_state_dict = module.lora.state_dict()
            
            # 为每个权重key添加模块名称前缀
            # 例如：将 "A.weight" 转换为 "layers.0.attention.q_proj.lora.A.weight"
            # 这样可以唯一标识每个LoRA适配器的权重
            lora_state = {
                f'{name}.lora.{k}': v 
                for k, v in lora_state_dict.items()
            }
            
            # 将当前模块的LoRA权重添加到总的state_dict中
            state_dict.update(lora_state)
    
    # 将所有LoRA权重保存到文件
    # 文件通常很小，只包含LoRA的A和B矩阵权重
    torch.save(state_dict, path)
