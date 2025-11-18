"""
独立agent网络的MAVAC模型变体
每个agent有独立的网络参数，支持联邦学习
"""
from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead


@MODEL_REGISTRY.register('mavac_independent')
class MAVACIndependent(nn.Module):
    """
    Overview:
        多智能体VAC模型的独立网络版本。
        每个agent有独立的actor和critic网络参数，支持联邦学习中的参数平均。
        
    Note:
        与标准MAVAC不同，这个版本为每个agent创建独立的网络实例。
        可以通过FederatedAveraging中间件定期对参数进行平均。
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        actor_head_hidden_size: int = 256,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 512,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
        encoder: Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None,
    ) -> None:
        """
        Overview:
            初始化独立agent网络的MAVAC模型
        
        Arguments:
            - agent_obs_shape: 单个agent的观察空间
            - global_obs_shape: 全局观察空间
            - action_shape: 动作空间
            - agent_num: agent数量
            - 其他参数与标准MAVAC相同
        """
        super(MAVACIndependent, self).__init__()
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape = global_obs_shape
        self.agent_obs_shape = agent_obs_shape
        self.action_shape = action_shape
        self.agent_num = agent_num
        self.action_space = action_space
        
        # 为每个agent创建独立的网络
        self.agent_models = nn.ModuleList()
        
        for agent_id in range(agent_num):
            # 为每个agent创建独立的encoder和head
            if encoder:
                actor_encoder, critic_encoder = encoder
                # 如果提供了encoder，需要为每个agent创建副本
                actor_enc = self._clone_module(actor_encoder)
                critic_enc = self._clone_module(critic_encoder)
            else:
                actor_enc = nn.Sequential(
                    nn.Linear(agent_obs_shape, actor_head_hidden_size),
                    activation,
                )
                critic_enc = nn.Sequential(
                    nn.Linear(global_obs_shape, critic_head_hidden_size),
                    activation,
                )
            
            # 创建head
            critic_head = RegressionHead(
                critic_head_hidden_size, 1, critic_head_layer_num, 
                activation=activation, norm_type=norm_type
            )
            
            if action_space == 'discrete':
                actor_head = DiscreteHead(
                    actor_head_hidden_size, action_shape, actor_head_layer_num,
                    activation=activation, norm_type=norm_type
                )
            elif action_space == 'continuous':
                actor_head = ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type=sigma_type,
                    activation=activation,
                    norm_type=norm_type,
                    bound_type=bound_type
                )
            else:
                raise ValueError(f"不支持的动作空间: {action_space}")
            
            # 将每个agent的网络组织成ModuleList
            agent_actor = nn.ModuleList([actor_enc, actor_head])
            agent_critic = nn.ModuleList([critic_enc, critic_head])
            
            # 存储每个agent的模型
            agent_model = nn.ModuleDict({
                'actor': agent_actor,
                'critic': agent_critic
            })
            self.agent_models.append(agent_model)
        
        # 为了兼容性，保留actor和critic属性（指向第一个agent的网络）
        self.actor = self.agent_models[0]['actor']
        self.critic = self.agent_models[0]['critic']
    
    def _clone_module(self, module: nn.Module) -> nn.Module:
        """克隆一个模块（深拷贝）"""
        import copy
        return copy.deepcopy(module)
    
    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        """
        Overview:
            前向传播，为每个agent使用其独立的网络
        
        Arguments:
            - inputs: 输入字典，包含 'agent_state', 'global_state', 'action_mask' 等
            - mode: 前向模式
        
        Returns:
            - outputs: 输出字典，包含 'logit' 和/或 'value'
        """
        assert mode in self.mode, f"不支持的前向模式: {mode}"
        return getattr(self, mode)(inputs)
    
    def compute_actor(self, x: Dict) -> Dict:
        """计算actor输出，每个agent使用自己的网络"""
        agent_state = x['agent_state']  # shape: (B, M, obs_dim)
        batch_size = agent_state.shape[0]
        
        if self.action_space == 'discrete':
            action_mask = x.get('action_mask', None)
        
        # 为每个agent分别计算
        logits = []
        for agent_id in range(self.agent_num):
            agent_obs = agent_state[:, agent_id, :]  # (B, obs_dim)
            agent_model = self.agent_models[agent_id]
            
            # 通过actor encoder和head
            x_enc = agent_model['actor'][0](agent_obs)
            x_head = agent_model['actor'][1](x_enc)
            
            if self.action_space == 'discrete':
                logit = x_head['logit']  # (B, action_dim)
                if action_mask is not None:
                    mask = action_mask[:, agent_id, :]  # (B, action_dim)
                    logit[mask == 0.0] = -99999999
                logits.append(logit)
            else:
                logits.append(x_head)
        
        # 堆叠所有agent的输出
        logit = torch.stack(logits, dim=1)  # (B, M, action_dim)
        return {'logit': logit}
    
    def compute_critic(self, x: Dict) -> Dict:
        """计算critic输出，每个agent使用自己的网络"""
        global_state = x['global_state']  # shape: (B, M, global_obs_dim)
        batch_size = global_state.shape[0]
        
        # 为每个agent分别计算
        values = []
        for agent_id in range(self.agent_num):
            agent_global_obs = global_state[:, agent_id, :]  # (B, global_obs_dim)
            agent_model = self.agent_models[agent_id]
            
            # 通过critic encoder和head
            x_enc = agent_model['critic'][0](agent_global_obs)
            x_head = agent_model['critic'][1](x_enc)
            value = x_head['pred']  # (B, 1)
            values.append(value.squeeze(-1))  # (B,)
        
        # 堆叠所有agent的输出
        value = torch.stack(values, dim=1)  # (B, M)
        return {'value': value}
    
    def compute_actor_critic(self, x: Dict) -> Dict:
        """同时计算actor和critic输出"""
        actor_output = self.compute_actor(x)
        critic_output = self.compute_critic(x)
        return {
            'logit': actor_output['logit'],
            'value': critic_output['value']
        }
    
    def get_agent_models(self) -> list:
        """
        Overview:
            获取所有agent的模型列表，用于联邦学习参数平均
        
        Returns:
            - agent_models: 所有agent模型的列表（每个模型包含actor和critic）
        """
        return list(self.agent_models)

