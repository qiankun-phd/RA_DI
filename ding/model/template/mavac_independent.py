"""
独立agent网络的MAVAC模型变体
每个agent有独立的网络参数，支持联邦学习
"""
from typing import Union, Dict, Tuple, Optional
import torch
import torch.nn as nn
from easydict import EasyDict

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, FCEncoder
from ding.torch_utils.network import fc_block
from ding.torch_utils.network.normalization import build_normalization


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
        encoder_hidden_size_list: Optional[SequenceType] = None,  # For 3-layer shared encoder (512->256->128)
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
        # For FL scenario without global_state, critic also uses agent_state
        # So global_obs_shape should equal agent_obs_shape
        if global_obs_shape is None or global_obs_shape == 0:
            global_obs_shape = agent_obs_shape
        else:
            global_obs_shape: int = squeeze(global_obs_shape)
        
        # 处理hybrid action space的action_shape
        if action_space == 'hybrid':
            if isinstance(action_shape, (dict, EasyDict)):
                action_shape = EasyDict(action_shape)
            else:
                raise ValueError("For hybrid action_space, action_shape must be EasyDict with 'action_type_shape' and 'action_args_shape'")
            action_shape.action_args_shape = squeeze(action_shape.action_args_shape)
            action_shape.action_type_shape = squeeze(action_shape.action_type_shape)
        else:
            action_shape: int = squeeze(action_shape)
        
        self.global_obs_shape = global_obs_shape
        self.agent_obs_shape = agent_obs_shape
        self.action_shape = action_shape
        self.agent_num = agent_num
        self.action_space = action_space
        
        # 为每个agent创建独立的网络
        self.agent_models = nn.ModuleList()
        
        # 创建encoder的辅助函数（参考vac.py的实现，支持BatchNorm）
        def new_encoder(hidden_size_list, activation, norm_type):
            if hidden_size_list:
                # 如果指定了norm_type，手动创建带BatchNorm的encoder（匹配TensorFlow实现）
                if norm_type and norm_type.upper() in ['BN', 'BATCHNORM']:
                    layers = []
                    # 第一层: obs -> hidden_size_list[0]
                    layers.append(fc_block(
                        agent_obs_shape, hidden_size_list[0],
                        activation=activation, norm_type='BN'  # fc_block会根据dim=1自动构建BN1
                    ))
                    # 后续层
                    for i in range(len(hidden_size_list) - 1):
                        layers.append(fc_block(
                            hidden_size_list[i], hidden_size_list[i + 1],
                            activation=activation, norm_type='BN'  # fc_block会根据dim=1自动构建BN1
                        ))
                    return nn.Sequential(*layers)
                else:
                    # 使用FCEncoder创建多层结构（不带BatchNorm）
                    return FCEncoder(
                        obs_shape=agent_obs_shape,
                        hidden_size_list=hidden_size_list,
                        activation=activation,
                        norm_type=norm_type
                    )
            else:
                return None
        
        for agent_id in range(agent_num):
            # 为每个agent创建独立的encoder和head
            if encoder:
                actor_encoder, critic_encoder = encoder
                # 如果提供了encoder，需要为每个agent创建副本
                actor_enc = self._clone_module(actor_encoder)
                critic_enc = self._clone_module(critic_encoder)
            else:
                if encoder_hidden_size_list:
                    # 共享编码器：obs -> 512 -> 256 (前两层)
                    # Actor分支：从256维输出 (mu, sigma, RB)
                    # Critic分支：256 -> 128 -> v (在RegressionHead中处理)
                    shared_hidden_list = encoder_hidden_size_list[:2]  # [512, 256]
                    shared_encoder = new_encoder(shared_hidden_list, activation, norm_type)
                    
                    # Actor encoder: 直接使用共享编码器（到256维）
                    actor_enc = shared_encoder
                    
                    # Critic encoder: 也使用共享编码器（到256维），256->128在RegressionHead中处理
                    critic_enc = shared_encoder
                else:
                    # 默认单层结构（不共享）
                    actor_enc = nn.Sequential(
                        nn.Linear(agent_obs_shape, actor_head_hidden_size),
                        activation,
                    )
                    # Critic uses agent_state instead of global_state (same shape as actor)
                    critic_enc = nn.Sequential(
                        nn.Linear(agent_obs_shape, critic_head_hidden_size),
                        activation,
                    )
            
            # 创建head
            # Critic head: 如果使用共享编码器，输入256维，通过hidden_size=128添加256->128层
            if encoder_hidden_size_list:
                critic_head_input_size = encoder_hidden_size_list[1]  # 256
                critic_head_hidden_size_param = encoder_hidden_size_list[2]  # 128
            else:
                critic_head_input_size = critic_head_hidden_size
                critic_head_hidden_size_param = critic_head_hidden_size
            critic_head = RegressionHead(
                critic_head_input_size, 1, critic_head_layer_num,
                hidden_size=critic_head_hidden_size_param,  # 256 -> 128 -> 1
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
            elif action_space == 'hybrid':
                # hybrid action space: action_type(discrete) + action_args(continuous)
                # action_type: 选择rb (0 到 n_rb-1)
                # action_args: 选择功率 (连续值)
                actor_action_type = DiscreteHead(
                    actor_head_hidden_size,
                    action_shape.action_type_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type,
                )
                actor_action_args = ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape.action_args_shape,
                    actor_head_layer_num,
                    sigma_type=sigma_type,
                    activation=activation,
                    norm_type=norm_type,
                    bound_type=bound_type,
                    hidden_size=actor_head_hidden_size,
                )
                actor_head = nn.ModuleList([actor_action_type, actor_action_args])
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
        action_mask = x.get('action_mask', None)
        
        # 为每个agent分别计算（由于参数独立，必须循环处理）
        if self.action_space == 'discrete':
            logits = []
            for agent_id in range(self.agent_num):
                agent_obs = agent_state[:, agent_id, :]  # (B, obs_dim)
                agent_model = self.agent_models[agent_id]
                x_enc = agent_model['actor'][0](agent_obs)
                x_head = agent_model['actor'][1](x_enc)
                logit = x_head['logit']  # (B, action_dim)
                if action_mask is not None:
                    logit[action_mask[:, agent_id, :] == 0.0] = -99999999
                logits.append(logit)
            return {'logit': torch.stack(logits, dim=1)}  # (B, M, action_dim)
        elif self.action_space == 'continuous':
            mus = []
            sigmas = []
            for agent_id in range(self.agent_num):
                agent_obs = agent_state[:, agent_id, :]  # (B, obs_dim)
                agent_model = self.agent_models[agent_id]
                x_enc = agent_model['actor'][0](agent_obs)
                x_head = agent_model['actor'][1](x_enc)  # {'mu': ..., 'sigma': ...}
                mus.append(x_head['mu'])  # (B, action_dim)
                sigmas.append(x_head['sigma'])  # (B, action_dim)
            return {'logit': {'mu': torch.stack(mus, dim=1), 'sigma': torch.stack(sigmas, dim=1)}}
        elif self.action_space == 'hybrid':
            # hybrid action space: action_type(discrete) + action_args(continuous)
            # actor_head 是 ModuleList[actor_action_type, actor_action_args]
            action_types = []
            action_args_mus = []
            action_args_sigmas = []
            for agent_id in range(self.agent_num):
                agent_obs = agent_state[:, agent_id, :]  # (B, obs_dim)
                agent_model = self.agent_models[agent_id]
                x_enc = agent_model['actor'][0](agent_obs)
                actor_head = agent_model['actor'][1]  # ModuleList[actor_action_type, actor_action_args]
                # 分别调用两个head
                action_type_output = actor_head[0](x_enc)  # {'logit': ...}
                action_args_output = actor_head[1](x_enc)  # {'mu': ..., 'sigma': ...}
                action_types.append(action_type_output['logit'])  # (B, action_type_dim)
                action_args_mus.append(action_args_output['mu'])  # (B, action_args_dim)
                action_args_sigmas.append(action_args_output['sigma'])  # (B, action_args_dim)
            return {
                'logit': {
                    'action_type': torch.stack(action_types, dim=1),  # (B, M, action_type_dim)
                    'action_args': {
                        'mu': torch.stack(action_args_mus, dim=1),  # (B, M, action_args_dim)
                        'sigma': torch.stack(action_args_sigmas, dim=1)  # (B, M, action_args_dim)
                    }
                }
            }
        else:
            raise ValueError(f"Unsupported action_space: {self.action_space}")
    
    def compute_critic(self, x: Dict) -> Dict:
        """计算critic输出，每个agent使用自己的网络和状态"""
        # Use agent_state instead of global_state (each agent has different state)
        agent_state = x['agent_state']  # shape: (B, M, agent_obs_dim)
        
        # 为每个agent分别计算（每个agent使用自己的状态）
        values = []
        for agent_id in range(self.agent_num):
            agent_obs = agent_state[:, agent_id, :]  # (B, agent_obs_dim)
            agent_model = self.agent_models[agent_id]
            
            # 通过critic encoder和head（使用agent自己的状态）
            x_enc = agent_model['critic'][0](agent_obs)
            x_head = agent_model['critic'][1](x_enc)
            value = x_head['pred']  # (B, 1) 或 (B,)
            # 确保 value 是 (B, 1) 形状，以便后续 stack
            if value.dim() == 1:
                value = value.unsqueeze(-1)  # (B,) -> (B, 1)
            elif value.dim() == 0:
                value = value.unsqueeze(0).unsqueeze(-1)  # scalar -> (1, 1)
            values.append(value)
        
        # 堆叠所有agent的输出: (B, 1) -> (B, M, 1) -> (B, M)
        value = torch.stack(values, dim=1).squeeze(-1)  # (B, M)
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

