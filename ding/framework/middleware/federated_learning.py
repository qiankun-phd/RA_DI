"""
联邦学习中间件：用于多智能体环境中的参数平均
实现每个agent独立网络参数的定期平均（Federated Learning）
"""
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from ditk import logging
from ding.framework import task
from time import sleep, time

if TYPE_CHECKING:
    from ding.framework.context import Context
    from torch.nn import Module


class FederatedAveraging:
    """
    Overview:
        多智能体联邦学习参数平均中间件。
        支持每个agent有独立的网络参数，并定期对所有agent的参数进行平均。
    
    Arguments:
        - agent_models (:obj:`List[torch.nn.Module]`): 每个agent的模型列表，长度为agent_num
        - agent_num (:obj:`int`): agent数量
        - aggregation_freq (:obj:`int`): 参数聚合频率，每N个训练步骤进行一次平均
        - aggregation_mode (:obj:`str`): 聚合模式，'average'（简单平均）或 'weighted_average'（加权平均）
        - agent_weights (:obj:`Optional[List[float]]`): 每个agent的权重（用于加权平均），如果为None则使用均匀权重
    """
    
    def __init__(
        self,
        agent_models: List["Module"],
        agent_num: int,
        aggregation_freq: int = 10,
        aggregation_mode: str = 'average',
        agent_weights: Optional[List[float]] = None,
    ) -> None:
        self._agent_models = agent_models
        self._agent_num = agent_num
        self._aggregation_freq = aggregation_freq
        self._aggregation_mode = aggregation_mode
        self._step_count = 0
        
        # 验证模型数量
        if len(agent_models) != agent_num:
            raise ValueError(
                f"模型数量 ({len(agent_models)}) 与agent数量 ({agent_num}) 不匹配"
            )
        
        # 设置权重
        if agent_weights is None:
            self._agent_weights = [1.0 / agent_num] * agent_num
        else:
            if len(agent_weights) != agent_num:
                raise ValueError(f"权重数量 ({len(agent_weights)}) 与agent数量 ({agent_num}) 不匹配")
            # 归一化权重
            total_weight = sum(agent_weights)
            self._agent_weights = [w / total_weight for w in agent_weights]
        
        # 验证所有模型结构相同
        self._validate_model_structure()
        
        logging.info(
            f"初始化联邦学习参数平均器: agent_num={agent_num}, "
            f"aggregation_freq={aggregation_freq}, mode={aggregation_mode}"
        )
    
    def _validate_model_structure(self):
        """验证所有agent模型的结构是否相同"""
        if len(self._agent_models) == 0:
            return
        
        reference_state_dict = self._agent_models[0].state_dict()
        for i, model in enumerate(self._agent_models[1:], 1):
            model_state_dict = model.state_dict()
            if set(reference_state_dict.keys()) != set(model_state_dict.keys()):
                raise ValueError(f"Agent {i} 的模型结构与 Agent 0 不一致")
            for key in reference_state_dict.keys():
                if reference_state_dict[key].shape != model_state_dict[key].shape:
                    raise ValueError(
                        f"Agent {i} 的参数 {key} 的形状与 Agent 0 不一致: "
                        f"{model_state_dict[key].shape} vs {reference_state_dict[key].shape}"
                    )
    
    def _average_parameters(self):
        """
        Overview:
            对所有agent的参数进行平均
        """
        if self._aggregation_mode == 'average':
            self._simple_average()
        elif self._aggregation_mode == 'weighted_average':
            self._weighted_average()
        else:
            raise ValueError(f"不支持的聚合模式: {self._aggregation_mode}")
    
    def _simple_average(self):
        """简单平均：所有agent参数的平均值"""
        # 获取第一个模型的参数作为基准
        averaged_params = {}
        first_model = self._agent_models[0]
        
        # 处理ModuleDict结构（如MAVACIndependent中的agent模型）
        if hasattr(first_model, 'keys'):
            # 如果是ModuleDict，需要遍历所有子模块
            for submodule_name in first_model.keys():
                submodule = first_model[submodule_name]
                for key, param in submodule.named_parameters():
                    full_key = f"{submodule_name}.{key}" if submodule_name else key
                    averaged_params[full_key] = param.data.clone()
        else:
            # 普通Module
            for key, param in first_model.named_parameters():
                averaged_params[key] = param.data.clone()
        
        # 累加所有agent的参数
        for model in self._agent_models[1:]:
            if hasattr(model, 'keys'):
                # ModuleDict结构
                for submodule_name in model.keys():
                    submodule = model[submodule_name]
                    for key, param in submodule.named_parameters():
                        full_key = f"{submodule_name}.{key}" if submodule_name else key
                        if full_key in averaged_params:
                            averaged_params[full_key] += param.data
            else:
                # 普通Module
                for key, param in model.named_parameters():
                    if key in averaged_params:
                        averaged_params[key] += param.data
        
        # 除以agent数量得到平均值
        for key in averaged_params:
            averaged_params[key] /= self._agent_num
        
        # 将平均后的参数赋值给所有agent
        for model in self._agent_models:
            if hasattr(model, 'keys'):
                # ModuleDict结构
                for submodule_name in model.keys():
                    submodule = model[submodule_name]
                    for key, param in submodule.named_parameters():
                        full_key = f"{submodule_name}.{key}" if submodule_name else key
                        if full_key in averaged_params:
                            param.data.copy_(averaged_params[full_key])
            else:
                # 普通Module
                for key, param in model.named_parameters():
                    if key in averaged_params:
                        param.data.copy_(averaged_params[key])
        
        logging.info(f"完成参数平均聚合 (简单平均模式)")
    
    def _weighted_average(self):
        """加权平均：根据权重对agent参数进行加权平均"""
        # 初始化加权平均参数
        averaged_params = {}
        first_model = self._agent_models[0]
        
        # 处理ModuleDict结构
        if hasattr(first_model, 'keys'):
            for submodule_name in first_model.keys():
                submodule = first_model[submodule_name]
                for key, param in submodule.named_parameters():
                    full_key = f"{submodule_name}.{key}" if submodule_name else key
                    averaged_params[full_key] = param.data.clone() * self._agent_weights[0]
        else:
            for key, param in first_model.named_parameters():
                averaged_params[key] = param.data.clone() * self._agent_weights[0]
        
        # 累加所有agent的加权参数
        for i, model in enumerate(self._agent_models[1:], 1):
            if hasattr(model, 'keys'):
                for submodule_name in model.keys():
                    submodule = model[submodule_name]
                    for key, param in submodule.named_parameters():
                        full_key = f"{submodule_name}.{key}" if submodule_name else key
                        if full_key in averaged_params:
                            averaged_params[full_key] += param.data * self._agent_weights[i]
            else:
                for key, param in model.named_parameters():
                    if key in averaged_params:
                        averaged_params[key] += param.data * self._agent_weights[i]
        
        # 将加权平均后的参数赋值给所有agent
        for model in self._agent_models:
            if hasattr(model, 'keys'):
                for submodule_name in model.keys():
                    submodule = model[submodule_name]
                    for key, param in submodule.named_parameters():
                        full_key = f"{submodule_name}.{key}" if submodule_name else key
                        if full_key in averaged_params:
                            param.data.copy_(averaged_params[full_key])
            else:
                for key, param in model.named_parameters():
                    if key in averaged_params:
                        param.data.copy_(averaged_params[key])
        
        logging.info(f"完成参数平均聚合 (加权平均模式, 权重={self._agent_weights})")
    
    def __call__(self, ctx: "Context") -> Any:
        """
        Overview:
            中间件调用函数，在每个训练步骤后检查是否需要聚合参数
        """
        self._step_count += 1
        
        # 检查是否到达聚合频率
        if self._step_count % self._aggregation_freq == 0:
            logging.info(f"步骤 {self._step_count}: 开始参数聚合...")
            self._average_parameters()
            logging.info(f"步骤 {self._step_count}: 参数聚合完成")
        
        yield


class MultiAgentFederatedAveraging:
    """
    Overview:
        多智能体联邦学习参数平均中间件（适用于MAPPO场景）。
        从单个MAVAC模型中提取每个agent的参数并进行平均。
        
    Note:
        这个版本假设使用共享参数的MAVAC模型，但可以通过hook机制实现每个agent的参数分离和平均。
        如果需要完全独立的agent网络，需要修改MAVAC模型结构。
    
    Arguments:
        - model (:obj:`torch.nn.Module`): 多智能体模型（如MAVAC）
        - agent_num (:obj:`int`): agent数量
        - aggregation_freq (:obj:`int`): 参数聚合频率
        - use_actor_only (:obj:`bool`): 是否只对actor网络进行平均（默认False，对actor和critic都平均）
    """
    
    def __init__(
        self,
        model: "Module",
        agent_num: int,
        aggregation_freq: int = 10,
        use_actor_only: bool = False,
    ) -> None:
        self._model = model
        self._agent_num = agent_num
        self._aggregation_freq = aggregation_freq
        self._use_actor_only = use_actor_only
        self._step_count = 0
        
        # 检查模型是否支持多智能体
        if not hasattr(model, 'actor') or not hasattr(model, 'critic'):
            raise ValueError("模型必须包含 'actor' 和 'critic' 属性（如MAVAC模型）")
        
        logging.info(
            f"初始化多智能体联邦学习参数平均器: agent_num={agent_num}, "
            f"aggregation_freq={aggregation_freq}, use_actor_only={use_actor_only}"
        )
    
    def _average_parameters(self):
        """
        Overview:
            对模型参数进行平均（在MAPPO中，所有agent共享参数，这里主要是占位）
        
        Note:
            在标准的MAPPO实现中，所有agent共享同一套参数，所以不需要平均。
            如果需要实现真正的联邦学习，需要：
            1. 修改MAVAC模型，为每个agent创建独立的网络
            2. 或者在训练过程中为每个agent维护独立的参数副本
        """
        # 在标准MAPPO中，参数是共享的，所以这里只是记录日志
        # 如果需要真正的联邦学习，需要修改模型结构
        logging.warning(
            "标准MAPPO使用参数共享，无法进行参数平均。"
            "如需实现联邦学习，请使用独立agent网络的模型结构。"
        )
    
    def __call__(self, ctx: "Context") -> Any:
        """中间件调用函数"""
        self._step_count += 1
        if self._step_count % self._aggregation_freq == 0:
            self._average_parameters()
        yield

