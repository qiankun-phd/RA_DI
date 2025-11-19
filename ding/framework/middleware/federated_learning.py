"""
联邦学习中间件：用于多智能体环境中的参数平均
实现每个agent独立网络参数的定期平均（Federated Learning）
"""
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from easydict import EasyDict
from ditk import logging
from ding.framework import task
from time import sleep, time

if TYPE_CHECKING:
    from ding.framework.context import Context
    from torch.nn import Module
    from ding.policy import Policy


class FederatedAveraging:
    """
    Overview:
        多智能体联邦学习参数平均中间件。
        支持每个agent有独立的网络参数，并定期对所有agent的参数进行平均。
    
    Arguments:
        - cfg (:obj:`EasyDict`): 配置字典，包含以下字段：
            - policy.model.agent_num: agent数量
            - policy.federated_learning.aggregation_mode: 聚合模式，'average'（简单平均）或 'weighted_average'（加权平均），默认为 'average'
            - policy.federated_learning.agent_weights: 每个agent的权重（用于加权平均），如果为None则使用均匀权重
        - policy (:obj:`Policy`): 策略对象，其模型必须支持 `get_agent_models()` 方法（如MAVACIndependent）
        - fl_freq (:obj:`int`): 参数聚合频率，每N个训练步骤进行一次平均，默认为0（从cfg中读取，如果cfg中没有则使用默认值10）
    """
    
    def __init__(
        self,
        cfg: EasyDict,
        policy: "Policy",
        fl_freq: int = 0,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self._step_count = 0
        
        # 从policy中获取模型（优先使用_learn_model，如果没有则使用_model）
        if hasattr(policy, '_learn_model') and policy._learn_model is not None:
            model = policy._learn_model
        elif hasattr(policy, '_model') and policy._model is not None:
            model = policy._model
        else:
            raise ValueError("无法从policy中获取模型，请确保policy已正确初始化")
        
        # 检查模型是否支持get_agent_models方法
        if not hasattr(model, 'get_agent_models'):
            raise ValueError(
                "Model must implement `get_agent_models()`. "
                "Please use a model with independent agent parameters (e.g., MAVACIndependent) instead of the shared-parameter MAPPO model."
            )
        
        # 获取所有agent的模型
        self._agent_models = model.get_agent_models()
        
        # 从配置中获取agent数量
        self._agent_num = cfg.policy.model.agent_num
        
        # 验证模型数量
        if len(self._agent_models) != self._agent_num:
            raise ValueError(
                f"Number of agent models ({len(self._agent_models)}) does not match configured agent_num ({self._agent_num})."
            )
        
        # 获取聚合频率（优先使用参数，其次从cfg读取，最后使用默认值）
        if fl_freq > 0:
            self._aggregation_freq = fl_freq
        else:
            # 从配置中读取，如果配置中没有则使用默认值10
            fl_cfg = getattr(cfg.policy, 'federated_learning', None)
            if fl_cfg and hasattr(fl_cfg, 'aggregation_freq'):
                self._aggregation_freq = fl_cfg.aggregation_freq
            elif isinstance(fl_cfg, dict) and 'aggregation_freq' in fl_cfg:
                self._aggregation_freq = fl_cfg['aggregation_freq']
            else:
                self._aggregation_freq = 10
        
        # 从配置中获取聚合模式和权重
        fl_cfg = getattr(cfg.policy, 'federated_learning', None)
        if fl_cfg:
            if hasattr(fl_cfg, 'aggregation_mode'):
                self._aggregation_mode = fl_cfg.aggregation_mode
            elif isinstance(fl_cfg, dict) and 'aggregation_mode' in fl_cfg:
                self._aggregation_mode = fl_cfg['aggregation_mode']
            else:
                self._aggregation_mode = 'average'
            
            if hasattr(fl_cfg, 'agent_weights'):
                agent_weights = fl_cfg.agent_weights
            elif isinstance(fl_cfg, dict) and 'agent_weights' in fl_cfg:
                agent_weights = fl_cfg['agent_weights']
            else:
                agent_weights = None
        else:
            self._aggregation_mode = 'average'
            agent_weights = None
        
        # 设置权重
        if agent_weights is None:
            self._agent_weights = [1.0 / self._agent_num] * self._agent_num
        else:
            if len(agent_weights) != self._agent_num:
                raise ValueError(f"权重数量 ({len(agent_weights)}) 与agent数量 ({self._agent_num}) 不匹配")
            # 归一化权重
            total_weight = sum(agent_weights)
            self._agent_weights = [w / total_weight for w in agent_weights]
        
        # 验证所有模型结构相同
        self._validate_model_structure()
        
        logging.info(
            f"Initialize federated averaging: agent_num={self._agent_num}, "
            f"aggregation_freq={self._aggregation_freq}, mode={self._aggregation_mode}"
        )
    
    def _validate_model_structure(self):
        """验证所有agent模型的结构是否相同"""
        if len(self._agent_models) == 0:
            return
        
        reference_state_dict = self._agent_models[0].state_dict()
        for i, model in enumerate(self._agent_models[1:], 1):
            model_state_dict = model.state_dict()
            if set(reference_state_dict.keys()) != set(model_state_dict.keys()):
                raise ValueError(f"Agent {i} model structure is inconsistent with agent 0.")
            for key in reference_state_dict.keys():
                if reference_state_dict[key].shape != model_state_dict[key].shape:
                    raise ValueError(
                        f"Shape mismatch for parameter {key} between agent {i} and agent 0: "
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
            raise ValueError(f"Unsupported aggregation mode: {self._aggregation_mode}")
    
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
        
        logging.info("Federated averaging done (simple average).")
    
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
        
        logging.info(f"Federated averaging done (weighted average, weights={self._agent_weights}).")
    
    def __call__(self, ctx: "Context") -> Any:
        """
        Overview:
            中间件调用函数，在每个训练步骤后检查是否需要聚合参数
        """
        self._step_count += 1

        # 优先使用 ctx.train_iter 作为日志和触发参考，以便与训练迭代保持一致
        current_iter = getattr(ctx, "train_iter", None)
        counter_for_trigger = current_iter if current_iter is not None else self._step_count
        log_prefix = (
            f"Train iter {current_iter}"
            if current_iter is not None else
            f"Step {self._step_count}"
        )

        # 检查是否到达聚合频率
        if counter_for_trigger % self._aggregation_freq == 0:
            logging.info(f"{log_prefix}: start federated averaging.")
            self._average_parameters()
            logging.info(f"{log_prefix}: federated averaging finished.")
        
        yield