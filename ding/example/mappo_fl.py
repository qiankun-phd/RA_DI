import os
import gym
from ditk import logging
from ding.model import MAVACIndependent
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger, termination_checker, FederatedAveraging
from ding.utils import set_pkg_seed
from dizoo.semcom.config.multi_general_mappo_config import main_config, create_config
from dizoo.semcom.envs.multi_general_env import MultiGeneralEnv


def main():
    """主函数"""
    logging.getLogger().setLevel(logging.INFO)
    
    # 设置随机数种子，匹配原始TensorFlow代码
    # 原始代码: args.seed=1 (用于Python/Numpy/Hash/环境), args.set_random_seed=2 (用于TensorFlow内部随机操作)
    seed_value = 1  # 匹配原始代码的args.seed，用于Python/Numpy/Hash/环境
    torch_seed = 2  # 匹配原始代码的args.set_random_seed，用于PyTorch网络初始化
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 匹配原始代码的设置
    
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, seed=seed_value)
    
    # 初始化DI-engine
    ding_init(cfg)
    
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        # 创建环境（使用Semcom环境）
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: MultiGeneralEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: MultiGeneralEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        # 设置环境种子（匹配原始代码，在环境创建后、首次reset前设置）
        collector_env.seed(seed_value, dynamic_seed=False)  # 固定种子，匹配原始代码的args.seed=1
        evaluator_env.seed(seed_value, dynamic_seed=False)  # 固定种子，匹配原始代码的args.seed=1
        
        # 设置全局随机数种子（匹配原始代码）
        # Python/Numpy使用seed_value=1，PyTorch使用torch_seed=2
        import random
        import numpy as np
        import torch
        random.seed(seed_value)  # 匹配原始代码的random.seed(args.seed)
        np.random.seed(seed_value)  # 匹配原始代码的np.random.seed(args.seed)
        torch.manual_seed(torch_seed)  # 匹配原始代码的set_random_seed(args.set_random_seed)
        if cfg.policy.cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)  
        
        model = MAVACIndependent(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)


        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=100))
        task.use(FederatedAveraging(cfg=cfg, policy=policy, fl_freq=100))  # 匹配原始TensorFlow代码的100个episode (9 * 11 episodes/train_iter ≈ 100 episodes)
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.use(online_logger(record_train_iter=True, train_show_freq=10))
        task.use(termination_checker(max_train_iter=int(1e3)))
        task.run()


if __name__ == "__main__":
    main()

