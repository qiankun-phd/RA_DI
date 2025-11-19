from easydict import EasyDict

n_user = 6
n_rb = 10
collector_env_num = 1  # 单个环境也可以运行，代码已修复支持 batch_size=1
evaluator_env_num = 1  # 评估时使用单个环境即可
max_env_step = 1e6

# Calculate observation dimensions
# state_dim = n_rb * 4 + 2
# cellular_fast (n_rb) + cellular_abs (n_rb) + channel_choice (n_rb) + user_vector (n_rb) + success (1) + time (1)
agent_obs_shape = n_rb * 4 + 2
# No global_state - each agent uses its own agent_state for both actor and critic
# In FL scenario, each agent has different state, allowing different strategies
global_obs_shape = agent_obs_shape  # Critic also uses agent_state, so same shape

main_config = dict(
    exp_name='multi_general_mappo_seed0',
    env=dict(
        n_user=n_user,
        n_rb=n_rb,
        max_episode_steps=100,
        agent_obs_only=False,  # Return dict with agent_state only (no global_state)
        normalize_reward=False,  # Set to True to enable reward normalization
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=float('inf'),  # 禁用基于episode return的提前停止，使用termination_checker控制训练时长
        manager=dict(shared_memory=True, ),
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='hybrid',  # hybrid: discrete(rb) + continuous(power)
        model=dict(
            action_space='hybrid',
            agent_num=n_user,
            agent_obs_shape=agent_obs_shape,
            global_obs_shape=global_obs_shape,
            action_shape=EasyDict({
                'action_type_shape': n_rb,  # 离散：选择rb (0 到 n_rb-1)
                'action_args_shape': 1,     # 连续：选择功率
            }),

            encoder_hidden_size_list=[512, 256, 128],  # 3层共享编码器结构
            actor_head_hidden_size=256,  # Actor head输入维度（来自encoder的256维输出）
            critic_head_hidden_size=128,  # Critic head输入维度（来自encoder的128维输出）
            actor_head_layer_num=2,  # 2 layers for actor head
            critic_head_layer_num=1,  # 1 layer for critic head
            norm_type='BN',  # Batch Normalization，匹配TensorFlow实现
            bound_type='tanh',  # mu使用tanh激活，匹配TensorFlow实现
            # Note: type and hidden_size_list are not needed when manually creating MAVACIndependent
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=32,
            batch_size=100,
            learning_rate=1e-6,  # lr_main = 1e-6
            value_weight=0.5,  # weight_for_L_vf = 0.5
            entropy_weight=0.01,  # weight_for_entropy = 0.01
            clip_ratio=0.5,  # epsilon = 0.5
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(
            n_sample=100,
            unroll_len=1,
            env_num=collector_env_num,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.9,  # gamma = 0.9
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.98,  # lambda_advantage = 0.98
        ),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=5, ),  # Evaluate every 5 training iterations
        ),
        other=dict(),
    ),
)
main_config = EasyDict(main_config)

create_config = dict(
    env=dict(
        import_names=['dizoo.semcom.envs.multi_general_env'],
        type='multi_general_env',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

multi_general_mappo_config = main_config
multi_general_mappo_create_config = create_config

if __name__ == '__main__':
    # Use command line to run: ding -m serial_onpolicy -c dizoo.semcom.config.multi_general_mappo_config -s 0
    # Or set PYTHONPATH and run: python -m dizoo.semcom.config.multi_general_mappo_config
    print("Configuration loaded successfully!")
    print(f"Experiment name: {main_config.exp_name}")
    print(f"Agent number: {main_config.policy.model.agent_num}")
    print(f"Agent obs shape: {main_config.policy.model.agent_obs_shape}")
    print(f"Global obs shape: {main_config.policy.model.global_obs_shape}")
    print(f"Action shape: {main_config.policy.model.action_shape}")
    print("\nTo run training, use:")
    print("  ding -m serial_onpolicy -c dizoo.semcom.config.multi_general_mappo_config -s 0")
    print("\nOr set PYTHONPATH=DI-engine and run:")
    print("  python -m dizoo.semcom.config.multi_general_mappo_config")

