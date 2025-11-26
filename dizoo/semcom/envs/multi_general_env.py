"""
General multi-agent semantic communication environment.
This file wraps the provided user communication simulator into a DI-engine BaseEnv.
"""
from typing import List, Tuple, Union, Optional

import math
import random

import numpy as np
import gym

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY


class BSchannels:
    """Simulator of the downlink channels."""

    def __init__(self) -> None:
        self.h_bs = 5  # 匹配indoor: h_bs=5
        self.h_ms = 1.5
        self.Decorrelation_distance = 25
        self.bs_position = [[12.5, 12.5]]  # 匹配indoor: BS_position=[[12.5, 12.5]] (25x25网格的中心)
        # carrier frequencies in GHz
        self.fc = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    def get_path_loss(self, position_a: List[float]) -> np.ndarray:
        path_loss = np.zeros((len(self.bs_position), len(self.fc)))
        for i in range(len(self.bs_position)):
            for k in range(len(self.fc)):
                d1 = abs(position_a[0] - self.bs_position[i][0])
                d2 = abs(position_a[1] - self.bs_position[i][1])
                d_3d = math.sqrt(math.hypot(d1, d2)**2 + (self.h_bs - self.h_ms)**2)
                path_loss[i, k] = 32.4 + 20 * np.log10(self.fc[k]) + 31.9 * np.log10(d_3d)
        return path_loss

    def get_shadowing(self, delta_distance: float, shadowing: float) -> float:
        coef = np.exp(-1 * (delta_distance / self.Decorrelation_distance))
        return (
            np.multiply(coef, shadowing) + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance)))
            * np.random.normal(0, 1)
        )


class User:
    """User simulator that keeps track of motion."""

    def __init__(self, start_position: List[float], start_direction: str, velocity: float) -> None:
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity


@ENV_REGISTRY.register('semcom_general')
class Environ:

    def __init__(self, n_user: int, n_rb: int) -> None:
        self.width = 25  # 匹配indoor: width=25
        self.height = 25  # 匹配indoor: height=25
        self.bs_channels = BSchannels()
        self.users: List[User] = []
        self.demand: List = []
        self.cellular_shadowing: np.ndarray = np.zeros(0)
        self.delta_distance: np.ndarray = np.zeros(0)
        self.cellular_channels_abs: np.ndarray = np.zeros((0, 0))
        self.cellular_power_db_list = [24, 21, 18, 15, 12, 9, 6, 3, 0]
        self.sig2_db = -160
        self.bs_ant_gain = 8
        self.bs_noise_figure = 5
        self.user_ant_gain = 3
        self.user_noise_figure = 9
        self.n_RB = n_rb
        self.n_User = n_user
        self.time_fast = 0.001
        self.time_slow = 0.1
        self.bandwidth = int(1e6)
        self.BW = [0.18, 0.18, 0.36, 0.36, 0.36, 0.72, 0.72, 0.72, 1.44, 1.44]
        self.channel_choice = np.zeros([self.n_RB])
        self.success = np.zeros([self.n_User])
        self.sig2 = [i * 1e6 * 10**(self.sig2_db / 10) for i in self.BW]
        # agent ids for multi-agent adapter
        self.agents = [f'agent_{i}' for i in range(self.n_User)]

    def add_new_users(self, start_position: List[float], start_direction: str, start_velocity: float) -> None:
        self.users.append(User(start_position, start_direction, start_velocity))

    def add_new_users_by_number(self, n: int) -> None:
        for _ in range(n):
            start_position = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
            start_direction = random.choice(['d', 'u', 'l', 'r'])
            self.add_new_users(start_position, start_direction, np.random.random())

        # initialize channels
        self.cellular_shadowing = np.random.normal(0, 8.29, len(self.users))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.users])

    def renew_positions(self) -> None:
        for user in self.users:
            delta_distance = user.velocity * self.time_slow
            if user.direction == 'u':
                user.position[1] += delta_distance
            elif user.direction == 'd':
                user.position[1] -= delta_distance
            elif user.direction == 'r':
                user.position[0] += delta_distance
            else:
                user.position[0] -= delta_distance

            # wrap-around behavior when reaching boundary
            if user.position[0] < 0:
                user.position[0] += self.width
                user.direction = 'r'
            if user.position[0] > self.width:
                user.position[0] -= self.width
                user.direction = 'l'
            if user.position[1] < 0:
                user.position[1] += self.height
                user.direction = 'u'
            if user.position[1] > self.height:
                user.position[1] -= self.height
                user.direction = 'd'

    def renew_bs_channel(self) -> None:
        """Renew slow fading channel."""
        self.cellular_pathloss = np.zeros((len(self.users), self.n_RB))
        for i in range(len(self.users)):
            self.cellular_shadowing[i] = self.bs_channels.get_shadowing(
                self.delta_distance[i], self.cellular_shadowing[i]
            )
            pathloss = self.bs_channels.get_path_loss(self.users[i].position)[0]
            self.cellular_pathloss[i] = pathloss
        self.cellular_channels_abs = self.cellular_pathloss + np.repeat(
            self.cellular_shadowing[:, np.newaxis], self.n_RB, axis=1
        )

    def renew_bs_channels_fastfading(self) -> None:
        """Renew fast fading channel."""
        cellular_channels_with_fastfading = self.cellular_channels_abs
        self.cellular_channels_with_fastfading = cellular_channels_with_fastfading - 20 * np.log10(
            np.abs(
                np.random.normal(0, 1, cellular_channels_with_fastfading.shape) +
                1j * np.random.normal(0, 1, cellular_channels_with_fastfading.shape)
            ) / math.sqrt(2)
        )

    def compute_performance_reward_failure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        transmit_power = np.zeros(len(self.users))
        # ------------ Compute cellular rate --------------------
        cellular_rate = np.zeros(self.n_User)
        cellular_sinr = np.zeros(self.n_User)
        ee = np.zeros(self.n_User)
        cellular_signals = np.zeros([self.n_User, self.n_RB])
        for i in range(len(self.users)):
            for l in range(self.n_RB):
                transmit_power[i] = self.cellular_power_db_list[0]
                cellular_signals[i, l] = 10**(
                    (
                        transmit_power[i] - self.cellular_channels_with_fastfading[i, l] + self.user_ant_gain +
                        self.bs_ant_gain - self.bs_noise_figure
                    ) / 10
                )

        # In failure case, SINR and Rate are calculated based on a default choice (e.g., RB 0)
        # as in the original code's logic.
        cellular_sinr_all = np.divide(cellular_signals, self.sig2[0]) # Assuming sig2 is constant across RBs for this calc
        for i in range(len(self.users)):
            cellular_sinr[i] = cellular_sinr_all[i, 0]

        cellular_rate_all = np.log2(1 + np.divide(cellular_signals, self.sig2[0])) * self.BW[0]

        for i in range(len(self.users)):
            cellular_rate[i] = cellular_rate_all[i, 0]

        for i in range(len(self.users)):
            # Using the fixed transmit_power for EE calculation
            ee[i] = np.divide(cellular_rate[i], 10**(transmit_power[i] / 10 + 0.06))

        return cellular_rate, cellular_sinr, ee

    def compute_performance_reward_train(
            self, actions_all: np.ndarray, is_ppo: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-user (rate, SINR, EE) given actions.

        Action format (is_ppo=True expected by wrapper): shape (n_user, 2)
        - actions_all[i, 0]: RB index (channel)
        - actions_all[i, 1]: transmit power in dB
        """
        actions_all = np.asarray(actions_all)
        if not (actions_all.ndim == 2 and actions_all.shape[0] == self.n_User and actions_all.shape[1] == 2):
            raise ValueError(f"action shape should be (n_user, 2), got {actions_all.shape}")

        # Parse and clamp actions to valid ranges
        channel_sel = np.clip(actions_all[:, 0].astype(int), 0, self.n_RB - 1)
        tx_power_db = actions_all[:, 1].astype(float)
        tx_power_db = np.clip(tx_power_db, min(self.cellular_power_db_list), max(self.cellular_power_db_list))

        # Initialize accumulators on RB axis
        cellular_signals = np.zeros((self.n_User, self.n_RB))
        channel_choice = np.zeros((self.n_RB,), dtype=int)

        for i in range(self.n_User):
            rb = channel_sel[i]
            channel_choice[rb] += 1
            gain_db = (
                tx_power_db[i]
                - self.cellular_channels_with_fastfading[i, rb]
                + self.user_ant_gain + self.bs_ant_gain - self.bs_noise_figure
            )
            cellular_signals[i, rb] = 10 ** (gain_db / 10.0)

        # Interference per RB: sum of signals on the RB
        interference = cellular_signals.sum(axis=0)

        # Compute SINR and Rate per user
        # NOTE: Matching RA_demo's SINR calculation (without interference in denominator)
        # RA_demo: cellular_SINR_all = np.divide(cellular_Signals, self.sig2)
        cellular_sinr = np.zeros(self.n_User)
        cellular_rate = np.zeros(self.n_User)
        ee = np.zeros(self.n_User)
        noise_power = np.array(self.sig2)  # length >= n_RB
        for i in range(self.n_User):
            rb = channel_sel[i]
            sig = cellular_signals[i, rb]
            # Match RA_demo: SINR = Signal / Noise (without interference)
            sinr = sig / noise_power[rb] if noise_power[rb] > 0 else 0.0
            cellular_sinr[i] = sinr
            rate = np.log2(1.0 + sinr) * self.BW[rb]
            cellular_rate[i] = rate
            # Match RA_demo: EE = Rate / (10^(power/10 + 0.06))
            # RA_demo: EE[i] = np.divide(cellular_Rate[i], 10 ** (transmit_power[i, 0] / 10 + 0.06))
            ee[i] = rate / (10 ** (tx_power_db[i] / 10.0 + 0.06))

        # Success metric: no collision on selected RB
        self.success = np.zeros([self.n_User])
        for i in range(self.n_User):
            rb = channel_sel[i]
            if channel_choice[rb] <= 1:
                self.success[i] = 1

        self.channel_choice = channel_choice
        return cellular_rate, cellular_sinr, ee

    def act_for_training(self, actions: np.ndarray, is_ppo: bool) -> float:
        action_temp = actions.copy()
        cellular_Rate, cellular_SINR, EE = self.compute_performance_reward_train(action_temp, is_ppo)
        _, _, failure_EE = self.compute_performance_reward_failure()

        EE_sum = 0.0
        for i in range(len(self.success)):
            if self.success[i] == 1 and cellular_SINR[i] > 3.16:
                EE_sum += EE[i]
            elif self.success[i] == 1:
                EE_sum += failure_EE[i]
            else:
                EE_sum = (np.sum(self.success) - self.n_User) / self.n_User
                break
        reward = EE_sum / self.n_User
        return reward

    def new_random_game(self, n_user: Optional[int] = None) -> None:
        self.users = []
        if n_user is not None:
            self.n_User = n_user
        self.add_new_users_by_number(self.n_User)
        self.renew_bs_channel()
        self.renew_bs_channels_fastfading()


@ENV_REGISTRY.register('multi_general_env')
class MultiGeneralEnv(BaseEnv):
    """
    DI-engine wrapper of the user semantic communication environment.
    Observation: positions of all users, shape (n_user, 2).
    Action: tensor of shape (n_user, 2) containing [rb_idx, power_db].
    """
    # Class variable to track global episode count across all instances
    _global_episode_counter = 0
    _max_episode = None  # Will be set from config

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._n_user = self._cfg.get('n_user', 10)
        self._n_rb = self._cfg.get('n_rb', 6)
        self._max_episode_steps = self._cfg.get('max_episode_steps', 200)
        self._agent_obs_only = self._cfg.get('agent_obs_only', False)
        self._agent_specific_global_state = self._cfg.get('agent_specific_global_state', False)
        
        # Set max_episode from config (max_train_iter, matching RA_demo's n_episode)
        if MultiGeneralEnv._max_episode is None:
            MultiGeneralEnv._max_episode = self._cfg.get('max_train_iter', 1000)
        
        self._env = Environ(
            n_user=self._n_user,
            n_rb=self._n_rb
        )
        self._agents = [f'agent_{i}' for i in range(self._n_user)]
        self._step_counter = 0
        self._cumulative_success = np.zeros(self._n_user, dtype=np.float32)
        self._episode_rewards = []  # Store rewards for the current episode
        self._eval_episode_return = 0.0  # Track cumulative reward for evaluator
        self._normalize_reward = self._cfg.get('normalize_reward', False)
        self._init_flag = False  # Flag to track if observation_space has been initialized
        
        # Action space and reward space can be set immediately
        # Hybrid action space: discrete (rb selection) + continuous (power)
        # action_type: Discrete(n_rb) - 选择资源块 (0 到 n_rb-1)
        # action_args: Box - 选择功率 (连续值)
        # 模型输出范围是 [-1, 1] (tanh激活)，在 _process_action 中映射到 [0, 24]
        action_type_space = gym.spaces.Discrete(self._n_rb)
        action_args_low = np.array([-1.0], dtype=np.float32)  # tanh输出范围
        action_args_high = np.array([1.0], dtype=np.float32)  # tanh输出范围
        action_args_space = gym.spaces.Box(low=action_args_low, high=action_args_high, dtype=np.float32)
        
        # Hybrid action space for each agent
        single_agent_hybrid = gym.spaces.Dict({
            'action_type': action_type_space,
            'action_args': action_args_space
        })
        self._action_space = gym.spaces.Dict({agent: single_agent_hybrid for agent in self._agents})
        self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        # observation_space will be initialized in reset() method (lazy initialization, like petting_zoo)
        # Don't set _observation_space here - let it raise AttributeError when accessed, 
        # so BaseEnvManager's try-except will catch it and call reset()

    def reset(self) -> Union[np.ndarray, dict]:
        # Initialize observation_space on first reset (lazy initialization, like petting_zoo)
        if not self._init_flag:
            # Calculate state dimension based on the new get_state logic
            # cellular_fast (n_rb) + cellular_abs (n_rb) + channel_choice (n_rb) + user_vector (n_rb) + success (1) + time (1)
            state_dim = self._n_rb * 4 + 2
            
            obs_low = -np.inf * np.ones((self._n_user, state_dim), dtype=np.float32)
            obs_high = np.inf * np.ones((self._n_user, state_dim), dtype=np.float32)
            
            # Define observation space - only agent_state, no global_state
            if self._agent_obs_only:
                self._observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            else:
                # Dict format with only agent_state (each agent has different state)
                self._observation_space = gym.spaces.Dict({
                    'agent_state': gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
                })
            self._init_flag = True
        
        # Increment global episode counter (matching RA_demo's i_episode)
        # Note: We increment AFTER the first episode completes, so:
        # - First episode: counter=0 (time_feature=0/max_episode=0)
        # - Second episode: counter=1 (time_feature=1/max_episode)
        # This matches RA_demo where i_episode starts at 0 for the first episode
        # The counter is incremented here, but the first call to reset() will use counter=0
        # Subsequent resets will use incremented counter values
        if self._init_flag:  # Only increment after first initialization
            MultiGeneralEnv._global_episode_counter += 1
        
        self._step_counter = 0
        self._env.new_random_game(self._n_user)
        self._cumulative_success.fill(0)  # Reset cumulative success at the beginning of each episode
        self._episode_rewards = []  # Reset episode rewards
        self._eval_episode_return = 0.0  # Reset episode return for evaluator
        return self._get_obs()

    def step(self, action: Union[np.ndarray, dict]) -> BaseEnvTimestep:
        self._env.renew_bs_channels_fastfading()
        self._step_counter += 1
        # process action (support ndarray, dict, or hybrid action format)
        action_arr = self._process_action(action)
        reward = self._env.act_for_training(action_arr, is_ppo=True)
        self._cumulative_success += self._env.success
        
        # Track cumulative reward for evaluator
        self._eval_episode_return += reward
        
        # Store reward for normalization
        self._episode_rewards.append(reward)
        
        # Use max_episode_steps to determine 'done'
        done = self._step_counter >= self._max_episode_steps
        
        # Normalize rewards if episode is done and normalization is enabled
        if done and self._normalize_reward and len(self._episode_rewards) > 1:
            rewards_array = np.array(self._episode_rewards, dtype=np.float32)
            rewards_mean = rewards_array.mean()
            rewards_std = rewards_array.std() + 1e-8
            normalized_rewards = (rewards_array - rewards_mean) / rewards_std
            # Use the normalized reward for the current step
            reward = normalized_rewards[-1]
            # Store normalized rewards in info for reference
            episode_info = self._collect_info(action_arr)
            episode_info['normalized_rewards'] = normalized_rewards
            episode_info['raw_rewards'] = rewards_array
        else:
            episode_info = self._collect_info(action_arr)
        
        # Add eval_episode_return to info when episode is done (required by evaluator)
        # Divide by n_user to get per-agent return
        if done:
            info = {
                'eval_episode_return': self._eval_episode_return / self._max_episode_steps,
                'episode_info': episode_info
            }
             # update environment for next episode
            self._env.renew_positions()
            self._env.renew_bs_channel()
            self._env.renew_bs_channels_fastfading()
        else:
            info = {}  

        obs = self._get_obs()
        return BaseEnvTimestep(obs, np.array([reward], dtype=np.float32), done, info)

    def _collect_info(self, action: np.ndarray) -> dict:
        rate, sinr, ee = self._env.compute_performance_reward_train(action, is_ppo=True)
        success_rate_per_user = self._cumulative_success / self._step_counter
        return {
            'rate': rate,
            'sinr': sinr,
            'ee': ee,
            'success_rate_per_user': success_rate_per_user
        }

    def _get_obs(self) -> Union[np.ndarray, dict]:
        """
        Get state from the environment for each agent, based on the user-provided logic.
        Returns dict with 'agent_state' and 'global_state' when agent_obs_only=False,
        otherwise returns array.
        """
        n_user = self._n_user
        n_rb = self._n_rb
        
        # Pre-calculate global parts
        channel_choice_norm = (self._env.channel_choice / n_user)
        
        user_vector = np.zeros(n_rb)
        # This logic is kept from the original snippet. It might need review if n_user > n_rb.
        for i in range(min(n_user, n_rb)):
            user_vector[i] = 1 / n_user
        
        # Use global episode progress instead of step progress (matching RA_demo: ind_episode / n_episode)
        # RA_demo uses: ind_episode / n_episode (global episode progress, 0 to 1)
        # Previous DI-engine used: step_counter / max_episode_steps (episode step progress, 0 to 1 each episode)
        # Now using: global_episode_counter / max_episode (global episode progress, 0 to 1)
        time_feature = MultiGeneralEnv._global_episode_counter / MultiGeneralEnv._max_episode

        # Build state for each agent
        agent_states = []
        for i in range(n_user):
            cellular_fast = (self._env.cellular_channels_with_fastfading[i, :] - self._env.cellular_channels_abs[i, :] + 10) / 35
            cellular_abs = (self._env.cellular_channels_abs[i, :] - 80) / 60.0
            success = self._env.success[i]
            
            state_i = np.concatenate((
                cellular_fast,
                cellular_abs,
                channel_choice_norm,
                user_vector,
                np.asarray([success, time_feature])
            ))
            agent_states.append(state_i)
            
        obs = np.stack(agent_states).astype(np.float32)
        
        # Return only agent_state (each agent has its own different state)
        # No global_state needed - each agent uses its own state for both actor and critic
        return {'agent_state': obs}

    def _process_action(self, action: Union[np.ndarray, dict]) -> np.ndarray:
        n = self._n_user
        low = np.array([0, float(min(self._env.cellular_power_db_list))], dtype=np.float32)
        high = np.array([float(self._n_rb - 1), float(max(self._env.cellular_power_db_list))], dtype=np.float32)
        
        arr = np.zeros((n, 2), dtype=np.float32)
        
        if isinstance(action, dict):
            # 检查是否是hybrid action格式
            if 'action_type' in action and 'action_args' in action:
                # Hybrid action format: {'action_type': (n_user,), 'action_args': (n_user, 1)}
                action_type = action['action_type']
                action_args = action['action_args']
                action_args = np.clip(action_args, -1, 1)
                # 转换为numpy数组
                if hasattr(action_type, 'cpu'):
                    action_type = action_type.cpu().numpy()
                if hasattr(action_args, 'cpu'):
                    action_args = action_args.cpu().numpy()
                
                # 处理形状
                if action_type.ndim > 1:
                    action_type = action_type.squeeze()
                if action_args.ndim > 1:
                    action_args = action_args.squeeze()
        
                
                arr[:, 0] = action_type.astype(np.float32)  # rb_idx (discrete -> float for compatibility)
                action_bound = 1.0  # tanh bound, action_args range is [-1, 1]
                max_power = float(self._env.cellular_power_db_list[0])  # 24
                amp = max_power / (2 * action_bound)  # 24 / 2 = 12
                power_action = (action_args + action_bound) * amp  # Map from [-1, 1] to [0, 24]
                arr[:, 1] = power_action.astype(np.float32)  # power_db
            else:
                # Per-agent dict format (like PettingZoo)
                for i, agent in enumerate(self._agents):
                    if agent in action:
                        agent_action = action[agent]
                        if isinstance(agent_action, dict) and 'action_type' in agent_action and 'action_args' in agent_action:
                            # Per-agent hybrid action
                            arr[i, 0] = float(agent_action['action_type'])
                            # Map action_args (power) from model output range to actual power range
                            action_bound = 1.0  # tanh bound
                            max_power = float(self._env.cellular_power_db_list[0])  # 24
                            amp = max_power / (2 * action_bound)  # 24 / 2 = 12
                            power_action = (float(agent_action['action_args']) + action_bound) * amp
                            arr[i, 1] = float(power_action)
                        else:
                            arr[i] = np.array(agent_action, dtype=np.float32)
        else:
            # ndarray format
            arr = np.asarray(action, dtype=np.float32)
            if arr.shape == (n,):
                # 如果只有一维，假设是rb_idx，需要补充power
                arr = np.column_stack([arr, np.zeros(n, dtype=np.float32)])
            elif arr.shape != (n, 2):
                raise ValueError(f'action shape should be ({n}, 2) or ({n},), got {arr.shape}')
        
        # clip per-dimension
        arr[:, 0] = np.clip(arr[:, 0], low[0], high[0])  # rb_idx: 0 to n_rb-1
        arr[:, 1] = np.clip(arr[:, 1], low[1], high[1])  # power_db
        return arr

    def close(self) -> None:
        return None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        np.random.seed(seed)
        random.seed(seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    @property
    def agents(self) -> List[str]:
        return self._agents

    def random_action(self) -> dict:
        """Generate random per-agent action (Dict[agent]=[rb, power_db])."""
        act_dict = {}
        low_vec = np.array([0, float(min(self._env.cellular_power_db_list))], dtype=np.float32)
        high_vec = np.array([float(self._n_rb - 1), float(max(self._env.cellular_power_db_list))], dtype=np.float32)
        for agent in self._agents:
            rb = np.random.randint(0, self._n_rb)
            p = np.random.choice(self._env.cellular_power_db_list)
            act = np.array([rb, float(p)], dtype=np.float32)
            act = np.clip(act, low_vec, high_vec)
            act_dict[agent] = act
        return act_dict
    def __repr__(self) -> str:
        return "DI-engine Multi-agent General Semantic Communication Environment"

