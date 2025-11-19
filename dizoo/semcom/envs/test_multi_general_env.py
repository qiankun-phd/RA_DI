import numpy as np
from dizoo.semcom.envs.multi_general_env import MultiGeneralEnv

def main():
    """
    A simple test script for the MultiGeneralEnv.
    It creates an environment, resets it, and runs a few steps with random actions.
    """
    print("--- Testing MultiGeneralEnv ---")

    # Configuration for the environment
    env_config = {
        'n_user': 6,
        'n_rb': 10,
        'max_episode_steps': 100, # Added for the new state logic
        'normalize_reward': True,  # Enable reward normalization
    }
    
    # Create the environment
    env = MultiGeneralEnv(cfg=env_config)
    
    # Print observation and action spaces
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print("-" * 20)

    # Reset the environment
    try:
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        # Expected shape: (n_user, n_rb * 4 + 2)
        expected_dim = env_config['n_rb'] * 4 + 2
        assert obs.shape == (env_config['n_user'], expected_dim), "Observation shape is incorrect"
        print("Reset successful.")
    except Exception as e:
        print(f"An error occurred during reset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run a full episode
    print(f"\nRunning a full episode...")
    obs = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        try:
            # Get a random action from the environment's helper function
            random_action = env.random_action()
            
            # Step the environment
            timestep = env.step(random_action)
            obs, reward, done, info = timestep.obs, timestep.reward, timestep.done, timestep.info
            
            # Print step information
            print(f"\n--- Step {step} ---")
            # Taking the first agent's action for display purposes
            print(f"Action taken (for agent_0): {random_action['agent_0']}")
            print(f"Next observation shape: {obs.shape}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            # Print the per-user success rate
            print(f"Info (success_rate_per_user): {info['success_rate_per_user']}")
            # Print normalized rewards if episode is done
            if done and 'normalized_rewards' in info:
                print(f"Normalized rewards (episode): {info['normalized_rewards']}")
                print(f"Raw rewards (episode): {info['raw_rewards']}")
                print(f"Reward mean: {info['raw_rewards'].mean():.6f}, std: {info['raw_rewards'].std():.6f}")

        except Exception as e:
            print(f"An error occurred during step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n--- Test finished ---")

if __name__ == "__main__":
    main()
