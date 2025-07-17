import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from battery_env import BatteryChargingEnv
from ppo_agent import PPOAgent # Import the new agent

# --- Configuration ---
MODEL_PATH = 'battery_multimodal_model.h5'
NUM_EPISODES = 1000 # More episodes might be needed for PPO
PRINT_EVERY = 50
TRAIN_BATCH_SIZE = 256 # Number of steps to collect before training

def main():
    """
    Main function to train the RL agent using PPO.
    """
    # 1. Load the pre-trained model for the environment
    try:
        fault_model = tf.keras.models.load_model(MODEL_PATH)
        print("Fault prediction model loaded successfully.")
    except Exception as e:
        print(f"Error loading fault model: {e}")
        return

    # 2. Initialize environment and agent
    env = BatteryChargingEnv(fault_prediction_model=fault_model)
    state_size = env._get_state().shape[0]
    action_size = len(env.action_space)
    agent = PPOAgent(state_size, action_size) # Use the PPO agent

    # 3. Training loop
    print("--- Starting PPO Reinforcement Learning Training ---")
    all_rewards = []
    
    state = env.reset()
    current_step = 0

    for episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        episode_rewards = 0
        done = False

        while not done:
            current_step += 1
            # Agent chooses an action
            action, action_prob = agent.choose_action(state)
            
            # Environment performs the action
            next_state, reward, done, _ = env.step(action)
            
            # Store this transition
            agent.store_transition(state, action, reward, next_state, done, action_prob[action])
            
            state = next_state
            episode_rewards += reward

            # If we have collected enough data, train the agent
            if current_step % TRAIN_BATCH_SIZE == 0:
                agent.learn()

        all_rewards.append(episode_rewards)

        # Print progress
        if (episode + 1) % PRINT_EVERY == 0:
            avg_reward = np.mean(all_rewards[-PRINT_EVERY:])
            print(f"Episode {episode + 1}/{NUM_EPISODES}, Average Reward: {avg_reward:.2f}")

    print("--- Training Finished ---")

    # 4. Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 