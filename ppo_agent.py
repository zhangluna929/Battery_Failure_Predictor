import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

class PPOAgent:
    """
    A Reinforcement Learning agent that uses the Proximal Policy Optimization (PPO) algorithm.
    It uses an Actor-Critic architecture.
    """
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_ratio=0.2, gae_lambda=0.95, epochs=10):
        """
        Initializes the PPO agent.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        self.epochs = epochs

        # Actor and Critic networks
        self.actor = self._build_actor_network()
        self.critic = self._build_critic_network()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Memory buffer to store transitions
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.action_probs = [], [], []

    def _build_actor_network(self):
        """Builds the policy network (the 'Actor')."""
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        return Model(inputs, outputs)

    def _build_critic_network(self):
        """Builds the value network (the 'Critic')."""
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation=None)(x)
        return Model(inputs, outputs)

    def store_transition(self, state, action, reward, next_state, done, action_prob):
        """Stores a single transition in the memory buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.action_probs.append(action_prob)

    def choose_action(self, state):
        """Chooses an action based on the current policy."""
        state = np.array(state).reshape([1, self.state_size])
        action_probs = self.actor.predict(state, verbose=0)[0]
        dist = tfp.distributions.Categorical(probs=action_probs)
        action = dist.sample().numpy()
        return action, action_probs

    def learn(self):
        """
        Updates the Actor and Critic networks using the stored batch of experience.
        This is where the PPO magic happens.
        """
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        next_states = np.array(self.next_states)
        dones = np.array(self.dones)
        old_action_probs = np.array(self.action_probs)
        
        # Calculate advantages using General Advantage Estimation (GAE)
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - dones[t])
        
        target_values = advantages + values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train the networks for a few epochs
        for _ in range(self.epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                # Critic Loss
                critic_values = self.critic(states, training=True)
                critic_loss = tf.keras.losses.mean_squared_error(target_values, tf.squeeze(critic_values))

                # Actor Loss (PPO-Clip objective)
                current_action_probs = self.actor(states, training=True)
                dist = tfp.distributions.Categorical(probs=current_action_probs)
                current_log_probs = dist.log_prob(actions)
                
                old_dist = tfp.distributions.Categorical(probs=old_action_probs)
                old_log_probs = old_dist.log_prob(actions)

                # Ratio of new policy to old policy
                ratio = tf.exp(current_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            # Calculate and apply gradients
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Clear the memory buffer for the next batch of experience
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.action_probs = [], [], [], [], [], [] 