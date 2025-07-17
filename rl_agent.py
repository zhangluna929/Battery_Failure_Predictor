import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

class PolicyGradientAgent:
    """
    A Reinforcement Learning agent that uses a policy gradient method (REINFORCE).
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        """
        Initializes the agent.

        Args:
            state_size (int): The number of dimensions in the state space.
            action_size (int): The number of possible actions.
            learning_rate (float): The learning rate for the optimizer.
            gamma (float): The discount factor for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.policy_network = self._build_policy_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # These lists will store the history of an episode
        self.states = []
        self.actions = []
        self.rewards = []

    def _build_policy_network(self):
        """
        Builds the neural network that represents the agent's policy.
        """
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_size, activation='softmax')(x)
        
        return Model(inputs, outputs)

    def choose_action(self, state):
        """
        Chooses an action based on the current state and the learned policy.

        Args:
            state: The current state of the environment.

        Returns:
            The action to take.
        """
        # Reshape state for the network and get action probabilities
        state = state.reshape([1, self.state_size])
        action_probs = self.policy_network.predict(state)[0]
        
        # Choose action based on the probability distribution
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def store_transition(self, state, action, reward):
        """
        Stores the experience of one time step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        """
        Updates the policy network's weights using the stored experience from an episode.
        This is where the REINFORCE algorithm is implemented.
        """
        # 1. Calculate discounted rewards
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add

        # 2. Normalize rewards (optional but recommended)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-9)

        # 3. Update the policy network
        with tf.GradientTape() as tape:
            loss = 0
            for i, state in enumerate(self.states):
                state = state.reshape([1, self.state_size])
                action = self.actions[i]
                reward = discounted_rewards[i]
                
                # Get the probabilities of actions from the network
                action_probs = self.policy_network(state)
                # Get the probability of the specific action that was taken
                action_prob = action_probs[0, action]
                
                # Calculate the loss for this step
                # We use negative because we want to perform gradient *ascent*
                loss -= tf.math.log(action_prob) * reward
        
        # Calculate gradients and apply them
        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

        # Clear the memory for the next episode
        self.states = []
        self.actions = []
        self.rewards = [] 