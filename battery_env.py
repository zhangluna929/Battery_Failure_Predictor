import numpy as np

class BatteryChargingEnv:
    """
    A simulation environment for a battery being charged.
    This environment will be used to train a Reinforcement Learning agent.
    """
    def __init__(self, fault_prediction_model):
        """
        Initializes the battery environment.

        Args:
            fault_prediction_model: The pre-trained multi-modal model used to predict battery health.
        """
        self.model = fault_prediction_model
        
        # Action space: 0 = Slow Charge, 1 = Normal Charge, 2 = Fast Charge
        self.action_space = [0, 1, 2]
        self.action_effects = {
            0: {'current': 0.5, 'temp_increase': 0.1}, # Slow
            1: {'current': 1.5, 'temp_increase': 0.5}, # Normal
            2: {'current': 3.0, 'temp_increase': 1.5}  # Fast
        }
        
        self.reset()

    def reset(self):
        """
        Resets the environment to an initial state.
        Returns:
            The initial state of the battery.
        """
        self.soc = np.random.uniform(0.1, 0.3)
        self.temperature = np.random.uniform(20, 25)
        self.age = 0 # Represents battery degradation over time
        self.done = False
        return self._get_state()

    def _get_state(self):
        """
        Gets the current state of the environment.
        The state includes SOC, temperature, and battery age.
        """
        return np.array([self.soc, self.temperature, self.age])

    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action (int): The action to take (0, 1, or 2).

        Returns:
            A tuple containing: (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0, self.done, {}

        # 1. Update battery state based on action
        charge_current = self.action_effects[action]['current']
        temp_increase = self.action_effects[action]['temp_increase']

        self.soc += charge_current * 0.01 # Simplified SOC increase
        self.temperature += temp_increase
        self.age += 0.001 # Battery ages with each step

        # Clip values to be within realistic bounds
        self.soc = np.clip(self.soc, 0, 1)
        self.temperature = np.clip(self.temperature, 20, 80)

        # 2. Calculate the reward
        reward = self._calculate_reward(charge_current)

        # 3. Check if the episode is finished
        if self.soc >= 0.99 or self.temperature > 70:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def _calculate_reward(self, charge_current):
        """
        Calculates the reward for the current state and action.
        """
        # --- Positive Reward ---
        # Reward for charging (proportional to how much it charged)
        charge_reward = charge_current * 1.5

        # --- Negative Reward (Penalties) ---
        # Penalty for high temperature
        temp_penalty = 0
        if self.temperature > 45:
            temp_penalty = - (self.temperature - 45) ** 2

        # Penalty based on predicted fault probability from our deep learning model
        health_penalty = self._get_health_penalty()
        
        total_reward = charge_reward + temp_penalty + health_penalty
        return total_reward

    def _get_health_penalty(self):
        """
        Uses the pre-trained model to predict the probability of a fault,
        and returns a penalty based on that probability.
        """
        # To use our multimodal model, we need to create dummy inputs for the other modalities.
        # This is a simplification; a more complex setup would have dynamic EIS/capacity data.
        dummy_ts = np.array([[3.7, self.soc, self.temperature, 0.8, 0, 0, 0, 0]]) # Example features
        dummy_eis = np.random.rand(1, 50)
        dummy_capacity = np.random.rand(1, 5)

        # The model expects scaled data, but we'll skip scaling here for simplicity.
        # In a full implementation, the scalers would be used.
        fault_probs, _ = self.model.predict([dummy_ts, dummy_eis, dummy_capacity])
        
        # We penalize for any fault probability (excluding 'Normal')
        non_normal_prob = 1 - fault_probs[0][0] # Probability of not being normal
        
        health_penalty = - (non_normal_prob * 10) ** 2 # Heavily penalize fault risk
        
        return health_penalty 