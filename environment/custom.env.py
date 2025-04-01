import gym
from gym import spaces
import numpy as np

class HypertensionMonitoringEnv(gym.Env):
    """
    Custom Gym environment for AI-powered remote monitoring of hypertension patients.
    The agent observes vital parameters and chooses an action from a discrete set to
    stabilize blood pressure.
    """
    def __init__(self):
        super(HypertensionMonitoringEnv, self).__init__()
        
        # Define action space: Discrete with 7 actions:
        # 0: No medication, 1: Low-dose medication, 2: Moderate-dose medication,
        # 3: High-dose medication, 4: Recommend rest, 5: Recommend exercise,
        # 6: Call emergency response.
        self.action_space = spaces.Discrete(7)
        
        # Define observation space as a Box representing:
        # [Systolic BP, Diastolic BP, Heart Rate, Stress Level, 
        #  Physical Activity Level, Last Medication Level, Time Since Last Dose, Sleep Quality]
        # For simplicity, we use normalized ranges for these values.
        low = np.array([50, 30, 40, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([200, 150, 180, 10, 2, 3, 120, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Initialize state variables: starting with some average values.
        self.state = None
        self.reset()
        self.time_step = 0

    def reset(self):
        # Reset the environment state to an initial condition.
        # [SBP, DBP, HR, Stress, Physical Activity, Last Medication, Time since dose, Sleep Quality]
        self.state = np.array([120, 80, 70, 5, 0, 0, 0, 1], dtype=np.float32)
        self.time_step = 0
        return self.state

    def step(self, action):
        """
        Perform one step in the environment given the action.
        """
        sbp, dbp, hr, stress, phys_act, last_med, time_since, sleep = self.state
        
        # Simulate the effect of random factors on patient vitals
        sbp += np.random.uniform(-2, 2)
        dbp += np.random.uniform(-1, 1)
        hr += np.random.uniform(-1, 1)
        stress += np.random.uniform(-0.5, 0.5)
        # Clamp stress level
        stress = np.clip(stress, 0, 10)
        
        # Define effects for each action:
        if action == 0:  # No medication
            # Let natural fluctuations happen
            reward = 0
        elif action == 1:  # Low-dose medication
            sbp -= 3
            dbp -= 2
            reward = 2
        elif action == 2:  # Moderate-dose medication
            sbp -= 5
            dbp -= 3
            reward = 4
        elif action == 3:  # High-dose medication
            sbp -= 8
            dbp -= 5
            reward = -2  # Risk of sudden drop causing hypotension
        elif action == 4:  # Recommend rest
            stress -= 1
            reward = 2
        elif action == 5:  # Recommend exercise
            # Exercise can initially raise BP but is beneficial over time.
            sbp += 2
            dbp += 1
            reward = 1
        elif action == 6:  # Call emergency response
            # Emergency response is beneficial only when BP is critical.
            if sbp > 160 or dbp > 100 or sbp < 80 or dbp < 50:
                reward = 10
            else:
                reward = -5  # Unnecessary emergency call
        
        # Simulate time passage (affecting time since last medication)
        time_since += 1
        if action in [1, 2, 3]:
            last_med = action  # record the medication level (1,2,or3)
            time_since = 0

        # Update state
        self.state = np.array([
            np.clip(sbp, 50, 200),
            np.clip(dbp, 30, 150),
            np.clip(hr, 40, 180),
            stress,
            np.clip(phys_act, 0, 2),   # activity level: 0 (none), 1 (moderate), 2 (high)
            last_med,
            np.clip(time_since, 0, 120),
            sleep  # assuming sleep quality remains constant in this simple model
        ], dtype=np.float32)

        # Additional reward shaping based on blood pressure stability:
        optimal_sbp = (90, 120)
        optimal_dbp = (60, 80)
        if optimal_sbp[0] <= self.state[0] <= optimal_sbp[1] and optimal_dbp[0] <= self.state[1] <= optimal_dbp[1]:
            reward += 10
        elif self.state[0] > 130 or self.state[1] > 90 or self.state[0] < 80 or self.state[1] < 50:
            reward -= 10

        # Termination condition: we could define an episode length or critical vitals
        self.time_step += 1
        done = self.time_step >= 100  # example: end episode after 100 steps
        
        info = {}
        return self.state, reward, done, info

    def render(self, mode='human'):
        # For now, we print the state; later, integrate with PyOpenGL for 3D visualization.
        print(f"Step: {self.time_step}, State: {self.state}")

    def close(self):
        pass

# Example usage:
if __name__ == "__main__":
    env = HypertensionMonitoringEnv()
    state = env.reset()
    done = False

    while not done:
        # Here the agent randomly selects an action.
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
    env.close()
