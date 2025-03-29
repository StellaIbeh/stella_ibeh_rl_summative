import gym
from gym import spaces
import numpy as np

class EvacuationEnv(gym.Env):
    """
    Custom Gym environment for a flood evacuation scenario.
    The agent directs groups from a flood-affected area to safe zones.
    """
    
    def __init__(self):
        super(EvacuationEnv, self).__init__()
        
        # Define discrete action space with 8 possible actions
        self.action_space = spaces.Discrete(8)
        
        # Define observation space:
        # State vector: [group_proximity, water_level, time_elapsed, safe_zone_occupancy]
        # Values are normalized between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Initialize state variables
        self.state = None
        self.time_step = 0
        self.max_time_steps = 50  # Example episode length
        
        self.reset()
    
    def reset(self):
        # Reset state: groups are near the flood (high proximity), water level moderate, time=0, safe zones empty
        self.state = np.array([0.8, 0.5, 0.0, 0.0], dtype=np.float32)
        self.time_step = 0
        return self.state
    
    def step(self, action):
        reward = 0
        
        # Unpack state variables for readability
        group_proximity, water_level, time_elapsed, safe_zone_occupancy = self.state
        
        # Action definitions:
        if action == 0:
            # Scan Environment: Update state by "observing" changes
            # (for simplicity, assume water level may change slightly)
            water_level = np.clip(water_level + np.random.uniform(-0.05, 0.05), 0, 1)
            reward += 0  # No immediate reward
        elif action == 1:
            # Direct Group to Safe Zone A
            if group_proximity > 0.6:
                # If group is close to hazard, proper instruction yields success
                reward += 15
                safe_zone_occupancy = np.clip(safe_zone_occupancy + 0.3, 0, 1)
                group_proximity = np.clip(group_proximity - 0.3, 0, 1)
            else:
                reward -= 10
        elif action == 2:
            # Direct Group to Safe Zone B
            if group_proximity > 0.6:
                reward += 15
                safe_zone_occupancy = np.clip(safe_zone_occupancy + 0.3, 0, 1)
                group_proximity = np.clip(group_proximity - 0.3, 0, 1)
            else:
                reward -= 10
        elif action == 3:
            # Direct Group to Safe Zone C
            if group_proximity > 0.6:
                reward += 15
                safe_zone_occupancy = np.clip(safe_zone_occupancy + 0.3, 0, 1)
                group_proximity = np.clip(group_proximity - 0.3, 0, 1)
            else:
                reward -= 10
        elif action == 4:
            # Optimize Route (re-route groups)
            # If water level is rising fast, good rerouting can save time
            if water_level > 0.7:
                reward += 5
            else:
                reward -= 15
        elif action == 5:
            # Send Rescue Team Alert: Provide extra support
            # Extra support works best if hazard is high and groups are still at risk
            if water_level > 0.6 and group_proximity > 0.5:
                reward += 15
            else:
                reward -= 10
        elif action == 6:
            # Monitor Water Levels: Update state based on dynamic hazard
            water_level = np.clip(water_level + np.random.uniform(-0.1, 0.1), 0, 1)
            reward += 0  # No immediate reward
        elif action == 7:
            # Wait/Do Nothing: Opportunity cost for inaction
            reward -= 5
        
        # Simulate time progression and state changes
        self.time_step += 1
        time_elapsed = self.time_step / self.max_time_steps
        
        # Natural increase in hazard if groups remain near danger
        if group_proximity > 0.5:
            water_level = np.clip(water_level + 0.02, 0, 1)
        
        # Update state
        self.state = np.array([group_proximity, water_level, time_elapsed, safe_zone_occupancy], dtype=np.float32)
        
        # Define termination condition: either time has run out or all groups are safe
        done = bool(self.time_step >= self.max_time_steps or safe_zone_occupancy >= 1.0)
        
        # Additional info can be provided here if needed
        info = {}
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        # Basic text-based rendering of the state
        group_proximity, water_level, time_elapsed, safe_zone_occupancy = self.state
        print(f"Time: {self.time_step}/{self.max_time_steps}")
        print(f"Group Proximity to Hazard: {group_proximity:.2f}")
        print(f"Water Level (Hazard Intensity): {water_level:.2f}")
        print(f"Safe Zone Occupancy: {safe_zone_occupancy:.2f}")
        print("-" * 30)

# Example usage:
if __name__ == "__main__":
    env = EvacuationEnv()
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        # For demonstration, we select a random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print("Episode finished. Total reward:", total_reward)
