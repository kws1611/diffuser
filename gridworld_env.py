import numpy as np
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=5, num_obstacles=3):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.reset()

    def reset(self):
        # Initialize the grid with zeros
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place obstacles randomly
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                obstacle = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if obstacle not in self.obstacles:
                    self.obstacles.append(obstacle)
                    self.grid[obstacle] = -1  # Mark obstacle cells
                    break
        
        # Randomize start and goal points
        while True:
            self.start = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if self.start not in self.obstacles:
                break
        
        while True:
            self.goal = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if self.goal not in self.obstacles and self.goal != self.start:
                break
        
        # Set the agent's start position
        self.agent_pos = self.start
        
        return self._get_observation()

    def _get_observation(self):
        obs = np.copy(self.grid)
        obs[self.agent_pos] = 1  # Mark agent's position
        obs[self.goal] = 2  # Mark goal position
        return obs

    def step(self, action):
        # Define movement directions
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        # Compute new position
        new_pos = (self.agent_pos[0] + moves[action][0], self.agent_pos[1] + moves[action][1])
        
        # Check if new position is within bounds
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.agent_pos = new_pos
        
        # Calculate reward and penalty
        reward = -1  # Default penalty for each step
        done = False
        
        if self.agent_pos in self.obstacles:
            reward -= 5  # Penalty for hitting an obstacle
        elif self.agent_pos == self.goal:
            reward = 10  # Reward for reaching the goal
            done = True
        
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        for i in range(self.grid_size):
            row = ''
            for j in range(self.grid_size):
                if (i, j) == self.agent_pos:
                    row += 'A '
                elif (i, j) == self.goal:
                    row += 'G '
                elif (i, j) in self.obstacles:
                    row += 'X '
                else:
                    row += '. '
            print(row)
        print()

