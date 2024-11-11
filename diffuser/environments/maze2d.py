import numpy as np

class RandomMazeEnv:
    def __init__(self):
        self.grid_size = (5, 5)
        self.obstacle_count = 3
        self.start = None
        self.goal = None
        self.obstacles = []

    def reset(self):
        # Randomly select start and goal positions
        self.start = tuple(np.random.randint(0, 5, 2))
        self.goal = tuple(np.random.randint(0, 5, 2))
        while self.goal == self.start:
            self.goal = tuple(np.random.randint(0, 5, 2))

        # Randomly place obstacles
        self.obstacles = []
        while len(self.obstacles) < self.obstacle_count:
            obstacle = tuple(np.random.randint(0, 5, 2))
            if obstacle != self.start and obstacle != self.goal and obstacle not in self.obstacles:
                self.obstacles.append(obstacle)

        # Return initial observation
        return self.start  # or other initial observation

    def compute_reward(self, current_position):
        # Penalty for hitting obstacles
        if current_position in self.obstacles:
            return -1  # Penalty for hitting an obstacle

        # Reward for reaching the goal
        if current_position == self.goal:
            return 10  # Positive reward for reaching the goal

        # Small step penalty to encourage efficiency (optional)
        return -0.1

