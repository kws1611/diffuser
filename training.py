import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pickle
import pdb
import torch.nn.functional as F
import sys
sys.path.append('~/project/diffuser')
import diffuser.utils as utils
import einops
from collections import namedtuple

# Define custom namedtuple for storing trajectories
Trajectories = namedtuple('Trajectories', 'actions observations')

# Define a custom dataset for PyTorch
class PathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        grid = torch.tensor(data["grid"], dtype=torch.float32).unsqueeze(0)  # Shape: (1, grid_size, grid_size)
        start = torch.tensor(data["start"], dtype=torch.float32)
        goal = torch.tensor(data["goal"], dtype=torch.float32)
        waypoints = torch.tensor(data["waypoints"], dtype=torch.float32)
        return grid, start, goal, waypoints

# Load the dataset for training
with open('grid_world_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
path_dataset = PathDataset(dataset)
print(f"Loaded dataset with {len(path_dataset)} samples.")

# Create DataLoader
batch_size = 16
dataloader = DataLoader(path_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x))

# Function to pad waypoints
def pad_waypoints(waypoints, max_len):
    padded_waypoints = []
    for w in waypoints:
        pad_size = max_len - len(w)
        padded_w = np.pad(w, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        padded_waypoints.append(padded_w)
    return torch.tensor(padded_waypoints, dtype=torch.float32)

# Collate function for dataloader
def collate_batch(batch):
    grids, starts, goals, waypoints = zip(*batch)
    max_len = max(len(w) for w in waypoints) * 5  # Increase the number of waypoints to generate a smoother path
    padded_waypoints = pad_waypoints(waypoints, max_len)
    return torch.stack(grids), torch.stack(starts), torch.stack(goals), padded_waypoints

# Define the Policy class
class Policy:
    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size)

        # Run reverse diffusion process
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample)

        # Extract actions [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        # Extract first action
        action = actions[0, 0]

        # Extract observations
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations)
        return action, trajectories

# Training the policy-guided diffusion model
def train_policy_model(dataset, grid_size, num_epochs=20, batch_size=16, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    model = PolicyModel(grid_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Reward function
    def compute_reward(waypoints, goal):
        final_position = waypoints[-1]
        distance_to_goal = torch.norm(final_position - goal)
        reward = -distance_to_goal  # Negative reward for distance to goal
        return reward

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            grid, start, goal, waypoints = batch

            # Use the grid as input and predict the next waypoint after start
            optimizer.zero_grad()
            predicted_waypoints = []
            current_position = start
            for _ in range(waypoints.size(1)):
                input_grid = grid
                predicted_waypoint = model(input_grid)
                predicted_waypoints.append(predicted_waypoint)
                current_position = predicted_waypoint

            predicted_waypoints = torch.stack(predicted_waypoints, dim=1)

            # Compute reward for the predicted path
            reward = torch.stack([compute_reward(predicted_waypoints[i], goal[i]) for i in range(goal.size(0))]).mean()
            loss = -reward  # Maximize reward

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model, 'trained_policy_model.pth')
    print("Training complete.")
    return model

# Train the policy-guided model
grid_size = 5
trained_policy_model = train_policy_model(path_dataset, grid_size)

# Visualization of generated paths
def visualize_samples(dataset, num_samples=10):
    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for i in range(num_samples):
        data = dataset[i]
        grid = data['grid']
        start = data['start']
        goal = data['goal']
        waypoints = data['waypoints']

        ax = axes[i]
        ax.imshow(grid, cmap='gray_r')
        ax.plot(start[1], start[0], "go", label="Start")
        ax.plot(goal[1], goal[0], "ro", label="Goal")
        ax.plot(waypoints[:, 1], waypoints[:, 0], "b.-", label="Path")
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# Generate and visualize 10 samples
visualize_samples(dataset, num_samples=10)

