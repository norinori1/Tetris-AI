"""
Deep Q-Network (DQN) Agent for Tetris
Implements DQN with Prioritized Experience Replay and target network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os


class SumTree:
    """
    Sum Tree data structure for Prioritized Experience Replay.
    Allows O(log n) sampling based on priorities.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find sample on leaf node based on priority value s"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new experience with priority"""
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority of existing experience"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get experience based on priority value s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on TD-error priority.
    Positive reward experiences are more likely to be sampled.
    """
    
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full)
        self.beta = beta_start  # Importance sampling weight (increases over time)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 0.01  # Small constant to avoid zero priority
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority (will be updated during training)"""
        experience = (state, action, reward, next_state, done)
        # Give higher initial priority to experiences with positive rewards
        if reward > 0:
            priority = self.max_priority * 2  # Boost positive reward experiences
        else:
            priority = self.max_priority
        self.tree.add(priority ** self.alpha, experience)
    
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        batch = []
        indices = []
        priorities = []
        
        # Update beta (importance sampling weight increases over time)
        self.frame += 1
        beta_progress = min(1.0, self.frame / self.beta_frames)
        self.beta = self.beta_start + beta_progress * (1.0 - self.beta_start)
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None or (isinstance(data, (int, float)) and data == 0):
                # Handle empty slots by resampling
                s = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        total = self.tree.total()
        probabilities = np.array(priorities) / total
        weights = (self.tree.n_entries * probabilities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(indices),
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries


class DQN(nn.Module):
    """
    Deep Q-Network neural network
    Input: observation (grid state + piece info)
    Output: Q-values for each action
    """
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(DQN, self).__init__()
        
        # Larger network for better feature extraction
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    DQN Agent with Prioritized Experience Replay and improved exploration
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters - optimized for line clearing
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Higher minimum for continued exploration
        self.epsilon_decay = 0.9995  # Slower decay for more exploration
        self.learning_rate = 0.0005  # Slightly higher learning rate
        self.batch_size = 128  # Larger batch size for stability
        self.target_update_freq = 500  # More frequent target updates
        
        # Networks with larger capacity
        self.policy_net = DQN(state_size, action_size, hidden_size=512).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size=512).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss for stability
        
        # Prioritized Replay buffer with larger capacity
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        
        # Training statistics
        self.steps = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            action: Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step with Prioritized Experience Replay
        
        Returns:
            loss: Training loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy net to select actions, target net to evaluate
        with torch.no_grad():
            # Select best action using policy network
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Evaluate using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for priority update
        td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # Compute weighted loss (importance sampling)
        element_wise_loss = self.criterion(current_q_values.squeeze(), target_q_values)
        loss = (element_wise_loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Use weights_only=True for security (prevents arbitrary code execution)
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {e}")
