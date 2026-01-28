"""
Deep Q-Network (DQN) Agent for Tetris
Implements DQN with experience replay and target network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque


class DQN(nn.Module):
    """
    Deep Q-Network neural network
    Input: observation (grid state + piece info)
    Output: Q-values for each action
    """
    
    def __init__(self, input_size, output_size, hidden_size=256):
        super(DQN, self).__init__()
        
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


class ReplayBuffer:
    """
    Experience Replay Buffer
    Stores transitions and samples mini-batches for training
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random mini-batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration
    """
    
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1  # 下限を0.1に（常に10%の探索を維持）
        self.epsilon_decay = 0.9995  # 以前: 0.995（減衰が速すぎた）→ 0.9995に変更
        # 計算: 0.9995^10000 ≈ 0.006 で約10000エピソード後に0.01に到達
        self.learning_rate = 0.0001  # v10: 0.003 → 0.0001（損失爆発防止のため30倍削減）
        self.batch_size = 64
        self.target_update_freq = 1000  # Update target network every N steps
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Huber Loss（外れ値に強く、大きな報酬値でも安定）
        self.criterion = nn.SmoothL1Loss()  # SmoothL1Loss = Huber Loss
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0  # エピソード数をカウント
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
        Perform one training step
        
        Returns:
            loss: Training loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping（強化: 1.0 → 0.5）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # ε減衰はエピソド終了時に実行（train_step()では実行しない）
        # train_gpu.py側で end_episode() メソッドを呼び出す
        
        self.losses.append(loss.item())
        return loss.item()
    
    def end_episode(self):
        """
        End of episode callback - decay epsilon per episode, not per step
        This is the correct way to decay exploration rate
        """
        self.episodes += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
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
            self.episodes = checkpoint.get('episodes', 0)  # 互換性のため
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {e}")
