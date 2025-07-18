"""
Machine Learning and Reinforcement Learning models module for the Indian stock trading bot.
Includes adversarial training, RL environments, and various ML models.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gym
from gym import spaces
import warnings
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class MLModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_adversarial_financial_data(self, history, epsilon=0.05, noise_factor=0.1, event_prob=0.1):
        try:
            adv_history = history.copy()
            price_std = history['Close'].std()
            volume_std = history['Volume'].std()
            
            min_price = 0.01
            max_price = history['Close'].max() * 2
            max_volume = history['Volume'].max() * 10
            
            for i in range(len(adv_history)):
                perturbation = np.random.uniform(-epsilon, epsilon) * price_std
                noise = np.random.normal(0, noise_factor * price_std)
                
                adv_history['Close'].iloc[i] += perturbation + noise
                adv_history['Open'].iloc[i] += perturbation + noise
                adv_history['High'].iloc[i] = max(adv_history['High'].iloc[i] + perturbation + noise, 
                                                adv_history['Open'].iloc[i], 
                                                adv_history['Close'].iloc[i])
                adv_history['Low'].iloc[i] = min(adv_history['Low'].iloc[i] + perturbation + noise, 
                                               adv_history['Open'].iloc[i], 
                                               adv_history['Close'].iloc[i])
                
                volume_perturbation = np.random.uniform(-epsilon, epsilon) * volume_std
                adv_history['Volume'].iloc[i] = max(0, adv_history['Volume'].iloc[i] + volume_perturbation)
                
                if np.random.random() < event_prob:
                    event_type = np.random.choice(['crash', 'spike'])
                    if event_type == 'crash':
                        drop_factor = np.random.uniform(0.85, 0.95)
                        adv_history['Close'].iloc[i] *= drop_factor
                        adv_history['Open'].iloc[i] *= drop_factor
                        adv_history['High'].iloc[i] *= drop_factor
                        adv_history['Low'].iloc[i] *= drop_factor
                        adv_history['Volume'].iloc[i] *= 1.5
                    else:
                        spike_factor = np.random.uniform(1.05, 1.15)
                        adv_history['Close'].iloc[i] *= spike_factor
                        adv_history['Open'].iloc[i] *= spike_factor
                        adv_history['High'].iloc[i] *= spike_factor
                        adv_history['Low'].iloc[i] *= spike_factor
                        adv_history['Volume'].iloc[i] *= 1.3
                
                adv_history['Close'].iloc[i] = np.clip(adv_history['Close'].iloc[i], min_price, max_price)
                adv_history['Open'].iloc[i] = np.clip(adv_history['Open'].iloc[i], min_price, max_price)
                adv_history['High'].iloc[i] = np.clip(adv_history['High'].iloc[i], min_price, max_price)
                adv_history['Low'].iloc[i] = np.clip(adv_history['Low'].iloc[i], min_price, max_price)
                adv_history['Volume'].iloc[i] = np.clip(adv_history['Volume'].iloc[i], 0, max_volume)
            
            adv_history.replace([np.inf, -np.inf], np.nan, inplace=True)
            adv_history.fillna({
                'Close': history['Close'].mean(),
                'Open': history['Open'].mean(),
                'High': history['High'].mean(),
                'Low': history['Low'].mean(),
                'Volume': history['Volume'].mean()
            }, inplace=True)
            
            return adv_history
        
        except Exception as e:
            logger.error(f"Error generating adversarial data: {e}")
            return history

class AdversarialStockTradingEnv(gym.Env):
    def __init__(self, history, current_price, ml_predicted_price, 
                 adversarial_freq, max_event_magnitude):
        super(AdversarialStockTradingEnv, self).__init__()
        self.history = history
        self.current_price = current_price
        self.ml_predicted_price = ml_predicted_price
        self.adversarial_freq = adversarial_freq
        self.max_event_magnitude = max_event_magnitude
        self.max_steps = len(history) - 1
        self.current_step = 0
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_shares = 100
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.event_occurred = 0
        return self._get_observation()
    
    def _get_observation(self):
        price = float(self.history["Close"].iloc[self.current_step])
        sma_50 = float(self.history["SMA_50"].iloc[self.current_step] or 0)
        sma_200 = float(self.history["SMA_200"].iloc[self.current_step] or 0)
        rsi = float(self.history["RSI"].iloc[self.current_step] or 50)
        macd = float(self.history["MACD"].iloc[self.current_step] or 0)
        volatility = float(self.history["Volatility"].iloc[self.current_step] or 0)
        ml_pred = self.ml_predicted_price if self.current_step == self.max_steps else price
        return np.array([
            price, sma_50, sma_200, rsi, macd, volatility,
            self.balance, self.shares_held, self.net_worth, ml_pred,
            self.event_occurred
        ], dtype=np.float32)
    
    def step(self, action):
        current_price = float(self.history["Close"].iloc[self.current_step])
        reward = 0
        
        if np.random.random() < self.adversarial_freq:
            event_magnitude = np.random.uniform(-self.max_event_magnitude, 
                                              self.max_event_magnitude)
            current_price *= (1 + event_magnitude)
            self.event_occurred = abs(event_magnitude)
        else:
            self.event_occurred = 0
        
        if action == 1:
            shares_to_buy = min(self.max_shares - self.shares_held, 
                              int(self.balance / current_price))
            cost = shares_to_buy * current_price
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
        elif action == 2:
            shares_to_sell = self.shares_held
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.balance += revenue
                self.shares_held = 0
                
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.initial_balance
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        if done:
            reward += (self.ml_predicted_price - current_price) * self.shares_held
            
        return self._get_observation(), reward, done, {}

class AdversarialQLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.event_threshold = 0.05
        
    def discretize_state(self, state):
        rounded_state = np.round(state[:-1], 2)
        event = 1 if state[-1] > self.event_threshold else 0
        return tuple(np.append(rounded_state, event))
    
    def get_action(self, state):
        state_key = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
            
        q_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_max - q_value)
        self.q_table[state_key][action] = new_q_value

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

    def adversarial_training_loop(self, X_train, y_train, X_test, y_test, input_size,
                                 seq_length=20, num_epochs=50, adv_lambda=0.1):
        try:
            logger.info("Cleaning input data...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

            logger.info("Converting data to tensors...")
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)

            logger.info("Creating data loaders...")
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            logger.info("Initializing models...")
            lstm_model = LSTMModel(input_size, 64, 2, 1, dropout=0.3).to(self.device)
            transformer_model = TransformerModel(input_size, 64, 8, 4, 1, dropout=0.2).to(self.device)

            lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
            transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001, weight_decay=1e-5)

            criterion = nn.MSELoss()

            def fgsm_attack(model, data, target, epsilon=0.01):
                data.requires_grad = True
                output = model(data)
                loss = criterion(output, target)
                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                perturbed_data = data + epsilon * data_grad.sign()
                return perturbed_data.detach()

            logger.info("Starting adversarial training...")
            lstm_losses = []
            transformer_losses = []

            for epoch in range(num_epochs):
                lstm_model.train()
                transformer_model.train()
                epoch_lstm_loss = 0
                epoch_transformer_loss = 0

                for batch_X, batch_y in train_loader:
                    # LSTM Training
                    lstm_optimizer.zero_grad()
                    lstm_output = lstm_model(batch_X)
                    lstm_loss = criterion(lstm_output.squeeze(), batch_y)

                    # Adversarial training for LSTM
                    adv_X = fgsm_attack(lstm_model, batch_X, batch_y)
                    adv_output = lstm_model(adv_X)
                    adv_loss = criterion(adv_output.squeeze(), batch_y)
                    total_lstm_loss = lstm_loss + adv_lambda * adv_loss

                    total_lstm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    lstm_optimizer.step()

                    # Transformer Training
                    transformer_optimizer.zero_grad()
                    transformer_output = transformer_model(batch_X)
                    transformer_loss = criterion(transformer_output.squeeze(), batch_y)

                    # Adversarial training for Transformer
                    adv_X_transformer = fgsm_attack(transformer_model, batch_X, batch_y)
                    adv_transformer_output = transformer_model(adv_X_transformer)
                    adv_transformer_loss = criterion(adv_transformer_output.squeeze(), batch_y)
                    total_transformer_loss = transformer_loss + adv_lambda * adv_transformer_loss

                    total_transformer_loss.backward()
                    torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
                    transformer_optimizer.step()

                    epoch_lstm_loss += total_lstm_loss.item()
                    epoch_transformer_loss += total_transformer_loss.item()

                lstm_losses.append(epoch_lstm_loss / len(train_loader))
                transformer_losses.append(epoch_transformer_loss / len(train_loader))

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs}, "
                                f"LSTM Loss: {lstm_losses[-1]:.6f}, "
                                f"Transformer Loss: {transformer_losses[-1]:.6f}")

            # Evaluation
            lstm_model.eval()
            transformer_model.eval()

            with torch.no_grad():
                lstm_pred = lstm_model(X_test_tensor).squeeze().cpu().numpy()
                transformer_pred = transformer_model(X_test_tensor).squeeze().cpu().numpy()

            lstm_mse = mean_squared_error(y_test, lstm_pred)
            transformer_mse = mean_squared_error(y_test, transformer_pred)

            # Ensemble prediction
            ensemble_pred = (lstm_pred + transformer_pred) / 2
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)

            logger.info(f"LSTM MSE: {lstm_mse:.6f}")
            logger.info(f"Transformer MSE: {transformer_mse:.6f}")
            logger.info(f"Ensemble MSE: {ensemble_mse:.6f}")

            return {
                "success": True,
                "lstm_prediction": float(lstm_pred[-1]) if len(lstm_pred) > 0 else 0,
                "transformer_prediction": float(transformer_pred[-1]) if len(transformer_pred) > 0 else 0,
                "ensemble_prediction": float(ensemble_pred[-1]) if len(ensemble_pred) > 0 else 0,
                "lstm_mse": float(lstm_mse),
                "transformer_mse": float(transformer_mse),
                "ensemble_mse": float(ensemble_mse),
                "training_losses": {
                    "lstm": [float(x) for x in lstm_losses],
                    "transformer": [float(x) for x in transformer_losses]
                }
            }

        except Exception as e:
            logger.error(f"Error in adversarial training: {e}")
            return {
                "success": False,
                "message": f"Error in adversarial training: {str(e)}"
            }
