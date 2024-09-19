import numpy as np
import pandas as pd
import random

# Load the Nifty50 data
nifty50_df = pd.read_csv('NIFTY 50 - Minute data 2015 to Aug 2024.csv')

# Convert date to datetime and sort the data
nifty50_df['date'] = pd.to_datetime(nifty50_df['date'])
nifty50_df.sort_values(by='date', inplace=True)

# Calculate short-term (5-min) and long-term (20-min) moving averages
short_window = 5
long_window = 20
nifty50_df['short_ma'] = nifty50_df['close'].rolling(window=short_window, min_periods=1).mean()
nifty50_df['long_ma'] = nifty50_df['close'].rolling(window=long_window, min_periods=1).mean()

# Add the state (features) for reinforcement learning
nifty50_df['state'] = nifty50_df[['close', 'short_ma', 'long_ma']].values.tolist()

# Reduce the dataset for training (first year)
nifty50_df_small = nifty50_df[nifty50_df['date'].dt.year == 2015].copy()

# Define the action space: 0 = Hold, 1 = Buy, 2 = Sell
actions = [0, 1, 2]

# Define parameters for Q-learning
state_space_size = 50  # Reduced discretization size
q_table = np.zeros((state_space_size, len(actions)))  # Q-table initialization
alpha = 0.01  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay factor for epsilon
min_epsilon = 0.01
episode_count = 50  # Number of episodes for training

# Discretize the state space
def discretize_state(state, bins=50):
    # Use the close price for discretization, ensuring it fits within the bin size
    close = state[0]
    return min(bins - 1, int(np.digitize(close, np.linspace(min(nifty50_df_small['close']), max(nifty50_df_small['close']), bins))))

# Simulate the Q-learning trading process
initial_capital = 100000
cash = initial_capital
holdings = 0
position = 0
total_reward = 0

for episode in range(episode_count):
    print(episode)
    state = discretize_state(nifty50_df_small['state'].iloc[0])  # Starting state
    
    for i in range(1, len(nifty50_df_small)):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit best action from Q-table
        
        # Execute action
        if action == 1 and position == 0:  # Buy
            cash -= nifty50_df_small['close'].iloc[i]
            holdings = nifty50_df_small['close'].iloc[i]
            position = 1
        elif action == 2 and position == 1:  # Sell
            cash += nifty50_df_small['close'].iloc[i]
            holdings = 0
            position = 0
        
        # Calculate reward (change in total portfolio value)
        new_total_value = cash + (holdings if position == 1 else 0)
        reward = new_total_value - (cash + holdings)  # Change in value
        total_reward += reward
        
        # Move to the next state
        next_state = discretize_state(nifty50_df_small['state'].iloc[i])
        
        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        
        # Update state
        state = next_state
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Final portfolio value and total reward
final_portfolio_value = cash + holdings
print("Final Portfolio Value:", final_portfolio_value)
print("Total Reward:", total_reward)
