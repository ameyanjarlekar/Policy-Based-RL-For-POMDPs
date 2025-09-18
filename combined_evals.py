import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, defaultdict

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# POMDP Setup
states = [0, 1]
actions = [0, 1]
observations = [0, 1]

# Transition probabilities: P(s' | s, a)
T = {
    (0, 0): [0.9, 0.1],
    (0, 1): [0.2, 0.8],
    (1, 0): [0.1, 0.9],
    (1, 1): [0.8, 0.2],
}

# Observation probabilities: P(o | s)
O = {
    0: [0.8, 0.2],
    1: [0.3, 0.7],
}

# Reward function: R(s, a)
R = {
    (0, 0): 1,
    (0, 1): 0,
    (1, 0): 0,
    (1, 1): 1,
}

def sample_from_distribution(dist):
    return np.random.choice(len(dist), p=dist)

def run_episode(policy, history_length, alpha, gamma, m_step=False, m=5):
    state = np.random.choice(states)
    observation = sample_from_distribution(O[state])
    history = deque(maxlen=history_length)
    history.append((None, observation))  # no action before first obs

    total_reward = 0
    memory = []

    for t in range(20):  # fixed episode length
        state_key = tuple(history)
        if state_key not in policy:
            policy[state_key] = np.random.choice(actions)
        action = policy[state_key]
        reward = R[(state, action)]
        next_state = sample_from_distribution(T[(state, action)])
        next_obs = sample_from_distribution(O[next_state])
        history.append((action, next_obs))

        memory.append((state_key, action, reward))

        total_reward += reward
        state = next_state

    # TD updates (1-step or m-step)
    for i in range(len(memory)):
        state_key, action, reward = memory[i]

        if m_step:
            G = 0
            discount = 1
            for j in range(i, min(i + m, len(memory))):
                G += discount * memory[j][2]
                discount *= gamma
        else:
            if i + 1 < len(memory):
                G = reward + gamma * memory[i + 1][2]
            else:
                G = reward

        q_values[state_key][action] = (1 - alpha) * q_values[state_key][action] + alpha * G

        # Update policy
        best_a = max(q_values[state_key], key=q_values[state_key].get)
        policy[state_key] = best_a

    return total_reward

def train(history_length, alpha=0.1, gamma=0.9, episodes=200, m_step=False, m=5):
    global q_values
    q_values = defaultdict(lambda: {a: 0.0 for a in actions})
    policy = {}
    rewards = []
    for ep in range(episodes):
        ep_reward = run_episode(policy, history_length, alpha, gamma, m_step=m_step, m=m)
        rewards.append(ep_reward)
    return rewards

def moving_avg(x, w):
    return [np.mean(x[max(0, i - w):i]) for i in range(1, len(x) + 1)]

# Run experiments
results = {}

for hist_len in [1, 2]:
    rewards_td1 = train(hist_len, m_step=False, episodes=2000)
    rewards_td5 = train(hist_len, m_step=True, m=20, episodes=2000)
    results[(hist_len, 'TD1')] = rewards_td1
    results[(hist_len, 'TD5')] = rewards_td5

# Plotting with moving average
plt.figure(figsize=(10, 6))
window = 100
for hist_len in [1, 2]:
    raw_td1 = results[(hist_len, 'TD1')]
    raw_td5 = results[(hist_len, 'TD5')]
    #plt.plot(raw_td1, alpha=0.2, label=f'History {hist_len}, TD(1) (raw)')
    plt.plot(moving_avg(raw_td1, window), label=f'History {hist_len}, Our', linewidth=2)
    #plt.plot(raw_td5, alpha=0.2, label=f'History {hist_len}, TD(5) (raw)')
    plt.plot(moving_avg(raw_td5, window), label=f'History {hist_len}, Cayci el al.', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve: History Lengths 1 & 2, Our vs Cayci et al.')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
