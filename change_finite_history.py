import random
import math
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# ----- Toy POMDP -----
class ToyPOMDP:
    def __init__(self):
        self.states = [0, 1]
        self.actions = [0, 1]
        self.observations = [0, 1]
        self.state = random.choice(self.states)

    def reset(self):
        self.state = random.choice(self.states)
        return self.observe()

    def observe(self):
        return self.state if random.random() < 0.8 else 1 - self.state

    def step(self, action):
        if random.random() > 0.9:
            self.state = 1 - self.state
        reward = 1 if self.state == 1 and action == 1 else 0
        obs = self.observe()
        return obs, reward

# ----- Softmax Policy -----
def softmax(q_vals, tau=1.0):
    exps = np.exp(np.array(q_vals) / tau)
    return exps / np.sum(exps)

def choose_action(policy_probs):
    return np.random.choice(len(policy_probs), p=policy_probs)

# ----- 1-step TD Learning -----
def train_policy_td1(env, episodes=200, history_len=2, alpha=0.1, beta=0.05, gamma=0.95):
    Q = defaultdict(lambda: [0.0, 0.0])
    policy = defaultdict(lambda: [0.5, 0.5])
    rewards = []
    history = deque(maxlen=history_len)

    for ep in range(episodes):
        obs = env.reset()
        history.clear()
        history.append((None, obs))
        ep_reward = 0

        for t in range(30):
            state = tuple(history)
            pi = softmax(policy[state])
            action = choose_action(pi)
            obs_next, reward = env.step(action)
            ep_reward += reward

            next_state = tuple(list(history)[-history_len+1:] + [(action, obs_next)])
            history.append((action, obs_next))

            td_target = reward + gamma * max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            baseline = sum(policy[state][a] * Q[state][a] for a in range(2))
            for a in range(2):
                grad = (1 if a == action else 0) - pi[a]
                policy[state][a] += beta * grad * (Q[state][action] - baseline)

        rewards.append(ep_reward)
    return rewards

# ----- m-step TD Learning -----
def train_policy_tdm(env, episodes=200, history_len=2, alpha=0.1, beta=0.05, gamma=0.95, m=5):
    Q = defaultdict(lambda: [0.0, 0.0])
    policy = defaultdict(lambda: [0.5, 0.5])
    rewards = []

    for ep in range(episodes):
        obs = env.reset()
        history = deque([(None, obs)], maxlen=history_len)
        trajectory = []

        for t in range(30):
            state = tuple(history)
            pi = softmax(policy[state])
            action = choose_action(pi)
            obs_next, reward = env.step(action)
            next_state = tuple(list(history)[-history_len+1:] + [(action, obs_next)])
            history.append((action, obs_next))
            trajectory.append((state, action, reward, next_state))

        ep_reward = sum(r for (_, _, r, _) in trajectory)
        rewards.append(ep_reward)

        for i in range(len(trajectory)):
            G = 0.0
            for j in range(m):
                if i + j >= len(trajectory): break
                _, _, rj, _ = trajectory[i + j]
                G += (gamma ** j) * rj

            if i + m < len(trajectory):
                s_boot = trajectory[i + m][0]
                G += (gamma ** m) * max(Q[s_boot])

            s, a, _, _ = trajectory[i]
            td_error = G - Q[s][a]
            Q[s][a] += alpha * td_error

            pi = softmax(policy[s])
            baseline = sum(policy[s][a_] * Q[s][a_] for a_ in range(2))
            for a_ in range(2):
                grad = (1 if a_ == a else 0) - pi[a_]
                policy[s][a_] += beta * grad * (Q[s][a] - baseline)

    return rewards

# ----- Plotting -----
def moving_avg(x, w):
    return [np.mean(x[max(0, i - w):i]) for i in range(1, len(x) + 1)]

def plot_comparison(all_curves, title, window=100):
    plt.figure(figsize=(10, 5))
    for h_len, rewards in all_curves.items():
        plt.plot(rewards, alpha=0.2, label=f"History {h_len} (raw)")
        plt.plot(moving_avg(rewards, window), label=f"History {h_len} (avg)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----- Run Everything -----
env = ToyPOMDP()
history_lengths = [1, 2, 3, 4]

results_td1 = {}
results_tdm = {}

print("Training TD(1)...")
for h in history_lengths:
    results_td1[h] = train_policy_td1(env, history_len=h, episodes=250)

print("Training TD(5)...")
for h in history_lengths:
    results_tdm[h] = train_policy_tdm(env, history_len=h, m=30, episodes=250)

plot_comparison(results_td1, "Policy Optimization with 1-step TD (TD(1))")
plot_comparison(results_tdm, "Policy Optimization with 30-step TD (TD(30))")
