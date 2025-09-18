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
    Q = defaultdict(lambda: [0.0, 0.0])  # Q[state][action]
    policy = defaultdict(lambda: [0.5, 0.5])  # policy[state][action]
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

            # 1-step TD target
            td_target = reward + gamma * max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # Policy update: gradient ascent on expected return
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

        # m-step TD updates
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

            # Policy gradient update
            pi = softmax(policy[s])
            baseline = sum(policy[s][a_] * Q[s][a_] for a_ in range(2))
            for a_ in range(2):
                grad = (1 if a_ == a else 0) - pi[a_]
                policy[s][a_] += beta * grad * (Q[s][a] - baseline)

    return rewards

# ----- Plotting -----
def plot_curves(curve1, curve2, label1="TD(1)", label2="TD(5)", window=20):
    def moving_avg(x, w):
        return [np.mean(x[max(0, i - w):i]) for i in range(1, len(x) + 1)]

    plt.figure(figsize=(10, 5))
    plt.plot(curve1, alpha=0.3, label=f"{label1} (raw)")
    plt.plot(curve2, alpha=0.3, label=f"{label2} (raw)")
    plt.plot(moving_avg(curve1, window), label=f"{label1} (avg)", linewidth=2)
    plt.plot(moving_avg(curve2, window), label=f"{label2} (avg)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Policy Optimization with 1-step vs 5-step TD Learning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----- Run -----
env = ToyPOMDP()
r_td1 = train_policy_td1(env, episodes=200, history_len=2)
r_tdm = train_policy_tdm(env, episodes=200, history_len=2, m=5)

plot_curves(r_td1, r_tdm, label1="TD(1)", label2="TD(5)")
