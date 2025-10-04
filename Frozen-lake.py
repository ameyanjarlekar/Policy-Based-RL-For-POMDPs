import numpy as np
import gym
from collections import deque, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
import time

start_time = time.time()
# --- POMDP Wrapper: Noisy discrete observations ---
class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_noise_prob=0.1):
        super().__init__(env)
        self.obs_noise_prob = obs_noise_prob
        self.n_states = env.observation_space.n

    def observation(self, obs):
        if np.random.rand() < self.obs_noise_prob:
            noisy_obs = np.random.choice([s for s in range(self.n_states) if s != obs])
            return noisy_obs
        return obs

# --- Utility to encode history as a tuple (for dict key) ---
def encode_history_tuple(history):
    return tuple(history)

# --- Tabular Q-function ---
class TabularQFunction:
    def __init__(self, gamma=0.99, alpha=0.1):
        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.num_actions = None  # to be set externally

    def set_action_space(self, num_actions):
        self.num_actions = num_actions

    def predict(self, history, action):
        return self.q[history][action]

    def predict_all(self, history):
        return self.q[history]

    def update(self, history, action, reward, next_history, next_action, done):
        q_current = self.q[history][action]
        q_next = 0 if done else self.q[next_history][next_action]
        target = reward + self.gamma * q_next
        self.q[history][action] = self.q[history][action] + self.alpha * (target - q_current)

    def reset(self):
        self.q = defaultdict(lambda: np.zeros(self.num_actions))

class PolitexTabularAgent:
    def __init__(self, num_actions, eta=1):
        self.num_actions = num_actions
        self.eta = eta
        self.policy_weights = defaultdict(lambda: np.zeros(num_actions))

    def policy(self, history):
        logits = self.policy_weights[history]
        max_logit = np.max(logits)
        exp_logits = np.exp(logits-max_logit)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def select_action(self, history):
        probs = self.policy(history)
        return np.random.choice(self.num_actions, p=probs)

    def update(self, history, q_values):
        for a in range(self.num_actions):
            self.policy_weights[history][a] += self.eta * q_values[a]

# --- Training Loop ---
def train(env_name='FrozenLake-v1', num_episodes=5000000, k=3, tau=1000, obs_noise_prob=0.3):
    env = gym.make(env_name, is_slippery=False)
    env = NoisyObservationWrapper(env, obs_noise_prob=obs_noise_prob)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    # Print environment details
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Print the actual map being used (if it's the default 4x4)
    print("\nFrozenLake Map Layout:")
    for row in env.unwrapped.desc:
        print(row.tobytes().decode("utf-8"))
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.P[state][action]
            print(f"From state {state}, taking action {action}:")
            for prob, next_state, reward, done in transitions:
                print(f"  -> Next state {next_state}, prob={prob}, reward={reward}, done={done}")
    agent = PolitexTabularAgent(num_actions)
    q_func = TabularQFunction()
    q_func.set_action_space(num_actions)

    rewards_per_episode = []
    episode_buffer = []

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        if isinstance(obs, tuple):  # for gymnasium compatibility
            obs = obs[0]
        done = False
        total_reward = 0
        #print("started")

        history_deque = deque([(-1,-1)] * k, maxlen=k)  # (obs, action) tuples
        current_history = encode_history_tuple(history_deque)
        action = agent.select_action(current_history)

        episode_transitions = []
        while not done:
            next_obs, reward, done, *_ = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            total_reward += reward
            history_deque.append((next_obs,action))
            next_history = encode_history_tuple(history_deque)
            next_action = agent.select_action(next_history)

            episode_transitions.append((current_history, action, reward, next_history, next_action, done))
            #obs = next_obs
            current_history = next_history
            action = next_action

        for h, a, r, h_next, a_next, d in episode_transitions:
            q_func.update(h, a, r, h_next, a_next, d)

        if (episode + 1) % tau == 0:
            for h in q_func.q.keys():
                q_vals = q_func.predict_all(h)
                agent.update(h, q_vals)
            q_func.reset()

        rewards_per_episode.append(total_reward)

    return agent, q_func, rewards_per_episode

# --- Plotting ---
def plot_rewards(rewards, window=1000):
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label='Episode reward')
    plt.plot(range(window - 1, len(rewards)), moving_avg, label=f'{window}-episode moving avg', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('TD + POLITEX with Tabular Q over History (POMDP on FrozenLake)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Last moving average value: {moving_avg[-1]}")

# --- Run everything ---
if __name__ == "__main__":
    agent, q_func, rewards = train()
    plot_rewards(rewards)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
