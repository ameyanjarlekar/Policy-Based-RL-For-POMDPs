def print_experiment_summary(results, window=20):
    print(f"{'History':>7} | {'Algo':>5} | {'Episodes':>8} | {'EpLen':>5} | {'Alpha':>5} | {'Gamma':>5} | {'AvgReward':>10} | {'Notes'}")
    print("-" * 80)
    for (hist_len, algo), rewards in results.items():
        avg_last = np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards)
        notes = "5-step TD" if algo == 'TD5' else "1-step TD"
        print(f"{hist_len:>7} | {algo:>5} | {200:>8} | {20:>5} | {0.1:>5.2f} | {0.9:>5.2f} | {avg_last:>10.3f} | {notes}")

def print_learning_curve_samples(results, window=20, step=20):
    print(f"{'Episode':>7}", end='')
    for hist_len in [1, 2]:
        for algo in ['TD1', 'TD5']:
            print(f" | H{hist_len} {algo:>3}", end='')
    print()
    print("-" * 70)
    for ep in range(step, 201, step):
        print(f"{ep:7d}", end='')
        for hist_len in [1, 2]:
            for algo in ['TD1', 'TD5']:
                rewards = results[(hist_len, algo)]
                mov_avg = np.mean(rewards[max(0, ep - window):ep])
                print(f" | {mov_avg:7.2f}", end='')
        print()

# Usage example:
print_experiment_summary(results)
print()
print_learning_curve_samples(results)
