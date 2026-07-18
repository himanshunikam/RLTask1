"""
Task 1 - Solving the SeaShanty gridworld with tabular Q-learning.

SeaShanty(size=12, period=6) is a dynamic gridworld: lethal "water" cells move
every step following a deterministic wave pattern. Stepping into water ends the
episode with -100; reaching the goal gives +100; every other step gives 0.

Key ideas used here (see scratchpad experiments / report for the full story):

  * Phase augmentation. The wave pattern is a deterministic function of the step
    count t, and reset() restarts t at 0 every episode. Plain position-based
    Q-learning therefore breaks (the same cell is safe or lethal depending on t)
    and solves only ~10% of environments. Augmenting the state with the in-episode
    phase (t mod K) restores the Markov property.

  * Potential-based reward shaping (Ng et al., 1999). With step_cost = 0 the true
    reward is sparse, so we add F = gamma*phi(s') - phi(s) with phi = -dist-to-goal.
    This preserves the optimal policy and only guides learning - it is NEVER added
    to the reported "real" cumulative reward.

  * Optimistic initialisation. Initialising Q optimistically makes the greedy
    policy try unvisited actions systematically, giving exploration without the
    random-walk deaths that a high epsilon-floor would cause in this lethal env.

Hyperparameters were tuned on a SEPARATE set of environments (seeds 0-59) from the
ones used for the final plot (seeds 1000-1099), as required by the task.
"""

from gridworld import SeaShanty
from plot import plot_q_table, plot_v_table
import numpy as np
import random
import matplotlib.pyplot as plt


def _seed_everything(seed):
    # SeaShanty places start/goal with the stdlib `random` module and picks the
    # wave rotation with `np.random`, so BOTH must be seeded for a reproducible env.
    np.random.seed(seed)
    random.seed(seed)


# ---- Tuned hyperparameters --------------------------------------------------
GAMMA      = 0.965
ALPHA      = 0.9     # env is deterministic per episode -> exact (alpha=1) updates
EPS_START  = 0.973
EPS_DECAY  = 0.600    # epsilon -> ~0 fast; the shaped/optimistic greedy policy explores
Q_INIT     = 90.0    # optimistic initialisation
SHAPE_W    = 125.0    # weight of the distance-to-goal shaping potential
K          = 38      # phase resolution ~ round(2*pi*period) = round(2*pi*6)
MAX_STEPS  = 100
N_EPISODES = 500
N_ENVS     = 100


def _find_goal(env):
    """Goal position read from the (public) character grid."""
    g = np.argwhere(env.grid == 'G')
    return tuple(g[0]) if len(g) else env._goal

def DoubleQLearning(env, n_episodes=N_EPISODES, return_tables= False):
    ""
    num_pos= env.num_states()
    num_act = env.num_actions()
    ncol = len(env.grid[0])
    goal = _find_goal(env)
    scale = 2*(env.size -1)

    def aug(obs, t):
        # phase-augmented state index: (position, step mod K)
        return obs * K + (t % K)

    def phi(obs):
        # shaping potential: closer to the goal -> higher potential
        r, c = divmod(obs, ncol)
        return -(abs(r - goal[0]) + abs(c - goal[1])) / scale * SHAPE_W

    q_1     = np.full((num_pos * K, num_act), Q_INIT, dtype=float)
    q_2     = np.full((num_pos * K, num_act), Q_INIT, dtype=float)
    visits = np.zeros((num_pos * K, num_act), dtype=np.int64)
    eps    = EPS_START
    real_rewards = np.empty(n_episodes)

    for ep in range(n_episodes):
        obs = env.reset()
        t =0
        s= aug(obs, t)
        done = False
        real =0.0

        while not done and t < MAX_STEPS:
            if np.random.rand() < eps:
                action = np.random.randint(num_act)
            else:
                action = np.argmax((q_1[s]+q_2[s])/2)

            next_obs, reward, done =env.step(action)
            t+=1
            real += reward

            next_s = aug(next_obs, t)
            shaped = reward + GAMMA * phi(next_obs) - phi(obs)
            if np.random.rand() < 0.5:
                target = shaped if done else shaped + GAMMA * q_2[next_s, np.argmax(q_1[next_s])]
                q_1[s, action] = q_1[s, action] + ALPHA*(target - q_1[s, action])
            else:
                target = shaped if done else shaped + GAMMA * q_1[next_s, np.argmax(q_2[next_s])]
                q_2[s, action] = q_2[s, action] + ALPHA*(target - q_2[s, action])
            
            visits[s, action] += 1
            s, obs = next_s, next_obs

        eps *= EPS_DECAY
        real_rewards[ep] = real

    if return_tables:
        return real_rewards, (q_1+q_2)/2, visits
    return real_rewards

def DoubleQLearningSimultaenous(env, n_episodes=N_EPISODES, return_tables= False):
    ""
    num_pos= env.num_states()
    num_act = env.num_actions()
    ncol = len(env.grid[0])
    goal = _find_goal(env)
    scale = 2*(env.size -1)

    def aug(obs, t):
        # phase-augmented state index: (position, step mod K)
        return obs * K + (t % K)

    def phi(obs):
        # shaping potential: closer to the goal -> higher potential
        r, c = divmod(obs, ncol)
        return -(abs(r - goal[0]) + abs(c - goal[1])) / scale * SHAPE_W

    q_1     = np.full((num_pos * K, num_act), Q_INIT, dtype=float)
    q_2     = np.full((num_pos * K, num_act), Q_INIT, dtype=float)
    visits = np.zeros((num_pos * K, num_act), dtype=np.int64)
    eps    = EPS_START
    real_rewards = np.empty(n_episodes)

    for ep in range(n_episodes):
        obs = env.reset()
        t =0
        s= aug(obs, t)
        done = False
        real =0.0

        while not done and t < MAX_STEPS:
            if np.random.rand() < eps:
                action = np.random.randint(num_act)
            else:
                action = np.argmax((q_1[s]+q_2[s])/2)

            next_obs, reward, done =env.step(action)
            t+=1
            real += reward

            next_s = aug(next_obs, t)
            shaped = reward + GAMMA * phi(next_obs) - phi(obs)
            if done:
                q_1[s, action] += ALPHA * (shaped - q_1[s, action])
                q_2[s, action] += ALPHA * (shaped - q_2[s, action])
            else:
                t1 = shaped + GAMMA * q_2[next_s, np.argmax(q_1[next_s])]
                t2 = shaped + GAMMA * q_1[next_s, np.argmax(q_2[next_s])]
                q_1[s, action] += ALPHA * (t1 - q_1[s, action])
                q_2[s, action] += ALPHA * (t2 - q_2[s, action])

            
            visits[s, action] += 1
            s, obs = next_s, next_obs

        eps *= EPS_DECAY
        real_rewards[ep] = real

    if return_tables:
        return real_rewards, (q_1+q_2)/2, visits
    return real_rewards
def q_learning_single(env, n_episodes=N_EPISODES, return_tables=False):
    """
    Run tabular Q-learning on ONE SeaShanty environment.

    :param return_tables: if True, also return the learned (augmented) Q-table and
                          a same-shaped array of visit counts per (state, action).
    :return: 1D array of the REAL cumulative reward (sum of true, non-shaped
             rewards) per episode; plus (q, visits) when return_tables is True.
    """
    num_pos = env.num_states()
    num_act = env.num_actions()
    ncol    = len(env.grid[0])
    goal    = _find_goal(env)
    scale   = 2 * (env.size - 1)                 # max manhattan distance, for normalising phi

    def aug(obs, t):
        # phase-augmented state index: (position, step mod K)
        return obs * K + (t % K)

    def phi(obs):
        # shaping potential: closer to the goal -> higher potential
        r, c = divmod(obs, ncol)
        return -(abs(r - goal[0]) + abs(c - goal[1])) / scale * SHAPE_W

    q      = np.full((num_pos * K, num_act), Q_INIT, dtype=float)
    visits = np.zeros((num_pos * K, num_act), dtype=np.int64)
    eps    = EPS_START
    real_rewards = np.empty(n_episodes)

    for ep in range(n_episodes):
        obs  = env.reset()
        t    = 0
        s    = aug(obs, t)
        done = False
        real = 0.0                                # sum of TRUE rewards this episode

        while not done and t < MAX_STEPS:
            if np.random.rand() < eps:
                action = np.random.randint(num_act)
            else:
                action = np.argmax(q[s])

            next_obs, reward, done = env.step(action)
            t += 1
            real += reward                        # accumulate the true reward only

            next_s = aug(next_obs, t)
            shaped = reward + GAMMA * phi(next_obs) - phi(obs)
            target = shaped if done else shaped + GAMMA * np.max(q[next_s])
            q[s, action] += ALPHA * (target - q[s, action])
            visits[s, action] += 1

            s, obs = next_s, next_obs

        eps *= EPS_DECAY
        real_rewards[ep] = real

    if return_tables:
        return real_rewards, q, visits
    return real_rewards

def main():
    # Final-plot environments: seeds 1000..1099, disjoint from the tuning seeds.
    all_curves = np.empty((N_ENVS, N_EPISODES))
    for i in range(N_ENVS):
        _seed_everything(1000 + i)                # fresh, reproducible, different env
        env = SeaShanty(size=12, period=6)
        all_curves[i] = DoubleQLearningSimultaenous(env)
        print(f"env {i + 1:3d}/{N_ENVS} | final-10 avg real reward: "
              f"{all_curves[i, -10:].mean():6.1f}")

    avg = all_curves.mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(avg, color='tab:orange', label='q-learning')
    plt.axhline(100, ls='--', lw=0.8, color='gray')
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.title('SeaShanty(size=12, period=6): real cumulative reward '
              f'(averaged over {N_ENVS} environments)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=120)
    print(f"\nSaved learning_curve.png | reward at ep50/100/150/199: "
          f"{avg[50]:.0f} / {avg[100]:.0f} / {avg[150]:.0f} / {avg[-1]:.0f}")
    plt.show()


if __name__ == '__main__':
    main()