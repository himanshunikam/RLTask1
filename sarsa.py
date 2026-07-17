from gridworld import SeaShanty
from plot import plot_q_table, plot_v_table
import numpy as np
import matplotlib.pyplot as plt
import random
# Hyperparams
GAMMA      = 0.85
ALPHA      = 1.0     # env is deterministic per episode -> exact (alpha=1) updates
EPS_START  = 1.0
EPS_DECAY  = 0.95    # epsilon -> ~0 fast; the shaped/optimistic greedy policy explores
Q_INIT     = 0
SHAPE_W = 100
K = 38
MAX_STEPS= 100
N_EPISODES = 500
N_ENVS= 100
N_STEP=4

def _seed_everything(seed):
    # SeaShanty places start/goal with the stdlib `random` module and picks the
    # wave rotation with `np.random`, so BOTH must be seeded for a reproducible env.
    np.random.seed(seed)
    random.seed(seed)

def _find_goal(env):
    """Goal position read from the (public) character grid."""
    g = np.argwhere(env.grid == 'G')
    return tuple(g[0]) if len(g) else env._goal


def sarsa(env, n_eps, return_tables):
    num_states = env.num_states()
    num_actions = env.num_actions()
    scale = 2*(env.size -1)
    ncol = len(env.grid[0])
    goal = _find_goal(env)

    def aug(obs, t):
        # phase-augmented state index: (position, step mod K)
        return obs * K + (t % K)
    
    def phi(obs):
        # shaping potential: closer to the goal -> higher potential
        r, c = divmod(obs, ncol)
        return -(abs(r - goal[0]) + abs(c - goal[1])) / scale * SHAPE_W

    
    q_table = np.full((num_states*K, num_actions), Q_INIT, dtype=float)
    visits = np.zeros((num_states*K, num_actions))
    epsilon = EPS_START
    real_rewards = np.empty(n_eps)

    for ep in range(n_eps):
        obs = env.reset()
        t=0
        state=aug(obs, t)
        done = False
        real =0

        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])
        
        while not done and t< MAX_STEPS:

            next_state, reward, done = env.step(action)
            t+=1
            shaped = reward + GAMMA * phi(next_state) - phi(obs)
            next_obs = next_state
            next_state = aug(next_state, t)
            real += reward

            if np.random.rand() < epsilon :
                next_action = np.random.randint(num_actions)
            else :
                next_action = np.argmax(q_table[next_state])
            
            

            if done :
                q_table[state, action] += ALPHA *(shaped - q_table[state, action])
            else:
                q_table[state, action] += ALPHA*(shaped + GAMMA*q_table[next_state, next_action] - q_table[state, action])

            visits[state, action] += 1
            state = next_state
            action = next_action     
            obs = next_obs
        epsilon *= EPS_DECAY
        real_rewards[ep] = real

    
    if return_tables:
        return real_rewards, q_table, visits
    
    return real_rewards


def n_step_sarsa(env, n_eps, return_tables):
    num_states = env.num_states()
    num_actions = env.num_actions()
    scale = 2*(env.size -1)
    ncol = len(env.grid[0])
    goal = _find_goal(env)
    epsilon = EPS_START

    q_table = np.full((num_states*K, num_actions), Q_INIT, dtype=float)
    visits = np.zeros((num_states*K, num_actions))

    def aug(obs, t):
        # phase-augmented state index: (position, step mod K)
        return obs * K + (t % K)
    
    def phi(obs):
        # shaping potential: closer to the goal -> higher potential
        r, c = divmod(obs, ncol)
        return -(abs(r - goal[0]) + abs(c - goal[1])) / scale * SHAPE_W
    

    def eps_greedy(s):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        return np.argmax(q_table[s])

    for ep in range(n_eps):
        obs   = env.reset()
        s     = aug(obs, 0)
        a     = eps_greedy(s)

        states  = [s]        # S_0
        actions = [a]        # A_0
        rewards = [0.0]      # index shift: rewards[i] == R_i (reward AFTER A_{i-1})
        raw     = [obs]      # raw obs, for phi() of each stored state

        T    = np.inf        # terminal time (unknown until we hit it)
        real = 0.0           # sum of TRUE rewards this episode
        tt   = 0             # current time index

        while True:
            if tt < T:
                next_obs, reward, done = env.step(actions[tt])
                real += reward
                shaped = reward + GAMMA * phi(next_obs) - phi(raw[tt])
                rewards.append(shaped)
                states.append(aug(next_obs, tt + 1))
                raw.append(next_obs)
                if done or (tt + 1) >= MAX_STEPS:
                    T = tt + 1                      # episode ends / truncates here
                else:
                    actions.append(eps_greedy(states[tt + 1]))

            tau = tt - N_STEP + 1                    # time whose estimate we update now
            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + N_STEP, T) + 1):
                    G += (GAMMA ** (i - tau - 1)) * rewards[i]
                if tau + N_STEP < T:                 # not past the end -> bootstrap
                    G += (GAMMA ** N_STEP) * q_table[states[tau + N_STEP],
                                                    actions[tau + N_STEP]]
                s_tau, a_tau = states[tau], actions[tau]
                q_table[s_tau, a_tau] += ALPHA * (G - q_table[s_tau, a_tau])
                visits[s_tau, a_tau]  += 1

            if tau == T - 1:                         # all states updated
                break
            tt += 1

        epsilon *= EPS_DECAY
        real_rewards[ep] = real


def _collapse_to_position(env, q, visits):
    """
    Collapse the phase-augmented Q-table (num_pos*K, num_act) down to a plain
    position-indexed table (num_pos, num_act) for plotting.

    Each position is a visit-count-weighted average of its phases, so unvisited
    phases (still at the optimistic init) contribute nothing and don't wash out
    the learned values. Returns (pos_q, v_table, greedy_policy).
    """
    num_pos = env.num_states()
    num_act = env.num_actions()

    q_pp   = q.reshape(num_pos, K, num_act)              # [pos, phase, action]
    w_pp   = visits.reshape(num_pos, K, num_act).sum(axis=2)   # visits per (pos, phase)
    denom  = w_pp.sum(axis=1)                            # total visits per position
    seen   = denom > 0

    pos_q = np.zeros((num_pos, num_act))
    pos_q[seen] = (q_pp[seen] * w_pp[seen, :, None]).sum(axis=1) / denom[seen, None]

    v_table = np.zeros(num_pos)
    v_table[seen] = pos_q[seen].max(axis=1)

    policy = np.zeros((num_pos, num_act))
    policy[seen, pos_q[seen].argmax(axis=1)] = 1.0
    return pos_q, v_table, policy





def main():
    # Final-plot environments: seeds 1000..1099, disjoint from the tuning seeds.
    all_curves = np.empty((N_ENVS, N_EPISODES))
    for i in range(N_ENVS):
        _seed_everything(1000 + i)                # fresh, reproducible, different env
        env = SeaShanty(size=12, period=6)
        all_curves[i] = sarsa(env, N_EPISODES, False)
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
    main()                       # required deliverable: averaged learning curve
