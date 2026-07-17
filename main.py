from gridworld import *
from config import *
from plot import *

def QLearning(alpha=0.15, gamma=0.94, max_episodes=5000):
    '''
    Optimized Q-Learning:
    - Optimistic initialization
    - Improved epsilon/alpha decay
    - Correct q_change calculation
    - Random tie-breaking for action selection
    - Max episode limit
    '''
    env = SeaShanty()
    num_states = env.num_states()
    num_actions = env.num_actions()

    # Optimistic initialization encourages exploration
    q_table = np.ones((num_states, num_actions))

    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995

    min_alpha = 0.03
    alpha_decay = 0.995

    threshold = 1e-6
    convergence_window = 100
    episode = 0
    cumulative_reward = []
    q_value_changes = []

    while episode < max_episodes:
        rewards = 0
        state = env.reset()
        done = False
        episode_q_changes = []
        while not done:
            # Random tie-breaking for action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                max_q = np.max(q_table[state])
                best_actions = np.where(q_table[state] == max_q)[0]
                action = np.random.choice(best_actions)
            old_q = q_table[state, action]
            next_state, reward, done = env.step(action)
            rewards += reward
            if done:
                q_table[state, action] += alpha * (reward - q_table[state, action])
            else:
                q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            q_change = np.abs(q_table[state, action] - old_q)
            episode_q_changes.append(q_change)
            state = next_state
        q_value_changes.append(np.max(episode_q_changes) if episode_q_changes else 0)
        cumulative_reward.append(rewards)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        alpha = max(min_alpha, alpha * alpha_decay)
        episode += 1

        if len(q_value_changes) >= 2 * convergence_window:
            recent_avg = np.mean(q_value_changes[-convergence_window:])
            prev_avg = np.mean(q_value_changes[-2 * convergence_window:-convergence_window])
            improvement = abs(recent_avg - prev_avg)
            if improvement < threshold:
                print(f"Converged after {episode} episodes (improvement < threshold)")
                break

    plt.plot(cumulative_reward)
    plt.title('Cumulative Reward per Episode (Optimized Q-Learning)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.show()
    plot_q_table(env, q_table)

    return episode

def double_qlearning():
    '''
    :param alpha: 0.1
    :param gamma: 0.9
    :param epsilon: 1
    :return: least number of episodes to reach goal state
    '''
    env = ExerciseWorld()
    num_states = env.num_states()
    num_actions = env.num_actions()
    
    q_1 = np.zeros((num_states, num_actions))
    q_2 = np.zeros((num_states, num_actions))
    
    alpha = 0.098
    gamma = 0.94
    epsilon = 1
    
    epsilon_decay=  0.995
    min_epsilon = 0.01
    threshold = 1e-3
    convergence_window = 100
    cumulitive_rewards=[]
    q_value_changes=[]

    episode =0

    while True:
        state = env.reset()
        done = False
        rewards=0
        episode_q_changes = []
        while not done:
            if np.random.rand() < epsilon:  # explore
                action = np.random.randint(num_actions)
            else:
                action = np.argmax((q_1[state] + q_2[state])/2)
            
            prev_q1_value = q_1[state, action]
            prev_q2_value = q_2[state, action]
            next_state, reward, done = env.step(action)
            rewards += reward
            if np.random.rand() < 0.5:
                if done:
                    q_1[state, action] = q_1[state, action] + alpha * (reward - q_1[state, action])
                else:
                    q_1[state, action] = q_1[state, action] + alpha * (
                            reward + gamma * q_2[next_state, np.argmax(q_1[next_state])] - q_1[
                        state, action])
                q_change = abs(q_1[state, action]-prev_q1_value)        
            else:
                if done:
                    q_2[state, action] = q_2[state, action] + alpha * (reward - q_2[state, action])
                else:
                    q_2[state, action] = q_2[state, action] + alpha * (
                            reward + gamma * q_1[next_state, np.argmax(q_2[next_state])] - q_2[
                        state, action])
                q_change = abs(q_2[state, action] - prev_q2_value)
            episode_q_changes.append(q_change)
            state = next_state 
            
        
        max_q_episode_change = max(episode_q_changes) if episode_q_changes else 0
        q_value_changes.append(max_q_episode_change)
        cumulitive_rewards.append(rewards)
        epsilon = max(min_epsilon, epsilon*epsilon_decay) if episode > 300 else 1
        episode += 1
        if len(q_value_changes) >= convergence_window:
            recent_rewards = cumulitive_rewards[-100:]
            avg_recent = np.mean(recent_rewards)
            max_seen = max(cumulitive_rewards)

            if avg_recent > 0.9 * max_seen:  # plateauing near best performance
                print(f"Converged after {episode} episodes with avg recent reward: {avg_recent:.2f}")
                break
    plt.plot(cumulitive_rewards)
    plt.show()
    plot_q_table(env, q_1)
    plot_q_table(env, q_2)

    return episode



print(QLearning())