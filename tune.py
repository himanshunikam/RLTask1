import optuna
import numpy as np
import random
import task1
from task1 import q_learning_single, DoubleQLearning
from gridworld import SeaShanty

TUNE_SEEDS    = range(0, 30)   # disjoint from final-plot seeds 1000-1099 (required)
TUNE_EPISODES = 400            # shorter than the final run, for speed


def _seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def evaluate(learner, n_episodes):
    """Average learning-curve quality across the tuning environments."""
    scores = []
    for seed in TUNE_SEEDS:
        _seed_everything(seed)
        env = SeaShanty(size=12, period=6)
        curve = learner(env, n_episodes=n_episodes)
        scores.append(curve.mean())        # mean over episodes -> rewards FAST learning
    return float(np.mean(scores))


def make_objective(learner):
    def objective(trial):
        task1.Q_INIT    = trial.suggest_int("Q_INIT", 50, 150)
        task1.EPS_START = trial.suggest_float("EPS_START", 0.8, 1.0)
        task1.EPS_DECAY = trial.suggest_float("EPS_DECAY", 0.5, 0.99)
        task1.GAMMA     = trial.suggest_float("GAMMA", 0.5, 1.0)
        task1.ALPHA     = trial.suggest_float("ALPHA", 0.5, 1.0)
        task1.SHAPE_W   = trial.suggest_float("SHAPE_W", 50, 150)
        return evaluate(learner, TUNE_EPISODES)
    return objective


if __name__ == "__main__":
    for name, learner in [("q_learning", q_learning_single),
                          ("double_q",   DoubleQLearning)]:
        study = optuna.create_study(direction="maximize", study_name=name)
        study.optimize(make_objective(learner), n_trials=50)
        print(f"\n=== {name} ===")
        print("best value :", study.best_value)
        print("best params:", study.best_params)
