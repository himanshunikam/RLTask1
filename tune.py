import optuna
import numpy as np
import random
import task1
from task1 import q_learning_single, DoubleQLearning
from gridworld import SeaShanty
import json
TUNE_SEEDS    = range(0, 50)   # disjoint from final-plot seeds 1000-1099 (required)
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

def tune_doubleQ():
    name, learner = "double_q",   DoubleQLearning
    study = optuna.create_study(direction="maximize", study_name=name, storage="sqlite:///tune_doubleq.db")
    study.optimize(make_objective(learner), n_trials=500)
    print(f"\n=== {name} ===")
    print("best value :", study.best_value)
    print("best params:", study.best_params)
    json.dump({"best_val": study.best_value, "best_params": study.best_params}, open("tune_results.json", "w"), indent=2)
    print(f"\nsaved -> tune_results.json")

def tune_all():
    for name, learner in [("q_learning", q_learning_single),
                          ("double_q",   DoubleQLearning)]:
        study = optuna.create_study(direction="maximize", study_name=name, storage="sqlite:///tune_doubleq.db", load_if_exists=True)
        study.optimize(make_objective(learner), n_trials=500)
        print(f"\n=== {name} ===")
        print("best value :", study.best_value)
        print("best params:", study.best_params)


if __name__ == "__main__":
    tune_doubleQ()
