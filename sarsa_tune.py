import optuna
import numpy as np
import random
import sarsa
from sarsa import sarsa_fn, n_step_sarsa
from gridworld import SeaShanty

TUNE_SEEDS    = range(0, 50)   # disjoint from final-plot seeds 1000-1099 (required)
TUNE_EPISODES = 1000            # shorter than the final run, for speed


def _seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def evaluate(learner, n_episodes):
    """Average learning-curve quality across the tuning environments."""
    scores = []
    for seed in TUNE_SEEDS:
        _seed_everything(seed)
        env = SeaShanty(size=12, period=6)
        curve = learner(env, n_episodes, False)
        scores.append(curve.mean())        # mean over episodes -> rewards FAST learning
    return float(np.mean(scores))


def make_objective(learner):
    def objective(trial):
        sarsa.Q_INIT    = trial.suggest_int("Q_INIT", 50, 150)
        sarsa.EPS_START = trial.suggest_float("EPS_START", 0.8, 1.0)
        sarsa.EPS_DECAY = trial.suggest_float("EPS_DECAY", 0.5, 0.99)
        sarsa.GAMMA     = trial.suggest_float("GAMMA", 0.5, 1.0)
        sarsa.ALPHA     = trial.suggest_float("ALPHA", 0.5, 1.0)
        sarsa.N_STEP =    trial.suggest_int("N_STEP", 2, 8)
        return evaluate(learner, TUNE_EPISODES)
    return objective


def tune_nstep():
    name, learner = "n_step_sarsa", n_step_sarsa
    study = optuna.create_study(direction="maximize", study_name=name, storage="sqlite:///tune_nstep.db", load_if_exists=True)
    study.optimize(make_objective(learner), n_trials=150)
    print(f"\n=== {name} ===")
    print("best value :", study.best_value)
    print("best params:", study.best_params)

if __name__ == "__main__":
    for name, learner in [("sarsa", sarsa_fn),
                          ("n_step_sarsa",   n_step_sarsa)]:
        study = optuna.create_study(direction="maximize", study_name=name)
        study.optimize(make_objective(learner), n_trials=50)
        print(f"\n=== {name} ===")
        print("best value :", study.best_value)
        print("best params:", study.best_params)
