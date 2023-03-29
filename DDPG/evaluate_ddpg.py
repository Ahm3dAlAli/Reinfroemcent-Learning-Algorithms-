import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/ahmed/Documents/UOE/Courses/Semester 2/Reinfrocment Leanring /Coursework/RLAgents')





import gym
from typing import List, Tuple

from rl2023.exercise4.agents import DDPG
from rl2023.exercise4.train_ddpg import PENDULUM_CONFIG, BIPEDAL_CONFIG, play_episode
from gym.envs import box2d

RENDER = False

#CONFIG = PENDULUM_CONFIG
CONFIG = BIPEDAL_CONFIG


def evaluate(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DDPG(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    try:
        agent.restore(config['save_filename'])
        #agent.restore("/Users/ahmed/Documents/UOE/Courses/Semester 2/Reinfrocment Leanring /Coursework/RLAgents/rl2023/exercise4/bipedal_q4_latest.pt")
        #agent.restore("/Users/ahmed/Documents/UOE/Courses/Semester 2/Reinfrocment Leanring /Coursework/RLAgents/rl2023/exercise5/bipedal_q5_latest.pt")
    except:
        raise ValueError(f"Could not find model to load at {config['save_filename']}")

    eval_returns_all = []
    eval_times_all = []


    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            0,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
