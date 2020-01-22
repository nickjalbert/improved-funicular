"""Adapted from https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import argparse
from envs.nick_gym_adapter import Nick2048Gym
import mlflow
import ray
from ray import tune
from ray.rllib.utils import try_import_tf

# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
# from ray.rllib.agents.dqn import DEFAULT_CONFIG
# from ray.rllib.agents.dqn.apex import ApexTrainer, APEX_DEFAULT_CONFIG
# from ray.rllib.models import ModelCatalog
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
# from gym.spaces import Discrete, Box


tf = try_import_tf()


def run_ppo():
    # config = {}
    # config['num_workers'] = 2
    # config['num_sgd_iter'] = 10
    # config['sgd_minibatch_size'] = 512
    # config['model'] = {'fcnet_hiddens': [10, 10]}
    # config['num_cpus_per_worker'] = 1
    # mlflow.log_params(config)
    # full_config = DEFAULT_CONFIG.copy()
    # for k, v in config.items():
    #     full_config[k] = v
    # agent = PPOTrainer(full_config, env=Nick2048)
    # for i in range(1000):
    #     res = agent.train()
    #     print(res["episode_reward_mean"])
    #     # mlflow.log_metrics(res) #<-- MLflow can't handle nested dictionaries of metrics.
    #     mlflow.log_metric("episode_reward_mean", res["episode_reward_mean"], step=i)

    tune.run(
        "PPO",
        stop={"timesteps_total": 25000000,},
        config={
            "env": Nick2048Gym,  # or "corridor" if registered above
            "num_workers": 9,  # parallelism
            "num_gpus": 1,

            "num_sgd_iter": 10,
            "sgd_minibatch_size": 512,
            # copied from https://github.com/ray-project/rl-experiments/blob/master/atari-ppo/atari-ppo.yaml
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.1,

            "vf_clip_param": tune.grid_search([10.0, 100.0, 1000.0, 10000.0]),
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sample_batch_size": 100,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "vf_share_layers": "true",
        },
    )




def     run_dqn():
        tune.run(
            "DQN",
        stop={"timesteps_total": 30000,},
        config={"env": Nick2048Gym, "num_workers": 2}
    )


def run_apex():
    tune.run(
        "APEX",
        stop={"timesteps_total": 1000000,},
        config={"env": Nick2048Gym, "num_workers": 27, "num_gpus": 1}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ec2",
        action="store_true",
        help="this is running on an EC2 Ray cluster",
    )
    args = parser.parse_args()
    #with mlflow.start_run():
    if args.ec2:
        print("running APEX in Ray on this EC2 cluster")
        ray.init(address="auto")
        run_apex()
    else:
        ray.init()
        run_ppo()

        # run_dqn()
