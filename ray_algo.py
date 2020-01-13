from envs.nick_2048 import Nick2048

# from https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import mlflow
import numpy as np
import gym
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search
from ray.tune.logger import pretty_print

tf = try_import_tf()



class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    #register_env("2048-v0", lambda config: Nick2048(config))
    with mlflow.start_run():
        ray.init(num_cpus=2)
        ModelCatalog.register_custom_model("my_model", CustomModel)

        config = {}
        config['num_workers'] = 2
        config['num_sgd_iter'] = 10
        config['sgd_minibatch_size'] = 512
        config['model'] = {'fcnet_hiddens': [10, 10]}
        config['num_cpus_per_worker'] = 1
        mlflow.log_params(config)

        full_config = DEFAULT_CONFIG.copy()
        for k, v in config.items():
            full_config[k] = v

        agent = PPOTrainer(full_config, env=Nick2048)
        for i in range(1000):
            res = agent.train()
            print(res["episode_reward_mean"])
            #mlflow.log_metrics(res) #<-- MLflow can't handle nested dictionaries of metrics.
            mlflow.log_metric("episode_reward_mean", res["episode_reward_mean"], step=i)

        #tune.run(
        #    "PPO",
        #    stop={
        #        "timesteps_total": 30000,
        #    },
        #    config={
        #        "env": Nick2048,  # or "corridor" if registered above
        #        "model": {
        #            "custom_model": "my_model",
        #        },
        #        "vf_share_layers": True,
        #        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        #        "num_workers": 3,  # parallelism
        #        #"env_config": {},
        #    },
        #)
