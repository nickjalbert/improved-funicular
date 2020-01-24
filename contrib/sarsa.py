# RL Agent that plays 2048 using Actor-Critic. Coded by Andy 12/18/2019
from envs.nick_2048 import Nick2048
import logging
import mlflow
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_probability as tfp
from ray.tune import Trainable

#logging.basicConfig(level=logging.DEBUG)


class Sarsa(Trainable):
    def _setup(self, config):
        self.params = config

        self.q_models = [
            keras.Sequential(
                [
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dense(256, activation="relu"),
                    keras.layers.Dense(1),
                ]
            )
        ] * Nick2048.action_space.n
        [m.build(input_shape=(1, 16)) for m in self.q_models]
        self.seen_states = (
            set()
        )  # track which states we've seen, used for rotation invariance

    def _save(self, tmp_checkpoint_dir):
        [
            self.q_models[i].save(os.path.join(tmp_checkpoint_dir, "/model-" + str(i)))
            for i in len(self.q_models)
        ]

    def _restore(self, tmp_checkpoint_dir):
        self.q_models = [
            load_model(os.path.join(tmp_checkpoint_dir, "/model-" + str(i)))
            for i in len(self.q_models)
        ]

    def _train(self):
        with mlflow.start_run():
            mlflow.log_params(self.params)
            optimizer = keras.optimizers.Adam(lr=self.params["learning_rate"])
            game_scores = []
            game_num_steps = []
            b = Nick2048()
            for episode_num in range(self.params["num_episodes"]):
                state = b.reset()
                game_score = 0
                # Pseudo code for our Sarsa learning algo:
                #   for each step in the rollout:
                #     action = fancy_argmax_a(q(get_afterstate(s),a))
                #     q_val = model(get_afterstate(s),action)
                #     next_s, r = b.step(a)
                #     next_action = fancy_argmax_a(q(get_afterstate(next_s), a))
                #     next_q_val = model(get_afterstate(new_s),next_action)
                #     update q_model using loss(q_val - (r + next_q_val)
                for step_num in range(self.params["max_steps_per_episode"]):
                    with tf.GradientTape() as q_tape:
                        logging.debug(f"state:\n{np.asarray(state).reshape([4,4])}")
                        candidate_actions = list(range(b.action_space.n))
                        canonical_afterstates = [
                            b.get_canonical_board(b.get_afterstate(state, a))
                            for a in candidate_actions
                        ]
                        q_vals = [
                            tf.squeeze(
                                self.q_models[i](
                                    np.array(canonical_afterstates[i])[np.newaxis]
                                )
                            )
                            for i in candidate_actions
                        ]
                        logging.debug(f"q_vals : {q_vals}")
                        # pick action by rolling dice according to relative values of canonical_afterstates
                        while True:
                            dice_roll = tfp.distributions.Multinomial(
                                total_count=5, logits=q_vals
                            ).sample(1)
                            action_index = np.argmax(dice_roll)
                            action = candidate_actions[action_index]
                            next_state, reward, done, _ = b.step(action)
                            if next_state != state:  # you found a valid move
                                break
                            else:  # that wasn't a valid move, but one must exist since we weren't done after last step.
                                logging.debug(
                                    f"action {action} was invalid, removing it from candidate and rolling dice again"
                                )
                                assert (
                                    len(candidate_actions) > 1
                                ), "No actions changed the board but we are not done."
                            q_vals = q_vals[:action_index] + q_vals[action_index + 1:]
                            candidate_actions = (
                                candidate_actions[:action_index]
                                + candidate_actions[action_index + 1:]
                            )
                        logging.debug(f"action: {action}")
                        logging.debug(
                            f"canonical_afterstate:\n{np.asarray(canonical_afterstates[action]).reshape([4,4])}"
                        )
                        q_val = q_vals[action_index]
                        logging.debug(f"q_val: {q_val}")
                        logging.debug(f"reward: {reward}")
                        logging.debug(
                            f"next_state:\n{np.asarray(next_state).reshape([4,4])}"
                        )

                        # update q_model via TD learning using q(s,a) (which we computed last loop iter) and q(s',a')
                        next_candidate_actions = list(range(b.action_space.n))
                        next_canonical_afterstates = [
                            b.get_canonical_board(b.get_afterstate(next_state, action))
                            for action in next_candidate_actions
                        ]
                        next_q_vals = [
                            tf.squeeze(
                                self.q_models[i](
                                    np.array(next_canonical_afterstates[i])[np.newaxis]
                                )
                            )
                            for i in next_candidate_actions
                        ]
                        logging.debug(f"next_q_vals: {next_q_vals}")
                        next_action = np.argmax(next_q_vals)
                        logging.debug(f"next_action: {next_action}")
                        next_q_val = next_q_vals[next_action]
                        target_q_val = (
                            reward + (1 - done) * self.params["alpha"] * next_q_val
                        )
                        logging.debug(f"next_q_val: {next_q_val}")
                        logging.debug(f"target_q_val: {target_q_val}")
                        val_loss_fn = tf.math.square(q_val - target_q_val)
                    val_grads = q_tape.gradient(
                        val_loss_fn, self.q_models[action].trainable_variables
                    )
                    optimizer.apply_gradients(
                        zip(val_grads, self.q_models[action].trainable_variables)
                    )
                    logging.debug("\n")

                    # get ready to loop
                    state = next_state
                    game_score += reward
                    if done:
                        break
                game_scores.append(game_score)
                game_num_steps.append(step_num + 1)
                avg_game_score = np.mean(game_scores)
                print(
                    "%s steps in episode %s, score: %s, running_avg_score: %.0f"
                    % (step_num + 1, episode_num, game_score, avg_game_score)
                )
                mlflow.log_metric("game scores", game_score, step=episode_num)
                mlflow.log_metric("avg game score", avg_game_score, step=episode_num)
                mlflow.log_metric("game num steps", step_num + 1, step=episode_num)
                mlflow.log_metric("avg num steps", np.mean(game_num_steps), step=episode_num)
            return {"avg_game_score": avg_game_score,
                    "avg_num_steps": np.mean(game_num_steps),
                    "episodes_total": episode_num + 1,
                    "timesteps_total": np.sum(game_num_steps)}


if __name__ == "__main__":
    params = {}
    params["num_episodes"] = 100000000
    params["max_steps_per_episode"] = 500
    params["alpha"] = 0.95
    params["learning_rate"] = 0.001
    sarsa = Sarsa(params)
    sarsa.train()
