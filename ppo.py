'''
PPO implementation for the Pong game
Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
'''

import gymnasium as gym
import ale_py
from data_utils import preprocess_observation_batch
from networks import Policy, Critic
import torch
import numpy as np
import wandb
import math
from dataclasses import dataclass

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def load_from_checkpoint(model_path, device, lr: float = 1e-6):
    checkpoint = torch.load(model_path)
    policy = Policy(input_dim=6400, action_space=2).to(device)
    critic = Critic(input_dim=6400).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, fused=True)
    policy.load_state_dict(checkpoint["policy"])
    critic.load_state_dict(checkpoint["critic"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "best_mean_rewards" in checkpoint:
        best_mean_rewards = checkpoint["best_mean_rewards"]
    else:
        best_mean_rewards = -np.inf
    if "iteration" in checkpoint:
        iteration = checkpoint["iteration"]
    else:
        iteration = 0
    return policy, critic, optimizer, best_mean_rewards, iteration


@dataclass
class TrainingConfig:
    num_envs: int
    run_name: str
    capture_video: bool
    warmup_iters: int  # warmup steps
    lr_decay_iters: int  # learning rate decay steps
    learning_rate: float  # initial learning rate
    min_lr: float  # minimum learning rate
    epochs: int  # training epochs per data collection
    batch_size: int  # batch size in the PPO training
    gamma: float  # discount factor for rewards
    gae_lambda: float  # lambda for GAE
    clip_epsilon: float  # clip epsilon in the PPO clipped objective function
    entropy_coef: float  # entropy coefficient for the entropy bonus loss
    value_coef: float  # value coefficient for the value network loss
    iterations: int  # number of iterations to collect data and optimize the policy/value network
    rollout_steps: int  # number of steps to collect data per iteration
    gradient_clip_max_norm: float  # gradient clip max norm
    resume: bool  # whether to resume from the last checkpoint
    model_path: str  # path to the model to resume from
    checkpoint_save_path: str  # path to save the checkpoint
    device: str  # device to run the training on


def main():
    config = TrainingConfig(
        num_envs=20,
        run_name="ppo_test_v2",
        capture_video=True,
        warmup_iters=20,
        lr_decay_iters=100,
        learning_rate=1e-5,
        min_lr=1e-6,
        epochs=50,
        batch_size=2000,
        gamma=0.99,
        gae_lambda=0.98,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        iterations=500,
        rollout_steps=1000,
        gradient_clip_max_norm=1.0,
        resume=True,
        model_path="checkpoints/ppo_147.pth",
        checkpoint_save_path="checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # env setup
    # readers should be aware of the autoreset behavior in vectorized environments, the default behavior is NEXT_STEP
    # for more details check https://farama.org/Vector-Autoreset-Mode
    # and source code https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/sync_vector_env.py
    envs = gym.vector.SyncVectorEnv(
        [
            make_env("ALE/Pong-v5", i, config.capture_video, config.run_name)
            for i in range(config.num_envs)
        ],
    )

    device = config.device
    checkpoint_save_path = config.checkpoint_save_path

    best_mean_rewards = -np.inf
    iteration = 0

    if config.resume:
        policy, critic, optimizer, best_mean_rewards, iteration = load_from_checkpoint(
            config.model_path, config.device, config.learning_rate
        )
    else:
        # Initialize the policy network, action space is UP and DOWN
        policy = Policy(input_dim=6400, action_space=2).to(device)
        critic = Critic(input_dim=6400).to(device)
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.learning_rate, fused=True
        )
    optimizer.zero_grad(set_to_none=True)

    wandb.init(
        project="ppo",
        name=config.run_name,
        config={
            "num_envs": config.num_envs,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_epsilon": config.clip_epsilon,
            "entropy_coef": config.entropy_coef,
            "value_coef": config.value_coef,
            "iterations": config.iterations,
            "rollout_steps": config.rollout_steps,
        },
    )

    # initialize the buffer for the rollout data
    observations = torch.zeros(config.rollout_steps, config.num_envs, 6400).to(device)
    actions = torch.zeros(config.rollout_steps, config.num_envs, dtype=torch.long).to(
        device
    )
    rewards = torch.zeros(config.rollout_steps, config.num_envs).to(device)
    log_probs = torch.zeros(config.rollout_steps, config.num_envs).to(device)
    dones = torch.zeros(config.rollout_steps, config.num_envs).to(device)
    values = torch.zeros(config.rollout_steps, config.num_envs).to(device)

    game_observation, info = envs.reset()
    for i in range(config.iterations):
        i = iteration + i
        ob = torch.zeros(config.num_envs, 6400).to(device)
        prev_ob = torch.zeros(config.num_envs, 6400).to(device)
        done = torch.zeros(config.num_envs).to(device)
        total_rewards = np.zeros(config.num_envs)
        lr = get_lr(
            i,
            config.warmup_iters,
            config.lr_decay_iters,
            config.learning_rate,
            config.min_lr,
        )
        optimizer.param_groups[0]["lr"] = lr
        # collecting rollout data esepcially the reward signal and action taken
        for j in range(config.rollout_steps):
            # Preprocess the current observation
            ob = preprocess_observation_batch(game_observation)
            ob = torch.from_numpy(ob).float().to(device)
            # if previous state is in terminal state,
            # we don't need to subtract it from the current state because they are in different games
            prev_done = (
                torch.zeros(config.num_envs).to(device)
                if j == 0
                else dones[j - 1].detach().clone()
            )
            # if the previous state is in terminal state, the current state is the initial state of the next game
            # so we don't need to subtract the previous observation from the current observation
            prev_ob[prev_done == 1] = 0
            x = ob - prev_ob
            # make a copy of the current state so that they are not sharing the same storage
            prev_ob = ob.clone().detach()
            observations[j] = x
            dones[j] = done
            # feed into the policy network and value network
            with torch.no_grad():
                probs = policy(x)  # [config.num_envs, 2]
                value = critic(x).flatten()  # [config.num_envs]
            values[j] = value
            # sample an action from the policy network
            # [config.num_envs, 2] -> [config.num_envs]
            action = torch.multinomial(probs, num_samples=1).view(-1)
            actions[j] = action
            log_prob = torch.log(probs[np.arange(config.num_envs), action]).to(device)
            log_probs[j] = log_prob
            # UP is 2 and DOWN is 5: https://ale.farama.org/env-spec/#1
            pong_action = (
                torch.where(action == 0, torch.tensor(2), torch.tensor(5)).cpu().numpy()
            )
            # game_observation is next state
            # reward is the reward of the current action
            # terminated or truncated is indicating whether the next state is terminal or truncated
            game_observation, reward, terminated, truncated, info = envs.step(
                pong_action
            )
            total_rewards += reward
            # [config.num_envs]
            rewards[j] = torch.from_numpy(reward).float().to(device)
            done = torch.from_numpy(np.logical_or(terminated, truncated)).to(device)
        mean_rewards = total_rewards.mean()
        print(f"Average total rewards for {i} iteration: {mean_rewards}")
        if mean_rewards > best_mean_rewards:
            best_mean_rewards = mean_rewards
            checkpoint = {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": i,
                "best_mean_rewards": best_mean_rewards,
                "lr": lr,
            }
            torch.save(checkpoint, f"{checkpoint_save_path}/ppo_{i}.pth")

        # data collection is done, compute the advantage
        ob = preprocess_observation_batch(game_observation)
        ob = torch.from_numpy(ob).float().to(device)
        prev_ob[dones[config.rollout_steps - 1] == 1] = 0
        x = ob - prev_ob
        with torch.no_grad():
            next_value = critic(x).flatten().view(1, -1)
        last_advantage = 0
        advantages = torch.zeros(config.rollout_steps, config.num_envs).to(device)
        for k in reversed(range(config.rollout_steps)):
            if k == config.rollout_steps - 1:
                # if next step is in done state, we don't need to consider the value of the next step
                delta = (
                    rewards[k]
                    + config.gamma * (1 - done.float()) * next_value
                    - values[k]
                )
                advantages[k] = last_advantage = delta
            else:
                # if next step is not in done state, we need to consider the value of the next step
                delta = (
                    rewards[k]
                    + config.gamma * (1 - dones[k + 1].float()) * values[k + 1]
                    - values[k]
                )
                advantages[k] = last_advantage = (
                    delta
                    + config.gae_lambda
                    * config.gamma
                    * (1 - dones[k + 1].float())
                    * last_advantage
                )
        # returns is the estimate of the expected rewards at step t taken action a_t
        # it will be used in the L2 loss of the value function estimation
        returns = advantages + values

        # next step is to sample from the collected data and optimize the policy and value network with mini-batch gradient descent
        # flatten the data
        b_dones = dones.view(-1)
        # only consider the data that is not in done state
        b_obs = observations.view(-1, 6400)[b_dones == 0]
        b_actions = actions.view(-1)[b_dones == 0]
        b_log_probs = log_probs.view(-1)[b_dones == 0]
        b_advantages = advantages.view(-1)[b_dones == 0]
        b_returns = returns.view(-1)[b_dones == 0]

        total_samples = len(b_obs)
        b_inds = np.arange(total_samples)
        losses = []
        clip_losses = []
        entropy_losses = []
        v_losses = []
        for epoch in range(config.epochs):
            np.random.shuffle(b_inds)
            for b_start in range(0, total_samples, config.batch_size):
                inds = b_inds[b_start : b_start + config.batch_size]
                batch_obs = b_obs[inds]
                batch_actions = b_actions[inds]
                batch_old_log_probs = b_log_probs[inds]
                batch_advantages = b_advantages[inds]
                # normalize the advantages
                # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                batch_returns = b_returns[inds]

                new_probs = policy(batch_obs)
                new_values = critic(batch_obs).flatten()
                new_log_probs = torch.log(
                    new_probs[np.arange(len(batch_actions)), batch_actions]
                )

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # equation (7) in https://arxiv.org/abs/1707.06347
                # since we use gradient descent instead of gradient ascent, we use negative sign and use max instead of min accordingly
                clipped_loss_1 = (
                    -torch.clamp(
                        ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                    )
                    * batch_advantages
                )
                clipped_loss_2 = -ratio * batch_advantages
                clipped_loss = torch.max(clipped_loss_1, clipped_loss_2).mean()
                clip_losses.append(clipped_loss.item())
                # loss for value network
                v_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                v_losses.append(v_loss.item())
                # entropy bonus loss
                entropy = -new_probs * torch.log(new_probs + 1e-8)
                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.item())
                # total loss, equation (9) in https://arxiv.org/abs/1707.06347
                loss = (
                    clipped_loss
                    - config.entropy_coef * entropy_loss
                    + config.value_coef * v_loss
                )
                losses.append(loss.item())
                # update the policy network parameters
                optimizer.zero_grad(set_to_none=True)
                # compute the gradient
                loss.backward()
                # add gradient clip
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=config.gradient_clip_max_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), max_norm=config.gradient_clip_max_norm
                )
                # update the parameters
                optimizer.step()
        wandb.log(
            {
                "loss": np.mean(losses),
                "clip_loss": np.mean(clip_losses),
                "entropy_loss": np.mean(entropy_losses),
                "v_loss": np.mean(v_losses),
                "lr": lr,
                "reward": mean_rewards,
            }
        )
        print(f"Average loss for {i} iteration: {np.mean(losses)}")
        print(f"Average clip loss for {i} iteration: {np.mean(clip_losses)}")
        print(f"Average entropy loss for {i} iteration: {np.mean(entropy_losses)}")
        print(f"Average v loss for {i} iteration: {np.mean(v_losses)}")
    envs.close()


if __name__ == "__main__":
    main()
