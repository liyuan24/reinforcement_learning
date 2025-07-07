import gymnasium as gym
import time
import ale_py
from data_utils import preprocess_observation_batch
from networks import Policy
import torch
import numpy as np


def discount_rewards(rewards: list[torch.Tensor], gamma: float) -> list[torch.Tensor]:
    """
    Discount the rewards.

    args:
        rewards: rewards of one episode of the game and may contains multiple rounds.
        gamma: The discount factor.

    returns:
        The discounted rewards.
    """
    num_envs = len(rewards[0])
    discounted_rewards = [torch.zeros(num_envs) for _ in range(len(rewards))]
    running_reward = torch.zeros(num_envs)
    for i in range(len(rewards) - 1, -1, -1):
        running_reward[rewards[i] != 0] = 0
        running_reward = rewards[i] + gamma * running_reward
        discounted_rewards[i] = running_reward
    return discounted_rewards


# reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py#L81
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


def main():
    # Create the environment with human rendering that will display the game in a window
    num_envs = 10
    run_name = "vanilla_policy_gradient_parallel_envs"
    capture_video = False
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env("ALE/Pong-v5", i, capture_video, run_name) for i in range(num_envs)],
    )
    observation, info = envs.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_save_interval = 100
    checkpoint_save_path = "checkpoints"

    model_path = "checkpoints/policy_400.pth"
    # # Initialize the policy network, action space is UP and DOWN
    policy = Policy(input_dim=6400, action_space=2).to(device)
    checkpoint = torch.load(model_path)
    prev_observation = None
    log_probs = []
    rewards = []
    total_rewards = np.zeros(num_envs)
    episodes = 0
    batch_size = 10  # number of episodes to update the policy network
    gamma = 0.99  # discount factor
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, fused=True)
    optimizer.zero_grad(set_to_none=True)
    policy.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    try:
        while True:
            # Preprocess the observation
            observation = preprocess_observation_batch(observation)
            x = (
                observation
                if prev_observation is None
                else observation - prev_observation
            )
            prev_observation = observation
            # convert observation to float32 tensor
            x = torch.from_numpy(x).float().to(device)
            # feed into the policy network
            # [num_envs, 6400] -> [num_envs, 2]
            probs = policy(x)
            # sample an action from the policy network
            # [num_envs, 2] -> [num_envs]
            action = torch.multinomial(probs, num_samples=1).view(-1)

            # UP is 2 and DOWN is 5: https://ale.farama.org/env-spec/#1
            pong_action = (
                torch.where(action == 0, torch.tensor(2), torch.tensor(5)).cpu().numpy()
            )

            # Step the environment
            observation, reward, terminated, truncated, info = envs.step(pong_action)
            total_rewards += reward
            # [num_envs]
            log_prob = torch.log(probs[np.arange(num_envs), action])
            log_probs.append(log_prob)
            rewards.append(torch.from_numpy(reward).float())

            # Check if the episode is done
            # each episode may contain multiple rounds of games
            if np.any(terminated) or np.any(truncated):
                episodes += 1
                discounted_rewards_tensor = torch.stack(
                    discount_rewards(rewards, gamma)
                ).to(device)
                # normalize the discounted rewards for each env
                discounted_rewards_tensor = (
                    discounted_rewards_tensor
                    - torch.mean(discounted_rewards_tensor, dim=0, keepdim=True)
                ) / torch.std(discounted_rewards_tensor, dim=0, keepdim=True)
                # torch.stack can maintain the gradient
                log_probs_tensor = torch.stack(log_probs)

                # compute the loss
                loss = -torch.sum(log_probs_tensor * discounted_rewards_tensor)

                # accumulate the gradient
                loss.backward()

                # update the policy network parameters
                if episodes % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                observation, info = envs.reset()
                # reset the log_probs and rewards
                log_probs = []
                rewards = []
                prev_observation = None

                # save the checkpoint
                if episodes % checkpoint_save_interval == 0:
                    checkpoint = {
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(
                        checkpoint,
                        f"{checkpoint_save_path}/policy_parallel_envs_{episodes}.pth",
                    )

                # print the total rewards
                print(f"Episode {episodes} mean total rewards: {total_rewards.mean()}")
                total_rewards = np.zeros(num_envs)

    except KeyboardInterrupt:
        print("\nStopping the game...")

    finally:
        envs.close()


if __name__ == "__main__":
    main()
