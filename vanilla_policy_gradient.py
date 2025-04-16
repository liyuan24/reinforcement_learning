import gymnasium as gym
import time
import ale_py
from data_utils import preprocess_observation
from policy_neural_network import Policy
import torch


def discount_rewards(rewards: list[float], gamma: float) -> list[float]:
    """
    Discount the rewards.

    args:
        rewards: rewards of one episode of the game and may contains multiple rounds.
        gamma: The discount factor.

    returns:
        The discounted rewards.
    """
    discounted_rewards = [0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        if rewards[i] != 0:
            running_reward = 0
        running_reward = rewards[i] + gamma * running_reward
        discounted_rewards[i] = running_reward
    return discounted_rewards


def main():
    # Create the environment with human rendering that will display the game in a window
    env = gym.make("ALE/Pong-v5", render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_save_interval = 100
    checkpoint_save_path = "checkpoints"

    # Reset the environment and get initial observation
    observation, info = env.reset()

    # Initialize the policy network, action space is UP and DOWN
    policy = Policy(input_dim=6400, action_space=2).to(device)
    prev_observation = None
    log_probs = []
    rewards = []
    total_rewards = 0
    episodes = 0
    batch_size = 10  # number of episodes to update the policy network
    gamma = 0.99  # discount factor
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, fused=True)
    optimizer.zero_grad(set_to_none=True)

    try:
        while True:
            # Preprocess the observation
            observation = preprocess_observation(observation)
            x = (
                observation
                if prev_observation is None
                else observation - prev_observation
            )
            prev_observation = observation
            # convert observation to float32 tensor
            x = torch.from_numpy(x).float().to(device)
            # feed into the policy network
            probs = policy(x)
            # sample an action from the policy network
            action = torch.multinomial(probs, num_samples=1).item()

            # UP is 2 and DOWN is 5: https://ale.farama.org/env-spec/#1
            pong_action = 2 if action == 0 else 5

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(pong_action)
            total_rewards += reward
            log_prob = torch.log(probs[action])
            log_probs.append(log_prob)
            rewards.append(reward)

            # Check if the episode is done
            # each episode may contain multiple rounds of games
            if terminated or truncated:
                episodes += 1
                discounted_rewards_tensor = torch.tensor(
                    discount_rewards(rewards, gamma), dtype=torch.float32
                ).to(device)
                discounted_rewards_tensor = (
                    discounted_rewards_tensor - torch.mean(discounted_rewards_tensor)
                ) / torch.std(discounted_rewards_tensor)
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

                observation, info = env.reset()
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
                        checkpoint, f"{checkpoint_save_path}/policy_{episodes}.pth"
                    )

                # print the total rewards
                print(f"Episode {episodes} total rewards: {total_rewards}")
                total_rewards = 0

    except KeyboardInterrupt:
        print("\nStopping the game...")

    finally:
        env.close()


if __name__ == "__main__":
    main()
