import gymnasium as gym
import time
from gymnasium.wrappers import RecordVideo
import ale_py

def main():
    # Create the environment with human rendering that will display the game in a window
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    
    # Wrap the environment with RecordVideo
    # Videos will be saved to './videos/' directory
    # https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/
    env = RecordVideo(env, "videos", episode_trigger=lambda x: True)
    
    # Reset the environment and get initial observation
    observation, info = env.reset()
    
    try:
        while True:
            # Take a random action
            action = env.action_space.sample()
            
            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Add a small delay to make the game viewable
            time.sleep(0.02)
            
            # Check if the episode is done
            if terminated or truncated:
                observation, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping the game...")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()
