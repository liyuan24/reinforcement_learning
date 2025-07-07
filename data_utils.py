import numpy as np


def preprocess_observation(observation):
    """
    Preprocess the observation of the pong game.
    Referenced Karpathy's preprocessing: https://karpathy.github.io/2016/05/31/rl/

    1. Remove the top score board and bottom
    2. Downsample the width and height by 2
    3. Only keep the first channel
    4. Set the background pixel(144) to 0
    5. Set other pixels to 1
    6. Flatten the image

    Args:
        observation: The observation of the pong game. Numpy array of shape (210, 160, 3).

    Returns:
        The preprocessed observation. One dimension numpy array of shape (6400, )
    """
    # Remove the top score board and bottom
    observation = observation[35:195, :, :]

    # Downsample the width and height by 2
    observation = observation[::2, ::2, :]

    # Only keep the first channel
    observation = observation[:, :, 0]

    # Set the background pixel(144) to 0
    observation[observation == 144] = 0

    # Set other pixels to 1
    observation[observation != 0] = 1
    return observation.ravel()


def preprocess_observation_batch(observation):
    """
    Preprocess the observation of the pong game.
    Referenced Karpathy's preprocessing: https://karpathy.github.io/2016/05/31/rl/

    1. Remove the top score board and bottom
    2. Downsample the width and height by 2
    3. Only keep the first channel
    4. Set the background pixel(144) to 0
    5. Set other pixels to 1
    6. Flatten the image

    Args:
        observation: The observation of the pong game. Numpy array of shape (210, 160, 3).

    Returns:
        The preprocessed observation. One dimension numpy array of shape (6400, )
    """
    # Remove the top score board and bottom
    observation = observation[:, 35:195, :, :]

    # Downsample the width and height by 2
    observation = observation[:, ::2, ::2, :]

    # Only keep the first channel
    observation = observation[:, :, :, 0]

    # Set the background pixel(144) to 0
    observation[observation == 144] = 0

    # Set other pixels to 1
    observation[observation != 0] = 1
    return observation.reshape(observation.shape[0], -1)


if __name__ == "__main__":
    observation = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    preprocessed_observation = preprocess_observation(observation)
    print(preprocessed_observation.shape)

    observation = np.random.randint(0, 256, (3, 210, 160, 3), dtype=np.uint8)
    preprocessed_observation = preprocess_observation_batch(observation)
    print(preprocessed_observation.shape)
