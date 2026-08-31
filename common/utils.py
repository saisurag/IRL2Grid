from argparse import ArgumentTypeError

from .imports import *

seed = 0    # Global variable to store the seed

class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        # When False, normalize() stops updating the running stats (frozen, for
        # reproducible evaluation). reset_stats() re-initialises them cold so an
        # eval pass over fixed chronics is a deterministic function of the policy.
        self.training = True

    def reset_stats(self):
        """Re-initialise the running mean/var to a cold state."""
        shape = self.single_observation_space.shape if self.is_vector_env else self.observation_space.shape
        self.obs_rms = RunningMeanStd(shape=shape)

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        if self.training:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

def set_torch(n_threads: int = 0, deterministic: bool = True, cuda: bool = False) -> th.device:
    """Configure PyTorch settings including the number of threads, determinism, and CUDA usage.

    Args:
        n_threads: Number of threads for PyTorch operations.
        deterministic: Whether to use deterministic algorithms.
        cuda: Whether to enable CUDA.

    Returns:
        The PyTorch device configured based on availability and input flags.
    """
    th.set_num_threads(n_threads)
    th.backends.cudnn.deterministic = deterministic
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

def set_random_seed(s: Optional[int] = None) -> None:
    """Set the random seed for reproducibility across different libraries.

    Args:
        s: Seed value to set. If None, the global seed value is used.
    """
    global seed
    if s is not None: seed = s
    rnd.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def str2bool(s: str) -> bool:
    """Convert a string representation of a boolean to an actual boolean value.

    Args:
        s: String to convert.

    Returns:
        The boolean value corresponding to the input string.

    Raises:
        ArgumentTypeError: If the string does not represent a boolean value.
    """
    if s.lower() == 'true':return True
    elif s.lower() == 'false': return False
    raise ArgumentTypeError('Boolean value expected.')


def Linear(input_dim: int, output_dim: int, act_fn: str = 'leaky_relu', init_weight_uniform: bool = True) -> nn.Linear:
    """Create and initialize a linear layer with appropriate weights.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/    
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        act_fn: Activation function.
        init_weight_uniform: Whether to uniformly sample initial weights.

    Returns:
        The initialized layer.
    """
    act_fn = act_fn.lower()
    gain = nn.init.calculate_gain(act_fn)
    layer = nn.Linear(input_dim, output_dim)
    if init_weight_uniform: nn.init.xavier_uniform_(layer.weight, gain=gain)
    else: nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.00)
    return layer

class Tanh(nn.Module):
    """In-place tanh module."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        """Forward pass for the in-place tanh activation function.

        Args:
            input (th.Tensor): Input tensor.

        Returns:
            Output tensor after applying in-place tanh.
        """
        return th.tanh_(input)

# Useful for picking up the right activation in the networks
th_act_fns = {
    'tanh': Tanh(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}

