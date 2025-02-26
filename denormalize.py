import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import matplotlib.pyplot as plt

from data_chandler_05 import times, values0, values1, values2, values3

# Transformation functions for different distributions


def normal_to_lognormal(parameters: npt.NDArray, mu: float, sigma: float):
    """Transform N(0,1) to LogN(mu,sigma)"""
    return np.exp(parameters * sigma + mu)


def lognormal_to_normal(parameters: npt.NDArray, mu: float, sigma: float):
    """Transform LogN(mu,sigma) to N(0,1)"""
    return (np.log(parameters) - mu) / sigma


def normal_to_uniform(parameters: npt.NDArray, a: float = 0, b: float = 1, mu: float = 0, sigma: float = 1):
    """Transform N(mu,sigma) to Uniform(a,b)"""
    tmp = stats.norm.cdf(parameters, mu, sigma)
    return a + tmp * (b - a)


def uniform_to_normal(parameters: npt.NDArray, a: float = 0, b: float = 1, mu: float = 0, sigma: float = 1):
    """Transform Uniform(a,b) to N(mu,sigma)"""
    tmp = (parameters - a) / (b - a)
    return stats.norm.ppf(tmp, mu, sigma)


def beta_to_uniform(parameters: npt.NDArray, a: float = 0, b: float = 1, alpha: float = 2, beta: float = 2):
    """Transform Beta(alpha,beta) to Uniform(a,b)"""
    tmp = stats.beta.cdf(parameters, alpha, beta)
    return a + tmp * (b - a)


def uniform_to_beta(parameters: npt.NDArray, a: float = 0, b: float = 1, alpha: float = 2, beta: float = 2):
    """Transform Uniform(a,b) to Beta(alpha,beta)"""
    tmp = (parameters - a) / (b - a)
    return stats.beta.ppf(tmp, alpha, beta)


def normal_to_beta(parameters: npt.NDArray, mu: float = 0, sigma: float = 1, alpha: float = 2, beta: float = 2):
    """Transform N(mu,sigma) to Beta(alpha,beta)"""
    tmp = normal_to_uniform(parameters, mu=mu, sigma=sigma)
    return uniform_to_beta(tmp, alpha=alpha, beta=beta)


def beta_to_normal(parameters: npt.NDArray, mu: float = 0, sigma: float = 1, alpha: float = 2, beta: float = 2):
    """Transform Beta(alpha,beta) to N(mu,sigma)"""
    tmp = beta_to_uniform(parameters, alpha=alpha, beta=beta)
    return uniform_to_normal(tmp, mu=mu, sigma=sigma)


class UnivariateComponent:
    """Base class for univariate distribution components"""

    def __init__(self):
        pass

    def transform(self, parameter):
        pass


class PriorIndependentComponents:
    """
    Transforms internal Gaussian prior N(0,1) to other distributions component by component.
    Available components: Uniform, Normal, Lognormal, Beta
    """

    def __init__(self, list_of_components: list[UnivariateComponent]):
        """
        Args:
            list_of_components: list of UnivariateComponent instances
        """
        self.list_of_components = list_of_components
        self.no_parameters = len(list_of_components)
        self.mean = np.zeros((self.no_parameters,))
        self.sd_approximation = np.ones((self.no_parameters,))

    def transform(self, sample):
        """Transform standard normal parameters to target distributions"""
        trans_sample = sample.copy()
        for i in range(self.no_parameters):
            trans_sample[i] = self.list_of_components[i].transform(sample[i])
        return trans_sample

    def logpdf(self, sample):
        """Returns logarithm of the N(0,1) pdf at given sample"""
        return -0.5 * np.dot(sample, sample)

    def rvs(self):
        """Returns a random sample from N(0,1)"""
        return np.random.randn(self.no_parameters)


class Uniform(UnivariateComponent):
    """Uniform(a,b) distribution component"""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = a
        self.b = b

    def transform(self, parameter):
        return normal_to_uniform(parameter, a=self.a, b=self.b)


class Lognormal(UnivariateComponent):
    """Lognormal(mu,sigma) distribution component"""

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return normal_to_lognormal(parameter, mu=self.mu, sigma=self.sigma)


class Beta(UnivariateComponent):
    """Beta(alpha,beta) distribution component"""

    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def transform(self, parameter):
        return normal_to_beta(parameter, alpha=self.alpha, beta=self.beta)


class Normal(UnivariateComponent):
    """Normal(mu,sigma) distribution component"""

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return parameter * self.sigma + self.mu


def create_denormalizer_tsx_9subdomains():
    """
    Creates a function that transforms normalized parameters to their original space

    Returns:
        function: A function that transforms normalized parameters to their physical domain
    """
    no_subdomains = 9

    # Create components for the prior distribution
    list_of_components = []
    # Permeability parameters (9)
    for _ in range(no_subdomains):
        list_of_components.append(Lognormal(np.log(2.63e-19), 2))
    # Young's modulus parameters (9)
    for _ in range(no_subdomains):
        list_of_components.append(Lognormal(np.log(140e9), 2))
    # Biot modulus parameters (9)
    for _ in range(no_subdomains):
        list_of_components.append(Lognormal(np.log(6e10), 1))
    # Poisson ratio parameters (9)
    for _ in range(no_subdomains):
        list_of_components.append(Uniform(0.1, 0.4))
    # Additional parameters (3)
    list_of_components.append(Lognormal(np.log(45e6), 1))  # Initial pressure
    list_of_components.append(Lognormal(np.log(11e6), 1))  # Production pressure
    list_of_components.append(Normal(0, np.pi / 8))  # Maximum principal stress direction

    prior = PriorIndependentComponents(list_of_components)

    return prior


def plot_observations_vs_reference(observation_vector, show_plot=True):
    """
    Plots observation vector against reference data

    Args:
        observation_vector (numpy.ndarray): Vector of observations (length 144)
        show_plot (bool): Whether to show the plot immediately

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if len(observation_vector) != 144:
        raise ValueError(f"Expected observation vector of length 144, got {len(observation_vector)}")

    # Split into 4 sensors (36 points each)
    obs_parts = [observation_vector[36 * i:36 * (i + 1)] for i in range(4)]
    reference_data = [values0, values1, values2, values3]

    # Check if times needs to be subsampled
    if len(times) != len(obs_parts[0]):
        print(f"Warning: times ({len(times)}) and observation data ({len(obs_parts[0])}) have different lengths")
        # Create appropriate time points for observations
        # Option 1: Use subset of times if it divides evenly
        if len(times) % len(obs_parts[0]) == 0:
            step = len(times) // len(obs_parts[0])
            obs_times = times[::step]
        # Option 2: Create new evenly spaced time points
        else:
            obs_times = np.linspace(times[0], times[-1], len(obs_parts[0]))
    else:
        obs_times = times

    fig = plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple']

    for i, (obs, ref) in enumerate(zip(obs_parts, reference_data)):
        # For reference data, use the full times array
        if len(times) == len(ref):
            plt.plot(times, ref, color=colors[i], label=f'Reference {i}')
        # If reference data also has a different length, use interpolation or subsampling
        else:
            ref_times = np.linspace(times[0], times[-1], len(ref))
            plt.plot(ref_times, ref, color=colors[i], label=f'Reference {i}')

        # For observation data, use the appropriate time points
        plt.plot(obs_times, obs, color=colors[i], linestyle=':', linewidth=2, label=f'Observation {i}')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if show_plot:
        plt.show()

    return fig
