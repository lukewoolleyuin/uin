import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Particle mass
x_min, x_max = -10, 10  # Spatial range
dx = 0.1  # Spatial step size
dt = 0.005  # Time step size
sigma = 1.0  # Initial Gaussian width
k = 5.0  # Initial momentum
time_steps = 500  # Number of time steps

# Spatial grid
x = np.arange(x_min, x_max, dx)
N = len(x)  # Number of spatial points

# Initial Gaussian wavefunction
A = 1 / (sigma * np.sqrt(2 * np.pi))  # Normalization constant
psi = A * np.exp(-x**2 / (2 * sigma**2)) * np.exp(1j * k * x)  # Gaussian wave packet

# Laplacian operator (second derivative) using finite difference
laplacian = (
    -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
) / dx**2
laplacian[0, -1] = laplacian[-1, 0] = 1 / dx**2  # Periodic boundary conditions

# Function to simulate wavefunction evolution
def simulate_environment(flux_type, flux_density=0.0, perturbation_strength=0.0, redundancy=None):
    """Simulate the wavefunction evolution under specific environmental conditions."""
    # Initialize random potential for chaotic environments
    random_potential = np.zeros_like(x)
    if flux_density > 0:
        num_perturbations = int(flux_density * N)
        perturbation_indices = np.random.choice(N, num_perturbations, replace=False)
        random_potential[perturbation_indices] = np.random.uniform(
            -perturbation_strength, perturbation_strength, size=num_perturbations
        )

    # Initialize wavefunction
    psi_sim = psi.copy()
    prob_density = []  # To store probability densities

    # Time evolution
    for t in range(time_steps):
        # Compute time derivative using Schrödinger equation
        dpsi_dt = -1j * (hbar / (2 * m)) * (laplacian @ psi_sim) + (-1j / hbar) * random_potential * psi_sim

        # Apply redundancy if specified
        if redundancy is not None:
            dpsi_dt += redundancy * psi_sim

        # Update wavefunction using Euler's method
        psi_sim += dpsi_dt * dt

        # Normalize wavefunction
        psi_sim /= np.sqrt(np.sum(np.abs(psi_sim)**2) * dx)

        # Store probability density
        if t % 10 == 0:
            prob_density.append(np.abs(psi_sim)**2)

    return prob_density, random_potential

# Run tests for different environments
results = {}
results['High Density'] = simulate_environment(
    flux_type="high", flux_density=0.0, perturbation_strength=0.0, redundancy=None
)
results['Low Density without Cij'] = simulate_environment(
    flux_type="low", flux_density=0.0, perturbation_strength=0.0, redundancy=None
)
results['Low Density with Cij'] = simulate_environment(
    flux_type="low", flux_density=0.0, perturbation_strength=0.0, redundancy=0.1
)
results['Chaotic with Cij'] = simulate_environment(
    flux_type="chaotic", flux_density=0.2, perturbation_strength=0.1, redundancy=0.05
)

# Plot results for visualization
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
axes = ax.flatten()
titles = list(results.keys())

for i, (key, (prob_density, random_potential)) in enumerate(results.items()):
    for p in prob_density[::20]:  # Plot every 20th time step
        axes[i].plot(x, p, alpha=0.5)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Position")
    axes[i].set_ylabel("Probability Density")
    axes[i].grid()

plt.tight_layout()
plt.show()
