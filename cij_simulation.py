import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 1.0  # Gravitational constant (arbitrary units)
decay_length = 2.0  # Decay length for connectivity
N = 50  # Number of particles
size = 10  # Size of the system (arbitrary units)

# Generate random positions and masses
np.random.seed(42)
positions = np.random.uniform(-size, size, (N, 2))
masses = np.random.uniform(1, 5, N)

# Compute distance matrix
distances = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=-1)
np.fill_diagonal(distances, 1e-3)  # Avoid division by zero for self-distance

# Compute Cij connectivity matrix
def compute_cij(alpha=0.5, beta=1.0):
    Cij = alpha * np.exp(-distances / decay_length) + beta * np.random.uniform(0, 0.1, distances.shape)
    np.fill_diagonal(Cij, 0)
    return Cij

# With Cij
Cij = compute_cij()

# Compute gravitational forces
def compute_forces(masses, positions, Cij=None):
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                r_hat = r_vec / r
                if Cij is not None:
                    F = G * masses[i] * masses[j] * Cij[i, j] / r**2
                else:
                    F = G * masses[i] * masses[j] / r**2
                forces[i] += F * r_hat
    return forces

forces_with_cij = compute_forces(masses, positions, Cij)
forces_without_cij = compute_forces(masses, positions)

# Plot gravitational forces
def plot_forces(positions, forces, title):
    plt.figure(figsize=(8, 6))
    plt.quiver(positions[:, 0], positions[:, 1], forces[:, 0], forces[:, 1], color="red", angles="xy", scale_units="xy", scale=0.1)
    plt.scatter(positions[:, 0], positions[:, 1], c="blue", label="Particles (Mass)")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

plot_forces(positions, forces_with_cij, "Gravitational Forces with Cij")
plot_forces(positions, forces_without_cij, "Gravitational Forces without Cij")

# Simulate galaxy rotation curves
def compute_rotation_curve(masses, distances, Cij=None):
    velocities = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                if Cij is not None:
                    velocities[i] += G * masses[j] * Cij[i, j] / distances[i, j]
                else:
                    velocities[i] += G * masses[j] / distances[i, j]
    return np.sqrt(velocities)

rotation_curve_with_cij = compute_rotation_curve(masses, distances, Cij)
rotation_curve_without_cij = compute_rotation_curve(masses, distances)

plt.figure(figsize=(8, 6))

# Overlay the two curves directly
plt.plot(range(N), rotation_curve_with_cij, label="With Cij", color="blue")
plt.plot(range(N), rotation_curve_without_cij, label="Without Cij", color="orange", alpha=0.7)

# Add titles and labels
plt.title("Galaxy Rotation Curves (With and Without Cij)")
plt.xlabel("Particle Index")
plt.ylabel("Orbital Velocity")
plt.legend()
plt.grid()
plt.show()


# Gravitational lensing effect (approximation)
def compute_lensing(masses, distances, Cij=None):
    lensing = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                if Cij is not None:
                    lensing[i] += G * masses[j] * Cij[i, j] / distances[i, j]**2
                else:
                    lensing[i] += G * masses[j] / distances[i, j]**2
    return lensing

lensing_with_cij = compute_lensing(masses, distances, Cij)
lensing_without_cij = compute_lensing(masses, distances)

# Plot lensing effects
plt.figure(figsize=(8, 6))
plt.bar(range(N), lensing_with_cij, alpha=0.7, label="With Cij")
plt.bar(range(N), lensing_without_cij, alpha=0.7, label="Without Cij")
plt.title("Gravitational Lensing Effects")
plt.xlabel("Particle Index")
plt.ylabel("Lensing Magnitude")
plt.legend()
plt.grid()
plt.show()
