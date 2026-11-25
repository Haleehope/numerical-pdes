# I used ChatGPT to generate all of this code.
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from scipy.sparse import eye
from scipy.sparse.linalg import factorized
from time import perf_counter


from scipy.sparse.linalg import factorized
from time import perf_counter


# Setup ---------------------------------------------------------------------------------------------------------------

# Plot settings
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'figure.dpi': 120
})


# Spatial parameters
N = 100
h = 1 / N
epsilon = h

# Time parameters
dt = h
T = 50
num_steps = int(np.ceil(T / dt))

# Adjust dt so that we end exactly at T
dt = T / num_steps

# Cell-centered spatial grid
x = (np.arange(N) + 0.5) * h
y = (np.arange(N) + 0.5) * h

X, Y = np.meshgrid(x, y)

# Time grid
times = np.linspace(0, T, num_steps + 1)

# Times at which we want to save the solution
save_times = [0, 5, 10, 25, 50]

print(f"N = {N}")
print(f"h = {h}")
print(f"epsilon = {epsilon}")
print(f"dt = {dt}")
print(f"Number of time steps = {num_steps}")


# Construct the 1D finite difference Laplacian
main_diag = -2 * np.ones(N)
off_diag = np.ones(N - 1)

# Modify the first and last diagonal entries
# to enforce homogeneous Neumann boundary conditions
main_diag[0] = -1
main_diag[-1] = -1

D = sp.diags(
    [off_diag, main_diag, off_diag],
    [-1, 0, 1],
    shape=(N, N),
    format='csr'
) / h**2

# Construct the 2D Laplacian using Kronecker products
I = sp.eye(N, format='csr')

L = sp.kron(I, D) + sp.kron(D, I)

L = L.tocsr()

# Check that the Laplacian of a constant is zero
print("Laplacian shape:", L.shape)
print("Constant function error:", np.max(np.abs(L @ np.ones(N**2))))

# Identity matrix for the full 2D system
I_2D = sp.eye(N**2, format='csc')

# Backward Euler matrix
A = I_2D - dt * epsilon**2 * L

# Factorize the matrix once so we can reuse it
solve_diffusion = factorized(A.tocsc())


# Set a seed for reproducibility
rng = np.random.default_rng(0)

# Generate random initial data
U0 = epsilon * rng.standard_normal(N**2)

# Subtract the mean so that the initial mass is zero
U0 = U0 - np.mean(U0)

# Calculate the initial discrete sum and mass
initial_sum = np.sum(U0)
initial_mass = h**2 * initial_sum

print(f"Initial mean = {np.mean(U0):.6e}")
print(f"Initial mass = {initial_mass:.6e}")


fig, ax = plt.subplots(figsize=(8, 7))

# Reshape the solution from a vector into a 2D grid
im = ax.pcolormesh(
    X, Y, U0.reshape(N, N),
    cmap='viridis',
    shading='auto'
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Initial Condition')
ax.set_aspect('equal')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\phi(x,y,0)$', fontsize=20)

plt.tight_layout()
plt.savefig('HW4/initial_condition.png', dpi=300, bbox_inches='tight')

def reaction_step(U, dt):
    # Exact solution to the nonlinear reaction ODE
    # phi_t = phi - phi^3

    denominator = np.sqrt(
        U**2 + (1 - U**2) * np.exp(-2 * dt)
    )

    return U / denominator


def F(U):
    # Double-well potential energy function
    return (U**2 - 1)**2 / 4


def compute_mass(U):
    # Approximate the integral of phi over the domain
    return h**2 * np.sum(U)

def plot_snapshots(snapshots, title, filename):

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)

    # Create a grid with 2 rows and 6 columns
    # Each plot occupies 2 columns
    gs = fig.add_gridspec(2, 6)

    # Three plots in the first row
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Two centered plots in the second row
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])

    axes = [ax1, ax2, ax3, ax4, ax5]

    # Plot the solution at each saved time
    for ax, t in zip(axes, sorted(snapshots)):

        im = ax.pcolormesh(
            X, Y,
            snapshots[t].reshape(N, N),
            cmap='viridis',
            shading='auto',
            vmin=-1,
            vmax=1
        )

        ax.set_title(f't = {t} s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # Shared colorbar for all five plots
    cbar = fig.colorbar(
        im, ax=axes,
        shrink=0.85,
        pad=0.02
    )

    cbar.set_label(r'$\phi(x,y,t)$', fontsize=20)

    plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_final_solution(U, title, filename):

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        X, Y, U.reshape(N, N),
        cmap='viridis',
        shading='auto',
        vmin=-1,
        vmax=1
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\phi(x,y,t)$', fontsize=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def solve_basic_dt(dt_test):

    # Adjust the time step to end exactly at T
    steps = int(np.ceil(T / dt_test))
    dt_test = T / steps

    # Construct and factorize the backward Euler matrix
    A_test = I_2D - dt_test * epsilon**2 * L
    solve_test = factorized(A_test.tocsc())

    # Reset to the original initial condition
    U = U0.copy()

    # Start timing
    start = perf_counter()

    # Perform operator splitting
    for n in range(steps):

        U_star = solve_test(U)
        U = reaction_step(U_star, dt_test)

    runtime = perf_counter() - start

    return U, runtime

# Problem 1 -----------------------------------------------------------------------------------------------------------
# Start with a copy of the initial condition
U = U0.copy()

# Store the mass at every time step
mass_basic = np.zeros(num_steps + 1)
mass_basic[0] = compute_mass(U)

# Dictionary to store solution snapshots
snapshots_basic = {0: U.copy()}

# Convert the desired snapshot times into step numbers
save_steps = {round(t / dt): t for t in save_times}

# Start timing the simulation
start = perf_counter()

# Time-stepping loop
for n in range(1, num_steps + 1):
    # Step 1: Solve the diffusion equation using backward Euler
    U_star = solve_diffusion(U)

    # Step 2: Solve the nonlinear reaction equation exactly
    U = reaction_step(U_star, dt)

    # Calculate the total mass
    mass_basic[n] = compute_mass(U)

    # Save the solution at selected times
    if n in save_steps:
        snapshots_basic[save_steps[n]] = U.copy()


# Save the final solution
U_basic = U.copy()

# Calculate the computational time
runtime_basic = perf_counter() - start

print(f"Runtime = {runtime_basic:.2f} seconds")
print(f"Initial mass = {mass_basic[0]:.6e}")
print(f"Final mass = {mass_basic[-1]:.6e}")
print(f"Minimum phi = {np.min(U_basic):.6f}")
print(f"Maximum phi = {np.max(U_basic):.6f}")


# Plot the basic Allen-Cahn solution
plot_snapshots(
    snapshots_basic,
    'Basic Allen-Cahn Equation',
    'HW4/basic_snapshots.png'
)


plot_final_solution(
    U_basic,
    'Basic Allen-Cahn Solution at t = 50',
    'HW4/basic_final.png'
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    times, mass_basic,
    color=plt.cm.viridis(0.2),
    linewidth=2.5,
    label='Basic Allen-Cahn'
)

# Reference line showing the initial mass
ax.axhline(
    initial_mass,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label='Initial Mass'
)

ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Total Mass $\int_\Omega \phi\,dx\,dy$')

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('HW4/basic_mass.png', dpi=300, bbox_inches='tight')

# time step size experiment -------------------------------------------------------------------------------------------

# Calculate a reference solution with a smaller time step
U_ref, runtime_ref = solve_basic_dt(h / 2)

# Time steps to test
dt_values = [h, 5*h, 10*h, 50*h]

# Store errors, runtimes, and solutions
dt_errors = []
dt_runtimes = []
dt_solutions = {}


for dt_test in dt_values:

    # Solve using the current time step
    U_test, runtime = solve_basic_dt(dt_test)

    # Calculate the relative L2 error
    error = (
        np.linalg.norm(U_test - U_ref)
        / np.linalg.norm(U_ref)
    )

    # Save the results
    dt_errors.append(error)
    dt_runtimes.append(runtime)
    dt_solutions[dt_test] = U_test.copy()

    print(f"dt = {dt_test:.4f}")
    print(f"Relative error = {error:.6e}")
    print(f"Runtime = {runtime:.2f} seconds")
    print(f"Minimum phi = {np.min(U_test):.6f}")
    print(f"Maximum phi = {np.max(U_test):.6f}")
    print()


fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 11),
    constrained_layout=True
)

# Flatten the 2D array of axes into a 1D array
axes = axes.flatten()

# Plot the solution for each time step
for ax, dt_test in zip(axes, dt_values):

    im = ax.pcolormesh(
        X, Y,
        dt_solutions[dt_test].reshape(N, N),
        cmap='viridis',
        shading='auto',
        vmin=-1,
        vmax=1
    )

    ax.set_title(rf'$\Delta t = {dt_test:.2f}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

# Shared colorbar for all four plots
cbar = fig.colorbar(
    im, ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(r'$\phi(x,y,t)$', fontsize=20)

plt.savefig('HW4/timestep_comparison.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(
    dt_values, dt_errors,
    'o-',
    color=plt.cm.viridis(0.2),
    linewidth=2.5,
    markersize=9
)

ax.set_xlabel(r'Time Step $\Delta t$')
ax.set_ylabel('Relative Error')

ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('HW4/timestep_error.png', dpi=300, bbox_inches='tight')



# Problem 2 -----------------------------------------------------------------------------------------------------------
def mass_correction(U, dt):

    # Calculate sqrt(F(phi)) at every grid point
    weights = np.sqrt(F(U))

    # Calculate how much mass was lost or gained
    numerator = initial_sum - np.sum(U)

    # Sum the correction weights
    denominator = np.sum(weights)

    # Avoid division by zero if there is no interface
    if denominator == 0:
        if abs(numerator) > 1e-12:
            raise ValueError("Mass correction is not possible.")
        return U

    # Calculate the mass correction coefficient
    beta = numerator / (dt * denominator)

    # Apply the correction to the solution
    U_corrected = U + dt * beta * weights

    return U_corrected


# Reset to the exact same initial condition
U = U0.copy()

# Store the mass at every time step
mass_conservative = np.zeros(num_steps + 1)
mass_conservative[0] = compute_mass(U)

# Store solution snapshots
snapshots_conservative = {0: U.copy()}

# Start timing the simulation
start = perf_counter()


# Time-stepping loop
for n in range(1, num_steps + 1):

    # Step 1: Solve the diffusion equation
    U_star = solve_diffusion(U)

    # Step 2: Solve the nonlinear reaction equation
    U_starstar = reaction_step(U_star, dt)

    # Step 3: Apply the mass conservation correction
    U = mass_correction(U_starstar, dt)

    # Calculate the total mass
    mass_conservative[n] = compute_mass(U)

    # Save the solution at selected times
    if n in save_steps:
        snapshots_conservative[save_steps[n]] = U.copy()


# Save the final solution
U_conservative = U.copy()

# Calculate the computational time
runtime_conservative = perf_counter() - start

print(f"Runtime = {runtime_conservative:.2f} seconds")
print(f"Initial mass = {mass_conservative[0]:.6e}")
print(f"Final mass = {mass_conservative[-1]:.6e}")
print(f"Minimum phi = {np.min(U_conservative):.6f}")
print(f"Maximum phi = {np.max(U_conservative):.6f}")

# Plot the solution at several times
plot_snapshots(
    snapshots_conservative,
    'Conservative Allen-Cahn Equation',
    'HW4/conservative_snapshots.png'
)


# Plot the final solution
plot_final_solution(
    U_conservative,
    'Conservative Solution at t = 50',
    'HW4/conservative_final.png'
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6),
    constrained_layout=True
)

# Plot the basic solution
im = axes[0].pcolormesh(
    X, Y, U_basic.reshape(N, N),
    cmap='viridis',
    shading='auto',
    vmin=-1,
    vmax=1
)

axes[0].set_title('Basic Allen-Cahn')


# Plot the conservative solution
im = axes[1].pcolormesh(
    X, Y, U_conservative.reshape(N, N),
    cmap='viridis',
    shading='auto',
    vmin=-1,
    vmax=1
)

axes[1].set_title('Conservative Allen-Cahn')


# Set axis labels and aspect ratios
for ax in axes:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')


# Shared colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.85)
cbar.set_label(r'$\phi(x,y,t)$', fontsize=20)

plt.savefig(
    'HW4/solution_comparison.png',
    dpi=300,
    bbox_inches='tight'
)



fig, ax = plt.subplots(figsize=(10, 6))

# Plot the mass for the basic method
ax.plot(
    times, mass_basic,
    color=plt.cm.viridis(0.15),
    linewidth=2.5,
    label='Basic Allen-Cahn'
)

# Plot the mass for the conservative method
ax.plot(
    times, mass_conservative,
    color=plt.cm.viridis(0.8),
    linewidth=2.5,
    label='Conservative Allen-Cahn'
)

ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Total Mass $\int_\Omega \phi\,dx\,dy$')

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('HW4/mass_comparison.png', dpi=300, bbox_inches='tight')

# Calculate the absolute mass error at each time step
error_basic = np.abs(mass_basic - initial_mass)

error_conservative = np.abs(
    mass_conservative - initial_mass
)

# Calculate the maximum mass error
max_error_basic = np.max(error_basic)
max_error_conservative = np.max(error_conservative)


print("Basic Allen-Cahn:")
print(f"Initial mass: {mass_basic[0]:.6e}")
print(f"Final mass: {mass_basic[-1]:.6e}")
print(f"Maximum mass error: {max_error_basic:.6e}")
print(f"Runtime: {runtime_basic:.2f} seconds")

print()

print("Conservative Allen-Cahn:")
print(f"Initial mass: {mass_conservative[0]:.6e}")
print(f"Final mass: {mass_conservative[-1]:.6e}")
print(f"Maximum mass error: {max_error_conservative:.6e}")
print(f"Runtime: {runtime_conservative:.2f} seconds")


fig, ax = plt.subplots(figsize=(10, 6))

# Use a small positive floor because log(0) is undefined
error_floor = 1e-18

ax.semilogy(
    times,
    np.maximum(error_basic, error_floor),
    color=plt.cm.viridis(0.15),
    linewidth=2.5,
    label='Basic Allen-Cahn'
)

ax.semilogy(
    times,
    np.maximum(error_conservative, error_floor),
    color=plt.cm.viridis(0.8),
    linewidth=2,
    label='Conservative Allen-Cahn'
)

ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Mass Error $|M(t)-M(0)|$')

ax.grid(alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('HW4/mass_error.png', dpi=300, bbox_inches='tight')



# Problem 3 -----------------------------------------------------------------------------------------------------------
# Effect of Initial Geometry on Phase Evolution
# Dictionary to store the initial conditions
initial_shapes = {}

# Coordinates centered at (0.5, 0.5)
xc = X - 0.5
yc = Y - 0.5

# Polar coordinates (useful for the star and flower)
r = np.sqrt(xc**2 + yc**2)
theta = np.arctan2(yc, xc)

# Function to create a smooth initial condition
def smooth_shape(mask):

    # Distance of each interior point from the boundary
    d_inside = distance_transform_edt(mask) * h

    # Distance of each exterior point from the boundary
    d_outside = distance_transform_edt(~mask) * h

    # Positive inside the shape and negative outside
    signed_distance = d_inside - d_outside

    # Smooth transition between the two phases
    phi = np.tanh(signed_distance / (np.sqrt(2) * epsilon))

    return phi.flatten()


# --------------------------------------------------
# Checkerboard
# --------------------------------------------------

# Number of squares in each direction
num_squares = 6

# Alternate between positive and negative phases
checkerboard = (
    np.floor(num_squares * X) +
    np.floor(num_squares * Y)
) % 2 == 0

initial_shapes['Checkerboard'] = smooth_shape(checkerboard)

# --------------------------------------------------
# Jellyfish
# --------------------------------------------------

# Create the rounded bell using the upper half of an ellipse
bell = (
    ((X - 0.5) / 0.34)**2
    + ((Y - 0.65) / 0.25)**2
) <= 1

bell = bell & (Y >= 0.65)

# Add a rounded underside to the bell
underside = (
    ((X - 0.5) / 0.34)**2
    + ((Y - 0.65) / 0.08)**2
) <= 1

# Combine the two parts of the bell
jellyfish = bell | underside

# Starting x-coordinates of the five tentacles
tentacle_positions = np.linspace(0.25, 0.75, 5)

# Width and waviness of the tentacles
tentacle_width = 0.025
amplitude = 0.035
frequency = 2.5

# Create five wavy tentacles
for x_start in tentacle_positions:

    # Use a sine function to create the wavy shape
    x_tentacle = x_start + amplitude * np.sin(
        2 * np.pi * frequency * (0.65 - Y)
    )

    # Define the width and length of each tentacle
    tentacle = (
        (np.abs(X - x_tentacle) <= tentacle_width)
        & (Y >= 0.15)
        & (Y <= 0.65)
    )

    # Add the tentacle to the jellyfish
    jellyfish = jellyfish | tentacle

# Smooth the interface between the two phases
initial_shapes['Jellyfish'] = smooth_shape(jellyfish)


# --------------------------------------------------
# Flower
# --------------------------------------------------

# Number of petals
num_petals = 5

# Radius varies with the angle
flower_radius = 0.26 + 0.12 * np.cos(num_petals * theta)

flower = r <= flower_radius

initial_shapes['Flower'] = smooth_shape(flower)


# --------------------------------------------------
# Crescent moon
# --------------------------------------------------

# Larger circle
outer_circle = (
    (X - 0.45)**2 + (Y - 0.5)**2
) <= 0.35**2

# Smaller circle shifted to the right
inner_circle = (
    (X - 0.65)**2 + (Y - 0.60)**2
) <= 0.30**2

# Remove the smaller circle from the larger circle
moon = outer_circle & ~inner_circle

initial_shapes['Crescent Moon'] = smooth_shape(moon)






# Plot all initial conditions
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()

for ax, (name, phi_initial) in zip(axes, initial_shapes.items()):

    im = ax.pcolormesh(
        X, Y,
        phi_initial.reshape(N, N),
        cmap='viridis',
        shading='auto',
        vmin=-1,
        vmax=1
    )

    ax.set_title(name, fontsize=22)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_aspect('equal')

fig.subplots_adjust(
    left=0.08, right=0.88,
    bottom=0.08, top=0.88,
    wspace=0.40, hspace=0.40
)

# Shared colorbar
cbar = fig.colorbar(
    im, ax=axes,
    shrink=0.85,
    pad=0.05,
    fraction=0.05
)

cbar.set_label(r'$\phi(x,y,0)$', fontsize=22)
cbar.ax.tick_params(labelsize=16)

plt.savefig(
    'HW4/charlie_initial_conditions.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
)


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------
dt_charlie = 0.01
T_charlie = 50

num_steps_charlie = int(round(T_charlie / dt_charlie))

times_charlie = np.linspace(
    0, T_charlie,
    num_steps_charlie + 1
)

# Times at which we want to save the solution
snapshot_times_charlie = [0, 10, 25, 50]

snapshot_steps_charlie = {
    int(round(t / dt_charlie)): t
    for t in snapshot_times_charlie
}

# Construct the Backward Euler matrix
A_charlie = eye(N**2, format='csc') - dt_charlie * epsilon**2 * L

# Factorize once and reuse for all simulations
solve_diffusion_charlie = factorized(A_charlie.tocsc())


# --------------------------------------------------
# Allen-Cahn solver for different initial conditions
# --------------------------------------------------
def solve_charlie(phi_initial, conservative=False):

    # Copy the initial condition so we do not modify it
    phi = phi_initial.copy()

    # Calculate the initial mass
    initial_sum = np.sum(phi)
    initial_mass = h**2 * initial_sum

    # Store mass at every time step
    mass = np.zeros(num_steps_charlie + 1)
    mass[0] = initial_mass

    # Store the initial condition
    snapshots = {0: phi.copy()}

    start = perf_counter()

    for n in range(1, num_steps_charlie + 1):

        # Step 1: Backward Euler diffusion
        phi_star = solve_diffusion_charlie(phi)

        # Step 2: Analytical reaction solution
        phi = phi_star / np.sqrt(
            phi_star**2
            + (1 - phi_star**2) * np.exp(-2 * dt_charlie)
        )

        # Step 3: Mass correction (conservative equation only)
        if conservative:

            # Calculate sqrt(F(phi))
            sqrt_F = np.abs(phi**2 - 1) / 2

            denominator = np.sum(sqrt_F)

            # Avoid dividing by zero
            if denominator > 1e-14:

                # Calculate the correction coefficient
                beta = (
                    initial_sum - np.sum(phi)
                ) / (dt_charlie * denominator)

                # Apply the mass correction
                phi = phi + dt_charlie * beta * sqrt_F

        # Calculate the total mass
        mass[n] = h**2 * np.sum(phi)

        # Save the solution at selected times
        if n in snapshot_steps_charlie:

            t = snapshot_steps_charlie[n]
            snapshots[t] = phi.copy()

    runtime = perf_counter() - start

    return snapshots, mass, runtime


# --------------------------------------------------
# Run both equations for each initial condition
# --------------------------------------------------
basic_results = {}
conservative_results = {}

for name, phi_initial in initial_shapes.items():

    print('\n====================================')
    print(name)
    print('====================================')

    # Solve the basic Allen-Cahn equation
    snapshots, mass, runtime = solve_charlie(
        phi_initial,
        conservative=False
    )

    basic_results[name] = {
        'snapshots': snapshots,
        'mass': mass,
        'runtime': runtime
    }

    print('Basic Allen-Cahn:')
    print(f'Initial mass = {mass[0]:.6e}')
    print(f'Final mass = {mass[-1]:.6e}')
    print(f'Runtime = {runtime:.2f} seconds')

    # Solve the conservative Allen-Cahn equation
    snapshots, mass, runtime = solve_charlie(
        phi_initial,
        conservative=True
    )

    conservative_results[name] = {
        'snapshots': snapshots,
        'mass': mass,
        'runtime': runtime
    }

    print('\nConservative Allen-Cahn:')
    print(f'Initial mass = {mass[0]:.6e}')
    print(f'Final mass = {mass[-1]:.6e}')
    print(f'Runtime = {runtime:.2f} seconds')

    # Check mass conservation
    mass_error = np.max(np.abs(mass - mass[0]))

    print(f'Maximum mass error = {mass_error:.6e}')


# --------------------------------------------------
# Function to plot the evolution of all six shapes
# --------------------------------------------------

def plot_shape_evolution(results, filename):

    fig, axes = plt.subplots(
        len(initial_shapes), len(snapshot_times_charlie),
        figsize=(16, 11)
    )

    # Loop through the different initial conditions
    for i, name in enumerate(initial_shapes):

        # Loop through the saved times
        for j, t in enumerate(snapshot_times_charlie):

            ax = axes[i, j]

            phi = results[name]['snapshots'][t]

            im = ax.pcolormesh(
                X, Y,
                phi.reshape(N, N),
                cmap='viridis',
                shading='auto',
                vmin=-1,
                vmax=1
            )

            ax.set_aspect('equal')
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=14)

            # Only put time titles above the first row
            if i == 0:
                ax.set_title(f't = {t} s', fontsize=22)

            # Label each shape in the first column
            if j == 0:
                ax.set_ylabel(name, fontsize=20)

            # Only label x on the bottom row
            if i == len(initial_shapes) - 1:
                ax.set_xlabel('x', fontsize=20)

    fig.subplots_adjust(
        left=0.08, right=0.85,
        bottom=0.05, top=0.92,
        wspace=0.40, hspace=0.15
    )

    # Shared colorbar for all subplots
    cbar = fig.colorbar(
        im,
        ax=axes.flatten().tolist(),
        shrink=0.8,
        pad=0.02
    )

    cbar.set_label(r'$\phi(x,y,t)$', fontsize=22)
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

# Basic Allen-Cahn equation
plot_shape_evolution(
    basic_results,
    'HW4/charlie_basic_evolution.png'
)

# Conservative Allen-Cahn equation
plot_shape_evolution(
    conservative_results,
    'HW4/charlie_conservative_evolution.png'
)

# --------------------------------------------------
# Compare total mass for each initial condition
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

# Viridis colors
cmap = plt.get_cmap('viridis')

basic_color = cmap(0.15)
conservative_color = cmap(0.80)

for ax, name in zip(axes, initial_shapes):

    # Get the mass arrays
    mass_basic = basic_results[name]['mass']
    mass_conservative = conservative_results[name]['mass']

    # Plot basic Allen-Cahn mass
    ax.plot(
        times_charlie,
        mass_basic,
        color=basic_color,
        linewidth=3,
        label='Basic'
    )

    # Plot conservative Allen-Cahn mass
    ax.plot(
        times_charlie,
        mass_conservative,
        color=conservative_color,
        linewidth=3,
        label='Conservative'
    )

    ax.set_title(name, fontsize=22)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Total Mass', fontsize=20)
    ax.tick_params(labelsize=16)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)

fig.subplots_adjust(
    left=0.10, right=0.96,
    bottom=0.09, top=0.90,
    wspace=0.30, hspace=0.35
)

plt.savefig(
    'HW4/charlie_mass_comparison.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
)

# --------------------------------------------------
# Calculate the area of the positive phase
# --------------------------------------------------

def positive_area(phi):

    return h**2 * np.sum(phi > 0)


# Store the areas
initial_areas = []
basic_areas = []
conservative_areas = []

for name in initial_shapes:

    area_initial = positive_area(initial_shapes[name])

    area_basic = positive_area(
        basic_results[name]['snapshots'][50]
    )

    area_conservative = positive_area(
        conservative_results[name]['snapshots'][50]
    )

    initial_areas.append(area_initial)
    basic_areas.append(area_basic)
    conservative_areas.append(area_conservative)

    print(f'\n{name}')
    print(f'Initial area = {area_initial:.4f}')
    print(f'Basic final area = {area_basic:.4f}')
    print(f'Conservative final area = {area_conservative:.4f}')

    print(
        f'Basic area change = '
        f'{100 * (area_basic - area_initial) / area_initial:.2f}%'
    )

    print(
        f'Conservative area change = '
        f'{100 * (area_conservative - area_initial) / area_initial:.2f}%'
    )


# --------------------------------------------------
# Plot the initial and final areas
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

positions = np.arange(len(initial_shapes))
width = 0.25

ax.bar(
    positions - width,
    initial_areas,
    width,
    color=cmap(0.45),
    label='Initial'
)

ax.bar(
    positions,
    basic_areas,
    width,
    color=basic_color,
    label='Basic at t = 50'
)

ax.bar(
    positions + width,
    conservative_areas,
    width,
    color=conservative_color,
    label='Conservative at t = 50'
)

ax.set_xticks(positions)
ax.set_xticklabels(initial_shapes.keys(), fontsize=16)

ax.set_ylabel('Area of Positive Phase', fontsize=22)

ax.tick_params(labelsize=16)
ax.legend(fontsize=16)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig(
    'HW4/charlie_area_comparison.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
)

for name in initial_shapes:
    phi_basic = basic_results[name]['snapshots'][50]
    phi_cons = conservative_results[name]['snapshots'][50]

    print(f'\n{name}')
    print(f'Basic min = {phi_basic.min():.8f}')
    print(f'Basic max = {phi_basic.max():.8f}')
    print(f'Conservative min = {phi_cons.min():.8f}')
    print(f'Conservative max = {phi_cons.max():.8f}')
