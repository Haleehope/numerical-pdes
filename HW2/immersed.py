# I used ChatGPT to give me all of this code, but I added more comments and made sure I understood everything
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix
import os

L = 2 * np.pi
colors = plt.cm.viridis([0.2, 0.8])

# helper functions
def exact(x, y):
    # exact solution
    return np.cos(2*x) * np.sin(2*y)

def setup(N):
    # N is the number of interior points in each direction.
    h = L / (N + 1)

    # construct the grid with interior points only
    x = np.linspace(h, L-h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    # we have N*N unknowns, so A is an N² by N² matrix.
    # ChatGPT says that using LIL format makes it easy to fill the matrix using loops.
    A = lil_matrix((N*N, N*N))

    for i in range(N):
        for j in range(N):
            # convert the 2D index (i,j) into a 1D index.
            k = i*N + j

            # Five-point finite-difference approximation:
            #
            #            u[i,j+1]
            #               |
            # u[i-1,j] -- u[i,j] -- u[i+1,j]
            #               |
            #            u[i,j-1]
            #
            # Laplacian = (sum of neighbors - 4*u[i,j]) / h²

            # Center coefficient.
            A[k, k] = -4 / h**2

            # Left neighbor (x direction).
            if i > 0:
                A[k, (i-1)*N + j] = 1 / h**2

            # Right neighbor (x direction).
            if i < N-1:
                A[k, (i+1)*N + j] = 1 / h**2

            # Bottom neighbor (y direction).
            if j > 0:
                A[k, i*N + j-1] = 1 / h**2

            # Top neighbor (y direction).
            if j < N-1:
                A[k, i*N + j+1] = 1 / h**2

    # convert to CSR format for efficient matrix multiplication.
    return h, X, Y, A.tocsr()



def circle(h, cx=np.pi, cy=np.pi, r=1, factor=1):
    # create points along a circular immersed boundary.
    # the default circle has center (pi, pi) and radius 1.

    # the circle's circumference is 2*pi*r.
    # we want approximately one IB point every factor*h units.
    # factor < 1 means more points; factor > 1 means fewer.
    n = max(8, round(2*np.pi*r / (factor*h)))

    # divide the circle into n equally spaced angles.
    # endpoint=False avoids duplicating the first point.
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)

    # parametric equation of a circle:
    # x = cx + r*cos(theta), y = cy + r*sin(theta).
    return np.column_stack((cx + r*np.cos(theta), cy + r*np.sin(theta)))


def plot_solution(X, Y, u, points=None, filename="solution.png"):
    # plot the numerical solution as a colored 2D surface.
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, u, shading="auto", cmap="viridis")
    plt.colorbar(label="u")

    # if we have an immersed boundary, draw its points in black.
    if points is not None:
        plt.plot(points[:, 0], points[:, 1], "k.", ms=3)

    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.axis("equal")

    # Save the figure before displaying it.
    plt.savefig(f"HW2/{filename}", dpi=300, bbox_inches="tight")
    plt.show()

# problem 1 ---------------------------------------------------------------------------------------------------------------
# test increasingly fine grids
Ns = [20, 40, 80, 160]
hs, errors = [], []

for N in Ns:
    # construct the grid and finite-difference Laplacian.
    h, X, Y, A = setup(N)

    # our exact solution is cos(2x)*sin(2y).
    # differentiating twice in x and y gives:
    # u_xx = -4*cos(2x)*sin(2y)
    # u_yy = -4*cos(2x)*sin(2y)
    # therefore, f = u_xx + u_yy = -8*u.
    f = -8 * exact(X, Y)

    # account for nonzero Dirichlet conditions at x=0 and x=2pi.
    # the y=0 and y=2pi boundaries are already zero.
    f[0, :] -= np.sin(2*Y[0, :]) / h**2
    f[-1, :] -= np.sin(2*Y[-1, :]) / h**2

    # solve the corrected system.
    u_poisson = spsolve(A, f.ravel()).reshape(N, N)

    # l2 error
    error = np.linalg.norm(u_poisson - exact(X, Y)) / np.linalg.norm(exact(X, Y))

    # store grid spacing and error for the convergence plot.
    hs.append(h)
    errors.append(error)

    print(f"N={N}, error={error:.3e}")

# if error = C*h^p, then log(error) = log(C) + p*log(h).
# so, the slope of the log-log plot is the order p.
order = np.polyfit(np.log(hs), np.log(errors), 1)[0]
print(f"Convergence order: {order:.2f}")

# compare our numerical errors against the theoretical O(h^2).
plt.loglog(hs, errors, "o-", color=colors[0],
           linewidth=2, markersize=6, label="Numerical error")
plt.loglog(hs, errors[0]*(np.array(hs)/hs[0])**2,
           "--", color=colors[1],
           linewidth=2, label=r"$O(h^2)$")
plt.legend()
slope = np.polyfit(np.log(hs), np.log(errors), 1)[0]
plt.text(
    0.05, 0.95,
    f"Slope = {slope:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    va="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)
plt.xlabel("Grid spacing $h$", fontsize=15)
plt.ylabel("L2 error", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig("HW2/problem1_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

plot_solution(X, Y, u_poisson, filename="problem1_solution.png")



# Problem 2 setup ---------------------------------------------------------------------------------------------------------
from scipy.sparse import coo_matrix

def ib_matrix(X, Y, points, h):
    # construct the Gaussian spreading matrix.
    a = 1.2 * h
    n = X.shape[0]

    rows, cols, values = [], [], []

    for k, (px, py) in enumerate(points):
        ix = np.where(abs(X[:, 0] - px) < 4*a)[0]
        iy = np.where(abs(Y[0, :] - py) < 4*a)[0]

        # collect weights for this boundary point.
        weights = []
        indices = []

        for i in ix:
            for j in iy:
                d2 = (X[i, j] - px)**2 + (Y[i, j] - py)**2
                weight = np.exp(-d2 / (2*a*a))

                weights.append(weight)
                indices.append(i*n + j)

        weights = np.array(weights)

        # normalize so that h^2 * sum(B[:,k]) = 1.
        weights /= h**2 * np.sum(weights)

        for index, weight in zip(indices, weights):
            rows.append(index)
            cols.append(k)
            values.append(weight)

    return coo_matrix((values, (rows, cols)), shape=(n*n, len(points))).tocsr()


def solve_ib(N, forcing=True, factor=1, points=None):
    h, X, Y, A = setup(N)

    if points is None:
        points = circle(h, factor=factor)

    B = ib_matrix(X, Y, points, h)

    m = N * N
    k = len(points)

    # Forcing and outer Dirichlet boundary conditions.
    if forcing:
        f = -8 * exact(X, Y)
        f[0, :] -= np.sin(2 * Y[0, :]) / h**2
        f[-1, :] -= np.sin(2 * Y[-1, :]) / h**2
    else:
        f = np.zeros((N, N))

    f = f.ravel()
    g = exact(points[:, 0], points[:, 1])

    # Scale the IB constraint equations by 1/h^2.
    # This does not change the mathematical solution.
    rhs = np.r_[f, g / h**2]

    # Matrix-free saddle-point operator.
    def multiply(z):
        u = z[:m]
        q = z[m:]

        return np.r_[
            A @ u - B @ q,
            B.T @ u
        ]

    M = LinearOperator(
        (m + k, m + k),
        matvec=multiply,
        dtype=float
    )

    # Preconditioner: sparse LU for the grid Laplacian
    # and an approximate Schur complement for the IB block.
    lu = splu(A.tocsc())

    W = lu.solve(B.toarray())
    S = B.T @ W

    # Use a small diagonal regularization only in the
    # preconditioner, not in the original equations.
    scale = max(np.linalg.norm(S, ord=np.inf), 1.0)
    S_pre = S + 1e-8 * scale * np.eye(k)

    def precondition(r):
        r1 = r[:m]
        r2 = r[m:]

        v = lu.solve(r1)

        # Approximate inverse of the saddle-point matrix.
        q = np.linalg.solve(S_pre, r2 - B.T @ v)
        u = v + W @ q

        return np.r_[u, q]

    P = LinearOperator(
        (m + k, m + k),
        matvec=precondition,
        dtype=float
    )

    # Solve using GMRES.
    solution, info = gmres(
        M,
        rhs,
        M=P,
        restart=100,
        maxiter=100,
        rtol=1e-10,
        atol=1e-12
    )

    u = solution[:m]
    q = solution[m:]

    # Verify the unscaled equations.
    pde_residual = np.linalg.norm(
        A @ u - B @ q - f,
        ord=np.inf
    )

    relative_pde_residual = pde_residual / max(
        np.linalg.norm(f, ord=np.inf)
        + np.linalg.norm(B @ q, ord=np.inf),
        1.0
    )

    ib_residual = np.linalg.norm(
        h**2 * (B.T @ u) - g,
        ord=np.inf
    )

    print(f"N={N}, IB points={k}")
    print(f"GMRES info: {info}")
    print(f"Relative PDE residual: {relative_pde_residual:.3e}")
    print(f"IB constraint residual: {ib_residual:.3e}")

    return X, Y, u.reshape(N, N), points, info


# use several grid sizes to estimate the convergence rate.
Ns = [20, 40, 80, 160]
hs, errors = [], []

for N in Ns:
    X, Y, u_ib, points, info = solve_ib(N, factor=1)

    h = L / (N + 1)
    error = np.linalg.norm(u_ib - exact(X, Y)) / np.linalg.norm(exact(X, Y))

    hs.append(h)
    errors.append(error)

    print(f"N={N}, error={error:.3e}")

order = np.polyfit(np.log(hs), np.log(errors), 1)[0]
print(f"Convergence order: {order:.2f}")

plt.figure()
plt.loglog(hs, errors, "o-", color=colors[0],
           linewidth=2, markersize=6, label="Numerical error")
plt.loglog(hs, errors[0]*(np.array(hs)/hs[0])**1.5,
           "--", color=colors[1],
           linewidth=2, label=r"$O(h^{1.5})$")
plt.legend()
slope = np.polyfit(np.log(hs), np.log(errors), 1)[0]
plt.text(
    0.05, 0.95,
    f"Slope = {slope:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    va="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)
plt.xlabel("Grid spacing $h$", fontsize=15)
plt.ylabel("L2 error", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig("HW2/problem2_convergence.png", dpi=300, bbox_inches="tight")
plt.show()

plot_solution(X, Y, u_ib, points, filename="problem2_solution.png")







# problem 2a ---------------------------------------------------------------------------------------------------------------
import time

# factor = IB spacing divided by grid spacing.
# 0.5 means twice as many points as factor=1, approximately.
# 4 means roughly four times fewer points.
#
# too few points may reduce boundary accuracy.
# too many points may cause slow or unstable GMRES convergence.

for factor in [0.5, 1, 2, 4]:
    # start timer
    start = time.perf_counter()

    # solve with a different number of IB points.
    X, Y, u, points, info = solve_ib(40, factor=factor)

    # calculate error against the exact solution.
    error = np.linalg.norm(u - exact(X, Y)) / np.linalg.norm(exact(X, Y))

    # subtract current time to get the total computation time
    elapsed = time.perf_counter() - start

    print(f"Spacing={factor}h, IB points={len(points)}, "
          f"error={error:.3e}, time={elapsed:.2f}s, info = {info}")



# problem 3 ---------------------------------------------------------------------------------------------------------------
# reuse the immersed-boundary solver from Problem 2.
# forcing=False sets f(x,y)=0 everywhere.
# the boundary sources q still enforce the values on the circle.
X, Y, u, points, info = solve_ib(60, forcing=False, factor=1)

# create a 3D plot of the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")

# the solution is harmonic away from the IB forcing.
# its shape resembles a minimal surface, but is not exactly one.
# a true minimal surface satisfies a nonlinear PDE.
ax.plot_surface(X, Y, u, cmap="viridis")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
plt.savefig("HW2/problem3_surface.png", dpi=300, bbox_inches="tight")
plt.show()

plot_solution(X, Y, u, points, filename="problem3_solution.png")




# Problem 4: Charlie's Rule --------------------------------------------------------------------------------------------
# Add two circular immersed boundaries instead of one.

N = 60
h = 2*np.pi / (N+1)

# create two smaller circles
theta = np.linspace(0, 2*np.pi, 40, endpoint=False)

circle1 = np.column_stack((
    2 + 0.5*np.cos(theta),
    np.pi + 0.5*np.sin(theta)
))

circle2 = np.column_stack((
    4 + 1*np.cos(theta),
    np.pi + 1*np.sin(theta)
))

# Combine the two sets of boundary points
points = np.vstack((circle1, circle2))

# Solve the same equation as Problem 3
X, Y, u, points, info = solve_ib(N, forcing=False, points=points)

# Plot the resulting surface
plot_solution(X, Y, u, filename="problem4_solution.png")


# Problem 4: 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot numerical solution
surf = ax.plot_surface(
    X, Y, u,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9
)

# Plot both immersed boundaries
for circle_points in [circle1, circle2]:
    px = circle_points[:, 0]
    py = circle_points[:, 1]
    pz = exact(px, py)

    # Close the circle for plotting
    ax.plot(
        np.r_[px, px[0]],
        np.r_[py, py[0]],
        np.r_[pz, pz[0]],
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=3
    )

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("y", fontsize=13)
ax.set_zlabel("u(x, y)", fontsize=13)

fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.1,
             label="u(x, y)")

ax.view_init(elev=30, azim=-60)

plt.savefig("HW2/problem4_surface.png", dpi=300, bbox_inches="tight")
plt.show()

