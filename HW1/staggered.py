# I asked ChatGPT to create all this code for me.
# I first made sure I understood how the staggered grid worked by writing it all out on my tablet.
# I basically just did what we did in class.
# Then I wrote out how I think the error should look based off the fact that we're using second order finite difference methods.
# I gave that writing to ChatGPT and the assignment, and asked it to check my work.
# Lastly, I asked for code based on what I wrote.

# import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time
from scipy.sparse.linalg import gmres

# Plot formatting
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12

# Analytical solutions
# These are the solutions we created with the method of manufactured solutions
def Uan(x, y):
    return np.sin(2*np.pi*y) * np.sin(2*np.pi*x)

def Van(x, y):
    return -3.5 + np.cos(2*np.pi*x)*(np.cos(2*np.pi*y) - 1)

def Pan(x, y):
    return np.sin(2*np.pi*y) * np.cos(2*np.pi*x)


# RHS functions 
# I calculated these by hand and then verified them with ChatGPT
def f(x, y):
    return (-8*np.pi**2 + 2*np.pi) * np.sin(2*np.pi*y)*np.sin(2*np.pi*x)

def g(x, y):
    return (-8*np.pi**2*np.cos(2*np.pi*y) + 4*np.pi**2
            - 2*np.pi*np.cos(2*np.pi*y)) * np.cos(2*np.pi*x)


# Now we will actually solve stokes equation
def solve_stokes(N, return_system=False):
    # we know h = 1/N
    h = 1/N

    # Number of unknown U, V, and P values
    # This makes the most sense when you look at figure 1 in the assignment
    nU = N*N
    nV = N*(N-1) # top/bottom V values are known
    nP = N*N
    n = nU + nV + nP # total number of unknowns

    # Convert (i,j) locations into positions in our big vector [U,V,P]
    # The "big vector" is the flattened vector 
    # This is similar to what we did in HW0 to find the indices in the flattened vector
    iu = lambda i,j: j*N + i
    iv = lambda i,j: nU + (j-1)*N + i
    ip = lambda i,j: nU + nV + j*N + i

    # ChatGPT used lil_matrix, which is apparently just a sparse matrix
    A = lil_matrix((n,n))
    b = np.zeros(n)

    # Create the U equation matrix
    # U equation: LU - Px = f
    for j in range(N):
        for i in range(N):
            row = iu(i,j)
            L, R = (i-1)%N, (i+1)%N     # periodic in x

            # Uxx
            A[row, iu(L,j)] += 1/h**2
            A[row, iu(i,j)] += -2/h**2
            A[row, iu(R,j)] += 1/h**2

            # Uyy; U=0 at top/bottom using ghost points
            if j == 0:
                A[row, iu(i,j)] += -3/h**2
                A[row, iu(i,j+1)] += 1/h**2
            elif j == N-1:
                A[row, iu(i,j)] += -3/h**2
                A[row, iu(i,j-1)] += 1/h**2
            else:
                A[row, iu(i,j-1)] += 1/h**2
                A[row, iu(i,j)] += -2/h**2
                A[row, iu(i,j+1)] += 1/h**2

            # -Px = -(Pright - Pleft)/h
            A[row, ip(L,j)] += 1/h
            A[row, ip(i,j)] -= 1/h

            b[row] = f(i*h, (j+0.5)*h)

    # Create the V equation matrix
    # V equation: LV - Py = g
    for j in range(1,N):
        for i in range(N):
            row = iv(i,j)
            L, R = (i-1)%N, (i+1)%N

            # Vxx
            A[row, iv(L,j)] += 1/h**2
            A[row, iv(i,j)] += -2/h**2
            A[row, iv(R,j)] += 1/h**2

            # Vyy
            A[row, iv(i,j)] += -2/h**2

            if j > 1:
                A[row, iv(i,j-1)] += 1/h**2
            else:
                b[row] += 3.5/h**2       # V=-3.5 boundary

            if j < N-1:
                A[row, iv(i,j+1)] += 1/h**2
            else:
                b[row] += 3.5/h**2

            # -Py
            A[row, ip(i,j-1)] += 1/h
            A[row, ip(i,j)] -= 1/h

            b[row] += g((i+0.5)*h, j*h)

    # Create the divergence matrix
    # Divergence: Ux + Vy = 0
    for j in range(N):
        for i in range(N):
            row = nU + nV + j*N + i
            R = (i+1)%N

            # Ux = (Uright - Uleft)/h
            A[row, iu(R,j)] += 1/h
            A[row, iu(i,j)] -= 1/h

            # Vy = (Vtop - Vbottom)/h
            if j > 0:
                A[row, iv(i,j)] -= 1/h
            else:
                b[row] -= 3.5/h

            if j < N-1:
                A[row, iv(i,j+1)] += 1/h
            else:
                b[row] += 3.5/h

    # Pressure needs one reference value because P+C gives the same pressure gradient as P
    # We are "pinning" pressure
    row = nU + nV          # choose one equation to replace (the first P equation after the U and V equations)
    A[row,:] = 0           # erase that equation
    A[row,ip(0,0)] = 1     # replace it with P[0,0]
    b[row] = 0             # require P[0,0] = 0

    # backslash
    A = A.tocsr()
    if return_system:
        return A, b
    sol = spsolve(A.tocsr(), b)

    # Make the solutions back into matrices from the flattened arrays
    U = sol[:nU].reshape(N,N)
    V = sol[nU:nU+nV].reshape(N-1,N)
    P = sol[nU+nV:].reshape(N,N)

    # Assignment says to subtract the mean pressure
    # This is essentially removing the added constant from the solution for the pressure values
    P -= np.mean(P)

    return U, V, P


# Problem 1 code
N = 40
U, V, P = solve_stokes(N)

h = 1/N

# Add the known V=-3.5 boundaries back in
Vfull = -3.5*np.ones((N+1,N))
Vfull[1:N,:] = V

# Average U and V onto the pressure-cell centers
# Because U and V live around P, we need to average them to find U and V where P is
Uc = 0.5*(U + np.roll(U,-1,axis=1))
Vc = 0.5*(Vfull[:-1,:] + Vfull[1:,:])

speed = np.sqrt(Uc**2 + Vc**2)

x = (np.arange(N)+0.5)*h
y = (np.arange(N)+0.5)*h

plt.pcolormesh(x, y, speed, shading="auto", cmap="viridis")
plt.colorbar(label=r"$||u||$")
plt.streamplot(x, y, Uc, Vc, color="black")

plt.xlabel("x")
plt.ylabel("y")
plt.show()



# Problem 2 code
# Problem 2 code
Ns = [10, 20, 30, 40, 50]

errU = []
errV = []
errP = []

for N in Ns:
    h = 1/N
    U, V, P = solve_stokes(N)

    # Add the known V=-3.5 boundaries back in
    Vfull = -3.5*np.ones((N+1, N))
    Vfull[1:N, :] = V

    # -------------------------------------------------------
    # Create the grids where U, V, and P actually live
    # -------------------------------------------------------

    # U: integer x, half-integer y
    xU = np.arange(N)*h
    yU = (np.arange(N)+0.5)*h
    XU, YU = np.meshgrid(xU, yU)

    # V: half-integer x, integer y
    xV = (np.arange(N)+0.5)*h
    yV = np.arange(N+1)*h
    XV, YV = np.meshgrid(xV, yV)

    # P: half-integer x, half-integer y
    xP = (np.arange(N)+0.5)*h
    yP = (np.arange(N)+0.5)*h
    XP, YP = np.meshgrid(xP, yP)

    # Analytical solutions evaluated on their correct grids
    Uexact = Uan(XU, YU)
    Vexact = Van(XV, YV)
    Pexact = Pan(XP, YP)

    # Remove pressure mean before comparing
    Pexact -= np.mean(Pexact)

    # Relative L2 errors
    errU.append(np.linalg.norm(U-Uexact) / np.linalg.norm(Uexact))
    errV.append(np.linalg.norm(Vfull-Vexact) / np.linalg.norm(Vexact))
    errP.append(np.linalg.norm(P-Pexact) / np.linalg.norm(Pexact))


# Convert to arrays so we can do calculations with all values
Ns = np.array(Ns)
errU = np.array(errU)
errV = np.array(errV)
errP = np.array(errP)


# Reference lines
# Scale them to start at the first U error so we compare slopes
first_order = errU[0]*(Ns[0]/Ns)
second_order = 1.3 * errU[0]*(Ns[0]/Ns)**2

# Plot errors and expected convergence rates
viridis = plt.cm.viridis

plt.loglog(Ns, errU, "o-", color=viridis(0.15), label="U error")
plt.loglog(Ns, errV, "s-", color=viridis(0.45), label="V error")
plt.loglog(Ns, errP, "^-", color=viridis(0.75), label="P error")

# Keep reference lines neutral
plt.loglog(Ns, first_order, "--", color="gray", label=r"$O(N^{-1})$")
plt.loglog(Ns, second_order, "--", color="black", label=r"$O(N^{-2})$")

plt.xlabel("N")
plt.ylabel("Relative error")
plt.legend()
plt.grid(True)
plt.show()



# Problem 3: Compare direct solver and GMRES
Ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

direct_times = []
gmres_times = []
differences = []

for N in Ns:

    # Build the same linear system for both methods
    A, b = solve_stokes(N, return_system=True)

    # Direct solver
    start = time.perf_counter()
    sol_direct = spsolve(A, b)
    direct_times.append(time.perf_counter() - start)

    # GMRES
    start = time.perf_counter()
    sol_gmres, info = gmres(A, b)
    gmres_times.append(time.perf_counter() - start)

    # Difference between the two solutions
    difference = (
        np.linalg.norm(sol_gmres - sol_direct)
        / np.linalg.norm(sol_direct)
    )
    differences.append(difference)

    print(
        f"N = {N}: "
        f"direct = {direct_times[-1]:.4f} s, "
        f"GMRES = {gmres_times[-1]:.4f} s, "
        f"info = {info}, "
        f"difference = {difference:.2e}"
    )

# Plot solver times
plt.figure(figsize=(8, 6))

plt.plot(
    Ns, direct_times, "o-",
    color=viridis(0.25),
    linewidth=2,
    label="Direct solver"
)

plt.plot(
    Ns, gmres_times, "s-",
    color=viridis(0.75),
    linewidth=2,
    label="GMRES"
)

plt.xlabel("Grid Resolution N")
plt.ylabel("Solver Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()