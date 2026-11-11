# All of this code was given to me using ChatGPT
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp

def setup(N):
    h = 2*np.pi/(N-1)
    x = np.linspace(0, 2*np.pi, N)[1:-1]
    X, Y = np.meshgrid(x, x)

    m = N-2

    # Fourth-order finite-difference Laplacian
    D = diags(
        [-np.ones(m-2), 16*np.ones(m-1),
         -30*np.ones(m), 16*np.ones(m-1),
         -np.ones(m-2)],
        [-2, -1, 0, 1, 2],
        shape=(m, m),
        format='csc'
    ) / (12*h**2)

    D = D.tolil()
    D[0, 0] += 1/(12*h**2)
    D[-1, -1] += 1/(12*h**2)
    D = D.tocsc()

    I = eye(m, format='csc')
    L = (kron(I, D) + kron(D, I)).tocsc()

    u0 = (np.sin(X)*np.sin(Y)).ravel()

    exact = lambda t: np.cos(10*t)*u0
    forcing = lambda t: (2*np.cos(10*t)-10*np.sin(10*t))*u0

    return L, u0, exact, forcing

def setup_second_order(N):
    h = 2*np.pi/(N-1)
    x = np.linspace(0, 2*np.pi, N)[1:-1]
    X, Y = np.meshgrid(x, x)

    m = N-2
    D = diags([np.ones(m-1), -2*np.ones(m), np.ones(m-1)],
              [-1, 0, 1], shape=(m, m), format='csc') / h**2

    I = eye(m, format='csc')
    L = (kron(I, D) + kron(D, I)).tocsc()

    u0 = (np.sin(X)*np.sin(Y)).ravel()
    exact = lambda t: np.cos(10*t)*u0
    forcing = lambda t: (2*np.cos(10*t)-10*np.sin(10*t))*u0

    return L, u0, exact, forcing

def FE(L, u0, f, dt, steps):
    u = u0.copy()

    for n in range(steps):
        t = n*dt
        u = u + dt*(L@u + f(t))

    return u

def BE(L, u0, f, dt, steps):
    u = u0.copy()
    solve = factorized(eye(len(u0), format='csc')-dt*L)

    for n in range(steps):
        t = n*dt
        u = solve(u + dt*f(t+dt))

    return u

def CN(L, u0, f, dt, steps):
    u = u0.copy()
    I = eye(len(u0), format='csc')
    solve = factorized(I-dt*L/2)

    for n in range(steps):
        t = n*dt
        u = solve(u + dt/2*(L@u + f(t) + f(t+dt)))

    return u

def RK2(L, u0, f, dt, steps):
    u = u0.copy()

    for n in range(steps):
        t = n*dt

        # Predictor
        u_half = u + dt/2*(L@u + f(t))

        # Corrector
        u = u + dt*(L@u_half + f(t+dt/2))

    return u

def BDF2(L, u0, f, dt, steps):
    u_old = u0.copy()
    I = eye(len(u0), format='csc')

    # First step: Crank-Nicolson
    solve0 = factorized(I-dt*L/2)
    u = solve0(u_old + dt/2*(L@u_old + f(0) + f(dt)))

    # Remaining steps: BDF2
    solve = factorized(3*I-2*dt*L)

    for n in range(1, steps):
        t = n*dt
        u_new = solve(4*u-u_old+2*dt*f(t+dt))
        u_old, u = u, u_new

    return u

def calculate_error(args):
    method, N, steps = args

    T = 2*np.pi
    dt = T/steps

    L, u0, exact, f = setup(N)

    solvers = {
        'FE': FE,
        'BE': BE,
        'CN': CN,
        'RK2': RK2,
        'BDF2': BDF2
    }

    u = solvers[method](L, u0, f, dt, steps)

    return np.linalg.norm(u-exact(T))/np.linalg.norm(exact(T))

def convergence_plot(method):
    T = 2*np.pi
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    colors = plt.cm.viridis([0.25, 0.75])

    for i, N in enumerate([64, 128]):
        h = 2*np.pi/(N-1)

        # Choose time steps
        if method in ['FE', 'RK2']:
            base = int(np.floor(16*T/(3*h**2))) + 1
            steps_list = base*np.array([1, 2, 3, 4])
        else:
            steps_list = np.array([160, 320, 640, 1280])

        dt_values = T/steps_list

        # Calculate errors in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            errors = list(executor.map(
                calculate_error,
                [(method, N, int(steps)) for steps in steps_list]
            ))

        # Estimate convergence order
        p = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
        print(f'{method}, N={N}: order = {p:.3f}')

        # Plot on corresponding subplot
        ax = axes[i]
        ax.loglog(dt_values, errors, 'o-', linewidth=3,
                  markersize=9, color=colors[i],
                  label=f'Order = {p:.4f}')

        ax.set_title(f'N = {N}', fontsize=22)
        ax.set_xlabel(r'Time step $\Delta t$', fontsize=19)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=16)

    axes[0].set_ylabel('Relative error', fontsize=19)
    fig.suptitle(f'{method} Convergence', fontsize=25)

    plt.tight_layout()
    plt.savefig(f"HW3/convergence_{method}.png", dpi=300, bbox_inches="tight")
    plt.show()


def compute_cost(method, N, steps):
    M = N**2

    mult = 5*M
    solve = M*np.log(M)
    factor = M**1.5

    if method == 'FE':
        return steps*mult
    if method == 'BE':
        return factor + steps*solve
    if method == 'CN':
        return factor + steps*(mult + solve)
    if method == 'RK2':
        return 2*steps*mult
    if method == 'BDF2':
        return 2*factor + mult + steps*solve


def problem2_test(args):
    method, N, tol = args
    T = 2*np.pi
    h = 2*np.pi/(N-1)

    # Eigenvalue of the corrected fourth-order Laplacian
    lam = -8*np.sin(h/2)**2/h**2

    def f(t):
        return 2*np.cos(10*t) - 10*np.sin(10*t)

    def run(steps):
        dt = T/steps
        a = 1.0

        if method == 'BDF2':
            old = a
            a = ((1+dt*lam/2)*a + dt/2*(f(0)+f(dt)))/(1-dt*lam/2)

        for n in range(1 if method == 'BDF2' else 0, steps):
            t = n*dt

            if method == 'FE':
                a += dt*(lam*a + f(t))

            elif method == 'BE':
                a = (a+dt*f(t+dt))/(1-dt*lam)

            elif method == 'CN':
                a = ((1+dt*lam/2)*a + dt/2*(f(t)+f(t+dt)))/(1-dt*lam/2)

            elif method == 'RK2':
                half = a + dt/2*(lam*a+f(t))
                a += dt*(lam*half+f(t+dt/2))

            elif method == 'BDF2':
                new = (4*a-old+2*dt*f(t+dt))/(3-2*dt*lam)
                old, a = a, new

        return abs(a-1)  # Same relative error at T=2*pi

    # Stability restriction
    if method in ['FE', 'RK2']:
        steps = int(np.floor(4*T/h**2)) + 1
    else:
        steps = 2

    while steps <= 500000:
        error = run(steps)

        if error <= tol:
            return method, N, tol, T/steps, error, compute_cost(method, N, steps)

        steps *= 2

    return method, N, tol, None, None, None

def problem2():
    methods = ['FE', 'BE', 'CN', 'RK2', 'BDF2']
    tolerances = [5e-2, 1e-3, 5e-6]

    jobs = [(method, N, tol)
            for N in [64, 128]
            for tol in tolerances
            for method in methods]

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(problem2_test, jobs))

    for N in [64, 128]:
        for tol in tolerances:
            print(f'\nN={N}, tolerance={tol:.1e}')

            valid = []

            for method, n, t, dt, error, cost in results:
                if n == N and t == tol:
                    if dt is None:
                        print(f'{method}: tolerance not reached')
                    else:
                        print(f'{method}: steps={round(2*np.pi/dt)}, dt={dt:.3e}, error={error:.3e}, cost={cost:.3e}')
                        valid.append((cost, method))

            if valid:
                print('Cheapest:', min(valid)[1])

def plot_error(args):
    method, N, steps = args
    T = 2*np.pi
    dt = T/steps
    h = 2*np.pi/(N-1)

    lam = -8*np.sin(h/2)**2/h**2

    def f(t):
        return 2*np.cos(10*t)-10*np.sin(10*t)

    a = 1.0

    if method == 'BDF2':
        old = a
        a = ((1+dt*lam/2)*a + dt/2*(f(0)+f(dt)))/(1-dt*lam/2)

    for n in range(1 if method == 'BDF2' else 0, steps):
        t = n*dt

        if method == 'FE':
            a += dt*(lam*a+f(t))
        elif method == 'BE':
            a = (a+dt*f(t+dt))/(1-dt*lam)
        elif method == 'CN':
            a = ((1+dt*lam/2)*a+dt/2*(f(t)+f(t+dt)))/(1-dt*lam/2)
        elif method == 'RK2':
            half = a+dt/2*(lam*a+f(t))
            a += dt*(lam*half+f(t+dt/2))
        elif method == 'BDF2':
            new = (4*a-old+2*dt*f(t+dt))/(3-2*dt*lam)
            old, a = a, new

    return abs(a-1)

def problem2_plot(method):
    T = 2*np.pi
    colors = plt.cm.viridis([0.25, 0.75])
    tolerances = [5e-2, 1e-3, 5e-6]

    plt.figure(figsize=(10, 7))

    for i, N in enumerate([64, 128]):
        h = 2*np.pi/(N-1)

        if method in ['FE', 'RK2']:
            base = int(np.floor(4*T/h**2)) + 1
            steps_list = np.unique(
                np.geomspace(base, base*1000, 45).astype(int)
            )
        else:
            steps_list = np.unique(
                np.geomspace(40, 100000, 60).astype(int)
            )

        dt_values = T/steps_list

        with ProcessPoolExecutor(max_workers=4) as executor:
            errors = list(executor.map(
                plot_error,
                [(method, N, int(s)) for s in steps_list]
            ))
        
        lam = -8*np.sin(h/2)**2/h**2
        plateau = abs((2+lam)*lam*(np.exp(lam*T)-1)/(lam**2+100))

        plt.loglog(
            dt_values, errors, 'o-',
            linewidth=2.5, markersize=5,
            color=colors[i], label=f'N = {N} (spatial error = {plateau:.2e})'
        )

        # Indicate approximate spatial-error plateau
        plt.axhline(
            abs(-8*np.sin(h/2)**2/h**2 + 2)/104,
            color=colors[i], linestyle=':', alpha=0.4
        )

    # Tolerance lines
    for tol in tolerances:
        plt.axhline(
            tol, color='gray', linestyle='--',
            linewidth=1.5, alpha=0.7,
            label=f'Tolerance = {tol:.0e}'
        )
    
    plt.gca().invert_xaxis()

    plt.xlabel(r'Time step $\Delta t$', fontsize=19)
    plt.ylabel('Relative error', fontsize=19)
    plt.title(f'{method} Error vs. Time Step', fontsize=23)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=13)
    plt.grid(True, which='both', alpha=0.25)
    plt.tight_layout()

    plt.savefig(f'HW3/problem2_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()



def extra_credit(N):
    T = 2*np.pi
    L, u0, exact, f = setup(N)

    # Heat equation as a system of ODEs
    def rhs(t, u):
        return L@u + f(t)

    # Solve using adaptive RK45
    sol = solve_ivp(
        rhs, (0, T), u0,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    # Calculate relative error at T
    error = np.linalg.norm(sol.y[:, -1] - exact(T)) / np.linalg.norm(exact(T))

    print(f'N = {N}')
    print(f'RK45 relative error = {error:.6e}')
    print(f'Accepted time steps = {len(sol.t)-1}')
    print(f'Function evaluations = {sol.nfev}')

    # Plot adaptive time steps, excluding startup steps
    dt_values = np.diff(sol.t)

    plt.figure(figsize=(10, 6))
    plt.scatter(sol.t[1:][5:], dt_values[5:], s=4, alpha=0.5, color=plt.cm.viridis(0.65))

    plt.xlabel('Time', fontsize=19)
    plt.ylabel(r'Adaptive time step $\Delta t$', fontsize=19)
    plt.title(f'RK45 Adaptive Time Steps (N={N})', fontsize=23)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'HW3/RK45_steps_{N}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return sol


if __name__ == "__main__":
    # for method in ['FE', 'BE', 'CN', 'RK2', 'BDF2']:
    #     convergence_plot(method)
    # for method in ['FE', 'BE', 'CN', 'RK2', 'BDF2']:
    #     problem2_plot(method)
    problem2()
    # extra_credit(64)
    # extra_credit(128)