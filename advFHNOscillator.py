import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit
import argparse
import os

#### Parameters #############
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default="WSp=0.txt", help='file name of network')
parser.add_argument('--N', type=int, default=90, help='num of FHN oscillator')
parser.add_argument('--sigma', type=float, default=0.0506, help='FHN model parameter')
parser.add_argument('--epsilon', type=float, default=0.05, help='FHN model parameter')
parser.add_argument('--a', type=float, default=0.5, help='FHN model parameter')
parser.add_argument('--tmax', type=float, default=10000, help='maximum simulation time')
parser.add_argument('--dt', type=float, default=0.5, help='size of dt')
parser.add_argument('--t_interval', type=float, default=1.0, help='perturbation interval')
parser.add_argument('--attack_eps', type=float, default=0.05, help='strength of perturbation')
parser.add_argument('--seed', type=int, default=128, help='random seed for reproducibility')
parser.add_argument('--random', action="store_true", help='perform random attacks with strength eps')
parser.add_argument('--save_path', type=str, default="results", help='save path of results')
parser.add_argument('--export_t', action="store_true", help='export r values')
args = parser.parse_args()

# Kuramoto model
@jit(nopython=True)
def fhn_ode(X, N, epsilon, sigma, a, A, B):
    u = X[:N]
    v = X[N:]
    du = np.zeros(N)
    dv = np.zeros(N)
    for k in range(N):
        sum_u = np.sum(A[k, :] * (B[0, 0] * (u - u[k]) + B[0, 1] * (v - v[k])))
        sum_v = np.sum(A[k, :] * (B[1, 0] * (u - u[k]) + B[1, 1] * (v - v[k])))
        du[k] = (u[k] - u[k]**3 / 3 - v[k] + sigma * sum_u) / epsilon
        dv[k] = u[k] + a + sigma * sum_v
    
    result = np.zeros(2 * N)
    result[:N] = du
    result[N:] = dv

    return result

# wrapper for solve_ivp
def fhn_wrapper(t, X, N, epsilon, sigma, a, A, B):
    return fhn_ode(X, N, epsilon, sigma, a, A, B)

def run_fhn_simulation(X0, A, B, attack_eps, dt=0.05):
    r_values = []
    t_values = []
    
    # pre-running for JIT
    _ = fhn_ode(X0, args.N, args.epsilon, args.sigma, args.a, A, B)
    
    t_current = 0
    uv_current = X0.copy()
    
    while t_current < args.tmax:
        t_end = round(min(t_current + args.t_interval, args.tmax), 6)
        t_eval = np.round(np.arange(t_current, t_end, dt), 6)

        sol = solve_ivp(
            fhn_wrapper,
            [t_current, t_end],
            uv_current,
            t_eval=t_eval,
            args=(args.N, args.epsilon, args.sigma, args.a, A, B)
        )
        
        uv_current = sol.y[:, -1]
        u = uv_current[:args.N]
        v = uv_current[args.N:]

        # add perturbation
        if t_end < args.tmax:
            if args.random:
                u = u + np.random.choice([-attack_eps, attack_eps], size=len(u))
                v = v + np.random.choice([-attack_eps, attack_eps], size=len(v))
            else:
                # dr/dgeophi
                geo_phi = np.arctan2(v, u)
                psi = np.angle(np.mean(np.exp(1j * geo_phi)))
                dr_dgeophi = np.sin(psi - geo_phi)

                # dr/du
                dgeophi_du = -v / (u**2 + v**2)
                dr_du = dr_dgeophi * dgeophi_du
                
                # dr/dv
                dgeophi_dv = u / (u**2 + v**2)
                dr_dv = dr_dgeophi * dgeophi_dv

                # add perturbation
                u = u + attack_eps * np.sign(dr_du)
                v = v + attack_eps * np.sign(dr_dv)

            uv_current[:args.N] = u
            uv_current[args.N:] = v

        t_current = t_end
        t_values.append(t_current)
    
        # compute order parameter
        for i in range(sol.y.shape[1]):
            u_i = sol.y[:args.N, i]
            v_i = sol.y[args.N:, i]
            geo_phase = np.arctan2(v_i, u_i)
            r_i = np.abs(np.mean(np.exp(1j * geo_phase)))
            r_values.append(r_i)
        #geo_phase = np.arctan2(v, u)
        #r = np.abs(np.mean(np.exp(1j * geo_phase)))
        #r_values.append(r)
    
    return t_values, r_values

if __name__ == "__main__":
    # set random seed
    np.random.seed(args.seed)
    
    # set A, B
    A = np.loadtxt(args.network)
    phi = np.pi/2 - 0.1
    B = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    # initial u, v
    u0 = np.random.uniform(-1, 1, size=args.N)
    v0 = np.random.uniform(-1, 1, size=args.N)
    X0 = np.concatenate([u0, v0])

    # simulation on FHN Oscillator Model
    t_values, r_values = run_fhn_simulation(X0, A, B, attack_eps=args.attack_eps, dt=args.dt)

    # save result
    np.savetxt(f"{args.save_path}/r_values_{args.network[:-4]}_eps={args.attack_eps}_seed={args.seed}_random={args.random}.txt", r_values)
    if args.export_t:
        np.savetxt(f"{args.save_path}/t_values.txt", t_values)
    