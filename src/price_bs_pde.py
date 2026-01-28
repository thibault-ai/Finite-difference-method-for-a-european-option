import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix as sparse
import time 


class BlackScholesPricer:
    def __init__(self, Smin, Smax, K, T, r, sigma, option_type='put'):
        self.Smin = Smin
        self.Smax = Smax
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

    def phi(self, s):
        """Payoff à l'échéance."""
        if self.option_type == 'put':
            return np.maximum(self.K - s, 0)
        return np.maximum(s - self.K, 0)

    def u_boundaries(self, t):
        """Conditions aux limites originales."""
        if self.option_type == 'put':
            left = self.K * np.exp(-self.r * t) - self.Smin
            right = 0
        else:
            left = 0
            right = self.Smax - self.K * np.exp(-self.r * t)
        return left, right

    def interpolate(self, s, U, S_val):
        """Interpolation linéaire pour trouver le prix en S_val."""
        s = np.array(s)
        U = np.array(U)
        if S_val < self.Smin or S_val > self.Smax:
            return None
        for i in range(len(s) - 1):
            if s[i] <= S_val <= s[i + 1]:
                h_i = s[i+1] - s[i]
                return ((s[i + 1] - S_val) / h_i) * U[i] + ((S_val - s[i]) / h_i) * U[i + 1]
        return None
    
    def black_scholes_analytical(self, S):
        """Formule de Black-Scholes théorique pour comparaison."""
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def solver(self, N, I, method='implicit', use_sparse=False):
        """
        Résout l'EDP et retourne (s, U, A).
        """
        h = (self.Smax - self.Smin) / (I + 1)
        dt = self.T / N
        s_nodes = self.Smin + h * np.arange(1, I + 1)
        U = self.phi(s_nodes).reshape(I, 1)

        alpha = (self.sigma**2 / 2) * (s_nodes**2 / h**2)
        beta = self.r * s_nodes / (2 * h)

        if not use_sparse:
            A = np.zeros((I, I))
            for i in range(I):
                if i > 0: A[i, i-1] = -alpha[i] + beta[i]
                A[i, i] = 2 * alpha[i] + self.r
                if i < I - 1: A[i, i+1] = -alpha[i] - beta[i]
        else:
            A = diags([-alpha[1:] + beta[1:], 2 * alpha + self.r, -alpha[:-1] - beta[:-1]], [-1, 0, 1], format='csc')

        for n in range(N):
            tn_next = (n + 1) * dt
            tn_curr = n * dt
            L_next, R_next = self.u_boundaries(tn_next)
            L_curr, R_curr = self.u_boundaries(tn_curr)

            if method == 'explicit':
                q_vec = np.zeros((I, 1))
                q_vec[0], q_vec[-1] = (-alpha[0] + beta[0]) * L_curr, (-alpha[-1] - beta[-1]) * R_curr
                U = U - dt * (A @ U + q_vec)
            elif method == 'implicit':
                q_vec = np.zeros((I, 1))
                q_vec[0], q_vec[-1] = (-alpha[0] + beta[0]) * L_next, (-alpha[-1] - beta[-1]) * R_next
                if not use_sparse:
                    U = solve(np.eye(I) + dt * A, U - dt * q_vec)
                else:
                    U = spsolve(speye(I, format='csc') + dt * A, U - dt * q_vec).reshape(I, 1)
                    
            else:#method == 'crank_nicolson':
                q_v = np.zeros((I, 1))
                q_v[0] = 0.5 * (-alpha[0] + beta[0]) * (L_next + L_curr)
                q_v[-1] = 0.5 * (-alpha[-1] - beta[-1]) * (R_next + R_curr)
                
                if not use_sparse:
                    B = np.eye(I) - (dt / 2) * A
                    C = np.eye(I) + (dt / 2) * A
                    rhs = B @ U - dt * q_v
                    U = solve(C, rhs).reshape(I, 1) # Reshape ici
                else:
                    B_mat = speye(I, format='csc') - (dt/2) * A
                    C_mat = speye(I, format='csc') + (dt/2) * A
                    U = spsolve(C_mat, B_mat @ U - dt * q_v).reshape(I, 1)

        final_s = np.concatenate(([self.Smin], s_nodes, [self.Smax]))
        L_final, R_final = self.u_boundaries(self.T)
        final_U = np.concatenate(([L_final], U.flatten(), [R_final]))
        return final_s, final_U, A

# --- FONCTIONS DE GRAPHIQUES ---

def plot_single_result(pricer, N, I, method='implicit', use_sparse=False):
    """Affiche le payoff initial et la solution obtenue avec 
    le schéma pour une configuration donnée."""
    s, U, _ = pricer.solver(N, I, method=method, use_sparse=use_sparse)
    S_ref = np.linspace(pricer.Smin, pricer.Smax, 500)
    
    # Dictionnaire de couleurs
    color_map = {
        'explicit': 'blue',
        'implicit': 'green',
        'crank-nicolson': 'red'
    }
    # On récupère la couleur associée à 'method', sinon rouge par défaut
    line_color = color_map.get(method.lower(), 'red')

    plt.figure(figsize=(8, 6))
    plt.plot(S_ref, pricer.phi(S_ref), 'k--', label="Payoff", alpha=0.5)
    
    # Utilisation de line_color pour la courbe numérique
    plt.plot(s, U, label=f"Numérique ({method}, N={N}, I={I})", color=line_color)
    
    plt.title(f"Prix de l'option {pricer.option_type.upper()} - Schéma {method.replace('_', ' ').title()}")
    plt.xlabel("Valeur du sous-jacent (S)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_four_graphs(pricer, params, method='implicit', title="Comparaison"):
    """Génère la grille 2x2 de graphiques."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title} schéma {method}.", fontsize=16)
    
    # Dictionnaire de couleurs
    color_map = {
        'explicit': 'blue',
        'implicit': 'green',
        'crank-nicolson': 'red'
    }
    line_color = color_map.get(method.lower(), 'black') # noir par défaut si non trouvé

    S_ref = np.linspace(pricer.Smin, pricer.Smax, 500)
    for i, (N, I) in enumerate(params):
        row, col = divmod(i, 2)
        s, U, _ = pricer.solver(N, I, method=method)
        
        axs[row, col].plot(S_ref, pricer.phi(S_ref), 'k--', alpha=0.4)
        # Application de la couleur ici
        axs[row, col].plot(s, U, label=f"N={N}, I={I}", color=line_color)
        
        axs[row, col].set_title(f"Simulation N={N}, I={I}")
        axs[row, col].legend()
        axs[row, col].grid(alpha=0.3)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- FONCTIONS POUR LES TABLEAUX DE CONVERGENCE ---

def generate_convergence_data(pricer, I_values, S_val, method='implicit', n_factor_func=None, use_sparse=False):
    """
    Calcule les données de convergence pour une méthode donnée.
    n_factor_func : une fonction qui définit N en fonction de I (ex: lambda I: I**2 // 10)
    """
    table = []
    
    # 1. Calcul des valeurs de base et temps CPU
    for I in I_values:
        # Calcul de N selon la règle passée en paramètre
        N = n_factor_func(I) if n_factor_func else I
        
        start_time = time.time()
        s, U, _ = pricer.solver(N, I, method=method, use_sparse=use_sparse)
        tcpu = time.time() - start_time
        
        U_val = pricer.interpolate(s, U, S_val)
        exact_val = pricer.black_scholes_analytical(S_val)
        errex = abs(U_val - exact_val)
        
        table.append({
            'I': I, 'N': N, 'U_val': U_val, 
            'errex': errex, 'tcpu': tcpu, 
            'alpha': 'N/A', 'error_succ': 'N/A'
        })

    # 2. Calcul des ordres alpha et erreurs successives
    Smax, Smin = pricer.Smax, pricer.Smin
    for k in range(1, len(table)):
        h_prev = (Smax - Smin) / (table[k-1]['I'] + 1)
        h_curr = (Smax - Smin) / (table[k]['I'] + 1)
        
        # Ordre de convergence alpha
        alpha = np.log(table[k-1]['errex'] / table[k]['errex']) / np.log(h_prev / h_curr)
        table[k]['alpha'] = alpha
        
        # Erreur entre deux approximations successives
        table[k]['error_succ'] = abs(table[k]['U_val'] - table[k-1]['U_val'])
        
    return table

def display_convergence_table(table, title="Tableau de Convergence"):
    """Affiche les données formatées avec 6 chiffres après la virgule."""
    print(f"\n--- {title} ---")
    header = f"{'I':<6} {'N':<8} {'U(s)':<15} {'errex':<15} {'tcpu':<15} {'alpha':<15} {'error_succ':<15}"
    print(header)
    print("-" * len(header))
    
    for row in table:
        u_str = f"{row['U_val']:<15.6f}"
        err_str = f"{row['errex']:<15.6f}"
        t_str = f"{row['tcpu']:<15.6f}"
        
        alpha = row['alpha']
        alpha_str = f"{alpha:<15.6f}" if isinstance(alpha, (int, float)) else f"{alpha:<15}"
        
        succ = row['error_succ']
        succ_str = f"{succ:<15.6f}" if isinstance(succ, (int, float)) else f"{succ:<15}"
        
        print(f"{row['I']:<6} {row['N']:<8} {u_str} {err_str} {t_str} {alpha_str} {succ_str}")