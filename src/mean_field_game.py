import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# =================================================================
# 1. FONCTIONS DE BASE (PHYSIQUE ET GRILLE)
# =================================================================

def phi(x):
    """Condition terminale pour U à T=1."""
    return -np.exp(-40 * (np.array(x) - 0.7)**2)

def m0(x):
    """Condition initiale pour M à T=0."""
    return np.sqrt(300 / np.pi) * np.exp(-300 * (np.array(x) - 0.2)**2)

def H_tilde(p1, p2, mu, beta, c0, c1, alpha, mode="mfg"):
    """Hamiltonien unifié. En mode MFC, inclut le terme d'anticipation du planificateur."""
    fac = (np.maximum(-p1, 0)**2 + np.maximum(p2, 0)**2)
    congestion = 1 / (c0 + c1 * mu)**alpha
    res = (1 / beta) * congestion * (fac**(beta / 2))
    
    if mode == "mfc":
        # Le terme correctif pour le contrôle de champ moyen
        factor = (c1 * alpha * mu) / (c0 + c1 * mu)
        res = res * (1 - factor)
    return res

def jacob_htilde(U, M_tilde, beta, c0, c1, alpha, h, D_curr, D_past, mode="mfg"):
    """Calcule la matrice Jacobienne du Hamiltonien."""
    p1 = (1/h) * (D_curr @ U)
    p2 = (1/h) * (D_past @ U) 
    fac = (np.maximum(-p1, 0)**2 + np.maximum(p2, 0)**2)
    
    coef = 1 / ((c0 + c1 * M_tilde)**alpha)
    if mode == "mfc":
        coef *= (1 - (c1 * alpha * M_tilde) / (c0 + c1 * M_tilde))
        
    # Sécurité pour la division par zéro
    factor = coef * np.power(fac, beta/2 - 1, out=np.zeros_like(fac), where=(fac != 0))

    main_diag = (1/h * (np.maximum(-p1, 0) + np.maximum(p2, 0)) * factor)
    sub_diag = (- 1/h * np.maximum(p2, 0) * factor)[1:]
    sup_diag = (- 1/h * np.maximum(-p1, 0) * factor)[:-1]
    
    return np.diag(sub_diag, k=-1) + np.diag(main_diag) + np.diag(sup_diag, k=+1)

# =================================================================
# 2. OUTILS MATRICIELS
# =================================================================

def get_operators(Nh):
    """Initialise les matrices de différences finies."""
    I = np.eye(Nh)
    # Laplacien avec conditions de Neumann
    A = (np.diag(-2*np.ones(Nh)) + np.diag(np.ones(Nh-1), 1) + np.diag(np.ones(Nh-1), -1))
    A[0, 0], A[-1, -1] = -1, -1
    # Dérivées décentrées
    D_curr = (np.diag(-np.ones(Nh), 0) + np.diag(np.ones(Nh-1), 1))
    D_curr[-1, -1] = 0
    D_past = (np.diag(-np.ones(Nh-1), -1) + np.diag(np.ones(Nh), 0))
    D_past[0, 0] = 0
    return I, A, D_curr, D_past

def convert_to_banded(matrix):
    """Convertit une matrice tridiagonale pour solve_banded."""
    ab = np.zeros((3, matrix.shape[0]))
    ab[0, 1:] = np.diag(matrix, k=1)  # Diagonale sup
    ab[1, :] = np.diag(matrix)       # Diagonale principale
    ab[2, :-1] = np.diag(matrix, k=-1) # Diagonale inf
    return ab

# =================================================================
# 3. SOLVEURS (HJB, KFP ET POINT FIXE)
# =================================================================

def solve_hjb(M, x_grid, params, mats, mode="mfg"):
    """Résout HJB à rebours dans le temps (Backward)."""
    U = np.zeros((params['NT']+1, params['Nh']))
    U[params['NT']] = phi(x_grid)
    
    for n in range(params['NT']-1, -1, -1):
        Un = U[n+1].copy()
        for _ in range(50): # Max Newton iterations
            p1, p2 = (1/params['h'])*(mats['D_curr'] @ Un), (1/params['h'])*(mats['D_past'] @ Un)
            H = H_tilde(p1, p2, M[n+1], params['beta'], params['c0'], params['c1'], params['alpha'], mode=mode)
            # Couplage : 0.1*M en MFG, 0 en MFC (selon structure PDF)
            f_coupling = (0.1 * M[n+1]) if mode == "mfg" else 0
            F = (1/params['dt'])*(U[n+1] - Un) + params['kappa']*(mats['A'] @ Un) - H + f_coupling
            
            Jac_H = jacob_htilde(Un, M[n+1], params['beta'], params['c0'], params['c1'], params['alpha'], params['h'], mats['D_curr'], mats['D_past'], mode=mode)
            J_F = (-1/params['dt'])*mats['I'] + params['kappa']*mats['A'] - Jac_H
            
            dU = scipy.linalg.solve_banded((1, 1), convert_to_banded(J_F), -F)
            Un += dU
            if np.linalg.norm(F) < 1e-10: break
        U[n] = Un
    return U

def solve_kfp(U, x_grid, params, mats):
    """Résout Kolmogorov vers l'avant (Forward)."""
    M = np.zeros((params['NT']+1, params['Nh']))
    M[0] = m0(x_grid)
    for n in range(params['NT']):
        # IMPORTANT : Pour KFP, on utilise toujours le mode 'mfg' pour la vitesse de transport
        Jac_H = jacob_htilde(U[n], M[n], params['beta'], params['c0'], params['c1'], params['alpha'], params['h'], mats['D_curr'], mats['D_past'], mode="mfg")
        P = (1/params['dt'])*mats['I'] - params['kappa']*mats['A'] + Jac_H.T
        M[n+1] = scipy.linalg.solve_banded((1, 1), convert_to_banded(P), (1/params['dt'])*M[n])
    return M

def run_simulation(mode, p_vals, Nh=201, NT=100):
    """Exécute une itération de point fixe complète avec relaxation de M et U."""
    h, dt = 1/(Nh-1), 1.0/NT
    x_grid = np.linspace(0, 1, Nh)
    I, A, D_curr, D_past = get_operators(Nh)
    
    params = {**p_vals, 'Nh': Nh, 'NT': NT, 'h': h, 'dt': dt}
    mats = {'I': I, 'A': A, 'D_curr': D_curr, 'D_past': D_past}
    
    # Initialisation de M (densité) et U (valeur)
    M = np.tile(m0(x_grid), (NT+1, 1))
    U = np.tile(phi(x_grid), (NT+1, 1)) # Initialisation de U pour la relaxation
    
    for k in range(p_vals['max_iter']):
        # 1. Résolution HJB complète (donne une proposition U_k+1)
        U_proposal = solve_hjb(M, x_grid, params, mats, mode=mode)
        
        # 2. Mise à jour de U avec relaxation (ton étape manquante)
        U_new = (1 - p_vals['theta']) * U + p_vals['theta'] * U_proposal
        
        # 3. Résolution KFP (on utilise le U_new relaxé pour le transport)
        M_proposal = solve_kfp(U_new, x_grid, params, mats)
        
        # 4. Mise à jour de M avec relaxation
        M_new = (1 - p_vals['theta']) * M + p_vals['theta'] * M_proposal
        
        # Calcul de l'erreur pour la convergence
        # On regarde la différence combinée entre les deux itérations
        err = np.linalg.norm(M_new - M) + np.linalg.norm(U_new - U)
        
        # Mise à jour des variables pour l'itération k+1
        M = M_new
        U = U_new
        
        if err < 1e-6:
            print(f"   -> [{mode.upper()}] Convergé en {k} itérations (err={err:.2e})")
            break
            
    return M, U, x_grid, np.linspace(0, 1.0, NT+1)


# =================================================================
# 4. RUN EXPERIMENTS & VISUALISATION
# =================================================================

def plot_u_m_contour_complete(M, U, x, t, idx, p, mode="mfg"):
    """Affiche les lignes de niveau pour U et M avec les paramètres en titre."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # Paramètres pour les titres (sigma est extrait de kappa pour l'affichage si besoin)
    # ou on peut ajouter 'sigma' dans le dictionnaire p lors de la boucle
    sigma_val = np.sqrt(p['kappa'] * 2 * (x[1]-x[0])**2) 
    
    title_params = f"Exp {idx} ({mode.upper()})\n"
    title_params += f"alpha={p['alpha']}, sigma={sigma_val:.3f}, c0={p['c0']}, c1={p['c1']}, theta={p['theta']}"
    
    # Graphique de gauche : Fonction de valeur U
    levels_u = 50
    cp1 = ax[0].contour(x, t, U, levels=levels_u, cmap='coolwarm')
    ax[0].clabel(cp1, inline=True, fontsize=8)
    fig.colorbar(cp1, ax=ax[0])
    ax[0].set_title(f"Fonction de Valeur $u(x,t)$\n{title_params}")
    ax[0].set_xlabel("Espace (x)")
    ax[0].set_ylabel("Temps (t)")
    
    # Graphique de droite : Densité M
    levels_m = 50
    cp2 = ax[1].contour(x, t, M, levels=levels_m, cmap='viridis')
    ax[1].clabel(cp2, inline=True, fontsize=8)
    fig.colorbar(cp2, ax=ax[1])
    ax[1].set_title(f"Densité de Population $m(x,t)$\n{title_params}")
    ax[1].set_xlabel("Espace (x)")
    ax[1].set_ylabel("Temps (t)")
    
    plt.tight_layout()
    plt.show()

def run_mfg_sensitivity():
    """Livre les 5 simulations MFG demandées pour l'étude de sensibilité."""
    c0_list = [0.1, 0.1, 0.01, 0.01, 1.0]
    c1_list = [1.0, 5.0, 2.0, 2.0, 3.0]
    alpha_list = [0.5, 1.0, 1.2, 1.5, 2.0]
    sigma_list = [0.02, 0.02, 0.1, 0.2, 0.002]
    theta_list = [0.01, 0.01, 0.2, 0.2, 0.001]
   
    Nh, NT = 201, 100
    h = 1.0 / (Nh - 1)
   
    print("=== PARTIE 1 : ÉTUDE DE SENSIBILITÉ (MODÈLE MFG) ===")
    for i in range(5):
        p = {
            'beta': 2, 'c0': c0_list[i], 'c1': c1_list[i],
            'alpha': alpha_list[i], 'theta': theta_list[i],
            'max_iter': 800,
            'kappa': (sigma_list[i]**2 / 2) / (h**2),
            'sigma': sigma_list[i]
        }
        print(f"Simulation MFG Exp {i+1} en cours...")
        M, U, x, t = run_simulation("mfg", p, Nh=Nh, NT=NT)
        plot_u_m_contour_complete(M, U, x, t, i+1, p, mode="mfg")


def run_comparison_set_1():
    """Lance la comparaison spécifique pour le Set de paramètres n°1."""
    # Paramètres du Set 1
    p = {
        'beta': 2, 'c0': 0.1, 'c1': 1.0, 'alpha': 0.5,
        'theta': 0.01, 'max_iter': 1000, 'sigma': 0.02
    }
   
    Nh, NT = 201, 100
    h = 1.0 / (Nh - 1)
    p['kappa'] = (p['sigma']**2 / 2) / (h**2)
   
    print("=== PARTIE 2 : COMPARAISON MFG VS MFC (SET 1) ===")
   
    # 1. Calcul MFG (déjà fait plus haut mais on le recalcule pour la clarté)
    print("Calcul du modèle MFG (Nash)...")
    M_mfg, U_mfg, x, t = run_simulation("mfg", p, Nh=Nh, NT=NT)
    plot_u_m_contour_complete(M_mfg, U_mfg, x, t, 1, p, mode="mfg")
   
    # 2. Calcul MFC (Optimum social)
    print("Calcul du modèle MFC (Social)...")
    M_mfc, U_mfc, _, _ = run_simulation("mfc", p, Nh=Nh, NT=NT)
    plot_u_m_contour_complete(M_mfc, U_mfc, x, t, 1, p, mode="mfc")

# Appel de la comparaison
if __name__ == "__main__":
    run_mfg_sensitivity()
    run_comparison_set_1()
