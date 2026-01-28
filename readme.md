# Méthodes Numériques : Finance et Jeux à Champ Moyen

Ce projet regroupe deux applications des méthodes de différences finies : le **pricing d'options financières** vie l'EDP de Black Scholes et la simulation de **Jeux à Champ Moyen (Mean Field Games)**.

---

## 🧐 À propos du projet

Ce dépôt est divisé en deux axes majeurs :

### 1. Pricing d'Options Européennes
Résolution numérique de l'équation de Black-Scholes pour évaluer le prix des options (Call et Put). L'accent est mis sur la comparaison de trois schémas de **Différences Finies** :
* **Explicite** : Simple, mais soumis à la condition de stabilité CFL (affiché en 🔵).
* **Implicite** : Inconditionnellement stable (affiché en 🟢).
* **Crank-Nicolson** : Précis à l'ordre 2 et stable (affiché en 🔴).

### 2. Mean Field Games (MFG) & Control (MFC)
Étude et résolution numérique de systèmes de Jeux à Champ Moyen et de Contrôle à Champ Moyen. Cette partie traite du comportement optimal d'un grand nombre d'agents en interaction, modélisé par le couplage de deux équations :
* L'équation de **Hamilton-Jacobi-Bellman (HJB)** (optimisation individuelle).
* L'équation de **Fokker-Planck (FP)** (évolution de la distribution de la population).

---

## 🛠️ Structure du Projet

Le projet est organisé de la manière suivante :

```text
.
├── src/                    # Scripts sources Python (.py)
│   ├── price_bs_pde.py     # Classes et moteurs de calcul pour les options
│   └──  mean_field_game.py # Algorithmes de résolution MFG/MFC
├── notebooks/              # Expérimentations et démonstrations interactives
│   ├── Option_europeenne.ipynb     # mplémentation du pricing d'options (Différences Finies)
│   └── solving_MFG_and_MFC.ipynb   # Résolution numérique des systèmes MFG et MFC
├── figures/                # Graphiques et visualisations générés
│   ├── bs_pde_figures.     # graphiques sur le pricing d'options européennes
│   └── mfg_figures         # graphiques sur les MFG et MFC
├── Option_europeenne.pdf   # Rapport détaillé des résultats sur le pricing d'options
├── Mean_Fields_Games.pdf   # Rapport détaillé des MFG et MFC
├── requirements.txt        # Dépendances
└── README.md               # Documentation du projet