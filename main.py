

"""Nous allons modélisier l'écoulement d'un fluide en 2D dans un tube avec un flux entrant dans le domaine de simulation
et donc un flux sortant
Source : Chaine youtube "Machine learning and simulation" 
 
1) Rappel équations en 2d
    On note : 
    U : vecteur vitesse (2D)
    u : composante de la vitesse selon x (→)
    v : composante de la vitesse selon y (↑) 
    p : la pression 
    ρ : masse volumique 
    ν : viscosité cinématique  
    f : Force volumique externe (ici nulle)

    Equation d'incompressibilité :  ∇U = 0 
    
    Equation de Navier-Stokes : ρ ( ∂U/∂t + (U  ∇) U ) = -∇p + ν ∇²U + f

2) Grille de simulation = superposition "décalée" de plusieurs grilles : même principe que la grille de Yee -
    Ici superposer la grille du champ des vitesse avec le champ des pression avec un incrément spatial entre les 2 : 
    
    + : sommet d'une cellule de la grille 
    ● : sommet de la grille du champ de pression (là où on stock les valeurs)
    → ou ↑ : sommet de la grille du champ des vitesses (idem)
    0 : origine du repère 
   
        +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +   
        |           |           |           |           |           |           |           |
        →     ●     →     ●     →     ●     →     ●     →     ●     →     ●     →     ●     →
        |           |           |           |           |           |           |           |
        +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +        
        |           |           |           |           |           |           |           |
        →     ●     →     ●     →     ●     →     ●     →     ●     →     ●     →     ●     →
        |           |           |           |           |           |           |           |
        +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +
        |           |           |           |           |           |           |           |
        →     ●     →     ●     →     ●     →     ●     →      ●    →     ●     →     ●     →
        |           |           |           |           |           |           |           |
        0  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +  -  ↑  -  +

    Dans cette exemple d'un espace de taille 4x8 (nombre de sommets)
     - p est de taille 3x7 
     - u est de taille 3x8
     - v est de taille 4x7 
    
    Généralisation - pour un espace de simulation de taille N_x par N_y 
     - p : N_x-1 par N_y-1 (Attention à inverser x et y en coordonée matricielle : nx = nb colonnes et ny = nb lignes)
     - u : N_x par N_y-1 
     - v : N_x-1 par N_y

3) Conditions aux limites : 
    Problème de la Grille décallé -> il faut rajouter deux colones aux extremintés droites et gauche du domaine ainsi que deux lignes 
    en haut et en bas du domaine pour pouvoir définir correctement les conditions aux limites avec la méthode des différences finie
    centrée. 

    Les tailles réelles de nos variables seront donc dans le cas générale : 
        - p : N_x+1 par N_y+1
        - u : N_x par N_y+1
        - v : N_x+1 par N_y

4) Implémentation : 

    4.1 - Initialisation : 
        u : initialisé à 1 partout 
        v : initialisé à 0 
        p : initialisé à 0 

    4.2 - Mise à jour du champ u (+ Conditions aux limites)
    
        u = u + dt * (-∂p/∂x + ν ∇²u - ∂u²/∂x - v ∂u/∂y ) 

    4.3 - Mise à jour du champ v (+conditions aux limites)

        Idem 

    4.4 - Calculer la divergence des vitesses provisoires 
      
      d = ∂u/∂x + ∂v/∂y

    4.5 - Calculer la correction de la pression q pour assurer la condition d'incompressibilité (Problème de Poisson) :

      Résoudre : ∇²q = d / dt (avec une condition aux limite de Neumann pour les parois de notre domaine sauf à droite, car flux
      sortant) 

    4.6 - Mettre à jour le champ de pression 

      p = p + q

    4.7 - Calcul des vitesses pour avoir un écoulement incompressible : 

      u = u - dt ⋅ ∂q/∂x 
      v = v - dt ⋅ ∂q/∂y   

    4.8 - Recommencer jusqu'à l'état stable 


    
Info prtaique : pour changer de repertoire de travail : ecrire cd .. dans le terminal, et ls pour voir les fichiers du repertoire
actuelle
pour entrer dans répertoire : cd .backslash nomrepértoire
"""

import numpy as np 
import matplotlib as plt 
import cmasher as cmr
from tqdm import tqdm


N_POINTS_Y = 15
ASPECT_RATIO = 10 # rapport longueur / largeur du domaine 
KINEMATIC_VISCOSITY = 0.01 
TIME_STEP_LENGTH = 0.0001
N_TIME_STEPS = 5000
PLOT_EVERY = 50

N_PRESSURE_POISSON_ITERATIONS = 50  # Utile pour l'étape 4.5

def main():
    cell_lenght = 1.0 / (N_POINTS_Y -1) 

    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x_range = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x) 
    y_range = np.linspace(0.0, 1.0 * ASPECT_RATIO, N_POINTS_Y)

    coordinate_x, coordinate_y = np.meshgrid(x_range, y_range) #création du maillage 
    
    # Conditions initiales :
    velocity_x_prev = np.ones((N_POINTS_Y + 1, n_points_x)) # _prev pour previous pour condition init 
    
    """On veut imposer une condition d'adhérences aux frontière haute et basse du domaine mais la grille des v_x n'est pas défini
      en ces points (du fait de notre grille de simu décalé), on arrange cela en calculant la valeur moyenne entre la valeur à l'int
      et celle à l'ext du domaine 
    """
    # On veut (u_ext + u_int)/2 = 0 soit u_ext = - u_int : 
    velocity_x_prev[0,:] = - velocity_x_prev[1,:]
    velocity_x_prev[-1,:] = - velocity_x_prev[-2,:]

    velocity_y_prev = np.zeros((N_POINTS_Y, n_points_x + 1))

    pressure_prev = np.zeros((N_POINTS_Y + 1, n_points_x +1))

    #Tableaus utiles pour le déroulement de la simu : 

    velocity_x_tent = np.zeros_like(velocity_x_prev) #_tent pour tentative = provisoir en français 
    velocity_x_next = np.zeros_like(velocity_x_prev)

    velocity_y_tent = np.zeros_like(velocity_y_prev)
    velocity_y_next = np.zeros_like(velocity_x_prev)

    
    for iter in tqdm(range(N_TIME_STEPS)) : #here we go :D 
        #Update de la vitesse intéreur u (composante v_x)
        diffusion_x = KINEMATIC_VISCOSITY *(
            
            (
                velocity_x_prev[1:-1, 2: ]            # Approximation du Laplacien en 2D avec le schéma des différences finies
                +
                velocity_x_prev[2:, 1:-1]
                +
                velocity_x_prev[1:-1, :-2]
                +
                velocity_x_prev[ :-2, 1:-1]
                -
                4 * velocity_x_prev[1:-1, 1:-1]
            ) / (
                cell_lenght**2
            )
        )
        convection_x = (
            (
                velocity_x_prev[1:-1, 2: ]**2    # schéma différences finies centré 
                -
                velocity_x_prev[1:-1, :-2]**2
            ) / (
                2 * cell_lenght
            )
            +
            (
                velocity_x_prev[2: , 1:-1]
                -
                velocity_x_prev[ :-2, 1:-1]
            ) / (
                2 * cell_lenght                   # Pour multiplier u par v sur la grille décalé, on fait la moyenne des 4 v autour d'un u  
            ) 
            * 
            (
                velocity_y_prev[ :-1, 1:-2]
                +
                velocity_y_prev[ :-1, 2:-1]
                +
                velocity_y_prev[1: , 1:-2]
                +
                velocity_y_prev[1: , 2:-1]
            ) / 4
        )
        pressure_gradient_x = (
            (
            pressure_prev[1:-1, 2:-1]
            -
            pressure_prev[1:-1, 1:-2]
            ) / (
                cell_lenght
            )
        )

        velocity_x_tent[1:-1, 1:-1] = (
            velocity_x_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_x
                +
                diffusion_x
                -
                convection_x
            )
        )

        #On applique les conditions aux limites 
        velocity_x_tent[1:-1, 0] = 1.0
        velocity_x_tent[1:-1, -1] = velocity_x_tent[1:-1, -2]   #Pour avoir condition Neumann (derivé selon x de v_x nulle en sortie de tube) On ecris simplement qu'il y a égalité entre la valeur (,-1) a la frontiere et sa voisine (,-2)              
        velocity_x_tent[0, :] = - velocity_x_tent[1, :] #condition adhérence en bas 
        velocity_x_tent[-1,:] = - velocity_x_tent[-2, :] #idem en haut
        
        #Update pour la vitesse v (composante v_y)
        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                +
                velocity_y_prev[1:-1, 2: ]
                +
                velocity_y_prev[2: , 1:-1]
                +
                velocity_y_prev[1:-1, :-2]
                +
                velocity_y_prev[ :-2, 1:-1]
                -
                4 * velocity_y_prev[1:-1, 1:-1]
            ) / (
               cell_lenght**2 
            )
        )
        convection_y = (
            (
                velocity_x_prev[2:-1, 1: ]
                +
                velocity_x_prev[2:-1, :-1]
                +
                velocity_x_prev[1:-2, 1: ]
                +
                velocity_x_prev[1:-2, :-1]
            ) / 4
            *
            (
                velocity_y_prev[1:-1, 2: ]
                -
                velocity_y_prev[1:-1, :-2]
            ) / (
                2 * cell_lenght
            )
            +
            (
                velocity_y_prev[2: , 1:-1]**2
                -
                velocity_y_prev[ :-2, 1:-1]**2
            ) / (
                2 * cell_lenght
            )
        )
        pressure_gradient_y = (
            (
                pressure_prev[2:-1, 1:-1]
                -
                pressure_prev[1:-2, 1:-1]
            ) / (
                cell_lenght
            )
        )

        velocity_y_tent[1:-1, 1:-1] = (
            velocity_y_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -pressure_gradient_y
                +
                diffusion_y
                -
                convection_y
            )
        )

        #Conditions aux limites : 
        velocity_y_tent[1:-1, 0] = - velocity_y_tent[1:-1, 1] # V_y nul en entrée de tube 
        velocity_y_tent[1:-1, -1] = velocity_y_tent[1:-1, -2]  # condition neumann en sortie de tube
        velocity_y_tent[0, : ] = 0 
        velocity_y_tent[-1, : ] = 0 
    



     



if __name__== "__main__": 
    main()
    




