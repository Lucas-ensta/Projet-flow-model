

"""Nous allons modélisier l'écoulement d'un fluide incompressible en 2D dans un tube avec un flux entrant dans le domaine de simulation
et donc un flux sortant
Source : Chaine youtube "Machine learning and simulation" 
 
1) Rappel équations en 2d
    On note : 
    U : vecteur vitesse (2D)
    u : composante de la vitesse selon x (→)
    v : composante de la vitesse selon y (↑) 
    p : la pression 
    ρ : masse volumique 
    ν : viscosité dynamique  
    f : Force volumique externe (ici nulle)

    Equation d'incompressibilité :  ∇U = 0 
    
    Equation de Navier-Stokes : ρ ( ∂U/∂t + (U ⋅ ∇) U ) = -∇p + ν ∇²U + f

2) Grille de simulation : même principe que la grille de Yee -
    superposer la grille du champ des vitesse avec le champ des pression avec un incrément spaciale entre les 2 : 
    
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
     - p : N_x-1 par N_y-1
     - u : N_x par N_y-1
     - v : N_x-1 par N_y

3) Conditions au limites : 
    Problème de la méthode FDVD -> il faut rajouter deux colones aux extremintés droites et gauche du domaine ainsi que deux lignes 
    en haut et en bas du domaine pour pouvoir définir correctement les conditions aux limmites avec la méthode des différences finie
    centrée. 

    Les tailles réelles de nos variables seront donc dans le cas générale : 
        - p : N_x+1 par N_y+1
        - u : N_x par N_y+1
        - v : N_x+1 par N_y




"""

print("push pls")


