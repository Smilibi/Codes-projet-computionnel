# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 17:04:37 2025

@author: GEHANT
"""

import numpy as np
import time
from math import pi, tan, atan, sin, cos
import matplotlib.pyplot as plt


# Paramètres

ecart_depart = 40.e-6       # Distance entre les cœurs des guides à l'entrée (en mètres)
z_rencontre = 800.e-6        # Distance de propagation où les deux guides se rejoignent (jonction)
demi_ecart = ecart_depart / 2
theta = atan(demi_ecart / z_rencontre) # Angle d'inclinaison des guides par rapport à l'axe Z

# Largeur totale de la fenêtre de simulation (axe X). On ajoute une marge de 100 µm (50 de chaque côté) pour éviter les réflexions sur les bords (repliement FFT).
largeur = max(100.e-6, ecart_depart + 100.e-6) 

dx_souhaite = 0.1e-6         # Résolution spatiale cible selon X
N = int(largeur / dx_souhaite) # Nombre total de points de la grille
if N % 2 != 0: N += 1        # N doit être pair pour optimiser la FFT

print(f"Ecart initial : {ecart_depart*1e6:.1f} µm")
print(f"Angle de convergence : {np.degrees(theta):.2f}°")

w0 = 2.e-6                   # Le waist en mètres
lambd = 633.e-9              # Longueur d'onde du laser
k = 2*pi/lambd               # Nombre d'onde
i = complex(0,1)             # Nombre imaginaire pur

n0 = 1.0                     # Indice de la gaine
n1 = 1.01                    # Indice du cœur du guide
dn = n1 - n0                 # Différence d'indice (contraste)
LargeurGuide = 5.e-6         # Largeur physique du guide d'onde (canal)

x = np.linspace(-largeur/2, largeur/2, N) # Tableau des coordonnées X
dx = x[1] - x[0]                          # Pas spatial réel (doit être proche de dx_souhaite)
milieu = N // 2                           # Indice du pixel central

dz = 0.1e-6                  # Pas de propagation longitudinal
distance = 1000.e-6          # Longueur totale de la simulation (1 mm)
Nb_cellules = int(distance/dz) # Nombre d'itérations de la boucle principale

# Création de la source

pos_g = -demi_ecart          # Position X du guide gauche (bas)
pos_d = +demi_ecart          # Position X du guide droit (haut)

# Profils gaussiens spatiaux
gauss_g = np.exp(-(x - pos_g)**2 / w0**2)
gauss_d = np.exp(-(x - pos_d)**2 / w0**2)

# Phase tilt
# On ajoute une pente de phase pour incliner le front d'onde.
# k_x = k * sin(theta) est la composante transverse du vecteur d'onde.
k_x_tilt = k * sin(theta)

phase_tilt_bas = np.exp(i * k_x_tilt * x)   # Faisceau montant (+theta)
phase_tilt_haut = np.exp(-i * k_x_tilt * x) # Faisceau descendant (-theta)

# Champ électrique initial total
y = gauss_g * phase_tilt_bas + gauss_d * phase_tilt_haut

# Préparation des opérateurs

nf = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) # Fréquences spatiales (kx)

# Opérateur de Diffraction - Propagateur de Fresnel

MFR_dz = np.exp(1j * pi * lambd * dz * nf**2)
MFR_dz_2 = np.exp(1j * pi * lambd * dz/2 * nf**2) # Demi-pas pour le Split-Step symétrique

TABLEAU = np.zeros([N, Nb_cellules], dtype=complex) # Stockage du champ pour affichage 2D
TABLEAU[:, 0] = y

# Boucle de propagation

Y_k = np.fft.fftshift(np.fft.fft(y))
Sortie_k = Y_k * MFR_dz_2

for j in range(1, Nb_cellules):
    z_actuel = j * dz
    
    if z_actuel < z_rencontre:
        # Phase 1 : le guide convergent
        dist_restante = z_rencontre - z_actuel
        exact_shift = dist_restante * tan(theta)
        c_haut = +exact_shift
        c_bas = -exact_shift
        
        # Correction géométrique : Le chemin optique est plus long en diagonale
        dz_local = dz / cos(theta)
        phase_guide_incline = k * dn * dz_local
        
        # Lissage des bordspour éviter l'effet d'escaliers
        dist_haut = np.abs(x - c_haut)
        taux_haut = np.clip((LargeurGuide/2 - dist_haut)/dx + 0.5, 0.0, 1.0)
        
        dist_bas = np.abs(x - c_bas)
        taux_bas = np.clip((LargeurGuide/2 - dist_bas)/dx + 0.5, 0.0, 1.0)
        
        taux_total = np.maximum(taux_haut, taux_bas)
        
        # Opérateur de lentille
        Lens = np.exp(-i * phase_guide_incline * taux_total)
        
    else:
        # Phase 2 : Guide droit central
        dist_centre = np.abs(x)
        taux_centre = np.clip((LargeurGuide/2 - dist_centre)/dx + 0.5, 0.0, 1.0)
        
        phase_guide_droit = k * dn * dz
        Lens = np.exp(-i * phase_guide_droit * taux_centre)
    
    # Algorithme Split-Step :
    Sortie_x = np.fft.ifft(np.fft.ifftshift(Sortie_k)) * Lens

    Sortie_k = np.fft.fftshift(np.fft.fft(Sortie_x)) * MFR_dz
    
    TABLEAU[:, j] = Sortie_x
    
    if j % (Nb_cellules//10) == 0:
        print(f"Progression : {int(j/Nb_cellules*100)}%")

# Dernière transformée inverse pour avoir le champ final
Sortie_Finale = np.fft.ifft(np.fft.ifftshift(Sortie_k))
print("Simulation terminée.")


# Analyse et graphique

# Définition des zones pour compter l'énergie
mask_entree = ((x > (pos_g - LargeurGuide/2)) & (x < (pos_g + LargeurGuide/2))) | \
              ((x > (pos_d - LargeurGuide/2)) & (x < (pos_d + LargeurGuide/2)))

mask_sortie = (x > (-LargeurGuide/2)) & (x < (LargeurGuide/2))

# Calcul des énergies
Energie_Total_Entree = np.sum(np.abs(y)**2) * dx
Energie_Coeur_Entree = np.sum(np.abs(y[mask_entree])**2) * dx
Energie_Coeur_Sortie = np.sum(np.abs(Sortie_Finale[mask_sortie])**2) * dx

Ratio_Conduit = Energie_Coeur_Sortie / Energie_Coeur_Entree * 100

print(f"Taux de transmission cœur à cœur : {Ratio_Conduit:.2f}%")

plt.figure(figsize=(12, 8))

# Visualisation 2D de la propagation
plt.subplot(2,1,1)
extent = [0, distance*1e6, x.min()*1e6, x.max()*1e6]
plt.imshow(np.abs(TABLEAU), aspect='auto', cmap='jet', extent=extent, origin='lower')
plt.title(f'Coupleur Y (Écart de {ecart_depart*1e6:.0f} en µm)')
plt.xlabel('Propagation z (en µm)')
plt.ylabel('Position x (en µm)')
plt.colorbar(label='Amplitude |E|')

# Entrée vs sortie au niveau de l'amplitude
plt.subplot(2,1,2)
plt.plot(x*1e6, np.abs(y), 'b-', label='Entrée (Inclinée)')
plt.plot(x*1e6, np.abs(Sortie_Finale), 'r-', label='Sortie')
plt.title('Comparaison des profils transverses')
plt.xlabel('Position x (µm)')
plt.ylabel('Amplitude')

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Étude sur l'efficacité en fonction de l'angle
print("Lancement de l'étude paramétrique")

def lancer_simulation_param(ecart_param):
#Fonction identique que celle d'avant. Elle recalcule 'largeur', 'N', 'theta' pour chaque nouvel écart.
    
    demi_ecart = ecart_param / 2
    theta = atan(demi_ecart / z_rencontre)

    largeur = max(100.e-6, ecart_param + 100.e-6)

    N = int(largeur / dx_souhaite)
    if N % 2 != 0: N += 1

    x = np.linspace(-largeur/2, largeur/2, N)
    dx = x[1] - x[0]

    pos_g = -demi_ecart
    pos_d = +demi_ecart
    
    gauss_g = np.exp(-(x - pos_g)**2 / w0**2)
    gauss_d = np.exp(-(x - pos_d)**2 / w0**2)
    
    k_x_tilt = k * sin(theta)
    phase_tilt_bas = np.exp(i * k_x_tilt * x)
    phase_tilt_haut = np.exp(-i * k_x_tilt * x)
    
    y = gauss_g * phase_tilt_bas + gauss_d * phase_tilt_haut

    nf = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    MFR_dz = np.exp(1j * pi * lambd * dz * nf**2)
    MFR_dz_2 = np.exp(1j * pi * lambd * dz/2 * nf**2)

    Y_k = np.fft.fftshift(np.fft.fft(y))
    Sortie_k = Y_k * MFR_dz_2
    
    for j in range(1, Nb_cellules):
        z_actuel = j * dz
        
        if z_actuel < z_rencontre:
            dist_restante = z_rencontre - z_actuel
            exact_shift = dist_restante * tan(theta)
            c_haut = +exact_shift
            c_bas = -exact_shift

            dz_local = dz / cos(theta)
            phase_guide_incline = k * dn * dz_local
            
            dist_haut = np.abs(x - c_haut)
            taux_haut = np.clip((LargeurGuide/2 - dist_haut)/dx + 0.5, 0.0, 1.0)
            
            dist_bas = np.abs(x - c_bas)
            taux_bas = np.clip((LargeurGuide/2 - dist_bas)/dx + 0.5, 0.0, 1.0)
            
            taux_total = np.maximum(taux_haut, taux_bas)
            Lens = np.exp(-i * phase_guide_incline * taux_total)
            
        else:

            dist_centre = np.abs(x)
            taux_centre = np.clip((LargeurGuide/2 - dist_centre)/dx + 0.5, 0.0, 1.0)
            
            phase_guide_droit = k * dn * dz
            Lens = np.exp(-i * phase_guide_droit * taux_centre)
        Sortie_x = np.fft.ifft(np.fft.ifftshift(Sortie_k)) * Lens
        Sortie_k = np.fft.fftshift(np.fft.fft(Sortie_x)) * MFR_dz

    Sortie_Finale = np.fft.ifft(np.fft.ifftshift(Sortie_k))

    mask_entree = ((x > (pos_g - LargeurGuide/2)) & (x < (pos_g + LargeurGuide/2))) | \
                  ((x > (pos_d - LargeurGuide/2)) & (x < (pos_d + LargeurGuide/2)))
    mask_sortie = (x > (-LargeurGuide/2)) & (x < (LargeurGuide/2))
    
    Energie_Coeur_Entree = np.sum(np.abs(y[mask_entree])**2) * dx
    Energie_Coeur_Sortie = np.sum(np.abs(Sortie_Finale[mask_sortie])**2) * dx
    
    return (Energie_Coeur_Sortie / Energie_Coeur_Entree * 100), np.degrees(theta)

# Boucle sur 10 valeurs
liste_ecarts = np.linspace(10.e-6, 100.e-6, 10)
resultats_eff = []
resultats_angles = []

temps_debut = time.time()

for idx, ecart_val in enumerate(liste_ecarts):
    eff_val, ang_val = lancer_simulation_param(ecart_val)
    resultats_eff.append(eff_val)
    resultats_angles.append(ang_val)
    print(f"Point {idx+1:02d}/10 : Écart = {ecart_val*1e6:5.1f}µm | Angle = {ang_val:4.2f}° | Eff = {eff_val:5.2f}%")

# Affichage de la courbe
plt.figure(figsize=(10, 6))
plt.plot(resultats_angles, resultats_eff, 'o-', linewidth=2, color='darkblue', label='Simulation BPM')
plt.title("Efficacité en fonction de l'angle de convergence", fontsize=14)
plt.xlabel("Angle de convergence (en degrés)", fontsize=12)
plt.ylabel("Efficacité de transmission (en %)", fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.ylim(0, 105)

# Zones indicatives (Vert = Bien, Rouge = Pertes)
plt.axvspan(0, 1.0, color='green', alpha=0.1, label='Zone Adiabatique')
plt.axvspan(2.5, max(resultats_angles), color='red', alpha=0.1, label='Fortes Pertes Radiatives')

plt.legend()
plt.tight_layout()
plt.show()