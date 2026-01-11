# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 09:32:47 2025

@author: GEHANT
"""

import numpy as np
from math import pi, tan, atan
import matplotlib.pyplot as plt

largeur = 100.e-6            
w0 = 2.e-6                   
lambd = 633.e-9              
k = 2*pi/lambd               
N = 1024                     
i = complex(0,1)             

n0 = 1.0                     
n1 = 1.01                    
LargeurGuide = 5.e-6         

LargeurGuidePt = int(N * LargeurGuide / largeur)
milieu = int((N-1)/2)

Lr = (pi*w0**2) / lambd      
dz = 0.1e-6   
# mettre le faisceau de depart  
# Distance
ecart_depart = 40.e-6         
z_rencontre = 800.e-6         
distance = 1000.e-6
Nb_cellules = int(distance/dz)

a = -(N-1)/2
b = (N-1)/2
x = np.linspace(a, b, N) * largeur/N
dx = largeur/N

# Angle de convergence
demi_ecart = ecart_depart / 2
theta = atan(demi_ecart / z_rencontre)

print(f"Angle : {np.degrees(theta):.2f}°")
print(f"Nombre de cellules : {Nb_cellules}")

# Source : 2 faisceaux séparés
pos_g = -demi_ecart
pos_d = +demi_ecart

y = np.exp(-(x - pos_g)**2 / w0**2) + np.exp(-(x - pos_d)**2 / w0**2)

nf = np.fft.fftshift(np.fft.fftfreq(N, d=dx))

MFR_dz = np.exp(1j * pi * lambd * dz * nf**2)
MFR_dz_2 = np.exp(1j * pi * lambd * dz/2 * nf**2)

TABLEAU = np.zeros([N, Nb_cellules], dtype=complex)
TABLEAU[:, 0] = y

# Première propagation AVEC fftshift
Y = np.fft.fftshift(np.fft.fft(y))
Sortie1 = Y * MFR_dz_2

# Paramètre du guide
dn = (n1/n0) - 1

for j in range(1, Nb_cellules):
    z_actuel = j * dz
    
    # Création dynamique du guide
    Lens = np.ones(N, dtype=complex)
    
    if z_actuel < z_rencontre:
        # Phase 1 : Les 2 guides convergent
        dist_restante = z_rencontre - z_actuel
        shift_m = dist_restante * tan(theta)
        shift_pt = int(shift_m / dx)
        
        c_haut = milieu + shift_pt
        c_bas = milieu - shift_pt
        
        # Guide du haut
        if 0 <= c_haut - LargeurGuidePt//2 and c_haut + LargeurGuidePt//2 < N:
            Lens[c_haut - LargeurGuidePt//2 : c_haut + LargeurGuidePt//2] = np.exp(-i * k * dz * dn)
        
        # Guide du bas
        if 0 <= c_bas - LargeurGuidePt//2 and c_bas + LargeurGuidePt//2 < N:
            Lens[c_bas - LargeurGuidePt//2 : c_bas + LargeurGuidePt//2] = np.exp(-i * k * dz * dn)
    
    else:
        # Un seul guide au centre
        Lens[milieu - LargeurGuidePt//2 : milieu + LargeurGuidePt//2] = np.exp(-i * k * dz * dn)
    

    # Propagation

    Sortie2 = np.fft.ifft(np.fft.ifftshift(Sortie1)) * Lens
    Sortie1 = np.fft.fftshift(np.fft.fft(Sortie2)) * MFR_dz
 
    TABLEAU[:, j] = Sortie2
    
    if j % (Nb_cellules//10) == 0:
        print(f"{int(j/Nb_cellules*100)}%")

# Dernière étape AVEC ifftshift
Sortie_Finale = np.fft.ifft(np.fft.ifftshift(Sortie1))
print("Terminé.")

# Pour le masque de sortie, on sélectionne les points qui sont dans les 2 guides
# Indices des centres
idx_g = milieu - int(demi_ecart / dx) 
idx_d = milieu + int(demi_ecart / dx)

masque_entree = np.zeros(N, dtype=bool)
# On allume les pixels du guide gauche
masque_entree[idx_g - LargeurGuidePt//2 : idx_g + LargeurGuidePt//2] = True
# On allume les pixels du guide droit
masque_entree[idx_d - LargeurGuidePt//2 : idx_d + LargeurGuidePt//2] = True

# Pour le masque de sortie, on sélectionne les points dans le guide central unique
masque_sortie = np.zeros(N, dtype=bool)
masque_sortie[milieu - LargeurGuidePt//2 : milieu + LargeurGuidePt//2] = True

# On ne somme que les pixels qui sont True dans les masques
Energie_Coeur_Entree = np.sum(np.abs(y[masque_entree])**2) * dx
Energie_Coeur_Sortie = np.sum(np.abs(Sortie_Finale[masque_sortie])**2) * dx

# 4. Calcul du Ratio
# Si ce nombre est < 100%, c'est que de la lumière est sortie du guide
Ratio_Conduit = Energie_Coeur_Sortie / Energie_Coeur_Entree * 100

print(f"Taux de conservation du conduit  : {Ratio_Conduit:.2f}%")

# Affichage

plt.figure(figsize=(12, 6))
plt.imshow(np.abs(TABLEAU), aspect='auto', cmap='jet', 
           extent=[0, distance*1e6, x.min()*1e6, x.max()*1e6], 
           origin='lower')

plt.xlabel('Propagation z (µm)')
plt.ylabel('Position x (µm)')
plt.title('Coupleur Y')
plt.colorbar(label='|E|')
plt.tight_layout()
plt.show()

# Comparaison 1D
plt.figure(figsize=(10, 5))
plt.plot(x*1e6, np.abs(y), 'b-', linewidth=2, label='Entrée (2 faisceaux)')
plt.plot(x*1e6, np.abs(Sortie_Finale), 'r-', linewidth=2, label='Sortie (1 faisceau)')


plt.xlabel('Position x (µm)')
plt.ylabel('Amplitude |E|')
plt.title('Comparaison Entrée / Sortie')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-50, 50])
plt.tight_layout()
plt.show()
