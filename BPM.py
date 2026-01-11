# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 16:45:21 2025

@author: GEHANT
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt

largeur = 100.e-6            
w0 = 1.e-6                   
lambd = 633.e-9              
k = 2*pi/lambd               
N = 1001                     
i = complex(0,1)             

n0 = 1.0                     
n1 = 1.01                    
LargeurGuide = 5.e-6         

LargeurGuidePt = int(N * LargeurGuide / largeur)     
milieu = int((N-1)/2)                                

Lr = (pi*w0**2) / lambd      
facteur = 0.001              
dz = facteur*Lr              

distance = 5 * Lr            
Nb_cellules = int(distance/dz)
z = np.linspace(0, distance, Nb_cellules)

a = -(N-1)/2
b = (N-1)/2
x = np.linspace(a, b, N) * largeur/N
dx = largeur / N

y = np.exp((-x**2)/(w0**2))
Energie_Initiale = np.sum(np.abs(y)**2) * dx

porte = np.zeros(N, dtype=int)
porte[milieu - LargeurGuidePt//2 : milieu + LargeurGuidePt//2] = 1
Lens = np.ones(N, dtype=complex)
Lens[milieu - LargeurGuidePt//2 : milieu + LargeurGuidePt//2] = np.exp(-i * k * dz * ((n1/n0) - 1))

nf = np.linspace(a, b, N)
MFR_dz = np.exp((i/(2*k)) * ((2*pi/largeur)**2) * (nf**2) * dz)
MFR_dz_2 = np.exp((i/(2*k)) * ((2*pi/largeur)**2) * (nf**2) * dz/2)

TABLEAU = np.zeros([N, Nb_cellules], dtype=complex)
TABLEAU[:, 0] = y

Energie_z = np.zeros(Nb_cellules)
Energie_z[0] = Energie_Initiale

Y = np.fft.fftshift(np.fft.fft(y))
Sortie1 = Y * MFR_dz_2

for j in range(1, Nb_cellules):
    Sortie2 = np.fft.ifft(np.fft.ifftshift(Sortie1)) * Lens
    Sortie1 = np.fft.fftshift(np.fft.fft(Sortie2)) * MFR_dz
 
    TABLEAU[:, j] = Sortie2

    Energie_z[j] = np.sum(np.abs(Sortie2)**2) * dx
    
    if j % (Nb_cellules//10) == 0:
        print(f"Calcul : {int(j/Nb_cellules*100)}%")

Sortie_Finale = np.fft.ifft(np.fft.ifftshift(Sortie1))

Energie_Finale = np.sum(np.abs(Sortie_Finale)**2) * dx
Conservation = (Energie_Finale / Energie_Initiale) * 100

print(f"Taux de Conservation : {Conservation:.4f} %")

plt.figure(figsize=(10, 6))
plt.imshow(np.abs(TABLEAU), aspect='auto', cmap='jet', 
           extent=[0, distance, x.min(), x.max()], origin='lower')
plt.axhline(LargeurGuide/2, color='w', linestyle='--', alpha=0.5)
plt.axhline(-LargeurGuide/2, color='w', linestyle='--', alpha=0.5)
plt.xlabel('Propagation z (m)')
plt.ylabel('Position x (m)')
plt.title('Propagation du faisceau')
plt.colorbar(label='Amplitude |E|')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x, np.abs(y), label='Entrée')
plt.plot(x, np.abs(Sortie_Finale), label='Sortie')
plt.plot(x, porte * np.max(np.abs(y)), 'k--', alpha=0.3, label='Guide')
plt.xlabel('x (m)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Profils des modes')
plt.show()