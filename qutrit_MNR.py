import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34  # J*s
kB = 1.380649e-23     # J/K
omega_02 = 2 * np.pi * 10e9   # 10 GHz (|0> <-> |2> transition)
T1_21 = 30e-6                 # (s) T1 for |2> -> |1>, e.g. 30 microseconds
Gamma_21 = 1.0 / T1_21
pi = np.pi

# Range of gate durations and temperatures
gate_times = np.linspace(20e-9, 100e-9, 200)    # (s) 20 ns to 100 ns
Temps = np.linspace(0.01, 0.1, 200)             # (K) 10 mK to 100 mK
TAU, TEMP = np.meshgrid(gate_times, Temps)

# Thermal occupation for transition frequency (typically |2> <-> |1> ~ 10 GHz)
n_th = 1.0 / (np.exp(hbar * omega_02 / (kB * TEMP)) - 1)

# For a population inversion (e.g., resonant pulse) average p2=0.5
gate_population_p2 = 0.5

# Infidelity: dominant relaxation + thermal effects
infidelity = gate_population_p2 * Gamma_21 * TAU * (1 + n_th)

# Power: minimal quantum-limited for qutrit population inversion
power = (hbar * omega_02 * pi**2) / (4 * Gamma_21 * TAU**2)

# Plot
plt.figure(figsize=(14, 6))

# Power heatmap
plt.subplot(1, 2, 1)
p1 = plt.pcolormesh(TAU * 1e9, TEMP * 1e3, np.log10(power), shading='auto', cmap='YlGnBu')
plt.colorbar(p1, label='log10 Power (W)')
plt.xlabel('Gate Duration (ns)')
plt.ylabel('Temperature (mK)')
plt.title('Qutrit: Power vs. Gate Duration and Temp')
power_levels = np.log10(np.array([1e-13, 1e-12, 1e-11]))
CS1 = plt.contour(TAU*1e9, TEMP*1e3, np.log10(power), levels=power_levels, colors='k', linewidths=0.8)
plt.clabel(CS1, fmt=lambda lv: f'$10^{{{int(lv)}}}$', fontsize=9)

# Infidelity heatmap
plt.subplot(1, 2, 2)
p2 = plt.pcolormesh(TAU * 1e9, TEMP * 1e3, np.log10(infidelity), shading='auto', cmap='OrRd')
plt.colorbar(p2, label='log10 Infidelity')
plt.xlabel('Gate Duration (ns)')
plt.ylabel('Temperature (mK)')
plt.title('Qutrit: Infidelity vs. Gate Duration and Temp')
inf_levels = np.log10(np.array([1e-5, 1e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2]))  # gives [ -5, -4, -3, -2 ]
CS2 = plt.contour(TAU*1e9, TEMP*1e3, np.log10(infidelity), levels=inf_levels, colors='k', linewidths=0.8)
plt.clabel(CS2, fmt=lambda lv: f'$10^{{{int(lv)}}}$', fontsize=9)

plt.tight_layout()
plt.show()
