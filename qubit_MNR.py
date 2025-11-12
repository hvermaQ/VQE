import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34         # (J*s)
kB = 1.380649e-23            # (J/K)
omega = 2 * np.pi * 5e9      # (rad/s), 5 GHz
T1 = 40e-6                   # (s) 40 microseconds (needed for comparison, but not plotted)
gamma = 1.0 / T1
pi = np.pi

# Gate times and temperatures
gate_times = np.linspace(10e-9, 100e-9, 200)  # Gate durations from 10 ns to 100 ns
Temps = np.linspace(0.01, 0.1, 200)           # Temperatures from 10 mK to 100 mK
TAU, TEMP = np.meshgrid(gate_times, Temps)

# Thermal photon occupation
n_th = 1.0 / (np.exp(hbar * omega / (kB * TEMP)) - 1)

# Infidelity (no explicit T1 axis)
infidelity = gamma * TAU * (1 + n_th)
# Power delivered to the qubit
power = (hbar * omega * pi ** 2) / (4 * gamma * TAU ** 2)

# Plot
plt.figure(figsize=(14, 6))

# Power heatmap
plt.subplot(1, 2, 1)
p1 = plt.pcolormesh(TAU * 1e9, TEMP * 1e3, np.log10(power), shading='auto', cmap='YlGnBu')
plt.colorbar(p1, label='log10 Power (W)')
plt.xlabel('Gate Duration (ns)')
plt.ylabel('Temperature (mK)')
plt.title('Power vs. Gate Duration and Temperature')
# Isopower curves
power_levels = np.log10(np.array([1e-13, 1e-12, 1e-11]))  # fW, pW, 10pW
CS1 = plt.contour(TAU*1e9, TEMP*1e3, np.log10(power), levels=power_levels, colors='k', linewidths=0.8)
plt.clabel(CS1, fmt=lambda lv: f'$10^{{{int(lv)}}}$', fontsize=9)

# Infidelity heatmap
plt.subplot(1, 2, 2)
p2 = plt.pcolormesh(TAU * 1e9, TEMP * 1e3, np.log10(infidelity), shading='auto', cmap='OrRd')
plt.colorbar(p2, label='log10 Infidelity')
plt.xlabel('Gate Duration (ns)')
plt.ylabel('Temperature (mK)')
plt.title('Infidelity vs. Gate Duration and Temperature')
# Isofidelity curves
inf_levels = np.log10(np.array([1e-5, 1e-4, 5e-4, 7e-4, 1e-3, 2e-3, 5e-3, 7e-3, 1e-2]))  # gives [ -5, -4, -3, -2 ]
CS2 = plt.contour(TAU*1e9, TEMP*1e3, np.log10(infidelity), levels=inf_levels, colors='k', linewidths=0.8)
plt.clabel(CS2, fmt=lambda lv: f'$10^{{{int(lv)}}}$', fontsize=9)

plt.tight_layout()
plt.show()
