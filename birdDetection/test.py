import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Weather data: days and corresponding temperatures for Chicago
days = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
temps = np.array([32, 35, 45, 58, 70, 80, 85, 83, 75, 62, 48, 38, 33])

# Create a spline interpolation
tck = interpolate.make_splrep(days, temps, s=0)

# New x-axis for smooth curve
days_fine = np.linspace(0, 360, 1000)

# Get interpolated temperatures
temps_smooth = interpolate.splev(days_fine, tck, der=0)

# Get first derivative
temps_derivative = interpolate.splev(days_fine, tck, der=1)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(days, temps, 'o', label='Data points')
plt.plot(days_fine, temps_smooth, label='Interpolated')
plt.title('Chicago Temperature Throughout the Year')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(days_fine, temps_derivative)
plt.title('Rate of Temperature Change')
plt.xlabel('Day of Year')
plt.ylabel('°F per day')
plt.tight_layout()
plt.show()