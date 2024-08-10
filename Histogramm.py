import numpy as np
import matplotlib.pyplot as plt

methode1 = []
methode2 = []
th_wert = 123

mittelwert_1 = np.mean(methode1)
mittelwert_2 = np.mean(methode2)

sta_1 = np.std(methode1)
sta_2 = np.std(methode2)

std_th_1 = np.sqrt(np.mean(np.array([x - th_wert for x in methode1]) ** 2))
std_th_2 = np.sqrt(np.mean(np.array([x - th_wert for x in methode2]) ** 2))

print(f"Methode 1: Mittelwert = {mittelwert_1}, Standardabweichung = {sta_1}")
print(f"Methode 2: Mittelwert = {mittelwert_2}, Standardabweichung = {sta_2}")
print(f"Methode 1: Standardabweichung vom theoretischen Wert = {std_th_1}")
print(f"Methode 2: Standardabweichung vom theoretischen Wert = {std_th_2}")

plt.hist(methode1, bins=int(len(methode1) / 10), alpha=0.5, label='Methode 1', color='blue', edgecolor='black')
plt.hist(methode2, bins=int(len(methode2) / 10), alpha=0.5, label='Methode 2', color='green', edgecolor='black')
plt.axvline(th_wert, color='red', linestyle='dashed', linewidth=1, label='Theoretischer Wert')
plt.xlabel('Messwerte')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der Messwerte')
plt.legend(loc='upper right')
plt.show()
