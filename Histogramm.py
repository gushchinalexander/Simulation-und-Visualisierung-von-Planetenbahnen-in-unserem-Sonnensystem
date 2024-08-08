import numpy as np
import matplotlib.pyplot as plt

methode1 = []
methode2 = []
th_wert = 123

mittelwert_m1 = np.mean(methode1)
mittelwert_m2 = np.mean(methode2)

std_m1 = np.std(methode1)
std_m2 = np.std(methode2)

# Abweichungen vom theoretischen Wert
abweichung_m1 = [x - th_wert for x in methode1]
abweichung_m2 = [x - th_wert for x in methode2]

# Standardabweichung vom theoretischen Wert
std_th_m1 = np.sqrt(np.mean(np.array(abweichung_m1) ** 2))
std_th_m2 = np.sqrt(np.mean(np.array(abweichung_m2) ** 2))

print(f"Methode 1: Mittelwert = {mittelwert_m1}, Standardabweichung = {std_m1}")
print(f"Methode 2: Mittelwert = {mittelwert_m2}, Standardabweichung = {std_m2}")
print(f"Methode 1: Standardabweichung vom theoretischen Wert = {std_th_m1}")
print(f"Methode 2: Standardabweichung vom theoretischen Wert = {std_th_m2}")

plt.hist(methode1, bins=int(len(methode1) / 10), alpha=0.5, label='Methode 1', color='blue', edgecolor='black')
plt.hist(methode2, bins=int(len(methode2) / 10), alpha=0.5, label='Methode 2', color='green', edgecolor='black')
plt.axvline(th_wert, color='red', linestyle='dashed', linewidth=1, label='Theoretischer Wert')
plt.xlabel('Messwerte')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der Messwerte')
plt.legend()
plt.show()
