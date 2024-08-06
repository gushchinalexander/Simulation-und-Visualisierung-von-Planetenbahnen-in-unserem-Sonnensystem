import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
from matplotlib.patches import Polygon
import tkinter as tk
from tkinter import simpledialog

# Konstanten:
tag = 24 * 60 * 60
jahr = 365.25 * tag
stunde = 60 * 60
minute = 60
sekunde = 1
AE = 1.495978707e11
M = 1.9885e30
G = 6.6743e-11
mu = G * M 
dt = 1 * tag
current_frame = 0

root = tk.Tk()
root.withdraw()  

abstand = simpledialog.askfloat("Abstand fiktiver Planet", "Abstand in Astronomischen Einheiten (AE):")
geschwindigkeit = simpledialog.askfloat("Geschwindigkeit fiktiver Planet", "Geschwindigkeit im Aphel in m/s:")

r0_fiktiv_wert = abstand * AE
v0_fiktiv_wert = geschwindigkeit

# Anfangsposition [m] und -Geschwindigkeit der Erde [m].
r0 = np.array([AE, 0.0])
v0 = np.array([0.0, 29.29e3])

# Anfangsposition [m] und -Geschwindigkeit des fiktiven Planeten
r0_fiktiv = np.array([r0_fiktiv_wert, 0.0])
v0_fiktiv = np.array([0.0, v0_fiktiv_wert])

# Zustandsvektor zum Zeitpunkt t=0
u0 = np.concatenate((r0, v0))

# Zustandsvektor für den fiktiven Planeten zum Zeitpunkt t=0
u0_fiktiv = np.concatenate((r0_fiktiv, v0_fiktiv))

def dgl(t, u):
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])

# spezifischen Orbitalenergie
e_tot = (v0_fiktiv_wert**2) / 2 - mu / r0_fiktiv_wert

# Grosse Halbachse
a = - mu / (2 * e_tot)

# spezifischer Drehimpuls
h = r0_fiktiv_wert * v0_fiktiv_wert

# Exzentrizität
e = np.sqrt(1 + (2 * e_tot * h**2) / mu**2)

#Abstand im Perihel und Aphel
r_perihel = a * (1 - e)
r_aphel = a * (1 + e)

# Berechnung der Umlaufzeit (T) nach dem dritten Keplerschen Gesetz
T = (2 * np.pi * np.sqrt(a**3 / mu)) / (60 * 60 * 24 * 365.25)

# Simulationszeit und Zeitschrittweite [s].
if T <= 1:
    t_max = 1 * jahr
else: 
    t_max = T * jahr

if r_aphel < AE:
    r = AE
else:
    r = r_aphel

# Fiktive Planet startet im Aphel oder Perihel
if abs(r_aphel / AE - abstand) < abs(r_perihel / AE - abstand):
    # Startposition ist im Aphel
    r_bp = (r_aphel - a) * 2

else:
    # Startposition im Perihel
    r_bp = (r_perihel - a) * 2


# Lösung der Bewegungsgleichungs
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   dense_output=True)
t_stuetz = result.t
r_stuetz, v_stuetz = np.split(result.y, 2)

# Lösnug der Bewegungsgleichung für den fiktiven Planeten.
result_fiktiv = scipy.integrate.solve_ivp(dgl, [0, t_max], u0_fiktiv, rtol=1e-9,
                                        dense_output=True)
t_stuetz_fiktiv = result_fiktiv.t
r_stuetz_fiktiv, v_stuetz_fiktiv = np.split(result_fiktiv.y, 2)

# Berechnung der Interpolation auf einem feinen Raster.
t_interp = np.arange(0, np.max(t_stuetz), dt)
r_interp, v_interp = np.split(result.sol(t_interp), 2)

# Berechnung der Interpolation des fiktiven Planeten auf einem feinen Raster.
t_interp_fiktiv = np.arange(0, np.max(t_stuetz_fiktiv), dt)
r_interp_fiktiv, v_interp_fiktiv = np.split(result_fiktiv.sol(t_interp_fiktiv), 2)

# Erzeugung einer Figure und einer Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.set_xlim(-(r / AE + 0.2), r / AE + 0.2)
ax.set_ylim(-(r / AE + 0.2), r / AE + 0.2)
ax.grid()

# Erzeugung einer separate Figure für den Text.
fig_kepler_1, ax_kepler_1 = plt.subplots()
ax_kepler_1.axis('off')

fig_kepler_2, ax_kepler_2 = plt.subplots()
ax_kepler_2.axis('off')

fig_kepler_3, ax_kepler_3 = plt.subplots()
ax_kepler_3.axis('off') 

# Plottung der Bahnkurve der Erde.
ax.plot(r_interp[0] / AE, r_interp[1] / AE, '-b', label='Bahnkurve Erde')

# Plottung der Bahnkurve des fiktiven Planeten.
ax.plot(r_interp_fiktiv[0] / AE, r_interp_fiktiv[1] / AE, '-r', label='Bahnkurve fiktiver Planet')

# Erzeugung des Punktplots für die Positionen der Himmelskörper.
plot_sonne, = ax.plot([0], [0], 'o', color='gold', markersize=15, label='Sonne')
plot_planet, = ax.plot([], [], 'o', color='green', label='Erde')
plot_fiktiv_planet, = ax.plot([], [], 'o', color='#A9A9A9', label='Fiktiver Planet')
plot_bp, = ax.plot([r_bp/AE], [0], 'o', color="black", markersize=4, label='2ter Brennpunkt')

ax.legend(loc='upper right', fontsize=8)

# Liste zur Speicherung der Koordinaten der Positionen des Planeten für die Polygone
kp2_punkte = []

poly_rot = mpl.patches.Polygon([(0,0),(0,0),(0,0)], closed=True, color='red', alpha=0.5)
poly_magenta = mpl.patches.Polygon([(0,0),(0,0),(0,0)], closed=True, color='magenta', alpha=0.5)

ax.add_patch(poly_rot)
ax.add_patch(poly_magenta)

# Variablen und Listen
eck_rot = [(0,0)]
eck_magenta = [(0,0)]

fläche_rot = 0
strecke_rot = 0
fläche_magenta = 0
strecke_magenta = 0

t_rot = 0
t_magenta = 0

flächen_rot = []
flächen_magenta = []

flächen_grau = []
flächen_schwarz = []

def kepler2_analyse(event):
    if event.key == 'a':
        differenzen = []
        proz_differenzen = []
        for rot, magenta in zip(flächen_rot, flächen_magenta):
            diff = abs(rot - magenta)
            differenzen.append(diff)

            proz_diff = (diff / rot) * 100
            proz_differenzen.append(proz_diff)
        print(len(flächen_grau))
        print("dif: ", np.mean(differenzen))
        print("dif proz: ", np.mean(proz_differenzen))

fig.canvas.mpl_connect('key_press_event', kepler2_analyse)

def add_point(event):
    global kp2_punkte, fläche_rot, fläche_magenta, strecke_rot, strecke_magenta, t_rot, t_magenta
    if event.key == 'x':
        if len(kp2_punkte) < 3:
            x = r_interp[0, current_frame] / AE
            y = r_interp[1, current_frame] / AE
            kp2_punkte.append((x, y))

        else:
            flächen_rot.append(fläche_rot * AE * AE)
            flächen_magenta.append(fläche_magenta * AE * AE)

            kp2_punkte.clear()
            poly_rot.set_xy([(0,0),(0,0),(0,0)])
            poly_magenta.set_xy([(0,0),(0,0),(0,0)])
            eck_rot.clear()
            eck_rot.append((0,0))
            eck_magenta.clear()
            eck_magenta.append((0,0))
            fläche_rot = 0
            strecke_rot = 0
            fläche_magenta = 0
            strecke_magenta = 0 
            t_rot = 0
            t_magenta = 0
    return poly_rot, poly_magenta

fig.canvas.mpl_connect('key_press_event', add_point)

def fläche_1():
    global kp2_punkte, fläche_rot, strecke_rot, eck_rot,  t_rot     

    # Berechnung der roten Fläche
    if len(kp2_punkte) == 1:
        t_rot += 1
        x1 = r_interp[0, current_frame] / AE
        y1 = r_interp[1, current_frame] / AE
        
        p1 = (x1, y1)
        
        x2 = r_interp[0, current_frame + 1] / AE
        y2 = r_interp[1, current_frame + 1] / AE
        
        p2 = (x2, y2)

        eck_rot.append(p1)
        eck_rot.append(p2)
        poly_rot.set_xy(eck_rot)

        dreiecksfläche = 0.5 * abs(x1 * y2 - x2 * y1)
        
        fläche_rot += dreiecksfläche 

        strecke_abschnitt = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        strecke_rot += strecke_abschnitt
          
def fläche_2():
    global fläche_magenta, strecke_magenta, kp2_punkte, t_rot, t_magenta, eck_magenta

    if len(kp2_punkte) == 3:
        t_magenta += 1 
    
    # Zeichne die magenta Fläche so lang, wie die rote Fläche gezeichnet wurde (Fläche in Abhängigkeit der Zeit)
    if len(kp2_punkte) == 3 and (t_magenta) <= (t_rot):
        x3 = r_interp[0, current_frame] / AE
        y3 = r_interp[1, current_frame] / AE
        
        p3 = (x3, y3)
        
        x4 = r_interp[0, current_frame + 1] / AE
        y4 = r_interp[1, current_frame + 1] / AE
        
        p4 = (x4, y4)
        
        eck_magenta.append(p3)
        eck_magenta.append(p4)
        poly_magenta.set_xy(eck_magenta)

        dreiecksfläche = 0.5 * abs(x3 * y4 - x4 * y3)
        
        fläche_magenta += dreiecksfläche

        strecke_abschnitt = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
        
        strecke_magenta += strecke_abschnitt
        
        if (t_magenta) >= (t_rot):
            x = r_interp[0, current_frame] / AE
            y = r_interp[1, current_frame] / AE
            kp2_punkte.append((x, y))

kp2_punkte_fiktiv = []

poly_grau = mpl.patches.Polygon([(0,0),(0,0),(0,0)], closed=True, color='grey', alpha=0.5)
poly_schwarz = mpl.patches.Polygon([(0,0),(0,0),(0,0)], closed=True, color='black', alpha=0.5)

ax.add_patch(poly_grau)
ax.add_patch(poly_schwarz)

# Variablen und Listen

eck_grau = [(0,0)]
eck_schwarz = [(0,0)]

fläche_grau = 0
strecke_grau = 0
fläche_schwarz = 0
strecke_schwarz = 0

t_grau = 0
t_schwarz = 0

def add_point_fiktiv(event):
    global kp2_punkte_fiktiv, fläche_grau, fläche_schwarz, strecke_grau, strecke_schwarz, t_grau, t_schwarz
    if event.key == 'v':
        if len(kp2_punkte_fiktiv) < 3: 
            x = r_interp_fiktiv[0, current_frame] / AE
            y = r_interp_fiktiv[1, current_frame] / AE
            kp2_punkte_fiktiv.append((x, y))

        else:
            flächen_grau.append(fläche_grau * AE * AE)
            flächen_schwarz.append(fläche_schwarz * AE * AE)

            kp2_punkte_fiktiv.clear()
            poly_grau.set_xy([(0,0),(0,0),(0,0)])
            poly_schwarz.set_xy([(0,0),(0,0),(0,0)])
            eck_grau.clear()
            eck_grau.append((0,0))
            eck_schwarz.clear()
            eck_schwarz.append((0,0))
            fläche_grau = 0
            strecke_grau = 0
            fläche_schwarz = 0
            strecke_schwarz = 0
            t_grau = 0
            t_schwarz = 0
    return poly_grau, poly_schwarz

fig.canvas.mpl_connect('key_press_event', add_point_fiktiv)

def fläche_1_fiktiv():
    global kp2_punkte_fiktiv, fläche_grau, strecke_grau, eck_blau,  t_grau     

    # Berechnung der grauen Fläche
    if len(kp2_punkte_fiktiv) == 1:
        t_grau += 1
        x1 = r_interp_fiktiv[0, current_frame] / AE
        y1 = r_interp_fiktiv[1, current_frame] / AE
        
        p1 = (x1, y1)
        
        x2 = r_interp_fiktiv[0, current_frame + 1] / AE
        y2 = r_interp_fiktiv[1, current_frame + 1] / AE
        
        p2 = (x2, y2)

        eck_grau.append(p1)
        eck_grau.append(p2)
        poly_grau.set_xy(eck_grau)

        dreiecksfläche = 0.5 * abs(x1 * y2 - x2 * y1)
        
        fläche_grau += dreiecksfläche 

        strecke_abschnitt = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        strecke_grau += strecke_abschnitt

def fläche_2_fiktiv():
    global fläche_schwarz, strecke_schwarz, kp2_punkte_fiktiv, t_grau, t_schwarz, eck_grün

    if len(kp2_punkte_fiktiv) == 3:
        t_schwarz += 1 
    
    # Zeichnung der schwarze Fläche so lang, wie die graue Fläche gezeichnet wurde (Fläche in Abhängigkeit der Zeit)
    if len(kp2_punkte_fiktiv) == 3 and (t_schwarz) <= (t_grau):
        x3 = r_interp_fiktiv[0, current_frame] / AE
        y3 = r_interp_fiktiv[1, current_frame] / AE
        
        p3 = (x3, y3)
        
        x4 = r_interp_fiktiv[0, current_frame + 1] / AE
        y4 = r_interp_fiktiv[1, current_frame + 1] / AE
        
        p4 = (x4, y4)
        
        eck_schwarz.append(p3)
        eck_schwarz.append(p4)
        poly_schwarz.set_xy(eck_schwarz)

        dreiecksfläche = 0.5 * abs(x3 * y4 - x4 * y3)
        
        fläche_schwarz += dreiecksfläche

        strecke_abschnitt = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
        
        strecke_schwarz += strecke_abschnitt
        
        if (t_schwarz) >= (t_grau):
            x = r_interp_fiktiv[0, current_frame] / AE
            y = r_interp_fiktiv[1, current_frame] / AE
            kp2_punkte_fiktiv.append((x, y))

r_umlaufszeiten = []
uz_mes = -4 # weil am Anfang 4 Tage lang die Animation lädt
uz_mes1 = 0
umlaufsstrecke = 0
runde_gemacht = False
toleranz = 0.01 * r0_fiktiv_wert / AE

def umlaufszeit():
    global runde_gemacht, toleranz, r_umlaufszeiten, uz_mes, dt, uz_mes1, umlaufsstrecke
    if not runde_gemacht:

        uz_mes += 1

        aktuelle_pos = r_interp_fiktiv[:, current_frame] / AE

        dis_zur_startposition = np.linalg.norm(aktuelle_pos - (r0_fiktiv / AE))
        if uz_mes >= 0:
            r_umlaufszeiten.append(dis_zur_startposition)

        if len(r_umlaufszeiten) > 3:
           r_umlaufszeiten.pop(0)
        
        #Umlaufsstrecke
        x1 = r_interp_fiktiv[0, current_frame] / AE
        y1 = r_interp_fiktiv[1, current_frame] / AE
        x2 = r_interp_fiktiv[0, current_frame + 1] / AE
        y2 = r_interp_fiktiv[1, current_frame + 1] / AE

        zwischenstrecke = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        umlaufsstrecke += zwischenstrecke

        if len(r_umlaufszeiten) == 3:
                if r_umlaufszeiten[1] < r_umlaufszeiten[0] and r_umlaufszeiten[1] < r_umlaufszeiten[2]:
                    uz_mes1 = uz_mes -1
                    runde_gemacht = True
                elif dis_zur_startposition == 0:
                    uz_mes1 = uz_mes 
                    runde_gemacht = True

kp1_linie_sonne, = ax.plot([], [], marker='o', color='black')
kp1_linie_bp, = ax.plot([], [], marker='o', color='black')

# Variablen und Listen
kp1_x = None
kp1_y = None
r_abstand = 0
kp1_key_press = False
kp1_mes = False
r_abstände = []
r_x_punkte = []
r_y_punkte = []
r_abstände_halb = []
r_abstände_total = []
zwischenstrecken = []
brennpunkte = []
zwischenstrecken_index = 0
kp1_strecke_sonne = 0
kp1_strecke_bp = 0
kp1_strecke_ds = 0
kp1_durchschnitt = 0

r_anstände_total_v2 = []
kp1_durchschnitt_v2 = 0

def kepler1_key_v1(event):
    global kp1_strecke_sonne, kp1_x, kp1_y, r_abstand, kp1_key_press, kp1_durchschnitt, zwischenstrecken_index, kp1_strecke_bp, kp1_durchschnitt_v2, kp1_mes, kp1_strecke_ds
    if event.key == 'y':
        if runde_gemacht and not kp1_strecke_ds:
            r_abstände.clear()
            r_abstände_halb.clear()
            zwischenstrecken.clear()
            r_x_punkte.clear()
            r_y_punkte.clear()
            brennpunkte.clear()
            kp1_durchschnitt = 0
            zwischenstrecken_index = 0
            kp1_strecke_bp = 0
            kp1_strecke_sonne = 0
            kp1_strecke_ds = 0

            r_anstände_total_v2.clear()
            kp1_durchschnitt_v2 = 0

            kp1_x = r_interp_fiktiv[0, current_frame] / AE
            kp1_y = r_interp_fiktiv[1, current_frame] / AE

            kp1_linie_sonne.set_data([0, kp1_x], [0, kp1_y])

            r_abstand = np.sqrt(kp1_x**2 + kp1_y**2)
            kp1_key_press = True
            kp1_mes = True
        else:   
            print("Bitte warten, bis die Rechnung abgeschlossen ist.")


fig.canvas.mpl_connect('key_press_event', kepler1_key_v1)

polygon_bp = mpl.patches.Polygon([(0,0),(0,0),(0,0)], closed=True, color='yellow', alpha=0.5)
ax.add_patch(polygon_bp)
r_abstand_halb = 0
r_abstand_mes = 0

def kepler1_v1():
    global kp1_strecke_sonne, kp1_key_press, brennpunkte, r_abstand_halb, r_abstand_mes
    if kp1_key_press:
        if kp1_strecke_sonne < (umlaufsstrecke /2):

            x1 = r_interp_fiktiv[0, current_frame] / AE
            y1 = r_interp_fiktiv[1, current_frame] / AE
            x2 = r_interp_fiktiv[0, current_frame + 1] / AE
            y2 = r_interp_fiktiv[1, current_frame + 1] / AE

            zwischenstrecke = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            kp1_strecke_sonne += zwischenstrecke
        else:
            x_halb = r_interp_fiktiv[0, current_frame] / AE
            y_halb = r_interp_fiktiv[1, current_frame] / AE
                
            kp1_linie_bp.set_data([kp1_x, kp1_x + x_halb], [kp1_y, kp1_y + y_halb])

            r_abstand_halb = np.sqrt(x_halb**2 + y_halb**2)
            r_abstand_mes = r_abstand + r_abstand_halb

            kp1_key_press = False

def kepler1durchschnitt_v1():
    global kp1_strecke_bp, zwischenstrecken_index, kp1_durchschnitt, kp1_mes, r_abstände_total, kp1_strecke_ds, kp1_durchschnitt_v2
    if kp1_mes:
        if kp1_strecke_ds < umlaufsstrecke:
            durchschnitt_kepler1_v2()

            x = r_interp_fiktiv[0, current_frame] / AE
            y = r_interp_fiktiv[1, current_frame] / AE
            abstand = np.sqrt(x**2 + y**2)
            r_abstände.append(abstand)
            r_x_punkte.append(x)
            r_y_punkte.append(y)

            x1 = r_interp_fiktiv[0, current_frame] / AE
            y1 = r_interp_fiktiv[1, current_frame] / AE
            x2 = r_interp_fiktiv[0, current_frame + 1] / AE
            y2 = r_interp_fiktiv[1, current_frame + 1] / AE

            zwischenstrecke = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            kp1_strecke_ds += zwischenstrecke
            zwischenstrecken.append(kp1_strecke_ds)

        if kp1_strecke_ds >= (umlaufsstrecke / 2):
            x1 = r_interp_fiktiv[0, current_frame] / AE
            y1 = r_interp_fiktiv[1, current_frame] / AE
            x2 = r_interp_fiktiv[0, current_frame + 1] / AE
            y2 = r_interp_fiktiv[1, current_frame + 1] / AE
            zwischenstrecke2 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            kp1_strecke_bp += zwischenstrecke2
            if zwischenstrecken_index <= len(zwischenstrecken) - 1:
                while kp1_strecke_bp > zwischenstrecken[zwischenstrecken_index]:  
                    x = r_interp_fiktiv[0, current_frame] / AE
                    y = r_interp_fiktiv[1, current_frame] / AE
                    r_abstand_halb2 = np.sqrt(x**2 + y**2)
                    r_abstände_halb.append(r_abstand_halb2)

                    brennpunkte.append((r_x_punkte[zwischenstrecken_index] + x, r_y_punkte[zwischenstrecken_index] + y))
                    if len(brennpunkte) > 2:
                        polygon_bp.set_xy(brennpunkte)
                    zwischenstrecken_index += 1
                    if zwischenstrecken_index == len(zwischenstrecken):
                        break
            if len(r_abstände) == len(r_abstände_halb):
                r_abstände_total = [r_abstände[i] + r_abstände_halb[i] for i in range(len(r_abstände))]
                kp1_durchschnitt = np.mean(r_abstände_total)

                kp1_durchschnitt_v2 = np.mean(r_anstände_total_v2)

                kp1_mes = False

kp1_linie_sonne_v2, = ax.plot([], [], marker='o', color='blue')
kp1_linie_bp_v2, = ax.plot([], [], marker='o', color='blue')

r_abstand_mes_v2 = 0

def kepler1_v2(event):
    global r_abstand_mes_v2
    if event.key == 'c':
        x = r_interp_fiktiv[0, current_frame] / AE
        y = r_interp_fiktiv[1, current_frame] / AE
        abstand_sonne = np.sqrt(x**2 + y**2)
        x_b = r_bp / AE
        y_b = 0
        abstand_b = np.sqrt((x - x_b)**2 + (y - y_b)**2)
        r_abstand_mes_v2 = abstand_sonne + abstand_b
        kp1_linie_sonne_v2.set_data([0, x], [0, y])
        kp1_linie_bp_v2.set_data([x_b, x], [y_b, y])

fig.canvas.mpl_connect('key_press_event', kepler1_v2)

def durchschnitt_kepler1_v2():
    x = r_interp_fiktiv[0, current_frame] / AE
    y = r_interp_fiktiv[1, current_frame] / AE
    abstand1 = np.sqrt(x**2 + y**2)
    x_2 = r_bp / AE
    y_2 = 0
    abstand2 = np.sqrt((x - x_2)**2 + (y - y_2)**2)
    tot = abstand1 + abstand2
    r_anstände_total_v2.append(tot)

# Textfeld
text_t = ax.text(0.01, 0.95, '', color='black',
                 transform=ax.transAxes)

def update(n):
    global current_frame
    current_frame = n

    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    t = t_interp[n]
    r = r_interp[:, n]
    r_fiktiver_planet = r_interp_fiktiv[:, n]

    plot_planet.set_data(r.reshape(-1, 1) / AE)
    plot_fiktiv_planet.set_data(r_fiktiver_planet.reshape(-1, 1) / AE)

    text_t.set_text(f'$t$ = {t / tag:.0f} d')

    fläche_1()
    fläche_2()
    fläche_1_fiktiv()
    fläche_2_fiktiv()
    umlaufszeit()
    kepler1_v1()
    kepler1durchschnitt_v1()

    return plot_planet, plot_fiktiv_planet, text_t, poly_rot, poly_magenta, poly_grau, poly_schwarz, kp1_linie_sonne, kp1_linie_bp, polygon_bp, kp1_linie_sonne_v2, kp1_linie_bp_v2

# Überschrift
fig_kepler_1.suptitle('1tes Keplersches Gesetz', fontsize=16)

# Textfelder
a_kepler_1_Rechung = ax_kepler_1.text(0.05, 0.85, '', color='black', fontsize=12)
a_kepler_1_Messung_v1 = ax_kepler_1.text(0.05, 0.75, '', color='black', fontsize=12)
a_kepler_1_Messung_v2 = ax_kepler_1.text(0.05, 0.65, '', color='black', fontsize=12)
a_kepler_1_Durchschnitt_v1 = ax_kepler_1.text(0.05, 0.55, '', color='black', fontsize=12)
a_kepler_1_Durchschnitt_v2 = ax_kepler_1.text(0.05, 0.45, '', color='black', fontsize=12)

def update_kepler_1(n):

    a_kepler_1_Rechung.set_text(f"Rechnung: {(r_aphel + r_perihel) / AE} AE")
    a_kepler_1_Messung_v1.set_text(f"Messung: {(r_abstand_mes)} AE")
    a_kepler_1_Messung_v2.set_text(f"Messung v2: {(r_abstand_mes_v2)} AE")
    a_kepler_1_Durchschnitt_v1.set_text(f"Durchschnitt: {(kp1_durchschnitt)} AE")
    a_kepler_1_Durchschnitt_v2.set_text(f"Durchschnitt v2: {(kp1_durchschnitt_v2)} AE")

    return a_kepler_1_Rechung, a_kepler_1_Messung_v1, a_kepler_1_Durchschnitt_v1, a_kepler_1_Durchschnitt_v2, a_kepler_1_Messung_v2

# Überschrift
fig_kepler_2.suptitle('2tes Keplersches Gesetz', fontsize=16)

# Textfelder
a_kepler_2 = ax_kepler_2.text(0.05, 0.85, '', color='black', fontsize=12)
a_kepler_2_strecke_rot = ax_kepler_2.text(0.05, 0.75, '', color='black', fontsize=12)
a_kepler_2_strecke_magenta = ax_kepler_2.text(0.05, 0.65, '', color='black', fontsize=12)
a_kepler_2_zeit = ax_kepler_2.text(0.05, 0.55, '', color='black', fontsize=12)
a_kepler_2_fiktiv = ax_kepler_2.text(0.05, 0.45, '', color='black', fontsize=12)
a_kepler_2_strecke_grau = ax_kepler_2.text(0.05, 0.35, '', color='black', fontsize=12)
a_kepler_2_strecke_schwarz = ax_kepler_2.text(0.05, 0.25, '', color='black', fontsize=12)
a_kepler_2_zeit_fiktiv = ax_kepler_2.text(0.05, 0.15, '', color='black', fontsize=12)

def update_kepler_2(n):

    a_kepler_2.set_text(f"Abschnittsfläche: {fläche_rot * AE * AE:e} $m^2$")
    a_kepler_2_strecke_rot.set_text(f"Strecke der roten Abschnittsfläche: {strecke_rot * AE:e} $m$")
    a_kepler_2_strecke_magenta.set_text(f"Strecke der magenta Abschnittsfläche: {strecke_magenta * AE:e} $m$")
    a_kepler_2_zeit.set_text(f"Benötigte Zeit: {t_rot} $Tage$")
    a_kepler_2_fiktiv.set_text(f"Abschnittsfläche fiktiv: {fläche_grau * AE * AE:e} $m^2$")
    a_kepler_2_strecke_grau.set_text(f"Strecke der grauen Abschnittsfläche: {strecke_grau * AE:e} $m$")
    a_kepler_2_strecke_schwarz.set_text(f"Strecke der schwarzen Abschnittsfläche: {strecke_schwarz * AE:e} $m$")
    a_kepler_2_zeit_fiktiv.set_text(f"Benötigte Zeit fiktiv: {t_grau} $Tage$")
    return a_kepler_2, a_kepler_2_strecke_rot, a_kepler_2_strecke_magenta, a_kepler_2_zeit, a_kepler_2_fiktiv, a_kepler_2_strecke_grau, a_kepler_2_strecke_schwarz, a_kepler_2_zeit_fiktiv

# Überschrift
fig_kepler_3.suptitle('3tes Keplersches Gesetz', fontsize=16)

# Textfelder
a_kepler_3_uz_rechnung = ax_kepler_3.text(0.05, 0.85, '', color='black', fontsize=12)
a_kepler_3_uz_messung = ax_kepler_3.text(0.05, 0.75, '', color='black', fontsize=12)

def update_kepler_3(n):

    a_kepler_3_uz_rechnung.set_text(f"Umlaufszeit Messung: {uz_mes1} $Tage$")
    a_kepler_3_uz_messung.set_text(f"Umlaufszeit Rechnung: {T * 365.25} $Tage$")
    return a_kepler_3_uz_rechnung, a_kepler_3_uz_messung

ani = mpl.animation.FuncAnimation(fig, update, frames=int(t_max / dt),
                                  interval=30, blit=True)
ani_kepler_1 = mpl.animation.FuncAnimation(fig_kepler_1, update_kepler_1, frames=int(t_max / dt), 
                                       interval=30, blit=True)

ani_kepler_2 = mpl.animation.FuncAnimation(fig_kepler_2, update_kepler_2, frames=int(t_max / dt), 
                                       interval=30, blit=True)

ani_kepler_3 = mpl.animation.FuncAnimation(fig_kepler_3, update_kepler_3, frames=int(t_max / dt), 
                                       interval=30, blit=True)

def stop(event):
    if event.key == 'q':
        for i in range(4):
            plt.close()

fig.canvas.mpl_connect('key_press_event', stop)

plt.show()
