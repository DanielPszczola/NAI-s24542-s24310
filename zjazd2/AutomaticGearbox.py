"""
Opis problemu:
System sterowania automatyczną skrzynią biegów oparty na fuzzy logic, wejściami są tu ilość obrotów,
prędkość oraz tryb skrzyni biegów (normal | eco). Użytkownik ma możliwość sterowania pedałem gazu oraz
wyboru trybu skrzyni biegów co wpływa na zmiane przełożeń przy różnych obrotach. Plansza z manipulatorami
oraz informacjami jest wyświetlana w oknie.

Autorzy: Michał Kaczmarek s24310, Daniel Pszczoła s24542

Przygotowanie środowiska, instalacja za pomocą np. pip:
- numpy
- tkinter
- skfuzzy
- networkx
- packaging
- scipy
- scikit-fuzzy
"""

import tkinter as tk
from tkinter import ttk
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np

current_gear = 0
current_rpm = 800
current_speed = 0
gear_mode = "normal"
previous_throttle = 0


def fuzzy_gear_logic(speed, rpm, mode, throttle):
    """
    Funkcja implementująca logikę rozmytą dla doboru biegu.

    Parametry:
    - speed (float): Aktualna prędkość pojazdu (0 - 200 km/h).
    - rpm (float): Aktualna wartość obrotów silnika (800 - 10000 obr/min).
    - mode (str): Wybrany tryb jazdy ("normal" lub "eco"), który wpływa na strategię zmiany biegów.
    - throttle (int): Wartość pedału gazu (0 - 100), która pośrednio wpływa na docelowe obroty.

    Zwraca:
    - int: Bieg na wyjściu (0 - 5), gdzie 0 oznacza bieg neutralny (N).

    Zasada działania:
    Funkcja definiuje zakresy prędkości, obrotów i biegów oraz tworzone są membership functions.
    Na podstawie trybu jazdy tworzone są reguły rozmyte, które określają rekomendowany
    bieg w oparciu o aktualne obroty i prędkość.
    """

    speed_range = ctrl.Antecedent(np.arange(0, 201, 1), 'speed')
    rpm_range = ctrl.Antecedent(np.arange(800, 10001, 1), 'rpm')
    gear_range = ctrl.Consequent(np.arange(0, 6, 1), 'gear')

    speed_range['low'] = fuzz.trapmf(speed_range.universe, [0, 0, 20, 40])
    speed_range['medium_low'] = fuzz.trimf(speed_range.universe, [25, 45, 60])
    speed_range['medium'] = fuzz.trimf(speed_range.universe, [50, 80, 110])
    speed_range['medium_high'] = fuzz.trimf(speed_range.universe, [100, 130, 160])
    speed_range['high'] = fuzz.trapmf(speed_range.universe, [150, 170, 200, 200])

    rpm_range['low'] = fuzz.trapmf(rpm_range.universe, [800, 800, 1000, 2000])
    rpm_range['medium_low'] = fuzz.trimf(rpm_range.universe, [1000, 2500, 3500])
    rpm_range['medium'] = fuzz.trimf(rpm_range.universe, [3000, 4500, 6000])
    rpm_range['medium_high'] = fuzz.trimf(rpm_range.universe, [5000, 6750, 8000])
    rpm_range['high'] = fuzz.trapmf(rpm_range.universe, [6500, 8250, 9000, 10000])

    gear_range['gear1'] = fuzz.trimf(gear_range.universe, [0, 1, 2])
    gear_range['gear2'] = fuzz.trimf(gear_range.universe, [1, 2, 3])
    gear_range['gear3'] = fuzz.trimf(gear_range.universe, [2, 3, 4])
    gear_range['gear4'] = fuzz.trimf(gear_range.universe, [3, 4, 5])
    gear_range['gear5'] = fuzz.trimf(gear_range.universe, [4, 5, 5])

    rules = []
    if mode == "normal":
        rules = [
            ctrl.Rule(speed_range['low'] & rpm_range['low'], gear_range['gear1']),
            ctrl.Rule(speed_range['low'] & rpm_range['medium_low'], gear_range['gear2']),
            ctrl.Rule(speed_range['medium_low'] & rpm_range['medium'], gear_range['gear3']),
            ctrl.Rule(speed_range['medium'] & rpm_range['medium_high'], gear_range['gear4']),
            ctrl.Rule(speed_range['medium_high'] & rpm_range['high'], gear_range['gear5']),
        ]
    elif mode == "eco":
        rules = [
            ctrl.Rule(speed_range['low'] & rpm_range['low'], gear_range['gear1']),
            ctrl.Rule(speed_range['low'] & rpm_range['low'], gear_range['gear2']),
            ctrl.Rule(speed_range['medium_low'] & rpm_range['medium_low'], gear_range['gear3']),
            ctrl.Rule(speed_range['medium'] & rpm_range['medium'], gear_range['gear4']),
            ctrl.Rule(speed_range['medium_high'] & rpm_range['medium'], gear_range['gear5']),
        ]

    gear_ctrl = ctrl.ControlSystem(rules)
    gear_sim = ctrl.ControlSystemSimulation(gear_ctrl)

    speed = min(max(speed, 0), 200)
    rpm = min(max(rpm, 800), 10000)

    gear_sim.input['speed'] = speed
    gear_sim.input['rpm'] = rpm

    try:
        gear_sim.compute()
        return round(gear_sim.output['gear'])
    except KeyError:
        print("Błąd: obroty i prędkość poza zakresem")
        return 0


def update_simulation():
    """
    Funkcja odpowiedzialna za aktualizację bieżącego stanu symulacji (bieg, obroty silnika, prędkość pojazdu).

    Zasada działania:
    Funkcja pobiera bieżące ustawienia pedału gazu oraz trybu jazdy, oblicza docelowe obroty (target_rpm),
    aktualizuje obroty oraz prędkość na podstawie aktualnego biegu i docelowych wartości. Wywołuje funkcję
    fuzzy_gear_logic do obliczenia rekomendowanego biegu w zależności od prędkości, obrotów i trybu jazdy.

    Zmienne:
    - current_gear: Zaktualizowany bieżący bieg na podstawie logiki rozmytej.
    - current_rpm: Zaktualizowane obroty silnika.
    - current_speed: Zaktualizowana prędkość pojazdu.
    - target_rpm: Obliczana wartość docelowych obrotów silnika, bazująca na ustawieniu pedału gazu.
    - target_speed: Docelowa prędkość pojazdu, zależna od bieżącego biegu i obrotów silnika.

    Wywołania cykliczne:
    Funkcja wywoływana jest cyklicznie co 0,1 sekundy.
    """
    global current_gear, current_rpm, current_speed, gear_mode, previous_throttle
    throttle_value = throttle_scale.get()
    mode_value = mode_selector.get()

    target_rpm = min(max(800 + (throttle_value ** 1.68) * 3, 800), 10000)

    if target_rpm > current_rpm:
        current_rpm += min((target_rpm - current_rpm) * 0.3, 300)
    else:
        current_rpm -= min((current_rpm - target_rpm) * 0.2, 300)

    if current_gear > 0:
        target_speed = (current_rpm / 100) * (current_gear * 0.4)
    else:
        target_speed = max(0, current_speed - 1)

    if target_speed > current_speed:
        current_speed += min((target_speed - current_speed) * 0.25, 5)
    else:
        current_speed -= min((current_speed - target_speed) * 0.15, 3)

    current_speed = min(current_speed, 200)

    new_gear = fuzzy_gear_logic(current_speed, current_rpm, mode_value, throttle_value)

    if new_gear > current_gear:
        current_rpm *= 0.95

    current_gear = new_gear
    gear_display = current_gear if current_gear != 0 else "N"

    gear_value.set(f"Bieg: {gear_display}")
    rpm_value.set(f"Obroty: {int(current_rpm)} obr/min")
    speed_value.set(f"Prędkość: {current_speed:.2f} km/h")
    mode_value_display.set(f"Tryb: {mode_value}")

    root.after(100, update_simulation)


root = tk.Tk()
root.title("Symulacja Skrzyni Biegów z Logiką Rozmytą")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

throttle_scale = tk.Scale(mainframe, from_=0, to=100, orient=tk.HORIZONTAL, label="Pedał Gazu")
throttle_scale.grid(column=1, row=1, sticky=(tk.W, tk.E))

mode_selector = tk.StringVar(value="normal")
mode_menu = ttk.Combobox(mainframe, textvariable=mode_selector, values=["normal", "eco"], state="readonly")
mode_menu.grid(column=1, row=2, sticky=(tk.W, tk.E))

gear_value = tk.StringVar()
gear_label = ttk.Label(mainframe, textvariable=gear_value)
gear_label.grid(column=1, row=3, sticky=(tk.W, tk.E))

rpm_value = tk.StringVar()
rpm_label = ttk.Label(mainframe, textvariable=rpm_value)
rpm_label.grid(column=1, row=4, sticky=(tk.W, tk.E))

speed_value = tk.StringVar()
speed_label = ttk.Label(mainframe, textvariable=speed_value)
speed_label.grid(column=1, row=5, sticky=(tk.W, tk.E))

mode_value_display = tk.StringVar()
mode_label = ttk.Label(mainframe, textvariable=mode_value_display)
mode_label.grid(column=1, row=6, sticky=(tk.W, tk.E))

root.after(100, update_simulation)
root.mainloop()
