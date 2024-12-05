"""
Skrypt do klasyfikacji samochodów na podstawie efektywności paliwowej (MPG) za pomocą metod Drzewa Decyzyjnego i SVM.

Opis:
Program klasyfikuje samochody na trzy klasy efektywności paliwowej (MPG): niskie, średnie i wysokie.
Wykorzystuje dane techniczne pojazdów takie jak liczba cylindrów, pojemność silnika, moc, waga,
przyspieszenie, rok produkcji oraz kraj pochodzenia.

Funkcjonalność:
1. Wczytanie danych z biblioteki Seaborn (zbiór Auto MPG).
2. Wstępne przetwarzanie danych: usunięcie braków, zmiana kolumn kategorii na zmienne wskaźnikowe.
3. Klasyfikacja MPG na trzy klasy przy użyciu podziału przedziałowego.
4. Trening modeli klasyfikacyjnych: Drzewo Decyzyjne i SVM.
5. Ocena skuteczności modeli za pomocą raportów klasyfikacyjnych i macierzy pomyłek.
6. Wizualizacja wyników: macierze pomyłek, porównanie dokładności, analiza rzeczywistych i przewidywanych wartości.
7. Predykcja klasy efektywności paliwowej dla nowych danych.

Autorzy: Michał Kaczmarek s24310, Daniel Pszczoła s24252

Link do zbioru danych: https://archive.ics.uci.edu/dataset/9/auto+mpg

Wymagania środowiska:
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Referencje:
https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
https://scikit-learn.org/stable/modules/svm.html#classification
https://scikit-learn.org/stable/modules/tree.html
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

data = sns.load_dataset('mpg')

print("Podgląd danych:")
print(data.head())

data = data.dropna()

data = data.drop('name', axis=1)

data = pd.get_dummies(data, columns=['origin'], prefix='origin', drop_first=True)

bins = [0, 20, 30, float('inf')]
labels = [0, 1, 2]  # 0: niskie MPG, 1: średnie MPG, 2: wysokie MPG
data['mpg_class'] = pd.cut(data['mpg'], bins=bins, labels=labels)

X = data[
    ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin_japan', 'origin_usa']]
y = data['mpg_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_svc = svm_model.predict(X_test)

print("Raport klasyfikacji - Drzewo Decyzyjne:")
print(classification_report(y_test, y_pred_dt))

print("Raport klasyfikacji - SVC:")
print(classification_report(y_test, y_pred_svc))

accuracy_dt = dt_model.score(X_test, y_test)
accuracy_svc = svm_model.score(X_test, y_test)

# Wykres porównujący dokładność metod - słupkowy
plt.figure(figsize=(6, 4))
plt.bar(['SVC', 'Drzewo Decyzyjne'], [accuracy_svc, accuracy_dt], color=['lightblue', 'orange'])
plt.ylim(0, 1)
plt.title("Porównanie dokładności modeli")
plt.ylabel("Dokładność")

for i, acc in enumerate([accuracy_svc, accuracy_dt]):
    plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', fontsize=12, color='black')

plt.show()

# Wykres macierzy pomyłek - Drzewo Decyzyjne
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['Niskie MPG', 'Średnie MPG', 'Wysokie MPG'])
disp_dt.plot(cmap=plt.cm.OrRd)
plt.title("Macierz pomyłek - Drzewo Decyzyjne")
plt.show()

# Wykres macierzy pomyłek - SVM
cm_svc = confusion_matrix(y_test, y_pred_svc)
disp_svc = ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=['Niskie MPG', 'Średnie MPG', 'Wysokie MPG'])
disp_svc.plot(cmap=plt.cm.OrRd)
plt.title("Macierz pomyłek - SVC")
plt.show()

# Wykres porównujący rzeczywiste i przewidywane wartosci
categories = ['Niskie MPG', 'Średnie MPG', 'Wysokie MPG']
real_values = cm_svc.sum(axis=1)
predicted_values = cm_svc.sum(axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width / 2, real_values, width, label='Rzeczywiste', color='lightblue')
rects2 = ax.bar(x + width / 2, predicted_values, width, label='Przewidywane', color='orange')

ax.set_xlabel('Klasy MPG')
ax.set_ylabel('Liczba przykładów')
ax.set_title('Porównanie rzeczywistych i przewidywanych klas')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()


def add_labels(rects):
    """
       Dodaje etykiety do słupków na wykresie.

       Parametry:
       - rects: Słupki wykresu.

       Zwraca:
       - Etykiety liczbowych wartości na słupkach.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


add_labels(rects1)
add_labels(rects2)

plt.show()

# Wywołanie klasyfikatorów dla nowych danych
new_data_path = 'test_mpg.txt'
new_data = pd.read_csv(new_data_path, header=None, sep=',')

new_data.columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin_japan',
                    'origin_usa']

print("\nNowe dane:")
print(new_data)

new_predictions_dt = dt_model.predict(new_data)
new_predictions_svm = svm_model.predict(new_data)

print("\nPrzewidywania drzewa decyzyjnego:")
for i, pred in enumerate(new_predictions_dt):
    print(f"Przykład {i + 1}: Klasa {pred}")

print("\nPrzewidywania SVM:")
for i, pred in enumerate(new_predictions_svm):
    print(f"Przykład {i + 1}: Klasa {pred}")
