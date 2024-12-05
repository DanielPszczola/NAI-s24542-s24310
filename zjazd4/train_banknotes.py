"""
Skrypt do klasyfikacji danych banknotów za pomocą metod Drzewa Decyzyjnego oraz SVM.

Opis:
Program odczytuje dane banknotów, przeprowadza ich podział na zbiory treningowe i testowe,
uczy modele klasyfikacyjne (Drzewo Decyzyjne i SVM), a następnie ocenia ich skuteczność
za pomocą raportów klasyfikacyjnych, macierzy pomyłek oraz porównawczych wykresów.

Funkcjonalność:
1. Wczytanie danych wejściowych z pliku tekstowego.
2. Podział danych na cechy (X) i etykiety (y).
3. Trening modeli klasyfikacyjnych.
4. Ewaluacja wyników na zbiorze testowym.
5. Wizualizacja macierzy pomyłek oraz porównanie skuteczności modeli.
6. Predykcja na podstawie nowych danych.

Autorzy: Michał Kaczmarek s24310, Daniel Pszczoła s24252

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

file_path = 'banknotes_data.txt'
column_names = [
    'Variance of Wavelet Transformed image',
    'Skewness of Wavelet Transformed image',
    'Kurtosis of Wavelet Transformed image',
    'Entropy of image',
    'Class'
]
data = pd.read_csv(file_path, header=None, names=column_names, sep=',')

X = data[['Variance of Wavelet Transformed image',
          'Skewness of Wavelet Transformed image',
          'Kurtosis of Wavelet Transformed image',
          'Entropy of image']]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)


def plot_confusion_matrix(y_true, y_pred, title):
    """
        Rysuje macierz pomyłek dla przewidywań modelu.

        Parametry:
        - y_true: Tablica rzeczywistych etykiet.
        - y_pred: Tablica przewidywanych etykiet.
        - title: Tytuł wykresu.

        Zwraca:
        - Wyświetla wykres macierzy pomyłek.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap=plt.cm.OrRd)
    plt.title(title)
    plt.show()


# Wykres macierzy pomyłek - Drzewo Decyzyjne
print("Drzewo decyzyjne:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))
plot_confusion_matrix(y_test, y_pred_dt, "Macierz pomyłek - Drzewo decyzyjne")

# Wykres macierzy pomyłek - SVM
print("SVM:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))
plot_confusion_matrix(y_test, y_pred_svm, "Macierz pomyłek - SVM")

# Wykres porównujący dokładność metod - słupkowy
models = ['Decision Tree', 'SVM']
accuracies = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_svm)]

plt.bar(models, accuracies, color=['blue', 'orange'])
plt.ylim(0, 1)
plt.title('Porównanie dokładności modeli', pad=20)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', fontsize=12, color='black')

plt.show()

# Wywołanie klasyfikatorów dla nowych danych
new_data_path = 'test_banknotes.txt'
new_data = pd.read_csv(new_data_path, header=None, sep=',')

new_data.columns = [['Variance of Wavelet Transformed image',
                     'Skewness of Wavelet Transformed image',
                     'Kurtosis of Wavelet Transformed image',
                     'Entropy of image']]

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
