import json
import numpy as np

"""
Problem:
    Aplikacja ma na celu rekomendowanie filmów na podstawie ocen innych użytkowników oraz identyfikowanie filmów, 
    które mogą być nieodpowiednie (antyrekomendacje). Wykorzystuje algorytm podobieństwa euklidesowego 
    do porównywania użytkowników na podstawie ocen filmów.

Autor:
    Imię i nazwisko: [Daniel Pszczoła, Michał Kaczmarek]

Instrukcja użycia:
    1. Upewnij się, że plik JSON jest poprawny i znajduje się w tym samym katalogu co skrypt.
    2. Uruchom skrypt i wprowadź nazwę użytkownika, dla którego chcesz wygenerować rekomendacje i antyrekomendacje.
    3. Otrzymasz listę rekomendowanych filmów oraz filmów do unikania wraz z ocenami.

Opcjonalnie:
    - Możesz zmodyfikować liczbę rekomendacji/antyrekomendacji (domyślnie 5) w funkcjach `get_recommendations` i 
      `get_anti_recommendations`, zmieniając parametr `top_n`.
    - Dostosuj dane wejściowe lub rozszerz funkcjonalność, jeśli wymagane.

Referencje:
    - Algorytm podobieństwa euklidesowego: https://en.wikipedia.org/wiki/Euclidean_distance
    - Dokumentacja JSON w Pythonie: https://docs.python.org/3/library/json.html
    - Numpy do obliczeń matematycznych: https://numpy.org/doc/
"""


def is_valid_rating(rating):
    """
    Sprawdza, czy podana ocena jest liczbą.

    Args:
        rating (int | float | None): Ocena filmu.

    Returns:
        bool: True, jeśli ocena jest liczbą (int lub float) i nie jest None, w przeciwnym razie False.
    """
    return isinstance(rating, (int, float)) and rating is not None

def euclidean_score(dataset, user1, user2):
    """
    Oblicza podobieństwo euklidesowe między dwoma użytkownikami na podstawie ich ocen.

    Args:
        dataset (dict): Słownik danych zawierający użytkowników i ich oceny filmów.
        user1 (str): Nazwa pierwszego użytkownika.
        user2 (str): Nazwa drugiego użytkownika.

    Returns:
        float: Wynik podobieństwa euklidesowego (0, jeśli brak wspólnych filmów).

    Raises:
        ValueError: Jeśli którykolwiek z użytkowników nie istnieje w dataset.
    """
    if user1 not in dataset:
        raise ValueError(f"User {user1} not found in the dataset")
    if user2 not in dataset:
        raise ValueError(f"User {user2} not found in the dataset")

    common_movies = set(dataset[user1].keys()).intersection(dataset[user2].keys())
    if not common_movies:
        return 0  # Brak wspólnych filmów

    squared_diff = 0
    for movie in common_movies:
        rating1 = dataset[user1].get(movie)
        rating2 = dataset[user2].get(movie)
        if is_valid_rating(rating1) and is_valid_rating(rating2):
            squared_diff += np.square(rating1 - rating2)

    if squared_diff == 0:
        return 0  # Użytkownicy mają identyczne oceny

    return 1 / (1 + np.sqrt(squared_diff))

def get_recommendations(dataset, target_user, top_n=5):
    """
    Generuje listę rekomendowanych filmów dla użytkownika na podstawie ocen podobnych użytkowników.

    Args:
        dataset (dict): Słownik danych zawierający użytkowników i ich oceny filmów.
        target_user (str): Nazwa użytkownika, dla którego generowane są rekomendacje.
        top_n (int): Maksymalna liczba rekomendacji do zwrócenia.

    Returns:
        list: Lista rekomendacji w formacie [(nazwa_filmu, ocena)].

    Raises:e
        ValueError: Jeśli użytkownik docelowy nie istnieje w dataset.
    """
    if target_user not in dataset:
        raise ValueError(f"User {target_user} not found in the dataset")

    scores = {}
    for user in dataset:
        if user != target_user:
            scores[user] = euclidean_score(dataset, target_user, user)

    sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    recommended_movies = {}
    target_movies = set(dataset[target_user].keys())
    for user, score in sorted_users:
        if score == 0:
            continue
        for movie, rating in dataset[user].items():
            if movie not in target_movies and is_valid_rating(rating):
                if movie not in recommended_movies:
                    recommended_movies[movie] = (rating, score)
                else:
                    recommended_movies[movie] = max(
                        recommended_movies[movie], (rating, score), key=lambda x: x[1]
                    )

    recommended_list = sorted(recommended_movies.items(), key=lambda x: -x[1][0])[:top_n]

    return [(movie, rating[0]) for movie, rating in recommended_list]

def get_anti_recommendations(dataset, target_user, top_n=5):
    """
    Generuje listę filmów, które użytkownik może unikać, na podstawie niskich ocen innych użytkowników.

    Args:
        dataset (dict): Słownik danych zawierający użytkowników i ich oceny filmów.
        target_user (str): Nazwa użytkownika, dla którego generowane są antyrekomendacje.
        top_n (int): Maksymalna liczba antyrekomendacji do zwrócenia.

    Returns:
        list: Lista antyrekomendacji w formacie [(nazwa_filmu, ocena)].

    Raises:
        ValueError: Jeśli użytkownik docelowy nie istnieje w dataset.
    """
    if target_user not in dataset:
        raise ValueError(f"User {target_user} not found in the dataset")

    scores = {}
    for user in dataset:
        if user != target_user:
            scores[user] = euclidean_score(dataset, target_user, user)

    sorted_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    anti_recommended_movies = {}
    target_movies = set(dataset[target_user].keys())
    for user, score in sorted_users:
        if score == 0:
            continue
        for movie, rating in dataset[user].items():
            if movie not in target_movies and is_valid_rating(rating):
                if movie not in anti_recommended_movies:
                    anti_recommended_movies[movie] = (rating, score)
                else:
                    anti_recommended_movies[movie] = min(
                        anti_recommended_movies[movie], (rating, score), key=lambda x: x[1]
                    )

    anti_recommended_list = sorted(anti_recommended_movies.items(), key=lambda x: x[1][0])[:top_n]

    return [(movie, rating[0]) for movie, rating in anti_recommended_list]

if __name__ == "__main__":
    """
    Główna część programu wczytująca dane z pliku JSON, pobierająca użytkownika
    i wyświetlająca rekomendacje oraz antyrekomendacje.
    """
    with open("dane.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    target_user = input("Enter the username: ").strip()
    if target_user not in data:
        print(f"User {target_user} does not exist in the dataset.")
    else:
        try:
            recommendations = get_recommendations(data, target_user)
            if recommendations:
                print("\nRecommended Movies:")
                for movie, rating in recommendations:
                    print(f"{movie}: {rating}")
            else:
                print("No recommendations available.")

            anti_recommendations = get_anti_recommendations(data, target_user)
            if anti_recommendations:
                print("\nAnti-Recommendations:")
                for movie, rating in anti_recommendations:
                    print(f"{movie}: {rating}")
            else:
                print("No anti-recommendations available.")
        except Exception as e:
            print(f"Error: {e}")
