import os
import simpleaudio as sa
from keras.models import load_model
import numpy as np
from datetime import datetime
import json


def play_sound(file_name):
    path_to_picture = os.path.join(f"dzwieki", file_name)
    folder_path = os.path.join(os.getcwd(), path_to_picture)
    if os.path.exists(folder_path):
        wave_obj = sa.WaveObject.from_wave_file(folder_path)  # Wymień na nazwę pliku dźwiękowego
        wave_obj.play()


# def flatten_landmarks(pose_landmarks_list):
#     # Spłaszczanie listy do jednego ciągu liczb
#
#     flattened_landmarks = [val for sublist in pose_landmarks_list for point in sublist for val in
#                            (point.x, point.y, point.z)]
#     print(flattened_landmarks)
#     return flattened_landmarks


# def flatten_normalized_landmarks(normalized_landmarks):
#     flattened_list = []
#
#     for landmark in normalized_landmarks:
#         flattened_list.extend([landmark.x, landmark.y, landmark.z])
#     return flattened_list


def get_model(model_name):
    path_to_picture = os.path.join(f"modele", model_name)
    model_path = os.path.join(os.getcwd(), path_to_picture)
    model = load_model(model_path)
    return model


# def get_max_value_index(vector):
#     max_index = np.argmax(vector)
#     print(max_index)
#     return max_index


def safe_to_file_history(history):
    if not os.path.exists("historia"):
        os.makedirs("historia")

    time_now = datetime.now()
    file_name = f"{time_now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    # file_name = f"{time_now.strftime('%d-%m-%Y_%H-%M-%S')}.json"
    path_file = os.path.join("historia", file_name)

    with open(path_file, 'w') as plik:
        json.dump(history, plik, indent=2)

    # Uaktualnienie danych w historii
    # self.add_history_data()
