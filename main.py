"""
Aplikacja do wspomagania treningow jogi. Wykorzystuje MediaPipe w celu wykrycia czesci ciala, a nastepnie przy
wykorzystaniu Keras nastepuje klasyfikacja pozycji. Na aplikacje skladaja sie ponizszy plik main.py wraz
z plikiem utils.py. Program ten jest skierowany do koncowego uzytkownika i razem w programami technicznymi
(przygotowujacego dane dla sieci oraz do trenowania sieci) stanowi system wspomagajacy aktywnosc fizyczna.

Autor: Michał Wójtowicz
Data utworzenia: 2024-01-09
"""


import json
import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import *
from tkinter import messagebox, ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from utils import play_sound, get_model, safe_to_file_history


class AplicationPoseEstimation:
    def __init__(self, master, video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.master = master
        self.master.title("Yoga Pose Estimation")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.menu_view_frame = None
        self.session_view_frame = None
        self.modify_view_frame = None
        self.history_view_frame = None

        self.session_left_frame = None
        self.photo_label = None
        self.pause_session_button = None
        self.start_session_button = None
        self.camera_label = None
        self.end_seesion_button = None
        self.history_tree = None
        self.back_to_menu_session_button = None

        self.checkbuttons_list = None
        self.check_var_list = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)

        # Pozyskanie modelu do klasyfikacji pozycji
        # self.model = self.get_model(f"perc.hdf5")
        self.model = get_model(f"conv.hdf5")

        # Prog pewnosci podczas klasyfikacji pozycji
        self.treshold = 0.50

        # Pozyskanie nazw uzywanych przy klasyfikajci
        with open('slownik_etykiet.json', 'r') as json_file:
            self.dictionary = json.load(json_file)

        print("Wczytany słownik etykiet:")
        print(self.dictionary)
        print(type(self.dictionary))
        self.dictionary = {val: key for key, val in self.dictionary.items()}
        self.new_dictionary = {}

        # Konfiguracja głównego okna
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.app_width = 4 * screen_width // 5
        self.app_height = 4 * screen_height // 5
        self.master.geometry(f"{self.app_width}x{self.app_height}")

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.camera_height = None
        self.camera_width = None

        self.list_of_positions = []

        # Utworzenie widoku z menu
        self.menu_view()
        # Utworzenie widoku sesji z przeprowadzaniem cwiczen
        self.session_view()
        # Utworzenie widoku umozliwiajacego modyfikacje sesji
        self.modify_view()
        # Utworzenie widoku umozliwiajacego podglad historii
        self.history_view()

        self.name_of_actual_position = None
        self.number_of_actual_position = -1

        self.number_positions_to_do = 0
        self.positions_to_do()

        # Parametry do zliczania i zarzadzania czasem
        self.timer = 0
        self.last_time = 0
        self.current_time = 0
        self.time_start = 0
        self.all_time = 5
        self.flag = 0

        self.performed_exercises = []
        self.time_session_begin = 0
        self.session_begin = False
        self.session_on = False

        self.video_running = False

    def menu_view(self):
        """Tworzenie widoku menu"""

        # Tworzenie glownej ramki
        self.menu_view_frame = tk.Frame(self.master)
        self.menu_view_frame.grid(row=0, column=0)

        # Tworzenie nagłówka
        label = tk.Label(self.menu_view_frame, text="Menu", font=("Helvetica", 40))
        label.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie etykiety
        desc_menu_label = tk.Label(self.menu_view_frame, text="Aplikacja do rozpoznawania ułożenia ciała w obrazie.",
                                   font=("Helvetica", 12))
        desc_menu_label.grid(row=1, column=1, pady=10, columnspan=1)

        # Tworzenie przyciskow
        button_font = ("Helvetica", 14)
        button_relief = "groove"
        button_width = 30

        session_view_button = tk.Button(self.menu_view_frame, text="Rozpocznij sesję",
                                        command=self.show_session_view,
                                        font=button_font, relief=button_relief, width=button_width, height=4)
        session_view_button.grid(row=2, column=1, pady=20, columnspan=1)

        modify_view_button = tk.Button(self.menu_view_frame, text="Modyfikuj sesję", command=self.show_modify_view,
                                       font=button_font, relief=button_relief, width=button_width, height=4)
        modify_view_button.grid(row=3, column=1, pady=20, columnspan=1)

        history_view_button = tk.Button(self.menu_view_frame, text="Historia", command=self.show_history_view,
                                        font=button_font, relief=button_relief, width=button_width, height=4)
        history_view_button.grid(row=4, column=1, pady=20, columnspan=1)

        shutdown_button = tk.Button(self.menu_view_frame, text="Zakończ", command=self.shutdown,
                                    font=button_font, relief=button_relief, width=button_width, height=4)
        shutdown_button.grid(row=5, column=1, pady=20, columnspan=1)

    def session_view(self):
        """Tworzenie widoku sesji"""

        # Tworzenie glownej ramki
        self.session_view_frame = tk.Frame(self.master)

        # Podzielenie glownego okna na 2 czesci
        self.session_view_frame.grid_rowconfigure(0, weight=1)
        self.session_view_frame.grid_columnconfigure(0, weight=3, minsize=300)
        self.session_view_frame.grid_columnconfigure(1, weight=7, minsize=700)

        # Lewa czesc
        self.session_left_frame = tk.Frame(self.session_view_frame, bg="lightblue")
        self.session_left_frame.grid(row=0, column=0, sticky="nsew")

        # Prawa czesc
        session_right_frame = tk.Frame(self.session_view_frame, bg="lightgreen")
        session_right_frame.grid(row=0, column=1, sticky="nsew")

        # Umieszczenie obrazu z kamery po środku prawego okna
        self.camera_label = tk.Label(session_right_frame, text="Testowy", bg='red')
        self.camera_label.grid(row=0, column=0)
        session_right_frame.grid_rowconfigure(0, weight=1)
        session_right_frame.grid_columnconfigure(0, weight=1)

        # Obraz z kamery
        self.camera_width = 9 * self.app_width // 10
        self.camera_height = 9 * self.app_height // 10

        # Dodanie 6 wierszy do lewej czesci
        for i in range(6):
            row_frame = tk.Frame(self.session_left_frame, bg="yellow", height=50)
            row_frame.pack(fill="both", expand=True)

        # Dodanie etykiet do lewej czesci widoku sesji
        self.add_labels()
        # Dodanie przyciskow do lewej czesci widoku sesji
        self.add_buttons()

    def add_labels(self):
        """Dodanie etykiet do widoku lewej czesci sesji"""

        # Label z nazwa pozycji
        row = self.session_left_frame.winfo_children()[0]
        label = tk.Label(row, text="Nazwa Pozycji", font=("Helvetica", 30, 'bold'), fg='blue', bg='yellow')
        label.pack(fill="both", expand=True)

        row = self.session_left_frame.winfo_children()[1]

        # Label ze zdjeciem pozycji
        self.photo_label = tk.Label(row)
        self.photo_label.pack()
        self.change_photo("None")

        # Label z liczba pozycji
        row = self.session_left_frame.winfo_children()[2]
        label = tk.Label(row, text="-/-", font=("Helvetica", 50, 'bold'), fg='blue', bg='yellow')
        label.pack()

        # Label z czasem
        row = self.session_left_frame.winfo_children()[3]
        label = tk.Label(row, text="0:00", font=("Helvetica", 50, 'bold'), fg='blue', bg='yellow')
        label.pack()

    def add_buttons(self):
        """Dodanie przyciskow Start, Wstrzymaj, Menu, Zakoncz sesje do lewej czesci widoku sesji"""

        row_buttons = self.session_left_frame.winfo_children()[4]
        # print(row_buttons)
        row_buttons = tk.Frame(row_buttons, bg="white")
        row_buttons.pack(fill="both", expand=True)

        # Przycisk odpowiedzialny za rozpoczecie i wznowienie sesji
        self.start_session_button = tk.Button(row_buttons, text="Start", relief="groove", width=10,
                                              command=self.start_capture)
        self.start_session_button.grid(row=0, column=0, sticky="nsew")
        row_buttons.grid_rowconfigure(0, weight=1)  # Ustawienie wiersza aby zajmowal dostępną przestrzen w pionnie

        # Przycisk odpowiedzialny za wstrzymanie sesji
        self.pause_session_button = tk.Button(row_buttons, text="Wstrzymaj", relief="groove", width=10,
                                              command=self.stop_capture)
        self.pause_session_button.grid(row=0, column=1, sticky="nsew")
        self.pause_session_button["state"] = "disabled"
        row_buttons.grid_columnconfigure(0, weight=1)  # Ustawienie kolumny aby zajmowala dostępna przestrzen w poziomie

        row_buttons.grid_columnconfigure(1, weight=1)
        row_buttons.grid_rowconfigure(0, weight=1)

        row_buttons = self.session_left_frame.winfo_children()[5]
        row_buttons = tk.Frame(row_buttons, bg="white")
        row_buttons.pack(fill="both", expand=True)

        # Przycisk odpowiedzialny za przejscie do menu
        self.back_to_menu_session_button = tk.Button(row_buttons, text="Menu", relief="groove", width=10,
                                                     command=self.back_to_menu_session)
        self.back_to_menu_session_button.grid(row=0, column=0, sticky="nsew")

        # Przycisk odpowiedzialny za zakończenie sesji cwiczen
        self.end_seesion_button = tk.Button(row_buttons, text="Zakończ sesję", relief="groove", width=10,
                                            command=self.end_session)
        self.end_seesion_button.grid(row=0, column=1, sticky="nsew")
        self.end_seesion_button["state"] = "disable"

        row_buttons.grid_rowconfigure(0, weight=1)     # Ustawienie wiersza aby zajmowal dostępną przestrzen w pionnie
        row_buttons.grid_columnconfigure(0, weight=1)  # Ustawienie kolumny aby zajmowala dostępna przestrzen w poziomie
        row_buttons.grid_columnconfigure(1, weight=1)

    def end_session(self):
        """ Funckja odpowiedzialna za zakonczenie sesji ćwiczeń i zapis historii do pliku"""

        # Obsluga przyciskow
        self.back_to_menu_session_button["state"] = "normal"
        self.pause_session_button["state"] = "disable"
        self.start_session_button["state"] = "normal"
        self.start_session_button.config(text="Start")
        self.end_seesion_button["state"] = "disable"
        self.number_of_actual_position = -1

        self.session_begin = False
        self.session_on = False

        # Ustawienie nazwy pozycji
        label = self.session_left_frame.winfo_children()[0].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="Nazwa pozycji")

        # Ustawienie numer pozycji
        label = self.session_left_frame.winfo_children()[2].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="-/-")

        # Ustawienie czasu pozycji
        label = self.session_left_frame.winfo_children()[3].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="00:00")

        # Zmiana obrazka na bazowy
        self.change_photo("None")

        # Zapytanie czy zapisać historie przeprowadzonej sesji
        answer = messagebox.askquestion("Pytanie", "Czy chcesz zapisać historię sesji?")

        # Obsluga zapisu historii sesji
        if answer == "yes":
            time_of_session = time.time() - self.time_session_begin
            hours = int(time_of_session // 3600)
            minutes = int(time_of_session // 60)
            seconds = int(time_of_session % 60)

            exercise_history = {
                'number_of_exercises': len(self.performed_exercises),
                'number_of_planed_exercises': self.number_positions_to_do,
                'exercise_list': self.performed_exercises,
                'time_of_session': f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            }
            safe_to_file_history(history=exercise_history)
            self.add_history_data()
        self.performed_exercises.clear()

    def modify_view(self):
        """Tworzenie widoku modyfikacji sesji"""

        # Tworzenie glownej ramki
        self.modify_view_frame = tk.Frame(self.master)

        # Tworzenie naglowka
        label_modify = tk.Label(self.modify_view_frame, text="Modyfikuj sesję ćwiczeń", font=("Helvetica", 40))
        label_modify.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie przycisków typu checkbutton
        self.check_var_list = []
        self.checkbuttons_list = []

        self.new_dictionary = {}
        # Uzyskanie listy wykrywanych pozycji
        values_list = [value for value in self.dictionary.values() if value != 'None']

        for i in range(0, len(values_list)):
            check_var = tk.IntVar(value=1)  # Ustawienie wartości na 1, czyli zaznaczone
            self.check_var_list.append(check_var)

            check_button = tk.Checkbutton(self.modify_view_frame, text=values_list[i], variable=check_var,
                                          font=("Helvetica", 16), command=lambda i=i: self.update_selected_options(i))
            self.checkbuttons_list.append(check_button)
            check_button.grid(row=i + 1, column=0, pady=10, sticky='w')

            self.new_dictionary[values_list[i - 1]] = self.check_var_list[i].get()

        # Przycisk odpowiedzialny za powrot do widoku menu
        back_to_menu_modify_button = tk.Button(self.modify_view_frame, text="Menu", command=self.back_to_menu_modify,
                                               font=("Helvetica", 14), relief="groove", width=40, height=4)
        back_to_menu_modify_button.grid(row=len(values_list) + 2, column=0, sticky='e')

    def history_view(self):
        """Tworzenie widoku historii"""

        # Tworzenie glownej ramki
        self.history_view_frame = tk.Frame(self.master)

        # Tworzenie nagłówka
        label_history = tk.Label(self.history_view_frame, text="Historia ćwiczeń", font=("Helvetica", 40))
        label_history.grid(row=0, column=0, columnspan=3, pady=30, sticky='s')

        # Utworzenie Treeview z nagłówkami kolumn
        self.history_tree = ttk.Treeview(self.history_view_frame,
                                         columns=('Data', 'Czas trwania sesji', 'Liczba wykonanych pozycji',
                                                  'Liczba zaplanowanych pozycji'), height=20)

        # Dodanie naglowkow kolumn
        self.history_tree.heading('#0', text='Nr')
        self.history_tree.heading('Data', text='Data')
        self.history_tree.heading('Czas trwania sesji', text='Czas trwania sesji')
        self.history_tree.heading('Liczba wykonanych pozycji', text='Liczba wykonanych pozycji')
        self.history_tree.heading('Liczba zaplanowanych pozycji', text='Liczba zaplanowanych pozycji')
        # self.history_tree.heading('Lista wykonanych pozcyji', text='Lista wykonanych pozcyji')

        # Dostosowanie stylu w Treeview
        style = ttk.Style()
        style.configure('Treeview.Heading', font=('Helvetica', 12), relief='solid')
        style.configure('Treeview', font=('Helvetica', 14), rowheight=30, relief='solid')

        self.history_tree.column('#0', width=50, anchor='center')
        self.history_tree.column('Data', width=200, anchor='center')
        self.history_tree.column('Czas trwania sesji', width=150, anchor='center')
        self.history_tree.column('Liczba wykonanych pozycji', width=200, anchor='center')
        self.history_tree.column('Liczba zaplanowanych pozycji', width=220, anchor='center')

        # Dodanie pionowego scrollbara
        scrollbar_y = ttk.Scrollbar(self.history_view_frame, orient='vertical', command=self.history_tree.yview)
        scrollbar_y.grid(row=1, column=3, sticky='ns')
        self.history_tree.configure(yscroll=scrollbar_y.set)

        # Pozyskanie danych o historii
        self.add_history_data()

        self.history_tree.grid(row=1, column=0, columnspan=3, sticky='nsew')

        # Przycisk odpowiedzialny za powrot do widoku menu
        back_to_menu_history_button = tk.Button(self.history_view_frame, text="Menu", command=self.back_to_menu_history,
                                                font=("Helvetica", 14), relief="groove", width=40, height=4)
        back_to_menu_history_button.grid(row=2, column=0, columnspan=3, pady=60)

        # Konfiguracja rozszerzania kolumn i wierszy
        self.history_view_frame.grid_columnconfigure(0, weight=1)
        self.history_view_frame.grid_rowconfigure(1, weight=1)

    def add_history_data(self):
        """Pobranie danych o historii wykonanych sesji z plikow z folderu /historia"""

        self.clear_history_tree_values()
        # Nazwa folderu
        folder_name = "historia"
        # Tworzenie pelnej ścieżki
        folder_path = os.path.join(os.getcwd(), folder_name)

        # Pozyskanie danych o hostorii przeprowadzonych sesji
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Pobranie historii z folderu w postaci nazw plikow
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            file_list.reverse()
            file_count = len(file_list)

            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Pobierz pozostałe informacje
                    number_of_exercises = data.get("number_of_exercises")
                    number_of_planed_exercises = data.get("number_of_planed_exercises")
                    # exercise_list = data.get("exercise_list")
                    time_of_session = data.get("time_of_session")

                    # Uzyskaj nazwę pliku bez rozszerzenia
                    time = os.path.splitext(os.path.basename(file_name))[0]

                    input_format = "%Y-%m-%d_%H-%M-%S"
                    output_format = "%d.%m.%Y %H:%M:%S"

                    # Parsowanie daty z wejściowego stringa
                    date_object = datetime.strptime(time, input_format)

                    # Formatowanie daty do żądanego formatu
                    time = date_object.strftime(output_format)

                    self.history_tree.insert('', 'end', text=str(file_count),
                                             values=(time, time_of_session, number_of_exercises,
                                                     number_of_planed_exercises))
                    file_count -= 1

    def clear_history_tree_values(self):
        """Usuniecie wszystkicj wartosci z historii"""

        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

    def update_selected_options(self, i):
        """Uaktualnienie listy zawierajacej pozycje do wykonania"""

        check_var = self.check_var_list[i]
        self.new_dictionary[self.checkbuttons_list[i].cget("text")] = check_var.get()

    def shutdown(self):
        """Zamkniecie aplikacji"""
        self.master.destroy()

    def start_capture(self):
        """Rozpoczecie oraz wznowienie sesji"""

        # Obsluga przyciskow
        self.start_session_button["state"] = "disabled"
        self.start_session_button.config(text="Wznów")

        self.pause_session_button["state"] = "normal"

        self.end_seesion_button["state"] = "normal"
        self.back_to_menu_session_button["state"] = "disabled"

        # Rozpoczecie po sesji
        if self.session_begin is False:
            self.time_session_begin = time.time()
            self.next_position()
            self.session_begin = True

        self.session_on = True

    def stop_capture(self):
        """Wstrzymanie sesji"""
        self.session_on = False

        # Obsluga przysickow
        self.start_session_button["state"] = "normal"
        self.pause_session_button["state"] = "disabled"

    def back_to_menu_session(self):
        """Przejscie z widoku sesji do menu"""

        self.video_running = False

        self.menu_view_frame.grid_forget()
        self.session_view_frame.grid_forget()
        self.menu_view_frame.grid(row=0, column=0)

    def show_session_view(self):
        """Przejscie z widoku menu do sesji"""

        try:
            ret, _ = self.cap.read()

            # Zabezpieczenie w przypadku braku wybrania pozycji do trenowania
            if self.number_positions_to_do < 1:
                messagebox.showwarning("Ostrzeżenie", "Nie wybrano żadnej pozycji! Przejdż do \"Modyfikuj sesję\".")
            elif ret is False:
                answer = messagebox.askquestion("Błąd", "Nie wykryto kamery, czy spróbować ją wykryć ponownie?")
                if answer == "yes":
                    self.cap = cv2.VideoCapture(self.video_path)
            else:
                self.video_running = True
                # Rozpoczecie nowego watku, aby w sposob ciągly czytac i wyswietlac klatki wideo
                threading.Thread(target=self.video_detection).start()

                self.session_view_frame.grid_forget()
                self.menu_view_frame.grid_forget()
                self.session_view_frame.grid(row=0, column=0, sticky="nsew")

        except IndexError as e:
            print(f"Wystąpił błąd IndexError: {str(e)}")

    def back_to_menu_modify(self):
        """Przejscie z widoku modyfikacji modify do menu"""
        self.positions_to_do()
        self.modify_view_frame.grid_forget()
        self.menu_view_frame.grid_forget()
        self.menu_view_frame.grid(row=0, column=0)

    def show_modify_view(self):
        """Przejscie z widoku menu do modyfikacji sesji"""
        self.modify_view_frame.grid_forget()
        self.menu_view_frame.grid_forget()
        self.modify_view_frame.grid(row=0, column=0)

    def back_to_menu_history(self):
        """Przejscie z widoku historii do menu"""
        self.history_view_frame.grid_forget()
        self.menu_view_frame.grid_forget()
        self.menu_view_frame.grid(row=0, column=0)

    def show_history_view(self):
        """Przejscie z widoku menu do historii"""
        self.history_view_frame.grid_forget()
        self.menu_view_frame.grid_forget()
        self.history_view_frame.grid(row=0, column=0, sticky='n')

    def positions_to_do(self):
        """Lista pozycji do wykonania przez uzytkownika w czasie sesji"""

        self.list_of_positions.clear()

        for key, value in self.new_dictionary.items():
            if value == 1:
                for key2, value2 in self.dictionary.items():
                    if value2 == key:
                        self.list_of_positions.append(key2)

        self.number_positions_to_do = len(self.list_of_positions)

    def update_timer(self):
        """Zliczanie czasu wykonywania pozycji oraz uaktualnienie etykiet"""
        # Zmiana tekstu w etykiecie
        self.timer = self.current_time - self.time_start

        minuty = int(self.timer // 60)
        sekundy = int(self.timer % 60)

        self.session_left_frame.winfo_children()[3].winfo_children()[0].config(text=f"{minuty:02d}:{sekundy:02d}")

        # Przejscie do nastepnej pozycji
        if int(self.timer) > self.all_time:
            self.next_position()

    def next_position(self):
        """Funckja obslugujaca przejscie do nastepnej pozycji podczas sesji"""

        self.timer = 0
        self.number_of_actual_position += 1

        # Stwierdzenie faktu wykonania cwiczenia
        if self.number_of_actual_position > 0 and self.number_of_actual_position <= self.number_positions_to_do:
            self.performed_exercises.append(self.name_of_actual_position)
            play_sound(f"signal.wav")

        # Sprawdzenie czy zostalo wykonane osatnie cwiczenie
        if self.number_of_actual_position <= self.number_positions_to_do:
            play_sound(f"signal.wav")
            self.name_of_actual_position = self.dictionary.get(self.list_of_positions[self.number_of_actual_position])

            # nazwa pozycji
            # Uzyskaj dostęp do etykiety
            label = self.session_left_frame.winfo_children()[0].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text=self.name_of_actual_position)

            # numer pozycji
            label = self.session_left_frame.winfo_children()[2].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text=str(self.number_of_actual_position + 1) + '/' + str(self.number_positions_to_do))

            # czas pozycji
            label = self.session_left_frame.winfo_children()[3].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text="00:00")

            # Zmiana wyswietlanego obrazka z popzycja
            self.change_photo(self.name_of_actual_position)

        else:
            self.end_session()

    def change_photo(self, name_of_photo):
        """Funckja zmienia wyswietlany obrazek w widoku sesji. Parametr wejciowy: name_of_photo - nazwa pliku ze
        zdjeciem"""
        path_to_picture = os.path.join(f"pozycje", f"{name_of_photo}.png")
        # Tworzenie pełnej ścieżki
        folder_path = os.path.join(os.getcwd(), path_to_picture)
        if os.path.exists(folder_path):
            img = ImageTk.PhotoImage(Image.open(folder_path))
            self.photo_label.configure(image=img)
            self.photo_label.image = img
        else:
            empty_image = Image.new("RGB", (469, 295), "white")
            img = ImageTk.PhotoImage(empty_image)
            self.photo_label.configure(image=img)
            self.photo_label.image = img

    def get_name_position(self, vector):
        """Pozyskanie nazwy pozycji na podstawie jej numeru"""
        # Progowanie pewnosci
        if np.max(vector) > self.treshold:
            # key = get_max_value_index(vector)
            key = np.argmax(vector)
            name = self.dictionary.get(key)
        else:
            name = 'None'

        print(name)
        return name

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """Opracowanie danych otrzymanych w wyniku szkieletyzacji, przeprowadzenie klasyfikacji danych oraz narysowanie
        szkieletu na otrzymanym obrazie"""

        # pose_landmarks_list_org = detection_result.pose_landmarks
        # print("Lista: ")
        # print(pose_landmarks_list_org)

        # Pozyskanie listy zawierajacej 3 wymiary: X, Y, Z
        new_list = [{'x': item.x, 'y': item.y, 'z': item.z} for item in detection_result.pose_landmarks.landmark]

        # Indeksy punktow do usunięcia
        indexes_to_remove = [1, 3, 4, 6, 8, 7, 22, 21]

        # Usunięcie elementów o podanych indeksach
        pose_landmarks_list = [pose_landmarks for i, pose_landmarks in enumerate(new_list)
                               if i not in indexes_to_remove]

        # Splaszczanie listy do jednowymiarowej listy liczb
        flattened_landmarks = [value for item in pose_landmarks_list for value in item.values()]

        # Dodadanie dodatkowego wymiaru dla batch_size
        flattened_landmarks_np = np.array([flattened_landmarks])

        annotated_image = np.copy(rgb_image)

        # Obsluga klasyfikacji pozycji
        if pose_landmarks_list:
            predictions = self.model.predict(flattened_landmarks_np)
            name_of_position = self.get_name_position(predictions)
            cv2.putText(annotated_image, name_of_position, (300, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
            # print(predictions)

            # Ocena zgodnosci wykrytej pozycji z zadana i zarzadzanie czasem
            if name_of_position == self.name_of_actual_position and self.flag == 0:
                self.time_start = time.time()
                if self.timer != 0:
                    self.time_start -= self.timer
                self.flag = 1
            elif name_of_position == self.name_of_actual_position and self.flag == 1:
                self.current_time = time.time()
                self.update_timer()
            elif self.flag == 1:
                self.flag = 0

        # Rysowanie wkrytych punktów na obrazie w postaci szkieletu
        self.mp_drawing.draw_landmarks(annotated_image, detection_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return annotated_image

    def video_detection(self):
        """"Pozyskanie danych w postaci obrazu z kamery, wykrywanie punktow charakterystycznych oraz wyswietlanie
        zmodyfikowanego obrazu na ekranie"""

        ret, frame = self.cap.read()

        # Sprawdzenie czy pozyskano obraz z kamery
        if ret:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = image.copy()

            try:
                # Sprawdzenie czy trwa sesja
                if self.session_on == True:
                    # Pozyskanie punktow charakterystycznych dla czesci ciala
                    results = self.pose.process(image)
                    if results.pose_landmarks is not None:
                        # Poddanie otrzymanych danych dalszej analizie
                        annotated_image = self.draw_landmarks_on_image(annotated_image, results)

                # Przygotowanie obrazu do wyswietlenia na ekranie
                frame = cv2.resize(annotated_image, (self.camera_width, self.camera_height))

                # Wyznaczanie nowego kadru obrazu
                h, w, _ = frame.shape
                h, w, _ = frame.shape
                min_dim = min(h, w)
                top = (h - min_dim) // 2

                left = (w - 4 * min_dim // 3) // 2

                # Wyswietlany obraz kamery ma proporcje 4:3
                frame = frame[top:top + min_dim, left:left + 4 * min_dim // 3]

                photo = ImageTk.PhotoImage(Image.fromarray(frame))
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo  # Aktualizacja referencji do obrazu w etykiecie

            except IndexError as e:
                print(f"Wystąpił błąd IndexError: {str(e)}")

            if self.video_running:
                self.master.after(10, self.video_detection)
        else:
            # Brak wykrycia kamery
            messagebox.showerror("Błąd!", "Błąd odczytu kamery.")
            self.end_session()
            self.back_to_menu_session()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = AplicationPoseEstimation(master=root, video_path=0)
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Tree.mp4')
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Squat.mp4')
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy3\Warrior.mp4')

    root.mainloop()
