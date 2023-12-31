import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import threading
from tkinter import messagebox, ttk
import json
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time
import os
from datetime import datetime
import simpleaudio as sa


class AplicationPoseEstimation:
    def __init__(self, master, video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.menu_view_frame = None
        self.history_view_button = None
        self.modify_view_button = None
        self.session_view_button = None
        self.photo_label = None
        self.img = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.master = master

        # nazwa modelu do klasyfikacji pozycji
        # self.model = load_model(r'C:\Inzynierka\Programy\Nauka\perc.hdf5')
        self.model = self.get_model(f"perc.hdf5")
        self.treshold = 0.50

        with open('slownik_etykiet.json', 'r') as json_file:
            self.dictionary = json.load(json_file)
        # Wyświetlamy wczytany słownik
        print("Wczytany słownik etykiet:")
        print(self.dictionary)
        print(type(self.dictionary))
        self.dictionary = {val: key for key, val in self.dictionary.items()}

        self.new_dictionary = {}

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.app_width = 4 * screen_width // 5
        self.app_height = 4 * screen_height // 5

        self.master.geometry(f"{self.app_width}x{self.app_height}")

        # Konfiguracja głównego okna
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.list_of_positions = []

        # utworzenie widoku z menu
        self.menu_view()
        # utworzenie widoku z przeprowadzaniem cwiczen
        self.session_view()
        # utworzenie widoku umozliwiajacego modyfikacje sesji
        self.modify_view()
        # utworzenie widoku umozliwiajacego podglad historii
        self.history_view()

        self.name_of_actual_position = None
        self.number_of_actual_position = -1
        # self.list_of_positions = self.positions_to_do()

        self.number_positions_to_do = 0
        self.positions_to_do()
        # self.number_positions_to_do = len(self.list_of_positions)
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
        # Tworzenie ramki
        self.menu_view_frame = tk.Frame(self.master)
        self.menu_view_frame.grid(row=0, column=0)

        # Tworzenie nagłówka
        self.label = tk.Label(self.menu_view_frame, text="Menu", font=("Helvetica", 40))
        self.label.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie etykiety
        self.desc_menu_label = tk.Label(self.menu_view_frame, text="To jest opis aplikacji.", font=("Helvetica", 12))
        self.desc_menu_label.grid(row=1, column=1, pady=10, columnspan=1)

        # Tworzenie przycisków
        button_font = ("Helvetica", 14)
        button_relief = "groove"
        button_width = 30

        self.session_view_button = tk.Button(self.menu_view_frame, text="Rozpocznij sesję",
                                             command=self.show_session_view,
                                             font=button_font, relief=button_relief, width=button_width, height=4)
        self.session_view_button.grid(row=2, column=1, pady=20, columnspan=1)

        self.modify_view_button = tk.Button(self.menu_view_frame, text="Modyfikuj sesję", command=self.show_modify_view,
                                            font=button_font, relief=button_relief, width=button_width, height=4)
        self.modify_view_button.grid(row=3, column=1, pady=20, columnspan=1)

        self.history_view_button = tk.Button(self.menu_view_frame, text="Historia", command=self.show_history_view,
                                             font=button_font, relief=button_relief, width=button_width, height=4)
        self.history_view_button.grid(row=4, column=1, pady=20, columnspan=1)

        self.shutdown_button = tk.Button(self.menu_view_frame, text="Zakończ", command=self.shutdown,
                                         font=button_font, relief=button_relief, width=button_width, height=4)
        self.shutdown_button.grid(row=5, column=1, pady=20, columnspan=1)

    def session_view(self):
        self.session_view_frame = tk.Frame(self.master)
        # self.session_view_frame.grid(row=0, column=0, sticky="nsew")

        # Podzielenie glownego okna na 2 czesci
        self.session_view_frame.grid_rowconfigure(0, weight=1)
        self.session_view_frame.grid_columnconfigure(0, weight=3, minsize=300)
        self.session_view_frame.grid_columnconfigure(1, weight=7, minsize=700)

        # Lewa czesc
        self.session_left_frame = tk.Frame(self.session_view_frame, bg="lightblue")
        self.session_left_frame.grid(row=0, column=0, sticky="nsew")

        # prawa czesc
        self.session_right_frame = tk.Frame(self.session_view_frame, bg="lightgreen")
        self.session_right_frame.grid(row=0, column=1, sticky="nsew")

        # Umieszczenie obrazu z kamery po środku prawego okna
        self.labelCamera = tk.Label(self.session_right_frame, text="Testowy", bg='red')
        self.labelCamera.grid(row=0, column=0)
        self.session_right_frame.grid_rowconfigure(0, weight=1)
        self.session_right_frame.grid_columnconfigure(0, weight=1)

        # Obraz z kamery
        self.camera_width = 9 * self.app_width // 10
        self.camera_height = 9 * self.app_height // 10
        self.empty_image = Image.new("RGB", (3 * self.camera_width // 4, self.camera_height),
                                     "green")  # Tworzenie pustego obrazu

        # Dodaj 6 wierszy do kolumny
        for i in range(6):
            row_frame = tk.Frame(self.session_left_frame, bg="yellow", height=50)
            row_frame.pack(fill="both", expand=True)

        self.add_labels()
        self.add_buttons()

    def add_labels(self):

        # Label z nazwa pozycji
        row = self.session_left_frame.winfo_children()[0]
        label = tk.Label(row, text="Nazwa Pozycji", font=("Helvetica", 30, 'bold'), fg='blue', bg='yellow')
        label.pack(fill="both", expand=True)

        row = self.session_left_frame.winfo_children()[1]

        # # Create an object of tkinter ImageTk
        # self.img = ImageTk.PhotoImage(Image.open("position.png"))

        # Create a Label Widget to display the text or Image
        self.photo_label = tk.Label(row, image=self.img)
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

        row_buttons = self.session_left_frame.winfo_children()[4]

        print(row_buttons)

        row_buttons = tk.Frame(row_buttons, bg="white")
        row_buttons.pack(fill="both", expand=True)

        self.start_session_button = tk.Button(row_buttons, text="Start", relief="groove", width=10,
                                              command=self.start_capture)
        self.start_session_button.grid(row=0, column=0, sticky="nsew")
        row_buttons.grid_rowconfigure(0, weight=1)  # Ustawienie wagi wiersza, aby zajmować dostępną przestrzeń pionową

        self.pause_session_button = tk.Button(row_buttons, text="Wstrzymaj", relief="groove", width=10,
                                              command=self.stop_capture)
        self.pause_session_button.grid(row=0, column=1, sticky="nsew")
        self.pause_session_button["state"] = "disabled"
        row_buttons.grid_columnconfigure(0,
                                         weight=1)  # Ustawienie wagi kolumny, aby zajmować dostępną przestrzeń poziomą

        row_buttons.grid_columnconfigure(1, weight=1)
        row_buttons.grid_rowconfigure(0, weight=1)

        row_buttons = self.session_left_frame.winfo_children()[5]
        row_buttons = tk.Frame(row_buttons, bg="white")
        row_buttons.pack(fill="both", expand=True)

        # przycisk odpowiedzialny za przejscie do menu
        self.back_to_menu_session_button = tk.Button(row_buttons, text="Menu", relief="groove", width=10,
                                                     command=self.back_to_menu_session)
        self.back_to_menu_session_button.grid(row=0, column=0, sticky="nsew")

        # przycisk odpowiedzialny za zakończenie sesji cwiczen
        self.end_seesion_button = tk.Button(row_buttons, text="Zakończ sesję", relief="groove", width=10,
                                            command=self.end_session)
        self.end_seesion_button.grid(row=0, column=1, sticky="nsew")
        self.end_seesion_button["state"] = "disable"

        row_buttons.grid_rowconfigure(0, weight=1)  # Ustawienie wagi wiersza, aby zajmować dostępną przestrzeń pionową
        row_buttons.grid_columnconfigure(0,
                                         weight=1)  # Ustawienie wagi kolumny, aby zajmować dostępną przestrzeń poziomą
        row_buttons.grid_columnconfigure(1, weight=1)

    def end_session(self):
        """ Funckja odpowiedzialna za zakonczenie sesji ćwiczeń i zapis historii do pliku"""

        # if self.session_begin is True:
        self.back_to_menu_session_button["state"] = "normal"
        self.pause_session_button["state"] = "disable"
        self.start_session_button["state"] = "normal"
        self.start_session_button.config(text="Start")
        self.end_seesion_button["state"] = "disable"
        self.number_of_actual_position = -1

        self.session_begin = False
        self.session_on = False

        # self.video_running = False
        # self.number_of_actual_position = -1
        # nazwa pozycji
        # Uzyskaj dostęp do etykiety
        label = self.session_left_frame.winfo_children()[0].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="Nazwa pozycji")

        # numer pozycji
        label = self.session_left_frame.winfo_children()[2].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="-/-")

        # czas pozycji
        label = self.session_left_frame.winfo_children()[3].winfo_children()[0]
        # Zmiana tekstu w etykiecie
        label.config(text="00:00")
        # Zmiana obrazka na bazowy
        self.change_photo("None")
        # zapytanie czy zapisać historie przeprowadzonej sesji
        answer = messagebox.askquestion("Pytanie", "Czy chcesz zapisać historię sesji?")
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
            self.safe_to_file_history(history=exercise_history)
        self.performed_exercises.clear()

    def modify_view(self):
        self.modify_view_frame = tk.Frame(self.master)
        # self.modify_view_frame.grid(row=0, column=0)

        # Tworzenie nagłówka
        self.label_modify = tk.Label(self.modify_view_frame, text="Modyfikuj sesję ćwiczeń", font=("Helvetica", 40))
        self.label_modify.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie przycisków typu Checkbutton
        self.check_var_list = []
        self.checkbuttons_list = []

        self.new_dictionary = {}

        # Uzyskaj listę wartości, pomijając wartość 'None'
        values_list = [value for value in self.dictionary.values() if value != 'None']

        for i in range(0, len(values_list)):
            check_var = tk.IntVar(value=1)  # Ustawienie wartości na 1, czyli zaznaczone
            self.check_var_list.append(check_var)

            check_button = tk.Checkbutton(self.modify_view_frame, text=values_list[i], variable=check_var,
                                          font=("Helvetica", 16), command=lambda i=i: self.update_selected_options(i))
            self.checkbuttons_list.append(check_button)
            check_button.grid(row=i + 1, column=0, pady=10, sticky='w')

            self.new_dictionary[values_list[i - 1]] = self.check_var_list[i].get()

        # Dodaj przycisk pod checkboxami
        self.back_to_menu_modify_button = tk.Button(self.modify_view_frame, text="Menu",
                                                    command=self.back_to_menu_modify,
                                                    font=("Helvetica", 14), relief="groove", width=40, height=4)
        self.back_to_menu_modify_button.grid(row=len(values_list) + 2, column=0, sticky='e')

    def history_view(self):
        self.history_view_frame = tk.Frame(self.master)
        # self.history_view_frame.grid(row=0, column=0, sticky='n')

        # Tworzenie nagłówka
        self.label_history = tk.Label(self.history_view_frame, text="Historia ćwiczeń", font=("Helvetica", 40))
        self.label_history.grid(row=0, column=0, columnspan=3, pady=30, sticky='s')

        # Utwórz Treeview z nagłówkami kolumn
        # self.tree = ttk.Treeview(self.history_view_frame, columns=('ID', 'Imię', 'Nazwisko'), height=20)
        self.history_tree = ttk.Treeview(self.history_view_frame,
                                         columns=('Data', 'Czas trwania sesji', 'Liczba wykonanych pozycji',
                                                  'Liczba zaplanowanych pozycji'), height=20)

        # Dodaj nagłówki kolumn
        self.history_tree.heading('#0', text='Nr')
        self.history_tree.heading('Data', text='Data')
        self.history_tree.heading('Czas trwania sesji', text='Czas trwania sesji')
        self.history_tree.heading('Liczba wykonanych pozycji', text='Liczba wykonanych pozycji')
        self.history_tree.heading('Liczba zaplanowanych pozycji', text='Liczba zaplanowanych pozycji')
        # self.history_tree.heading('Lista wykonanych pozcyji', text='Lista wykonanych pozcyji')

        # Dostosuj styl tekstu w Treeview
        style = ttk.Style()
        style.configure('Treeview.Heading', font=('Helvetica', 12), relief='solid')  # Ustaw nagłówki z większą czcionką
        style.configure('Treeview', font=('Helvetica', 14), rowheight=30,
                        relief='solid')  # Ustaw tekst komórek z większą czcionką

        self.history_tree.column('#0', width=50, anchor='center')  # Ustaw szerokość kolumny
        self.history_tree.column('Data', width=200, anchor='center')  # Ustaw szerokość kolumny
        self.history_tree.column('Czas trwania sesji', width=150, anchor='center')  # Ustaw szerokość kolumny
        self.history_tree.column('Liczba wykonanych pozycji', width=200, anchor='center')  # Ustaw szerokość kolumny
        self.history_tree.column('Liczba zaplanowanych pozycji', width=220, anchor='center')  # Ustaw szerokość kolumny

        # Dodaj pionowy scrollbar
        scrollbar_y = ttk.Scrollbar(self.history_view_frame, orient='vertical', command=self.history_tree.yview)
        scrollbar_y.grid(row=1, column=3, sticky='ns')
        self.history_tree.configure(yscroll=scrollbar_y.set)

        # Dodaj  dane
        self.add_history_data()

        self.history_tree.grid(row=1, column=0, columnspan=3, sticky='nsew')

        # # Dodaj przycisk powrotu do menu
        # self.btn_powrot = tk.Button(self.history_view_frame, text="Powrót do menu", command=self.powrot_do_menu)
        # self.btn_powrot.grid(row=2, column=0, columnspan=3, pady=10)

        # Dodaj przycisk pod checkboxami
        self.button_history = tk.Button(self.history_view_frame, text="Menu", command=self.back_to_menu_history,
                                        font=("Helvetica", 14), relief="groove", width=40, height=4)
        self.button_history.grid(row=2, column=0, columnspan=3, pady=60)

        # Konfiguracja rozszerzania kolumn i wierszy
        self.history_view_frame.grid_columnconfigure(0, weight=1)
        self.history_view_frame.grid_rowconfigure(1, weight=1)

    def add_history_data(self):
        """Pobranie danych o historii wykonanych sesji z plikow z folderu /historia"""

        self.clear_history_tree_values()
        # Nazwa folderu
        folder_name = "historia"
        # Tworzenie pełnej ścieżki
        folder_path = os.path.join(os.getcwd(), folder_name)

        # Sprawdź, czy folder istnieje
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Pobierz listę plików w folderze
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

                    self.history_tree.insert('', 'end', text=str(file_count), values=(
                    time, time_of_session, number_of_exercises, number_of_planed_exercises))
                    file_count -= 1

    def clear_history_tree_values(self):
        """Usuwa wszystkie wartosci z historii"""
        # Usuń wszystkie elementy (rzędy) z drzewa
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

    # uaktualnienie dictionarya odpowiedzialnego za pozycje do wykonania
    def update_selected_options(self, i):
        check_var = self.check_var_list[i]
        self.new_dictionary[self.checkbuttons_list[i].cget("text")] = check_var.get()

    def shutdown(self):
        self.master.destroy()

    # # module for upload video from directory
    # def upload_video(self):
    #     file_path = filedialog.askopenfilename()
    #     if file_path:
    #         self.cap = cv2.VideoCapture(file_path)

    # module for start video
    def start_capture(self):
        # if(len(self.positions_to_do())):

        # self.video_running = True
        self.start_session_button["state"] = "disabled"
        self.start_session_button.config(text="Wnów")

        self.pause_session_button["state"] = "normal"

        self.end_seesion_button["state"] = "normal"
        self.back_to_menu_session_button["state"] = "disabled"

        # # Start a new thread to read and display video frames continuously
        # threading.Thread(target=self.video_detection).start()

        if self.session_begin is False:
            self.time_session_begin = time.time()
            self.next_position()
            self.session_begin = True

        self.session_on = True

    # module for stop or pause video
    def stop_capture(self):
        # self.video_running = False
        self.session_on = False

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
            # if self.cap.isOpened():
            ret, _ = self.cap.read()

            if self.number_positions_to_do < 1:
                messagebox.showwarning("Ostrzeżenie", "Nie wybrano żadnej pozycji! Przejdż do \"Modyfikuj sesję\".")
            elif ret is False:
                answer = messagebox.askquestion("Błąd", "Nie wykryto kamery, czy spróbować ją wykryć ponownie?")
                if answer == "yes":
                    self.cap = cv2.VideoCapture(self.video_path)
            else:
                self.video_running = True
                # Start a new thread to read and display video frames continuously
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

        # list_of_positions = []
        self.list_of_positions.clear()

        for key, value in self.new_dictionary.items():
            if value == 1:
                for key2, value2 in self.dictionary.items():
                    if value2 == key:
                        self.list_of_positions.append(key2)

        self.number_positions_to_do = len(self.list_of_positions)
        # self.list_of_positions.append(0)
        # self.list_of_positions.append(1)
        # # list_of_positions = [0, 1]
        # return list_of_positions

    def update_timer(self):
        # Zmiana tekstu w etykiecie
        self.timer = self.current_time - self.time_start  # + self.timer

        minuty = int(self.timer // 60)
        sekundy = int(self.timer % 60)

        self.session_left_frame.winfo_children()[3].winfo_children()[0].config(text=f"{minuty:02d}:{sekundy:02d}")

        # przejscie do nastepnej pozycji
        if int(self.timer) > self.all_time:
            self.next_position()

    def next_position(self):
        """Funckja obslugujaca przejscie do nastepnej pozycji podczas sesji"""

        self.timer = 0
        self.number_of_actual_position += 1

        # stwierdzenie faktu wykonania cwiczenia
        if self.number_of_actual_position > 0 and self.number_of_actual_position <= self.number_positions_to_do:
            self.performed_exercises.append(self.name_of_actual_position)
            self.play_sound()

        # sprawdzenie czy zostalo wykonane osatnie cwiczenie
        if self.number_of_actual_position <= self.number_positions_to_do:
            self.play_sound()
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
        """Funckja zmienia wyswietlany obrazek w widoku sesji. Parametr wejciowy: name_of_photo - nazwa pliku ze zdjeciem"""
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

    def play_sound(self):
        path_to_picture = os.path.join(f"dzwieki", f"signal.wav")
        folder_path = os.path.join(os.getcwd(), path_to_picture)
        if os.path.exists(folder_path):
            wave_obj = sa.WaveObject.from_wave_file(folder_path)  # Wymień na nazwę pliku dźwiękowego
            wave_obj.play()

    def safe_to_file_history(self, history):
        if not os.path.exists("historia"):
            os.makedirs("historia")

        time_now = datetime.now()
        file_name = f"{time_now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        # file_name = f"{time_now.strftime('%d-%m-%Y_%H-%M-%S')}.json"
        path_file = os.path.join("historia", file_name)

        with open(path_file, 'w') as plik:
            json.dump(history, plik, indent=2)

        # Uaktualnienie danych w historii
        self.add_history_data()

    def get_model(self, model_name):
        path_to_picture = os.path.join(f"modele", model_name)
        model_path = os.path.join(os.getcwd(), path_to_picture)
        model = load_model(model_path)
        return model

    def get_max_value_index(self, vector):
        max_index = np.argmax(vector)
        print(max_index)
        return max_index

    def get_name_position(self, vector):

        # tmp = np.max(vector)
        # Progowanie pewnosci
        if np.max(vector) > self.treshold:
            # key = get_max_value_index(vector)
            key = np.argmax(vector)
            name = self.dictionary.get(key)
        else:
            name = 'None'

        print(name)
        return name

    def flatten_landmarks(self, pose_landmarks_list):
        # Spłaszczanie listy do jednego ciągu liczb

        flattened_landmarks = [val for sublist in pose_landmarks_list for point in sublist for val in
                               (point.x, point.y, point.z)]
        print(flattened_landmarks)
        return flattened_landmarks

    def flatten_normalized_landmarks(self, normalized_landmarks):
        flattened_list = []

        for landmark in normalized_landmarks:
            flattened_list.extend([landmark.x, landmark.y, landmark.z])
        return flattened_list

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list_org = detection_result.pose_landmarks

        # Assuming NormalizedLandmark has attributes x, y, and z
        new_list = [{'x': item.x, 'y': item.y, 'z': item.z} for item in detection_result.pose_landmarks.landmark]

        # Indeksy do usunięcia
        indexes_to_remove = [1, 3, 4, 6, 8, 7, 22, 21]

        print("Lista: ")
        print(pose_landmarks_list_org)

        # Usunięcie elementów o podanych indeksach
        pose_landmarks_list = [pose_landmarks for i, pose_landmarks in enumerate(new_list)
                               if i not in indexes_to_remove]

        # Spłaszczanie listy do jednowymiarowej listy liczb
        flattened_landmarks = [value for item in pose_landmarks_list for value in item.values()]

        flattened_landmarks_np = np.array([flattened_landmarks])  # Dodajemy dodatkowy wymiar dla batch_size

        '''
        # Dla resnet
        flattened_landmarks_np = [np.array_split(np.array(item), 25) for item in flattened_landmarks_np]

        flattened_landmarks_np = np.array(flattened_landmarks_np)

        desired_shape = (1, 32, 32, 3)
        expanded_data = np.zeros(desired_shape)

        # Kopiowanie istniejących danych do nowej tablicy
        expanded_data[:, :25, :25, :] = flattened_landmarks_np[:, :, np.newaxis, :]

        flattened_landmarks_np = expanded_data
        '''

        annotated_image = np.copy(rgb_image)
        if pose_landmarks_list:
            predictions = self.model.predict(flattened_landmarks_np)
            name_of_position = self.get_name_position(predictions)
            cv2.putText(annotated_image, name_of_position, (300, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 0, 0),
                        4)
            print(predictions)

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

        self.mp_drawing.draw_landmarks(annotated_image, detection_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return annotated_image

    def video_detection(self):
        ret, frame = self.cap.read()

        if ret:

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = image.copy()

            # Make a detection
            # results = self.holistic.process(image)

            try:
                if self.session_on == True:
                    results = self.pose.process(image)
                    if results.pose_landmarks is not None:
                        # Draw the detection points on the image
                        annotated_image = self.draw_landmarks_on_image(annotated_image, results)

                frame = cv2.resize(annotated_image, (self.camera_width, self.camera_height))

                # Oblicz obszar centralny i zapisz go do nowej ramki
                h, w, _ = frame.shape
                h, w, _ = frame.shape
                min_dim = min(h, w)
                top = (h - min_dim) // 2

                left = (w - 4 * min_dim // 3) // 2

                frame = frame[top:top + min_dim,
                        left:left + 4 * min_dim // 3]  # wyswietlany obraz kamery ma proporcje 4:3

                self.photo = ImageTk.PhotoImage(Image.fromarray(frame))
                self.labelCamera.configure(image=self.photo)
                self.labelCamera.image = self.photo  # Aktualizacja referencji do obrazu w etykiecie



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
    app = AplicationPoseEstimation(root, 0)
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Butterfly.mp4')
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Squat.mp4')
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Chair.mp4')
    # app = AplicationPoseEstimation(root, r'C:\Inzynierka\Programy\Filmy\Warrior.mp4')

    root.mainloop()
