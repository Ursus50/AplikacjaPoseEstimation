import cv2
import math
import tkinter as tk
from tkinter import *
import mediapipe as mp
from PIL import Image, ImageTk
import threading
from tkinter import filedialog, messagebox, ttk
import json
from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
from keras.models import load_model
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import os
from datetime import datetime



class AplicationPoseEstimation:
    def __init__(self, master, video_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose

        self.master = master

        # nazwa modelu do klasyfikacji pozycji
        self.model = load_model(r'C:\Inzynierka\Programy\Nauka\perc.hdf5')
        self.prog = 0.50

        with open('slownik_etykiet.json', 'r') as json_file:
            self.slownik = json.load(json_file)
        # Wyświetlamy wczytany słownik
        print("Wczytany słownik etykiet:")
        print(self.slownik)
        print(type(self.slownik))
        self.slownik = {wartosc: klucz for klucz, wartosc in self.slownik.items()}

        self.new_slownik = {}

        # self.name_of_actual_position = None
        # self.number_of_actual_position = -1
        # # self.list_of_positions = self.positions_to_do()
        # self.list_of_positions = []
        # self.positions_to_do()
        # self.number_positions_to_do = len(self.list_of_positions)
        # self.timer = 0
        # self.last_time = 0
        # self.current_time = 0
        # self.time_start = 0
        # self.all_time = 5
        # self.flag = 0

        self.cap = cv2.VideoCapture(video_path)

        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                                  min_tracking_confidence=min_tracking_confidence)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.app_width = 4*screen_width//5
        self.app_height = 4*screen_height//5

        self.master.geometry(f"{self.app_width}x{self.app_height}")

        # Konfiguracja głównego okna
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.list_of_positions = []

        # utworzenie widoku z menu
        self.strona_menu()
        # utworzenie widoku z przeprowadzaniem cwiczen
        self.strona_sesji()
        # utworzenie widoku umozliwiajacego modyfikacje sesji
        self.strona_modify()
        # utworzenie widoku umozliwiajacego podglad historii
        self.strona_historia()

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

        self.video_running = False

    def strona_menu(self):
        # Tworzenie ramki
        self.frame_strona1 = tk.Frame(self.master)
        self.frame_strona1.grid(row=0, column=0)

        # Tworzenie nagłówka
        self.label = tk.Label(self.frame_strona1, text="Menu", font=("Helvetica", 40))
        self.label.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie etykiety
        self.opis_label = tk.Label(self.frame_strona1, text="To jest opis aplikacji.", font=("Helvetica", 12))
        self.opis_label.grid(row=1, column=1, pady=10, columnspan=1)

        # Tworzenie przycisków
        button_font = ("Helvetica", 14)
        button_relief = "groove"
        button_width = 30

        self.rozpocznij_button = tk.Button(self.frame_strona1, text="Rozpocznij sesję", command=self.ukryj_menu,
                                           font=button_font, relief=button_relief, width=button_width, height=4)
        self.rozpocznij_button.grid(row=2, column=1, pady=20, columnspan=1)

        self.modyfikuj_button = tk.Button(self.frame_strona1, text="Modyfikuj sesję", command=self.modyfikuj_sesje,
                                          font=button_font, relief=button_relief, width=button_width, height=4)
        self.modyfikuj_button.grid(row=3, column=1, pady=20, columnspan=1)

        self.historia_button = tk.Button(self.frame_strona1, text="Historia", command=self.pokaz_historie,
                                         font=button_font, relief=button_relief, width=button_width, height=4)
        self.historia_button.grid(row=4, column=1, pady=20, columnspan=1)
    def strona_sesji(self):
        self.frame_strona2 = tk.Frame(self.master)
        # self.frame_strona2.grid(row=0, column=0, sticky="nsew")

        # Podzielenie glownego okna na 2 czesci
        self.frame_strona2.grid_rowconfigure(0, weight=1)
        self.frame_strona2.grid_columnconfigure(0, weight=3, minsize=300)
        self.frame_strona2.grid_columnconfigure(1, weight=7, minsize=700)

        # Lewa czesc
        self.leftFrame = tk.Frame(self.frame_strona2, bg="lightblue")
        self.leftFrame.grid(row=0, column=0, sticky="nsew")

        # prawa czesc
        self.rightFrame = tk.Frame(self.frame_strona2, bg="lightgreen")
        self.rightFrame.grid(row=0, column=1, sticky="nsew")

        # Umieszczenie obrazu z kamery po środku prawego okna
        self.labelCamera = tk.Label(self.rightFrame, text="Testowy", bg='red')
        self.labelCamera.grid(row=0, column=0)
        self.rightFrame.grid_rowconfigure(0, weight=1)
        self.rightFrame.grid_columnconfigure(0, weight=1)

        # Obraz z kamery
        self.camera_width = 9 * self.app_width // 10
        self.camera_height = 9 * self.app_height // 10
        self.empty_image = Image.new("RGB", (3 * self.camera_width // 4, self.camera_height),
                                     "green")  # Tworzenie pustego obrazu

        # Dodaj 6 wierszy do kolumny
        for i in range(6):
            row_frame = tk.Frame(self.leftFrame, bg="yellow", height=50)
            row_frame.pack(fill="both", expand=True)

        self.add_labels()
        self.add_buttons()

    def add_labels(self):

        # Label z nazwa pozycji
        row = self.leftFrame.winfo_children()[0]
        label = tk.Label(row, text="Nazwa Pozycji", font=("Helvetica", 30, 'bold'), fg='blue', bg='yellow')
        label.pack(fill="both", expand=True)

        row = self.leftFrame.winfo_children()[1]

        # Create an object of tkinter ImageTk
        self.img = ImageTk.PhotoImage(Image.open("position.png"))

        # Create a Label Widget to display the text or Image
        label = tk.Label(row, image=self.img)
        label.pack()


        # Label z liczba pozycji
        row = self.leftFrame.winfo_children()[2]
        label = tk.Label(row, text="-/-", font=("Helvetica", 50, 'bold'), fg='blue', bg='yellow')
        label.pack()

        # Label z czasem
        row = self.leftFrame.winfo_children()[3]
        label = tk.Label(row, text="0:00", font=("Helvetica", 50, 'bold'), fg='blue', bg='yellow')
        label.pack()

    def add_buttons(self):

        rowButtons = self.leftFrame.winfo_children()[4]

        print(rowButtons)

        rowButtons = tk.Frame(rowButtons, bg="white")
        rowButtons.pack(fill="both", expand=True)

        self.start_button = tk.Button(rowButtons, text="Start", relief="groove", width=10, command=self.start_capture)
        self.start_button.grid(row=0, column=0, sticky="nsew")
        rowButtons.grid_rowconfigure(0, weight=1)  # Ustawienie wagi wiersza, aby zajmować dostępną przestrzeń pionową

        self.stop_button = tk.Button(rowButtons, text="Stop", relief="groove", width=10, command=self.stop_capture)
        self.stop_button.grid(row=0, column=1, sticky="nsew")
        self.stop_button["state"] = "disabled"
        rowButtons.grid_columnconfigure(0, weight=1)  # Ustawienie wagi kolumny, aby zajmować dostępną przestrzeń poziomą

        rowButtons.grid_columnconfigure(1, weight=1)
        rowButtons.grid_rowconfigure(0, weight=1)

        rowButtons = self.leftFrame.winfo_children()[5]
        rowButtons = tk.Frame(rowButtons, bg="white")
        rowButtons.pack(fill="both", expand=True)

        # przycisk odpowiedzialny za przejscie do menu
        self.menu_button = tk.Button(rowButtons, text="Menu", relief="groove", width=10, command=self.ukryj)
        self.menu_button.grid(row=0, column=0, sticky="nsew")

        # przycisk odpowiedzialny za zakończenie sesji cwiczen
        self.zakoncz_button = tk.Button(rowButtons, text="Zakończ sesję", relief="groove", width=10, command=self.zakoncz_sesje)
        self.zakoncz_button.grid(row=0, column=1, sticky="nsew")

        rowButtons.grid_rowconfigure(0, weight=1)  # Ustawienie wagi wiersza, aby zajmować dostępną przestrzeń pionową
        rowButtons.grid_columnconfigure(0, weight=1)  # Ustawienie wagi kolumny, aby zajmować dostępną przestrzeń poziomą
        rowButtons.grid_columnconfigure(1, weight=1)

    def zakoncz_sesje(self):
        """ Funckja odpowiedzialna za zakonczenie sejji ćwiczeń i zapis historii do pliku"""

        if self.session_begin is True:
            self.menu_button["state"] = "normal"
            self.stop_button["state"] = "disable"
            self.start_button["state"] = "normal"
            self.zakoncz_button["state"] = "disable"
            self.number_of_actual_position = -1
            self.session_begin = False

        czas_sesji = time.time() - self.time_session_begin
        hours = int(czas_sesji // 3600)
        minutes = int(czas_sesji // 60)
        seconds = int(czas_sesji % 60)

        historia_cwiczen = {
            'liczba_cwiczen': len(self.performed_exercises),
            'liczba_wszystkich_cwiczen': self.number_positions_to_do,
            'lista_cwiczen': self.performed_exercises,
            'czas_sesji': f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        }
        self.zapisz_do_pliku(historia=historia_cwiczen)

    def strona_modify(self):
        self.frame_strona3 = tk.Frame(self.master)
        # self.frame_strona3.grid(row=0, column=0)

        # Tworzenie nagłówka
        self.label_modify = tk.Label(self.frame_strona3, text="Modyfikuj sesję ćwiczeń", font=("Helvetica", 40))
        self.label_modify.grid(row=0, column=0, columnspan=3, pady=20)

        # Tworzenie przycisków typu Checkbutton
        self.check_var_list = []
        self.checkbuttons_list = []

        self.new_slownik = {}

        # Uzyskaj listę wartości, pomijając wartość 'None'
        values_list = [value for value in self.slownik.values() if value != 'None']

        for i in range(0, len(values_list)):
            check_var = tk.IntVar(value=1)  # Ustawienie wartości na 1, czyli zaznaczone
            self.check_var_list.append(check_var)

            check_button = tk.Checkbutton(self.frame_strona3, text=values_list[i], variable=check_var,
                                          font=("Helvetica", 16), command=lambda i=i: self.update_selected_options(i))
            self.checkbuttons_list.append(check_button)
            check_button.grid(row=i+1, column=0, pady=10, sticky='w')

            self.new_slownik[values_list[i-1]] = self.check_var_list[i].get()


        # Dodaj przycisk pod checkboxami
        self.button_modify = tk.Button(self.frame_strona3, text="Menu", command=self.ukryj_modify,
                                          font=("Helvetica", 14), relief="groove", width=40, height=4)
        self.button_modify.grid(row=len(values_list)+2, column=0,  sticky='e')


    def strona_historia(self):
        self.frame_strona4 = tk.Frame(self.master)
        # self.frame_strona4.grid(row=0, column=0, sticky='n')

        # Tworzenie nagłówka
        self.label_history = tk.Label(self.frame_strona4, text="Historia ćwiczeń", font=("Helvetica", 40))
        self.label_history.grid(row=0, column=0, columnspan=3, pady=30, sticky='s')

        # Utwórz Treeview z nagłówkami kolumn
        # self.tree = ttk.Treeview(self.frame_strona4, columns=('ID', 'Imię', 'Nazwisko'), height=20)
        self.tree = ttk.Treeview(self.frame_strona4, columns=('Data', 'Czas trwania sesji', 'Liczba wykonanych pozycji',
                                                              'Liczba zaplanowanych pozycji'), height=20)

        # Dodaj nagłówki kolumn
        self.tree.heading('#0', text='Nr')
        self.tree.heading('Data', text='Data')
        self.tree.heading('Czas trwania sesji', text='Czas trwania sesji')
        self.tree.heading('Liczba wykonanych pozycji', text='Liczba wykonanych pozycji')
        self.tree.heading('Liczba zaplanowanych pozycji', text='Liczba zaplanowanych pozycji')
        # self.tree.heading('Lista wykonanych pozcyji', text='Lista wykonanych pozcyji')


        # Dostosuj styl tekstu w Treeview
        style = ttk.Style()
        style.configure('Treeview.Heading', font=('Helvetica', 12), relief='solid')  # Ustaw nagłówki z większą czcionką
        style.configure('Treeview', font=('Helvetica', 14), rowheight=30, relief='solid')  # Ustaw tekst komórek z większą czcionką

        self.tree.column('#0', width=50, anchor='center')  # Ustaw szerokość kolumny
        self.tree.column('Data', width=200, anchor='center')  # Ustaw szerokość kolumny
        self.tree.column('Czas trwania sesji', width=150, anchor='center')  # Ustaw szerokość kolumny
        self.tree.column('Liczba wykonanych pozycji', width=200, anchor='center')  # Ustaw szerokość kolumny
        self.tree.column('Liczba zaplanowanych pozycji', width=220, anchor='center')  # Ustaw szerokość kolumny

        # Dodaj pionowy scrollbar
        scrollbar_y = ttk.Scrollbar(self.frame_strona4, orient='vertical', command=self.tree.yview)
        scrollbar_y.grid(row=1, column=3, sticky='ns')
        self.tree.configure(yscroll=scrollbar_y.set)

        # Dodaj  dane
        self.dodaj_dane()

        self.tree.grid(row=1, column=0, columnspan=3, sticky='nsew')

        # # Dodaj przycisk powrotu do menu
        # self.btn_powrot = tk.Button(self.frame_strona4, text="Powrót do menu", command=self.powrot_do_menu)
        # self.btn_powrot.grid(row=2, column=0, columnspan=3, pady=10)

        # Dodaj przycisk pod checkboxami
        self.button_history = tk.Button(self.frame_strona4, text="Menu", command=self.ukryj_historie,
                                          font=("Helvetica", 14), relief="groove", width=40, height=4)
        self.button_history.grid(row=2, column=0,  columnspan=3, pady=60)

        # Konfiguracja rozszerzania kolumn i wierszy
        self.frame_strona4.grid_columnconfigure(0, weight=1)
        self.frame_strona4.grid_rowconfigure(1, weight=1)

    def dodaj_dane(self):
        """Pobranie danych o historii wykonanych sesji z plikow z folderu /historia"""

        self.clear_tree_values()
        # Nazwa folderu
        folder_name = "historia"
        # Tworzenie pełnej ścieżki
        folder_path = os.path.join(os.getcwd(), folder_name)

        # Sprawdź, czy folder istnieje
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Pobierz listę plików w folderze
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            file_count = 0

            for file_name in file_list:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    file_count += 1
                    data = json.load(file)
                    # Pobierz pozostałe informacje
                    liczba_cwiczen = data.get("liczba_cwiczen")
                    liczba_wszystkich_cwiczen = data.get("liczba_wszystkich_cwiczen")
                    # lista_cwiczen = data.get("lista_cwiczen")
                    czas_sesji = data.get("czas_sesji")

                    # Uzyskaj nazwę pliku bez rozszerzenia
                    time = os.path.splitext(os.path.basename(file_name))[0]

                    input_format = "%Y-%m-%d_%H-%M-%S"
                    output_format = "%d.%m.%Y %H:%M:%S"

                    # Parsowanie daty z wejściowego stringa
                    date_object = datetime.strptime(time, input_format)

                    # Formatowanie daty do żądanego formatu
                    time = date_object.strftime(output_format)

                    self.tree.insert('', 'end', text=str(file_count), values=(time, czas_sesji, liczba_cwiczen, liczba_wszystkich_cwiczen))
    def clear_tree_values(self):
        """Usuwa wszystkie wartosci z historii"""
        # Usuń wszystkie elementy (rzędy) z drzewa
        for item in self.tree.get_children():
            self.tree.delete(item)

    # uaktualnienie slownika odpowiedzialnego za pozycje do wykonania
    def update_selected_options(self, i):
        check_var = self.check_var_list[i]
        self.new_slownik[self.checkbuttons_list[i].cget("text")] = check_var.get()



    # module for upload video from directory
    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.cap = cv2.VideoCapture(file_path)

    # module for start video
    def start_capture(self):
        self.video_running = True
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"

        self.zakoncz_button["state"] = "normal"
        self.menu_button["state"] = "disabled"

        # Start a new thread to read and display video frames continuously
        threading.Thread(target=self.video_detection).start()

        if self.session_begin is False:
            self.time_session_begin = time.time()
            self.next_position()
            self.session_begin = True


    # module for stop or pause video
    def stop_capture(self):
        self.video_running = False
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"

    def ukryj(self):
        """Przejscie z widoku sesji do menu"""
        self.frame_strona1.grid_forget()
        self.frame_strona2.grid_forget()
        self.frame_strona1.grid(row=0, column=0)

    def ukryj_menu(self):
        """Przejscie z widoku menu do sesji"""
        self.frame_strona2.grid_forget()
        self.frame_strona1.grid_forget()
        self.frame_strona2.grid(row=0, column=0, sticky="nsew")

    def ukryj_modify(self):
        """Przejscie z widoku modyfikacji sesji do menu"""
        self.positions_to_do()
        self.frame_strona3.grid_forget()
        self.frame_strona1.grid_forget()
        self.frame_strona1.grid(row=0, column=0)

    def modyfikuj_sesje(self):
        """Przejscie z widoku menu do modyfikacji sesji"""
        self.frame_strona3.grid_forget()
        self.frame_strona1.grid_forget()
        self.frame_strona3.grid(row=0, column=0)

    def ukryj_historie(self):
        """Przejscie z widoku historii do menu"""
        self.frame_strona4.grid_forget()
        self.frame_strona1.grid_forget()
        self.frame_strona1.grid(row=0, column=0)

    def pokaz_historie(self):
        """Przejscie z widoku menu do historii"""
        self.frame_strona4.grid_forget()
        self.frame_strona1.grid_forget()
        self.frame_strona4.grid(row=0, column=0, sticky='n')

    def positions_to_do(self):
        """Lista pozycji do wykonania przez uzytkownika w czasie seesji"""

        # list_of_positions = []
        self.list_of_positions.clear()

        for key, value in self.new_slownik.items():
            if value == 1:
                for key2, value2 in self.slownik.items():
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

        self.leftFrame.winfo_children()[3].winfo_children()[0].config(text=f"{minuty:02d}:{sekundy:02d}")

        # przejscie do nastepnej pozycji
        if int(self.timer) > self.all_time:
            self.next_position()

    def next_position(self):
        self.timer = 0
        self.number_of_actual_position += 1

        # stwierdzenie faktu wykonania cwiczenia
        if self.number_of_actual_position > 0:
            self.performed_exercises.append(self.name_of_actual_position)

        # sprawdzenie czy zostalo wykonane osatnie cwiczenie
        if self.number_of_actual_position <= self.number_positions_to_do:
            self.name_of_actual_position = self.slownik.get(self.list_of_positions[self.number_of_actual_position])

            # nazwa pozycji
            # Uzyskaj dostęp do etykiety
            label = self.leftFrame.winfo_children()[0].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text=self.name_of_actual_position)

            # numer pozycji
            label = self.leftFrame.winfo_children()[2].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text=str(self.number_of_actual_position + 1) + '/' + str(self.number_positions_to_do))

            # czas pozycji
            label = self.leftFrame.winfo_children()[3].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text="00:00")
        else:
            self.stop_capture()
            self.number_of_actual_position = -1
            # nazwa pozycji
            # Uzyskaj dostęp do etykiety
            label = self.leftFrame.winfo_children()[0].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text="Nazwa pozycji")

            # numer pozycji
            label = self.leftFrame.winfo_children()[2].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text="-/-")

            # czas pozycji
            label = self.leftFrame.winfo_children()[3].winfo_children()[0]
            # Zmiana tekstu w etykiecie
            label.config(text="00:00")

    def zapisz_do_pliku(self, historia):
        if not os.path.exists("historia"):
            os.makedirs("historia")

        teraz = datetime.now()
        nazwa_pliku = f"{teraz.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        # nazwa_pliku = f"{teraz.strftime('%d-%m-%Y_%H-%M-%S')}.json"
        sciezka_pliku = os.path.join("historia", nazwa_pliku)

        with open(sciezka_pliku, 'w') as plik:
            json.dump(historia, plik, indent=2)

        # Uaktualnienie danych w historii
        self.dodaj_dane()

    def get_max_value_index(self, vector):
        max_index = np.argmax(vector)
        print(max_index)
        return max_index

    def get_name_position(self, vector):

        # tmp = np.max(vector)
        # Progowanie pewnosci
        if np.max(vector) > self.prog:
            # key = get_max_value_index(vector)
            key = np.argmax(vector)
            name = self.slownik.get(key)
        else:
            name = 'None'

        print(name)
        return name

    def flatten_landmarks(self,pose_landmarks_list):
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
            results = self.pose.process(image)

            try:

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

            # Exit if the user presses the 'q' key
        if self.video_running:
            self.master.after(10, self.video_detection)

        # Release the webcam and close the window

    # def next_positions(self):
    #     for pos in self.positions_to_do():
    #         self.video_detection()


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
