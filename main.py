import cv2
import math
import tkinter as tk
from tkinter import *
import mediapipe as mp
from PIL import Image, ImageTk
import threading
from tkinter import filedialog
import json
from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import mediapipe as mp
import numpy as np
from keras.models import load_model
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time




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


        self.name_of_actual_position = None
        self.number_of_actual_position = -1
        self.list_of_positions = self.positions_to_do()
        self.number_positions_to_do = len(self.list_of_positions)
        self.timer = 0
        self.last_time = 0
        self.current_time = 0
        self.time_start = 0
        self.all_time = 5
        self.flag = 0




        self.cap = cv2.VideoCapture(video_path)
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                                  min_tracking_confidence=min_tracking_confidence)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.app_width = 4*screen_width//5
        self.app_height = 4*screen_height//5

        self.master.geometry(f"{self.app_width}x{self.app_height}")

        # Podzielenie glownego okna na 2 czesci
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=3, minsize=300)
        self.master.grid_columnconfigure(1, weight=7, minsize=700)

        # Lewa czesc
        self.leftFrame = tk.Frame(master, bg="lightblue")
        self.leftFrame.grid(row=0, column=0, sticky="nsew")

        #prawa czesc
        self.rightFrame = tk.Frame(master, bg="lightgreen")
        self.rightFrame.grid(row=0, column=1, sticky="nsew")

        # Umieszczenie obrazu z kamery po środku prawego okna
        self.labelCamera = tk.Label(self.rightFrame, text="Testowy", bg='red')
        self.labelCamera.grid(row=0, column=0)
        self.rightFrame.grid_rowconfigure(0, weight=1)
        self.rightFrame.grid_columnconfigure(0, weight=1)

        # Obraz z kamery
        self.camera_width = 9 * self.app_width // 10
        self.camera_height = 9 * self.app_height // 10
        self.empty_image = Image.new("RGB", (3 * self.camera_width // 4, self.camera_height), "green")  # Tworzenie pustego obrazu

        # Dodaj 6 wierszy do kolumny
        for i in range(6):
            row_frame = tk.Frame(self.leftFrame, bg="yellow", height=50)
            row_frame.pack(fill="both", expand=True)

        self.add_labels()
        self.add_buttons()

        self.DabMove_image = None
        self.video_running = False

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

        rowButtons = self.leftFrame.winfo_children()[5]

        print(rowButtons)

        rowButtons = tk.Frame(rowButtons, bg="white")
        rowButtons.pack(fill="both", expand=True)

        self.start_button = tk.Button(rowButtons, text="Start", width=10, command=self.start_capture)
        self.start_button.grid(row=0, column=0, sticky="nsew")
        rowButtons.grid_rowconfigure(0, weight=1)  # Ustawienie wagi wiersza, aby zajmować dostępną przestrzeń pionową

        self.stop_button = tk.Button(rowButtons, text="Stop", width=10, command=self.stop_capture)
        self.stop_button.grid(row=0, column=1, sticky="nsew")
        self.stop_button["state"] = "disabled"
        rowButtons.grid_columnconfigure(0, weight=1)  # Ustawienie wagi kolumny, aby zajmować dostępną przestrzeń poziomą

        rowButtons.grid_columnconfigure(1, weight=1)
        rowButtons.grid_rowconfigure(0, weight=1)


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
        # Start a new thread to read and display video frames continuously
        threading.Thread(target=self.video_detection).start()

        self.next_position()


    # module for stop or pause video
    def stop_capture(self):
        self.video_running = False
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"


    def positions_to_do(self):
        list_of_positions = [0, 1]
        return list_of_positions

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
    root.configure(background='black')
    root.mainloop()