from tkinter import *
from tkinter import messagebox, ttk
import cv2
import PIL.Image
import PIL.ImageTk
from keras.models import load_model
import time
import numpy as np
from collections import deque

class SignLanguageDetector:

    def __init__(self):
        
        # Main window
        
        self.window = Tk()
        self.window.title("Sign Language Detector")
        self.window.geometry("1250x750")
        self.window.configure(bg="#F5F5DC")  # beige background
        self.window.resizable(False, False)

        
        # Top frame for video and hand preview
        
        self.top_frame = Frame(self.window, bg="#F5F5DC")
        self.top_frame.pack(pady=10)

        # Canvas sizes
        self.canvas_width = 450
        self.canvas_height = 400

        # Left canvas - main video with border
        self.canvas_video = Canvas(self.top_frame, width=self.canvas_width, height=self.canvas_height, bg="black", bd=3, relief="ridge")
        self.canvas_video.pack(side=LEFT, padx=10)

        # Middle - word box for saved letters
        self.word_box_frame = Frame(self.top_frame, bg="#F5F5DC", bd=2, relief="ridge")
        self.word_box_frame.pack(side=LEFT, padx=10)
        Label(self.word_box_frame, text="Letters Saved", font=("Arial", 14), bg="#F5F5DC").pack(pady=5)
        self.word_box = Text(self.word_box_frame, height=10, width=20, font=("Arial", 16), bg="#FFF8DC")
        self.word_box.pack()
        self.word_box.config(state=DISABLED)

        # Right canvas - hand preview + gesture label inside with border
        self.canvas_hand = Canvas(self.top_frame, width=self.canvas_width, height=self.canvas_height, bg="black", bd=3, relief="ridge")
        self.canvas_hand.pack(side=LEFT, padx=10)
        self.gesture_label = self.canvas_hand.create_text(self.canvas_width//2, self.canvas_height-30,
                                                         text="None", fill="yellow",
                                                         font=("Arial", 40, "bold"))

        
        # Buttons frame
        
        self.button_frame = Frame(self.window, bg="#F5F5DC")
        self.button_frame.pack(pady=10)

        # Create styled buttons with hover effects
        self.btn_start = Button(self.button_frame, text="Start", font=("Arial", 14, "bold"),
                                bg="#3E8E41", fg="white", bd=0, padx=25, pady=10,
                                command=self.on_start, relief="raised")
        self.btn_start.grid(row=0, column=0, padx=20)
        self.add_hover(self.btn_start, "#4CAF50", "#3E8E41")

        self.btn_adjust = Button(self.button_frame, text="Adjust", font=("Arial", 14, "bold"),
                                 bg="#1E90FF", fg="white", bd=0, padx=25, pady=10,
                                 command=self.on_adjust, relief="raised")
        self.btn_adjust.grid(row=0, column=1, padx=20)
        self.add_hover(self.btn_adjust, "#63B8FF", "#1E90FF")

        self.btn_stop = Button(self.button_frame, text="Stop", font=("Arial", 14, "bold"),
                               bg="#FF4500", fg="white", bd=0, padx=25, pady=10,
                               command=self.on_stop, relief="raised")
        self.btn_stop.grid(row=0, column=2, padx=20)
        self.add_hover(self.btn_stop, "#FF6347", "#FF4500")

        self.btn_reset = Button(self.button_frame, text="Reset Letters", font=("Arial", 14, "bold"),
                                bg="#FFA500", fg="white", bd=0, padx=25, pady=10,
                                command=self.reset_letters, relief="raised")
        self.btn_reset.grid(row=0, column=3, padx=20)
        self.add_hover(self.btn_reset, "#FFB84D", "#FFA500")

        
        # Footer
        
        Label(self.window, text="Made by Arun", font=("Arial", 10), bg="#F5F5DC").pack(side=BOTTOM, pady=5)

        
        # Video capture and model
        
        self.vid = cv2.VideoCapture(0)
        self.bg = None
        self.aWeight = 0.5
        self.num_frames = 0
        self.run_once = 0

        self.classifier = load_model('Trained_model.h5')

        
        # Recognized letters logic
       
        self.recognized_text = []
        self.last_letter = None
        self.letter_start_time = None
        self.save_delay = 4
        self.vote_queue = deque(maxlen=10)
        self.photo_threshold = None

        # ROI coordinates
        self.roi_top = 50
        self.roi_bottom = 350
        self.roi_left = 100
        self.roi_right = 400

    
    # Hover effect helper
    
    def add_hover(self, button, color_on_hover, color_default):
        button.bind("<Enter>", lambda e: button.config(bg=color_on_hover))
        button.bind("<Leave>", lambda e: button.config(bg=color_default))

   
    # Running average
    
    def run_avg(self, image):
        if image is None:
            return
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.aWeight)

    
    # Segment hand
    
    def segment(self, image, threshold=25):
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)
        thresholded = cv2.erode(thresholded, kernel, iterations=2)
        thresholded = cv2.dilate(thresholded, kernel, iterations=2)
        thresholded = cv2.medianBlur(thresholded, 5)

        cnts = cv2.findContours(thresholded.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if len(cnts) == 0:
            return
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented

   
    # Predict gesture
   
    def predictor(self):
        from keras.utils.image_utils import load_img, img_to_array
        try:
            test_image = load_img('1.png', target_size=(64, 64))
        except:
            return None
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.classifier.predict(test_image)
        for idx, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            if result[0][idx] == 1:
                return letter
        return None

   
    # Start scanning
    
    def on_start(self):
        if self.run_once == 0:
            self.delay = 10
            self.update()
            self.run_once += 1

    
    # Stop
    
    def on_stop(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.vid.release()
            cv2.destroyAllWindows()
            self.window.destroy()

   
    # Reset background
    
    def on_adjust(self):
        self.vid.release()
        self.num_frames = 0
        self.bg = None
        self.vid = cv2.VideoCapture(0)

    
    # Reset letters
    
    def reset_letters(self):
        self.recognized_text = []
        self.last_letter = None
        self.letter_start_time = None
        self.vote_queue.clear()
        self.update_word_box()

    
    # Update UI
   
    def update(self):
        ret, frame = self.vid.read()
        if not ret:
            self.window.after(10, self.update)
            return

        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        # Draw ROI rectangle
        cv2.rectangle(frame, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)

        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.num_frames < 30:
            self.run_avg(gray)
            self.canvas_hand.itemconfig(self.gesture_label, text="Loading...")
        else:
            hand = self.segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                cv2.imwrite('1.png', thresholded)

                # Resize thresholded image to fit right canvas
                img_resized = cv2.resize(thresholded, (self.canvas_width, self.canvas_height - 50))
                self.photo_threshold = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_resized))
                self.canvas_hand.create_image(0, 0, image=self.photo_threshold, anchor=NW)

                gesture = self.predictor()
                if gesture:
                    self.vote_queue.append(gesture)
                    most_common = max(set(self.vote_queue), key=self.vote_queue.count)
                    current_time = time.time()
                    if most_common == self.last_letter:
                        if self.letter_start_time and (current_time - self.letter_start_time) >= 4:
                            self.recognized_text.append(most_common)
                            self.letter_start_time = current_time
                    else:
                        self.last_letter = most_common
                        self.letter_start_time = current_time

                    self.canvas_hand.itemconfig(self.gesture_label, text=most_common)

                cv2.drawContours(clone, [segmented + (self.roi_left, self.roi_top)], -1, (0, 0, 255))

        self.num_frames += 1

        # Display main video
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo_video = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
        self.canvas_video.create_image(0, 0, image=self.photo_video, anchor=NW)

        self.window.after(self.delay, self.update)
        self.update_word_box()

    
    # Update word box
    
    def update_word_box(self):
        self.word_box.config(state=NORMAL)
        self.word_box.delete(1.0, END)
        self.word_box.insert(END, ''.join(self.recognized_text))
        self.word_box.config(state=DISABLED)

  
    # Run
    
    def run(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_stop)
        self.window.mainloop()



# Run the app
detector = SignLanguageDetector()
detector.run()
