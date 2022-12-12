from winsound import PlaySound
import face_recognition
import time
import os
import sys
import cv2
import numpy as np
import math
import pyttsx3


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"


def calculate_time(time_passed):
    minutes = time_passed / 60
    hours = minutes / 60

    if hours > 1:
        return f'{round(hours)} hours'
    elif minutes > 1:
        return f'{round(minutes)} minutes'
    else:
        return f'{round(time_passed)} seconds'


def load_photos():
    print('Loading photos...')

    face_photos = len(os.listdir('faces'))
    if face_photos >= 5:
        return

    print('Not enough photos, taking new ones...')

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        sys.exit('Video source not found')

    while face_photos < 5:
        ret, frame = video_capture.read()
        if not ret:
            sys.exit('Could not read frame')

        cv2.putText(frame, 'Press "S" to take photo', (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        photo_text = f'Photos: {face_photos}'
        cv2.putText(frame, photo_text, (150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Photoshoot - Press "q" to Quit', frame)

        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite(f'faces/daniel{face_photos}.jpg', frame)
            face_photos = len(os.listdir('faces'))

        if cv2.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


class FaceRecognition:
    face_recognition = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        print('Encoding faces...')
        self.encode_faces()
        self.init_speech_engine()

    def init_speech_engine(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('language', 'en')
        self.engine.setProperty('rate', '80')

    def encode_faces(self):
        for image in os.listdir('faces'):
            print(f'> {image}')
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(f'Total photos: {len(self.known_face_names)}')

    def run_recognition(self):
        sound_dir = 'sounds/'
        pomodoro_timer = 0
        found_face = False
        greetings = False

        print('Starting face recognition...')
        video_capture = cv2.VideoCapture(0)

        # Video optimizations
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
        video_capture.set(cv2.CAP_PROP_FPS, 30)

        if not video_capture.isOpened():
            sys.exit('Video source not found')

        while True:
            ret, frame = video_capture.read()

            if not ret:
                sys.exit('Could not read frame')

            if self.process_current_frame:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Find all faces
                self.face_locations = face_recognition.face_locations(
                    rgb_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_frame, self.face_locations)

                self.face_names = []

                # Append names to faces while a face is detected
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

                time_passed = time.time() - pomodoro_timer

                # Face is detected
                if len(self.face_names) > 0 and found_face == False:
                    print('Face detected')
                    found_face = True
                    pomodoro_timer = time.time()

                    if greetings == False:
                        greetings = True
                        PlaySound(f'{sound_dir}hello.wav', 0)

                # Face is no longer detected
                elif len(self.face_names) == 0 and time_passed > 5 and found_face == True:
                    print('Face not detected')
                    greetings = False
                    found_face = False
                    pomodoro_timer = 0

                    PlaySound(f'{sound_dir}seeyousoon.wav', 0)

                    self.engine.say(
                        f"Time in computer {calculate_time(time_passed)}")
                    self.engine.runAndWait()

                    time.sleep(5)

            # Only process every second frame
            self.process_current_frame = not self.process_current_frame

            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Draw the rectangle on the frame
                cv2.rectangle(rgb_frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(rgb_frame, (left, bottom - 15),
                              (right, bottom), (0, 0, 255), -1)
                cv2.putText(rgb_frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow('Face recognition - Press "q" to Quit', rgb_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    load_photos()
    fr = FaceRecognition()
    fr.run_recognition()
