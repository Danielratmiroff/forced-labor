from winsound import PlaySound
from gtts import gTTS
import face_recognition
import time
import os
import sys
import cv2
import numpy as np
import math


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + "%"


class FaceRecognition:
    face_recognition = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    sound_dir = 'sounds'

    def __init__(self):
        print('Encoding faces...')
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(f'Images: {self.known_face_names}')

    def play_hello_sound(self):
        print('Saying hello...')
        PlaySound(f'{self.sound_dir}/hello.wav', 0)

    def play_bye_sound(self):
        print('Saying goodbye...')
        PlaySound(f'{self.sound_dir}/seeyousoon.wav', 0)
        time.sleep(5)
        self.run_recognition()

    def run_recognition(self):
        print('Starting face recognition...')
        video_capture = cv2.VideoCapture(0)

        # Video optimizations
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
        video_capture.set(cv2.CAP_PROP_FPS, 30)

        if not video_capture.isOpened():
            sys.exit('Video source not found')

        pomodoro_timer = time.time()
        self.found_face = False
        self.greetings = False

        while True:
            ret, frame = video_capture.read()

            if not ret:
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.process_current_frame:

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

                if len(self.face_names) > 0 and self.found_face == False:
                    print('Face detected')
                    self.found_face = True

                    if self.greetings == False:
                        self.greetings = True
                        self.play_hello_sound()
                # Face is no longer detected
                elif len(self.face_names) == 0 and time.time() - pomodoro_timer > 5 and self.found_face == True:
                    print('Face not detected')
                    self.greetings = False
                    self.found_face = False
                    self.play_bye_sound()
                    break

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

            cv2.imshow('Face recognition', rgb_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
