import numpy
import streamlit as st
import cv2
import pickle
import face_recognition


# Load the face's dataset
with open('face_encodings.pkl', 'rb') as file:
    known_face_encodings = pickle.load(file)

with open('face_names.pkl', 'rb') as file:
    known_face_names = pickle.load(file)
    
    
#def the css and applying it 
def local_css(style):
    with open(style) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True )

local_css("style.css")

st.title("New Face Recognition App!")
run = st.button("Open Camera")

#variables for the cam 
frame_window = st.image([])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.write("Camera not working.")
            break

        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face detection
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = numpy.argmin(distances)
            
            
            
            if distances[best_match_index] < 0.8:
                name = known_face_names[best_match_index]
            else:
                name = "unknown"

            # Drawing a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Drawing a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 3, (255, 255, 255), 2)

        # Display the resulting frame
        frame_window.image(frame, channels='BGR')

    cap.release()

# # Outside the main loop
# cv2.destroyAllWindows()
