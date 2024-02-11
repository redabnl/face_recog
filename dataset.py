import face_recognition
import os 
import pickle

#image directory
image_dir = r"C:\Users\surface\Desktop\face_recognition_app\images"

face_encod = []
face_name = []

#loop throu the image directory
for file_name in os.listdir(image_dir):
    #skip if ther's a nom .jpg .. file
    if not file_name.endswith(('.png', '.jpg', '.jpeg')):
        continue
    #load the images
    image_path= os.path.join(image_dir, file_name)
    image = face_recognition.load_image_file(image_path)
    
    #face encodings
    encodings = face_recognition.face_encodings(image)
    
    if encodings:
        face_encod.append(encodings[0])
        # Use file name (without extension) as the name
        face_name.append(os.path.splitext(file_name)[0])
        
with open('face_encodings.pkl', 'wb') as file :
    pickle.dump(face_encod, file)
    
with open('face_names.pkl','wb' ) as file:
    pickle.dump(face_name, file)
    