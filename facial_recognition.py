import face_recognition
import cv2
import os

def load_faces(folder):
    encodings, names = [], []
    
    for file in os.listdir(folder):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(folder, file)
            image = face_recognition.load_image_file(path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                encodings.append(face_encoding)
                names.append(os.path.splitext(file)[0])
                print(f"[+] Loaded {file}")
            except IndexError:
                print(f"[!] No face found in {file}")
    
    return encodings, names

def recognize_faces(known_encodings, known_names):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[!] Webcam not accessible.")
        return
    print("[INFO] Press 'q' to quit.")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = face_distances.argmin()
                name = known_names[best_match_index]

            top, right, bottom, left = [coord * 4 for coord in (top, right, bottom, left)]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = "\images"
    encodings, names = load_faces(folder)
    
    if not encodings:
        print("[!] No faces loaded. Check your images folder.")
    else:
        print(f"[+] Loaded {len(encodings)} faces")
        recognize_faces(encodings, names)