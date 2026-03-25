import os
import re
from datetime import date, datetime

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, template_folder='templates')

NIMGS = 12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


def attendance_file_path():
    return os.path.join(ATTENDANCE_DIR, f"Attendance-{datetoday()}.csv")


def ensure_directories():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(FACES_DIR, exist_ok=True)

    csv_path = attendance_file_path()
    if not os.path.isfile(csv_path):
        pd.DataFrame(columns=["Name", "Roll", "Time"]).to_csv(csv_path, index=False)


def sanitize_text(value):
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_-]", "", value)
    return value


def totalreg():
    ensure_directories()
    return len([name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))])


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    except Exception:
        return []


def identify_face(facearray):
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    ensure_directories()
    faces = []
    labels = []

    for user in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    if not faces:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, MODEL_PATH)
    return True


def extract_attendance():
    ensure_directories()
    csv_path = attendance_file_path()
    df = pd.read_csv(csv_path)
    if df.empty:
        return [], [], [], 0

    names = df["Name"].tolist() if "Name" in df.columns else []
    rolls = df["Roll"].tolist() if "Roll" in df.columns else []
    times = df["Time"].tolist() if "Time" in df.columns else []
    return names, rolls, times, len(df)


def add_attendance(label):
    ensure_directories()
    if "_" not in label:
        return

    username, userid = label.rsplit("_", 1)
    current_time = datetime.now().strftime("%H:%M:%S")

    csv_path = attendance_file_path()
    df = pd.read_csv(csv_path)

    existing_rolls = set(df["Roll"].astype(str).tolist()) if not df.empty and "Roll" in df.columns else set()
    if str(userid) in existing_rolls:
        return

    new_row = pd.DataFrame([{"Name": username, "Roll": userid, "Time": current_time}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def getallusers():
    ensure_directories()
    userlist = [name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))]
    names = []
    rolls = []

    for user in userlist:
        if "_" in user:
            name, roll = user.rsplit("_", 1)
        else:
            name, roll = user, ""
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, len(userlist)


def deletefolder(folder_path):
    if not os.path.isdir(folder_path):
        return

    for file_name in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file_name))
    os.rmdir(folder_path)


def render_home(mess=""):
    names, rolls, times, l = extract_attendance()
    return render_template(
        "home.html",
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2(),
        mess=mess,
    )


@app.route("/")
def home():
    return render_home()


@app.route("/listusers")
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template(
        "listusers.html",
        userlist=userlist,
        names=names,
        rolls=rolls,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2(),
    )


@app.route("/deleteuser", methods=["GET"])
def deleteuser():
    duser = request.args.get("user", "")
    if duser:
        deletefolder(os.path.join(FACES_DIR, duser))

    train_model()
    return redirect(url_for("listusers"))


@app.route("/start", methods=["GET"])
def start():
    if not os.path.exists(MODEL_PATH):
        return render_home("No trained model found. Please add a new user first.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_home("Cannot access webcam. Check camera permissions.")

    cv2.namedWindow("Attendance Capture - Press ESC to Stop", cv2.WINDOW_NORMAL)
    recognized_person = ""
    max_frames = 180
    frames_checked = 0
    try:
        while frames_checked < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (86, 32, 251), -1)

                face = cv2.resize(frame[y : y + h, x : x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                recognized_person = identified_person

                cv2.putText(
                    frame,
                    identified_person,
                    (x + 5, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Attendance marked. Closing camera...",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Attendance Capture - Press ESC to Stop", frame)
                cv2.waitKey(700)
                break
            else:
                cv2.putText(
                    frame,
                    "Looking for face... Keep your face in frame",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Attendance Capture - Press ESC to Stop", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            frames_checked += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if recognized_person:
        display_name = recognized_person.rsplit("_", 1)[0] if "_" in recognized_person else recognized_person
        return render_home(f"Attendance marked for {display_name}. Camera closed automatically.")

    return render_home("No face detected for attendance. Please try again.")


@app.route("/add", methods=["POST"])
def add():
    newusername = sanitize_text(request.form.get("newusername", ""))
    newuserid = sanitize_text(request.form.get("newuserid", ""))

    if not newusername or not newuserid:
        return render_home("Name and ID are required.")

    user_folder = f"{newusername}_{newuserid}"
    userimagefolder = os.path.join(FACES_DIR, user_folder)
    os.makedirs(userimagefolder, exist_ok=True)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_home("Cannot access webcam. Check camera permissions.")

    cv2.namedWindow(f"Adding New User: {newusername} (ID: {newuserid}) - Press ESC to stop", cv2.WINDOW_NORMAL)
    try:
        while i < NIMGS:
            ret, frame = cap.read()
            if not ret:
                break

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(
                    frame,
                    f"Captured: {i}/{NIMGS}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if j % 5 == 0 and i < NIMGS:
                    file_name = f"{newusername}_{i}.jpg"
                    face_crop = frame[y : y + h, x : x + w]
                    resized = cv2.resize(face_crop, (256, 256))
                    cv2.imwrite(os.path.join(userimagefolder, file_name), resized)
                    i += 1
                j += 1

            cv2.imshow(f"Adding New User: {newusername} (ID: {newuserid}) - Press ESC to stop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or i >= NIMGS:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if i == 0:
        return render_home("No face captured. Please try again.")

    msg = f"Captured {i} face samples. Training model..."
    train_model()
    return render_home(f"User {newusername} (ID: {newuserid}) added successfully!")


if __name__ == "__main__":
    ensure_directories()
    print("\n" + "="*60)
    print("  Attendance System")
    print("="*60)
    print("\n  Starting Flask server...")
    print("  Open your browser and go to: http://127.0.0.1:5000")
    print("\n  When capturing faces, a webcam window will open.")
    print("  Press ESC in the webcam window to stop capturing.\n")
    print("="*60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
