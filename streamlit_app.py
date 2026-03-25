import os
import re
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = ""
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(exc)

NIMGS = 12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if CV2_AVAILABLE else None


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


def extract_faces(img):
    if not CV2_AVAILABLE or face_detector is None:
        return []

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    except Exception:
        return []


def identify_face(facearray):
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    if not CV2_AVAILABLE:
        return False

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


def extract_attendance_df():
    ensure_directories()
    csv_path = attendance_file_path()
    return pd.read_csv(csv_path)


def add_attendance(label):
    ensure_directories()
    if "_" not in label:
        return False

    username, userid = label.rsplit("_", 1)
    current_time = datetime.now().strftime("%H:%M:%S")

    csv_path = attendance_file_path()
    df = pd.read_csv(csv_path)
    existing_rolls = set(df["Roll"].astype(str).tolist()) if not df.empty and "Roll" in df.columns else set()

    if str(userid) in existing_rolls:
        return False

    new_row = pd.DataFrame([{"Name": username, "Roll": userid, "Time": current_time}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return True


def get_all_users():
    ensure_directories()
    userlist = [name for name in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, name))]
    rows = []
    for user in sorted(userlist):
        if "_" in user:
            name, roll = user.rsplit("_", 1)
        else:
            name, roll = user, ""
        rows.append({"Folder": user, "Name": name, "ID": roll})
    return rows


def delete_user_folder(user_folder):
    folder_path = os.path.join(FACES_DIR, user_folder)
    if not os.path.isdir(folder_path):
        return False

    for file_name in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file_name))
    os.rmdir(folder_path)
    train_model()
    return True


def capture_new_user(newusername, newuserid):
    if not CV2_AVAILABLE:
        return False, f"OpenCV import failed in this environment: {CV2_IMPORT_ERROR}"

    user_folder = f"{newusername}_{newuserid}"
    userimagefolder = os.path.join(FACES_DIR, user_folder)
    os.makedirs(userimagefolder, exist_ok=True)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Cannot access webcam. Check camera permissions."

    window_name = f"Adding New User: {newusername} (ID: {newuserid}) - Press ESC to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or i >= NIMGS:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if i == 0:
        return False, "No face captured. Please try again."

    train_model()
    return True, f"User {newusername} (ID: {newuserid}) added with {i} face samples."


def capture_attendance_once():
    if not CV2_AVAILABLE:
        return False, f"OpenCV import failed in this environment: {CV2_IMPORT_ERROR}"

    if not os.path.exists(MODEL_PATH):
        return False, "No trained model found. Please add a user first."

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Cannot access webcam. Check camera permissions."

    window_name = "Attendance Capture - Press ESC to Stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
                cv2.imshow(window_name, frame)
                cv2.waitKey(700)
                break

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
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            frames_checked += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if recognized_person:
        display_name = recognized_person.rsplit("_", 1)[0] if "_" in recognized_person else recognized_person
        return True, f"Attendance marked for {display_name}."

    return False, "No face detected for attendance. Please try again."


def app_main():
    ensure_directories()
    st.set_page_config(page_title="Attendance System", layout="wide")
    st.title("Attendance System")
    st.caption(f"Date: {datetoday2()}")

    if not CV2_AVAILABLE:
        st.error("OpenCV failed to load in this deployment environment.")
        st.info("For Streamlit Cloud, use Python 3.11 and opencv-python-headless. Also note: webcam capture via cv2.VideoCapture(0) is local-machine only.")
        st.code(CV2_IMPORT_ERROR)

    attendance_df = extract_attendance_df()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Today Attendance")
        if attendance_df.empty:
            st.info("No attendance marked yet.")
        else:
            st.dataframe(attendance_df, use_container_width=True, hide_index=True)

        if st.button("Take Attendance", type="primary", use_container_width=True):
            ok, message = capture_attendance_once()
            if ok:
                st.success(message)
            else:
                st.warning(message)
            st.rerun()

    with col2:
        st.subheader("Add New User")
        with st.form("add_user_form"):
            newusername = st.text_input("Full Name")
            newuserid = st.text_input("User ID")
            submitted = st.form_submit_button("Capture Face Samples", use_container_width=True)

        if submitted:
            clean_name = sanitize_text(newusername)
            clean_id = sanitize_text(newuserid)
            if not clean_name or not clean_id:
                st.error("Name and ID are required.")
            else:
                ok, message = capture_new_user(clean_name, clean_id)
                if ok:
                    st.success(message)
                else:
                    st.warning(message)
                st.rerun()

        st.caption("Webcam opens for capture. Press ESC to stop early.")

    st.subheader("Manage Users")
    users = get_all_users()
    st.write(f"Total Users: {len(users)}")

    if users:
        users_df = pd.DataFrame(users)
        st.dataframe(users_df[["Name", "ID", "Folder"]], use_container_width=True, hide_index=True)
        options = [u["Folder"] for u in users]
        selected = st.selectbox("Select user folder to delete", options)
        if st.button("Delete Selected User"):
            if delete_user_folder(selected):
                st.success("User deleted and model retrained.")
            else:
                st.warning("User folder not found.")
            st.rerun()
    else:
        st.info("No users registered yet.")


if __name__ == "__main__":
    app_main()