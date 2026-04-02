import os
import re
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Page configuration
st.set_page_config(
    page_title="Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS to match Flask styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    
    * {
        font-family: 'Manrope', sans-serif;
    }
    
    :root {
        --bg1: #f4f7fc;
        --bg2: #e9eff9;
        --accent: #3f6ca6;
        --text: #1f334f;
        --muted: #61748f;
    }
    
    .main {
        background: linear-gradient(180deg, #f4f7fc, #e9eff9);
    }
    
    .title-text {
        font-family: 'DM Serif Display', serif !important;
        color: #1e3557 !important;
        font-size: 2.5rem !important;
        letter-spacing: 0.4px;
    }
    
    .subtitle-text {
        color: #61748f;
        font-weight: 500;
    }
    
</style>
""", unsafe_allow_html=True)

# Configuration
NIMGS = 12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACES_DIR = os.path.join(STATIC_DIR, "faces")
MODEL_PATH = os.path.join(STATIC_DIR, "face_recognition_model.pkl")

_cv2 = None
_cv2_import_error = None
face_detector = None


def get_cv2():
    global _cv2, _cv2_import_error, face_detector

    if _cv2 is not None:
        return _cv2

    try:
        import cv2 as cv2_module
    except Exception as exc:
        _cv2_import_error = exc
        return None

    _cv2 = cv2_module
    face_detector = _cv2.CascadeClassifier(_cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cv2


def require_cv2():
    cv2_module = get_cv2()
    if cv2_module is None:
        st.error(
            "OpenCV could not be loaded in this deployment. Streamlit Cloud currently needs a compatible cv2 build to use the webcam features."
        )
        if _cv2_import_error is not None:
            st.caption(str(_cv2_import_error))
    return cv2_module

# Initialize session state
if "captured_count" not in st.session_state:
    st.session_state.captured_count = 0
if "adding_user" not in st.session_state:
    st.session_state.adding_user = False


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
    cv2_module = require_cv2()
    if cv2_module is None:
        return []

    global face_detector
    if face_detector is None:
        face_detector = cv2_module.CascadeClassifier(
            cv2_module.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    try:
        gray = cv2_module.cvtColor(img, cv2_module.COLOR_BGR2GRAY)
        return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    except Exception:
        return []


def identify_face(facearray):
    if not os.path.exists(MODEL_PATH):
        return None
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


def train_model():
    ensure_directories()
    cv2_module = require_cv2()
    if cv2_module is None:
        return False

    faces = []
    labels = []

    for user in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = cv2_module.imread(img_path)
            if img is None:
                continue

            resized_face = cv2_module.resize(img, (50, 50))
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


def home_page():
    # Header
    col_title, col_nav = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="title-text">Attendance System</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="subtitle-text">Date: {datetoday2()}</p>', unsafe_allow_html=True)
    with col_nav:
        st.write("")
        st.write("")
        if st.button("👥 Manage Users", use_container_width=True):
            st.session_state.page = "manage"
            st.rerun()
        st.metric("Total Users", totalreg())

    st.divider()

    # Main content - 2 columns
    col_attendance, col_add_user = st.columns([7, 5], gap="large")

    with col_attendance:
        st.subheader("Today's Attendance")
        
        if st.button("Take Attendance", use_container_width=True, key="take_attendance"):
            attendance_recognition_streamlit()

        names, rolls, times, l = extract_attendance()
        
        if l > 0:
            attendance_data = []
            for i in range(l):
                attendance_data.append({
                    "S No": i + 1,
                    "Name": names[i],
                    "ID": rolls[i],
                    "Time": times[i]
                })
            df_attendance = pd.DataFrame(attendance_data)
            st.dataframe(df_attendance, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance marked yet.")

    with col_add_user:
        st.subheader("Add New User")
        
        with st.form("add_user_form", clear_on_submit=True):
            new_username = st.text_input("Full Name", placeholder="e.g., John Doe")
            new_userid = st.text_input("User ID", placeholder="e.g., 101")
            
            submitted = st.form_submit_button("📷 Capture Face Samples", use_container_width=True)
            
            if submitted:
                if not new_username or not new_userid:
                    st.error("Please enter both name and ID.")
                else:
                    st.session_state.adding_user = True
                    st.session_state.new_username = sanitize_text(new_username)
                    st.session_state.new_userid = sanitize_text(new_userid)
                    st.session_state.captured_count = 0
                    st.rerun()
        
        st.info("""
        **How it works:**
        1. Enter name and ID
        2. Click capture button
        3. A webcam window opens
        4. Face samples are captured (12 samples)
        5. Press ESC to finish or wait for auto-completion
        """)

    if st.session_state.adding_user:
        capture_faces_streamlit()


def capture_faces_streamlit():
    cv2_module = require_cv2()
    if cv2_module is None:
        return

    st.divider()
    st.warning(f"Capturing faces for: {st.session_state.new_username} (ID: {st.session_state.new_userid})")
    st.info(f"Need to capture {NIMGS} face samples. Please look at your camera.")
    
    user_folder = f"{st.session_state.new_username}_{st.session_state.new_userid}"
    userimagefolder = os.path.join(FACES_DIR, user_folder)
    os.makedirs(userimagefolder, exist_ok=True)

    cap = cv2_module.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Check camera permissions.")
        st.session_state.adding_user = False
        return

    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    i = 0
    j = 0
    max_frames = 500

    try:
        frame_count = 0
        while i < NIMGS and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            frame = cv2_module.flip(frame, 1)
            faces = extract_faces(frame)

            for (x, y, w, h) in faces:
                cv2_module.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2_module.putText(
                    frame,
                    f"Captured: {i}/{NIMGS}",
                    (20, 30),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2_module.LINE_AA,
                )

                if j % 5 == 0 and i < NIMGS:
                    file_name = f"{st.session_state.new_username}_{i}.jpg"
                    face_crop = frame[y : y + h, x : x + w]
                    resized = cv2_module.resize(face_crop, (256, 256))
                    cv2_module.imwrite(os.path.join(userimagefolder, file_name), resized)
                    i += 1
                j += 1

            frame_rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
            progress_bar.progress(i / NIMGS)
            status_text.text(f"Faces captured: {i}/{NIMGS}")
            frame_count += 1

    finally:
        cap.release()

    if i > 0:
        status_text.success(f"Captured {i} face samples. Training model...")
        progress_bar.progress(1.0)
        
        train_model()
        st.success(f"User {st.session_state.new_username} (ID: {st.session_state.new_userid}) added successfully!")
    else:
        st.error("No faces captured. Please try again.")

    st.session_state.adding_user = False


def attendance_recognition_streamlit():
    cv2_module = require_cv2()
    if cv2_module is None:
        return

    if not os.path.exists(MODEL_PATH):
        st.error("No trained model found. Please add a new user first.")
        return

    cap = cv2_module.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam. Check camera permissions.")
        return

    st.warning("Looking for faces... Keep your face in frame within 30 seconds.")
    
    stframe = st.empty()
    status_text = st.empty()
    recognized_person = ""
    max_frames = 180
    frames_checked = 0

    try:
        while frames_checked < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2_module.flip(frame, 1)
            faces = extract_faces(frame)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2_module.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 2)
                cv2_module.rectangle(frame, (x, y - 40), (x + w, y), (86, 32, 251), -1)

                face = cv2_module.resize(frame[y : y + h, x : x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))
                
                if identified_person is not None:
                    identified_person = identified_person[0]
                    add_attendance(identified_person)
                    recognized_person = identified_person

                    cv2_module.putText(
                        frame,
                        identified_person,
                        (x + 5, y - 15),
                        cv2_module.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2_module.LINE_AA,
                    )

                    cv2_module.putText(
                        frame,
                        "Attendance marked. Closing camera...",
                        (10, 65),
                        cv2_module.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                        cv2_module.LINE_AA,
                    )
                    
                    frame_rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, use_container_width=True)
                    break
            else:
                cv2_module.putText(
                    frame,
                    "Looking for face... Keep your face in frame",
                    (10, 30),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2_module.LINE_AA,
                )

            frame_rgb = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
            stframe.image(frame_rgb, use_container_width=True)
            status_text.text(f"Scanning... {frames_checked}/{max_frames}")
            frames_checked += 1

    finally:
        cap.release()

    if recognized_person:
        display_name = recognized_person.rsplit("_", 1)[0] if "_" in recognized_person else recognized_person
        st.success(f"Attendance marked for **{display_name}**. Camera closed automatically.")
    else:
        st.warning("No face detected for attendance. Please try again.")


def manage_users_page():
    # Header
    col_title, col_nav = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="title-text">User Management</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="subtitle-text">Manage registered users | Date: {datetoday2()}</p>', unsafe_allow_html=True)
    with col_nav:
        st.write("")
        st.write("")
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.metric("Total Users", totalreg())

    st.divider()

    userlist, names, rolls, l = getallusers()

    if l == 0:
        st.info("No users enrolled yet.")
    else:
        st.subheader("Registered Users")
        
        users_data = []
        for i in range(l):
            users_data.append({
                "S No": i + 1,
                "Name": names[i],
                "ID": rolls[i],
                "User ID": userlist[i]
            })
        df_users = pd.DataFrame(users_data)
        
        # Display table without User ID column (it's just for reference)
        display_df = df_users[["S No", "Name", "ID"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Delete section
        st.divider()
        st.subheader("Delete User")
        user_to_delete = st.selectbox("Select user to delete:", userlist, key="delete_select")
        
        if st.button("Delete", use_container_width=True, key="delete_btn"):
            deletefolder(os.path.join(FACES_DIR, user_to_delete))
            train_model()
            st.success(f"User {user_to_delete} deleted successfully!")
            st.rerun()


def main():
    ensure_directories()
    
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Navigation
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "manage":
        manage_users_page()


if __name__ == "__main__":
    main()

