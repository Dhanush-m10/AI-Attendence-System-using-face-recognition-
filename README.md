# Attendance System

This project is rebuilt from scratch with the same core workflow as the reference Flask project:
- Add user by capturing face images from webcam
- Train a KNN model from saved face samples
- Start attendance recognition from webcam
- Store attendance in daily CSV files
- List and delete registered users

## Project Structure

- `app.py`: Main Flask application
- `templates/home.html`: Dashboard (attendance + add user)
- `templates/listusers.html`: User list and delete page
- `static/faces/`: Dataset folder in `Name_ID/` structure
- `static/face_recognition_model.pkl`: Trained model file (auto-generated)
- `Attendance/Attendance-MM_DD_YY.csv`: Daily attendance log (auto-generated)

## Dataset Format (same style as reference)

Each user has a dedicated folder under `static/faces`:

- `static/faces/Alice_101/`
- `static/faces/Bob_102/`

Each folder contains captured face images for that user.

## Setup

1. Create/activate virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run

### Option 1: Flask Web Application

```bash
python app.py
```

Then open: `http://127.0.0.1:5000`

**Why Flask?**
- Traditional web app with native browser webcam integration
- Desktop-friendly interface
- Press `ESC` in webcam window to stop capture

### Option 2: Streamlit Application

```bash
streamlit run streamlit_app.py
```

Then open: `http://localhost:8501`

**Why Streamlit?**
- Modern, responsive UI
- Automatic hot reload during development
- Interactive dashboard with better metrics display
- Easier deployment on Streamlit Cloud

## Usage

### Both Versions Support:

1. **Add New User**: Capture 12 face samples from webcam
2. **Mark Attendance**: Real-time face recognition
3. **Manage Users**: View, list, and delete registered users
4. **Daily Reports**: Attendance logged in CSV format

### Key Features:
- ✅ KNN Face Recognition
- ✅ Real-time Webcam Capture
- ✅ Daily CSV Attendance Logs
- ✅ User Management
- ✅ Model Training on the fly
