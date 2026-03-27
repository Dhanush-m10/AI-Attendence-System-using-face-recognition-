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

```bash
python app.py
```

Then open:

- `http://127.0.0.1:5000`

## Usage

1. Open home page and add a new user (name + ID)
2. Webcam captures samples and trains model automatically
3. Click "Take Attendance" to start recognition
4. Press `ESC` in webcam window to stop
5. Open "Manage Users" to review/delete users
