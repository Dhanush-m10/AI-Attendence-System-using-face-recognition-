# MDroid Attendance System - Dual Platform Guide

## ✨ Features Available on BOTH Platforms

Your attendance system is now fully compatible with both **Flask** and **Streamlit**. All features are identical:

### Core Functionalities
- ✅ Face Recognition using KNN classifier
- ✅ Real-time Webcam Capture (12 samples per user)
- ✅ Automatic Model Training
- ✅ Daily CSV Attendance Logs
- ✅ User Management (Add/Delete/List)
- ✅ Attendance Dashboard
- ✅ Responsive UI

### Shared Backend
Both platforms use the exact same:
- `static/faces/` - User face database
- `static/face_recognition_model.pkl` - Trained model
- `Attendance/*.csv` - Daily logs (in `Name_Roll_Time` format)
- Core logic and algorithms

---

## 🚀 Running the Applications

### **Flask Version** (Traditional Web App)

```bash
python app.py
```

- Opens at: `http://127.0.0.1:5000`
- **Best for**: Desktop deployment, native webcam integration
- Press `ESC` in webcam window to exit capture

### **Streamlit Version** (Modern Dashboard)

```bash
streamlit run streamlit_app.py
```

- Opens at: `http://localhost:8501`
- **Best for**: Quick development, cloud deployment (Streamlit Cloud)
- Hot reload enabled during development

---

## 📁 Project Structure

```
├── app.py                          # Flask main app
├── streamlit_app.py                # Streamlit main app
├── requirements.txt                # All dependencies
├── README.md                       # Documentation
├── DEPLOYMENT.md                   # This file
│
├── templates/
│   ├── home.html                   # Flask dashboard
│   └── listusers.html              # Flask user management
│
├── static/
│   ├── faces/                      # User face samples (SHARED)
│   │   ├── Alice_101/
│   │   ├── Bob_102/
│   │   └── ...
│   └── face_recognition_model.pkl  # Trained model (SHARED)
│
├── Attendance/
│   ├── Attendance-04_02_26.csv     # Daily logs (SHARED)
│   └── ...
│
└── .streamlit/
    └── config.toml                 # Streamlit styling
```

---

## 🔄 Switching Between Platforms

Since both apps share the same backend (`static/faces/`, models, and CSVs):

1. Add a user in **Flask** → Model trains
2. Switch to **Streamlit** → Can immediately use new user for attendance
3. Mark attendance in **Streamlit** → Logs appear in both interfaces
4. Go back to **Flask** → Sees all attendance records

**No data sync needed!** Everything is automatically shared.

---

## 📱 API Endpoints (Flask Only)

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home dashboard |
| `/add` | POST | Add new user |
| `/start` | GET | Mark attendance |
| `/listusers` | GET | View users |
| `/deleteuser` | GET | Delete user |

---

## 🌐 Deployment Options

### **Deploy Flask**
- Traditional VPS/Server
- Docker container
- Heroku, Railway, Render

### **Deploy Streamlit**
- **Streamlit Cloud** (easiest, free tier available)
- Docker container
- Any cloud platform (AWS, Heroku, Railway)

---

## 🛠️ Troubleshooting

### Webcam Not Working
Both apps check camera access at startup:
- **Flask**: Returns error on `/start` route
- **Streamlit**: Shows error in UI
- Solution: Check Windows Privacy Settings → Camera

### Model Not Found
Appears when no users have been added:
- **Flask**: "No trained model found. Please add a new user first."
- **Streamlit**: Same message in dashboard
- Solution: Add at least one user first

### Attendance Already Marked
If duplicate entries happen:
- Same user within same session (caught by app logic)
- Solution: Reload page/app to reset

---

## 📊 File Formats

### Face Samples Directory
```
static/faces/
├── Alice_101/
│   ├── Alice_0.jpg
│   ├── Alice_1.jpg
│   └── ... (up to 11)
└── Bob_102/
    ├── Bob_0.jpg
    └── ...
```

### Attendance CSV
```csv
Name,Roll,Time
Alice,101,09:30:45
Bob,102,10:15:22
Alice,101,14:20:10
```

---

## ⚙️ Configuration

### Capture Settings
- **NIMGS = 12**: Number of face samples per user
- **n_neighbors = 5**: KNN classifier neighbors

To modify, edit both:
- `app.py` (line 19)
- `streamlit_app.py` (line 14)

### Resize Dimensions
- Face samples: 256×256 (capture)
- Model input: 50×50 (training & recognition)

---

## 💡 Tips

1. **Best Practice**: Use the same user data with both platforms
2. **Development**: Use Streamlit for faster iteration
3. **Production**: Choose Flask for stability, Streamlit for cloud
4. **Backup**: Keep `static/faces/` and attendance CSVs safe
5. **GPU Support**: Both support CUDA-enabled OpenCV for faster processing

---

*Last Updated: 2026-04-02 | MDroid Attendance System v2.0*
