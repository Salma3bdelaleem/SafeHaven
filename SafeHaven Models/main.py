from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import joblib
import shutil
from pathlib import Path

from pathlib import Path
if not Path("random_forest_model 96.7 .pkl").exists():
    print("Model file not found!")

# تحميل النموذج
model = joblib.load("random_forest_model 96.7 .pkl")
print("done1")

# إعداد MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("done2")

# إنشاء تطبيق FastAPI
app = FastAPI()

# تحسين جودة الصورة
# def improve_image_quality(image):
#     channels = cv2.split(image)
#     eq_channels = [cv2.equalizeHist(channel) for channel in channels]
#     hist_eq_image = cv2.merge(eq_channels)
#     blurred_image = cv2.GaussianBlur(hist_eq_image, (5, 5), 0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_image = np.zeros_like(blurred_image)
#     for i in range(3):
#         clahe_image[:, :, i] = clahe.apply(blurred_image[:, :, i])
#     return clahe_image    

# استخراج المعالم والتنبؤ
def predict_fall_from_image(image):
    # enhanced_image = improve_image_quality(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    
    if not results.pose_landmarks:
        raise ValueError("No landmarks detected in the image.")
        
    landmarks = results.pose_landmarks.landmark
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    label = "fall" if prediction[0] == 1 else "not fall"
    return label

# نقطة النهاية لاستقبال الصور
@app.post("/predict_ml")
async def predict_fall(file: UploadFile = File(...)):
    try:
        # حفظ الصورة المرفوعة بشكل مؤقت
        temp_file = Path(f"temp_{file.filename}")
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # قراءة الصورة باستخدام OpenCV
        image = cv2.imread(str(temp_file))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # إجراء التنبؤ
        result = predict_fall_from_image(image)
        
        # حذف الملف المؤقت
        temp_file.unlink(missing_ok=True)

        # إرجاع النتيجة
        return JSONResponse(content={"prediction": result})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
