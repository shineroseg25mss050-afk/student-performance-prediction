
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "student_model.pkl"

if not model_path.exists():
    raise FileNotFoundError(
        "Model not found! Run mlflow_tracking.py first."
    )

model = joblib.load(model_path)

# Create FastAPI application
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student performance using Random Forest",
    version="1.0"
)

# Student input fields
class StudentData(BaseModel):
    age: float
    study_hours: float
    attendance: float


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running!"
    }


@app.post("/predict")
def predict(student: StudentData):
    try:
        input_data = pd.DataFrame([{
            "age": student.age,
            "study_hours": student.study_hours,
            "attendance": student.attendance
        }])

        prediction = model.predict(input_data)

        return {
            "student_details": {
                "age": student.age,
                "study_hours": student.study_hours,
                "attendance": student.attendance
            },
            "predicted_result": str(prediction[0]),
            "message": "Prediction completed successfully!"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
