from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# 🔹 Cargar el modelo entrenado
model = joblib.load("modelo_calorias_rf_features.pkl")

# 🔹 Crear la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Calorías Mejorada",
    version="2.0",
    description="Predice las calorías quemadas en un entrenamiento con validaciones avanzadas."
)

# 🔹 Definir la estructura de los datos de entrada con validaciones
class WorkoutInput(BaseModel):
    Age: int = Field(..., gt=0, lt=100, description="Edad debe estar entre 1 y 99 años.")
    Weight: float = Field(..., gt=30, lt=200, description="Peso debe estar entre 30 y 200 kg.")
    Height: float = Field(..., gt=1.2, lt=2.5, description="Altura debe estar entre 1.2 y 2.5 metros.")
    Max_BPM: int = Field(..., gt=50, lt=220, description="Frecuencia máxima debe estar entre 50 y 220 BPM.")
    Avg_BPM: int = Field(..., gt=50, lt=220, description="Frecuencia promedio debe estar entre 50 y 220 BPM.")
    Resting_BPM: int = Field(..., gt=30, lt=120, description="Frecuencia en reposo debe estar entre 30 y 120 BPM.")
    Session_Duration: float = Field(..., gt=0, lt=5, description="Duración debe estar entre 0 y 5 horas.")
    Workout_Type: int = Field(..., ge=0, le=1, description="Workout_Type debe ser 0 (Cardio) o 1 (Strength).")
    Fat_Percentage: float = Field(..., ge=0, le=50, description="Porcentaje de grasa debe estar entre 0 y 50%.")
    Water_Intake: float = Field(..., ge=0, le=10, description="Ingesta de agua debe estar entre 0 y 10 litros.")
    Workout_Frequency: int = Field(..., ge=1, le=7, description="Frecuencia de entrenamiento debe estar entre 1 y 7 días/semana.")
    BMI: float = Field(..., gt=10, lt=50, description="BMI debe estar entre 10 y 50.")
    Intensity: float = Field(..., gt=0, le=1, description="Intensidad debe estar entre 0 y 1.")
    BMI_Workout: float
    Calories_per_hour: float

# 🔹 Ruta de bienvenida
@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Calorías Mejorada 🔥"}

# 🔹 Ruta para hacer predicciones con validaciones avanzadas
@app.post("/predict")
def predict_calories(data: WorkoutInput):
    # Validación adicional para asegurarse de que Max_BPM sea mayor que Resting_BPM
    if data.Max_BPM < data.Resting_BPM:
        raise HTTPException(status_code=400, detail="Max_BPM no puede ser menor que Resting_BPM.")

    # Convertir los datos de entrada en un DataFrame
    input_data = pd.DataFrame([data.dict()])

    # 🔹 Renombrar columnas para que coincidan con el modelo
    input_data.rename(columns={
        "Weight": "Weight (kg)",
        "Height": "Height (m)",
        "Session_Duration": "Session_Duration (hours)",
        "Water_Intake": "Water_Intake (liters)",
        "Workout_Frequency": "Workout_Frequency (days/week)"
    }, inplace=True)

    # 🔹 Hacer la predicción
    prediction = model.predict(input_data)[0]

    return {
        "Calories_Burned_Predicted": round(prediction, 2),
        "message": "Predicción exitosa ✅"
    }
