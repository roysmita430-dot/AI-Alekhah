# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
from scipy.interpolate import UnivariateSpline

app = FastAPI(title="Alekhah AI Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Points(BaseModel):
    x: list[float]
    y: list[float]

@app.get("/")
def home():
    return JSONResponse(content={"message": "Alekhah AI backend is running!"})

@app.post("/predict")
def predict(points: Points):
    x = np.array(points.x)
    y = np.array(points.y)
    
    # Parameter for spline (0 → 1)
    t = np.linspace(0, 1, len(x))
    
    # Smooth splines for x(t) and y(t)
    spline_x = UnivariateSpline(t, x, s=0)
    spline_y = UnivariateSpline(t, y, s=0)
    
    # Generate 500 smooth points
    t_smooth = np.linspace(0, 1, 500)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)
    
    return JSONResponse(content={
        "fitted_x": x_smooth.tolist(),
        "fitted_y": y_smooth.tolist(),
        "function_type": "Parametric Spline (Exact shape)"
    })

