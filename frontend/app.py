# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.title("Alekhah Shape Fitter")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)", 
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Extract points from canvas
    data = canvas_result.json_data
    if data and "objects" in data:
        all_x, all_y = [], []
        for obj in data["objects"]:
            if "path" in obj:
                path = obj["path"]
                for point in path:
                    all_x.append(point[1])
                    all_y.append(point[2])
        if all_x:
            points = {"x": all_x, "y": all_y}
            
            # Send to backend
            response = requests.post("http://localhost:8000/predict", json=points)
            if response.status_code == 200:
                res = response.json()
                fitted_x = res["fitted_x"]
                fitted_y = res["fitted_y"]
                
                # Plot
                plt.figure(figsize=(6,6))
                plt.scatter(all_x, all_y, color="blue", label="Drawn Points")
                plt.plot(fitted_x, fitted_y, color="red", linewidth=2, label="Fitted Curve")
                plt.legend()
                st.pyplot(plt)

