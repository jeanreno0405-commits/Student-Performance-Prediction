
import pandas as pd
import gradio as gr
import joblib

model = joblib.load("model.pkl")

def predict_score(study_hours, attendance, previous_score, sleep_hours):
    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score],
        "Sleep_Hours": [sleep_hours]
    })

    prediction = model.predict(student)
    return round(float(prediction[0]), 2)

app = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Slider(1, 8, value=5, label="Study Hours"),
        gr.Slider(60, 100, value=90, label="Attendance"),
        gr.Slider(40, 95, value=75, label="Previous Score"),
        gr.Slider(5, 9, value=7, label="Sleep Hours")
    ],
    outputs=gr.Number(label="Predicted Math Score"),
    title="Student Performance Prediction System"
)

if __name__ == "__main__":
    app.launch()
