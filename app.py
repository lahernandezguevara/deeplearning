import gradio as gr
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
import torch
import time
import requests
import re
import pandas as pd
import datetime

# ==========================================
# Configuración
# ==========================================
# Triton Server Endpoints
TRITON_URL = "localhost:8000"
METRICS_URL = "http://localhost:8002/metrics"
MODEL_NAME = "cnn"

# Variables globales para el "salón" (todas las sesiones)
total_requests = 0
total_time = 0.0
prediction_history = []
# DataFrame para guardar el historial de la carga para el gráfico
load_history = pd.DataFrame(columns=["time", "load"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_server_load():
    """
    Realiza una petición GET a Triton /metrics y extrae la carga actual.
    Prioriza nv_cpu_utilization, luego nv_inference_pending_request_count, y por ultimo request_success.
    """
    try:
        resp = requests.get(METRICS_URL, timeout=2)
        if resp.status_code == 200:
            # 1. Utilización de CPU de Triton (si está expuesta)
            m_cpu = re.search(r'nv_cpu_utilization\s+([0-9.]+)', resp.text)
            if m_cpu: return float(m_cpu.group(1))
            
            # 2. Peticiones pendientes / encoladas
            m_queue = re.search(r'nv_inference_pending_request_count{.*?}\s+([0-9.]+)', resp.text)
            if m_queue: return float(m_queue.group(1))
            
            # 3. Éxitos en inferencia acumulativos
            m_succ = re.search(r'nv_inference_request_success{.*?}\s+([0-9.]+)', resp.text)
            if m_succ: return float(m_succ.group(1))
    except Exception as e:
        pass
    return 0.0

def format_history():
    if not prediction_history:
        return "Sin historial aún."
    return "\n".join(prediction_history)

def update_metrics():
    """Añade un nuevo punto al gráfico de carga del servidor"""
    global load_history
    val = extract_server_load()
    now = datetime.datetime.now()
    new_row = pd.DataFrame([{"time": now, "load": val}])
    load_history = pd.concat([load_history, new_row]).tail(20)  # Mantener los últimos 20 puntos de carga
    return load_history

def predict(image):
    global total_requests, total_time, prediction_history
    
    if image is None:
        plot_df = update_metrics()
        return "Sube una imagen válida.", "0 ms", total_requests, "0.00 ms", format_history(), plot_df
        
    total_requests += 1

    try:
        # Preprocesamiento
        input_tensor = transform(image).unsqueeze(0)
        input_data = input_tensor.numpy()
        
        # Conexión con Triton
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [httpclient.InferRequestedOutput("output")]
        
        # Inferencia con medición de Latencia Exacta
        infer_start = time.perf_counter()
        response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        infer_end = time.perf_counter()
        
        # Cálculos de tiempo
        infer_time_ms = (infer_end - infer_start) * 1000
        total_time += infer_time_ms
        avg_time = (total_time / total_requests)
        
        # Procesamiento de la salida
        logits = response.as_numpy("output")[0]
        predicted_class_idx = np.argmax(logits)
        prob = np.max(logits)
        
        latency_str = f"{infer_time_ms:.2f} ms"
        result_text = f"Clase Predicha: {predicted_class_idx}\nConfianza: {prob:.4f}"
        
        # Actualizar Historial
        prediction_history.insert(0, f"Clase {predicted_class_idx} (Conf: {prob:.2f}) | {latency_str}")
        if len(prediction_history) > 10:
            prediction_history.pop()  # Guardar máx 10
            
        plot_df = update_metrics()
        return result_text, latency_str, total_requests, f"{avg_time:.2f} ms", format_history(), plot_df
        
    except Exception as e:
        plot_df = update_metrics()
        return f"Error: {str(e)}", "0 ms", total_requests, "0.00 ms", format_history(), plot_df

# ==========================================
# Interfaz Gráfica con Panel de Observabilidad
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 VGG11 Classifier con Panel de Observabilidad (Triton)")
    
    with gr.Row():
        # Lado Izquierdo: Modelo y Predicción
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Clasificación de Imágenes")
            image_input = gr.Image(type="pil", label="Sube tu imagen aquí")
            submit_btn = gr.Button("Predecir Imagen", variant="primary")
            text_output = gr.Textbox(label="Resultado Predictivo")
            latency_out = gr.Textbox(label="⏱️ Latencia Exacta de Inferencia")
            
        # Lado Derecho: Dashboard de Observabilidad
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Dashboard del Salón (Métricas Globales)")
            with gr.Row():
                req_out = gr.Number(label="Total Peticiones Recibidas", value=0)
                avg_out = gr.Textbox(label="Tiempo Promedio de Respuesta")
            
            history_out = gr.Textbox(label="📝 Historial Reciente de Predicciones", lines=6, value="Sin historial aún")
            
            gr.Markdown("#### Carga del Servidor Triton (Extraído de *:8002/metrics*)")
            load_plot = gr.LinePlot(
                x="time", 
                y="load", 
                title="Monitoreo de Carga/Peticiones a Triton",
                height=250
            )
            refresh_plot_btn = gr.Button("🔄 Refrescar Gráfico Abierto")

    # Callbacks
    submit_btn.click(
        fn=predict, 
        inputs=image_input, 
        outputs=[text_output, latency_out, req_out, avg_out, history_out, load_plot]
    )
    
    # También permitimos actualizar la gráfica de forma manual sin clasificar imágenes
    refresh_plot_btn.click(
        fn=lambda: update_metrics(),
        inputs=None,
        outputs=load_plot
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
