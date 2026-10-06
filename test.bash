#!/bin/bash

curl localhost:8000/v2/health/ready
curl localhost:8000/v2/models/cnn/ready

# P1. Buscar un buen dataset
# -> de preferencia de imagenes
# -> Transfer Learnin
# -> Desplegar a produccion (Docker)
# -> Agente -> Crear GUI para consumir el modelo
# AGY -> Gemini pro