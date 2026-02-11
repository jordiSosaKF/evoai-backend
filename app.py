import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Union  # <--- IMPORTANTE: Agregamos esto

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CARGAR EL CEREBRO DE LA IA (SVM) ---
print("Cargando sistema EvoAI (SVM)...")
try:
    modelo = joblib.load('modelo_evoai.pkl')
    encoder = joblib.load('encoder_objetivo.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Cerebro cargado: SVM (94% Precisión)")
except Exception as e:
    print(f"❌ Error cargando archivos .pkl: {e}")
    print("⚠️ ASEGURATE DE SUBIR: modelo_evoai.pkl, encoder_objetivo.pkl, scaler.pkl")

# --- 2. DEFINIR DATOS DE ENTRADA (SOLUCIÓN HÍBRIDA) ---
class UserData(BaseModel):
    Genero: int = 1
    Edad: int
    TiempG: int
    Peso: float
    Estatura: float
    Lesion: int = 0
    Enfermedad: int = 0
    
    # ⚠️ CAMBIO CLAVE: Aceptamos String O Entero para que no falle nunca
    Objetivo: Union[str, int] 
    
    FrecEntren: int

# --- 3. BASE DE DATOS DE RUTINAS ---
RUTINAS = {
    0: {
        "id_rutina": "rut_fullbody_01",
        "nombre": "Full Body: Cimientos",
        "nivel": "Principiante",
        "objetivo_principal": "Adaptación anatómica y técnica",
        "frecuencia_sugerida": "3 días por semana",
        "descripcion": "Rutina de cuerpo completo para dominar los patrones básicos.",
        "ejercicios": [
            "1. Sentadilla Globet (Copa) - 3x12 (Cuádriceps)",
            "2. Press de Banca con Mancuernas - 3x12 (Pecho)",
            "3. Jalón al Pecho (Polea Alta) - 3x12 (Espalda)",
            "4. Press Militar Sentado - 3x10 (Hombros)",
            "5. Peso Muerto Rumano con Mancuernas - 3x12 (Isquios)",
            "6. Plancha Abdominal - 3 series al fallo técnico"
        ]
    },
    1: {
        "id_rutina": "rut_ppl_01",
        "nombre": "Push / Pull / Legs",
        "nivel": "Intermedio",
        "objetivo_principal": "Hipertrofia y Simetría",
        "frecuencia_sugerida": "4 a 6 días por semana",
        "descripcion": "División clásica moderna: Empuje, Tracción y Pierna.",
        "ejercicios": [
            "--- SESIÓN A: PUSH (Empuje) ---",
            "1. Press Inclinado con Barra - 4x8-10",
            "2. Press Militar con Barra (De pie) - 4x8-10",
            "3. Fondos en Paralelas (Dips) - 3x12",
            "4. Elevaciones Laterales - 3x15",
            "5. Extensiones de Tríceps en Polea - 3x15",
            "--- SESIÓN B: PULL (Tracción) ---",
            "1. Dominadas (o Jalón al pecho) - 4x8-10",
            "2. Remo con Barra - 4x10",
            "3. Face Pulls - 3x15",
            "4. Curl de Bíceps con Barra Z - 3x12",
            "5. Curl Martillo - 3x12",
            "--- SESIÓN C: LEGS (Pierna) ---",
            "1. Sentadilla Libre (High Bar) - 4x6-8",
            "2. Prensa de Piernas - 3x12",
            "3. Peso Muerto Rumano - 4x10",
            "4. Extensiones de Cuádriceps - 3x15",
            "5. Elevación de Talones - 4x20"
        ]
    },
    2: {
        "id_rutina": "rut_arnold_01",
        "nombre": "Arnold Split (Élite)",
        "nivel": "Avanzado",
        "objetivo_principal": "Volumen Máximo",
        "frecuencia_sugerida": "6 días por semana",
        "descripcion": "La división de la vieja escuela para máxima intensidad.",
        "ejercicios": [
            "--- DÍA 1: PECHO Y ESPALDA ---",
            "1. Press Banca Plano (Fuerza) - 5x5",
            "2. Dominadas Lastradas - 4x8",
            "3. Superserie: Press Inclinado + Remo con Mancuerna - 4x10",
            "4. Aperturas (Flyes) - 3x15",
            "5. Pullover - 3x15",
            "--- DÍA 2: HOMBROS Y BRAZOS ---",
            "1. Press Militar sentado - 4x8",
            "2. Elevaciones Laterales - 4x15",
            "3. Superserie: Curl Barra + Press Francés - 4x10",
            "4. Superserie: Curl Predicador + Extensión Copa - 3x12",
            "5. Curl de Muñeca - 3x20",
            "--- DÍA 3: PIERNA COMPLETA ---",
            "1. Sentadilla Frontal o Hack - 4x10",
            "2. Peso Muerto Convencional - 3x5",
            "3. Zancadas Búlgaras - 3x12",
            "4. Curl Femoral Tumbado - 4x15",
            "5. Gemelo en máquina - 4x15"
        ]
    }
}

@app.get("/")
def home():
    return {"mensaje": "Servidor EvoAI (SVM) Activo 🚀"}

@app.post("/asignar_rutina")
def predecir(usuario: UserData):
    try:
        # --- A. TRADUCTOR INTELIGENTE (LA SOLUCIÓN AL ERROR 422) ---
        # Si Flutter manda un número, lo convertimos a texto aquí mismo.
        objetivo_input = usuario.Objetivo
        
        if isinstance(objetivo_input, int):
            # Diccionario de traducción ID -> Texto
            mapa_objetivos = {
                1: "Ganar masa muscular",
                2: "Bajar de peso",
                3: "Tonificar",
                4: "Ganar fuerza"
            }
            # Si manda un número raro, por defecto ponemos "Tonificar"
            objetivo_input = mapa_objetivos.get(objetivo_input, "Tonificar")
            print(f"🔄 Convertido {usuario.Objetivo} (int) a -> {objetivo_input} (str)")
        
        # --- B. FEATURE ENGINEERING ---
        estatura_m = usuario.Estatura / 100.0 if usuario.Estatura > 3 else usuario.Estatura
        bmi = usuario.Peso / (estatura_m ** 2)

        # Usamos el objetivo ya "limpio" (sea texto original o traducido)
        try:
            objetivo_n = encoder.transform([objetivo_input])[0]
        except:
            objetivo_n = 0 # Fallback de seguridad

        features = pd.DataFrame([[
            usuario.TiempG, 
            objetivo_n, 
            bmi, 
            usuario.Edad
        ]], columns=['TiempG', 'Objetivo_n', 'BMI', 'Edad'])

        features_scaled = scaler.transform(features)

        # --- C. PREDICCIÓN ---
        prediccion_num = int(modelo.predict(features_scaled)[0])
        
        # --- D. RESULTADO ---
        rutina_seleccionada = RUTINAS.get(prediccion_num)
        
        # Copia segura para no modificar la original
        respuesta_final = rutina_seleccionada.copy()
        
        # Añadimos nota si la frecuencia del usuario es muy alta
        dias_sugeridos = 3
        if prediccion_num == 1: dias_sugeridos = 4
        if prediccion_num == 2: dias_sugeridos = 6
        
        if usuario.FrecEntren > dias_sugeridos:
            respuesta_final['descripcion'] += f"\n⚠️ Nota: Tu quieres ir {usuario.FrecEntren} días, pero esta rutina es de {dias_sugeridos}. ¡Descansa bien!"

        return {
            "cluster_asignado": prediccion_num,
            "rutina": respuesta_final,
            "debug_info": f"BMI: {bmi:.2f} | Objetivo: {objetivo_input}"
        }

    except Exception as e:
        print(f"ERROR INTERNO: {e}")
        return {"error": str(e), "mensaje": "Hubo un error en el servidor"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
