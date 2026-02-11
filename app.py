import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS para que Flutter pueda conectarse
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
    # Asegúrate de que estos 3 archivos estén en la misma carpeta
    modelo = joblib.load('modelo_evoai.pkl')
    encoder = joblib.load('encoder_objetivo.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Cerebro cargado: SVM (94% Precisión)")
except Exception as e:
    print(f"❌ Error cargando archivos .pkl: {e}")
    print("⚠️ ASEGURATE DE SUBIR: modelo_evoai.pkl, encoder_objetivo.pkl, scaler.pkl")

# --- 2. DEFINIR DATOS DE ENTRADA (Desde Flutter) ---
class UserData(BaseModel):
    # Solo necesitamos lo que Flutter manda
    # Nota: Aunque no usemos Genero o Lesion en el SVM, si Flutter los manda,
    # los recibimos pero los ignoramos en el cálculo matemático.
    Genero: int = 1
    Edad: int
    TiempG: int      # 1, 2, 3
    Peso: float
    Estatura: float  # CM o Metros
    Lesion: int = 0
    Enfermedad: int = 0
    Objetivo: str    # OJO: Ahora recibimos STRING (ej: "Ganar masa"), no int.
    FrecEntren: int

# --- 3. BASE DE DATOS DE RUTINAS (TUS RUTINAS PRO) ---
RUTINAS = {
    # PRINCIPIANTE (Cluster 0)
    0: {
        "id_rutina": "rut_fullbody_01",
        "nombre": "Full Body: Cimientos",
        "nivel": "Principiante",
        "objetivo_principal": "Adaptación anatómica y técnica",
        "frecuencia_sugerida": "3 días por semana",
        "descripcion": "Rutina de cuerpo completo para dominar los patrones básicos. Ideal para ganar fuerza inicial.",
        "ejercicios": [
            "1. Sentadilla Globet (Copa) - 3x12",
            "2. Press de Banca con Mancuernas - 3x12",
            "3. Jalón al Pecho (Polea Alta) - 3x12",
            "4. Press Militar Sentado - 3x10",
            "5. Peso Muerto Rumano con Mancuernas - 3x12",
            "6. Plancha Abdominal - 3 series al fallo"
        ]
    },
    # INTERMEDIO (Cluster 1)
    1: {
        "id_rutina": "rut_ppl_01",
        "nombre": "Push / Pull / Legs",
        "nivel": "Intermedio",
        "objetivo_principal": "Hipertrofia y Simetría",
        "frecuencia_sugerida": "4 a 5 días por semana",
        "descripcion": "División clásica moderna. Agrupa músculos por función biomecánica.",
        "ejercicios": [
            "--- DÍA PUSH (Empuje) ---",
            "1. Press Inclinado Barra - 4x10",
            "2. Press Militar - 4x10",
            "3. Fondos - 3x12",
            "4. Elev. Laterales - 3x15",
            "--- DÍA PULL (Tracción) ---",
            "1. Dominadas - 4x8",
            "2. Remo Barra - 4x10",
            "3. Curl Bíceps - 3x12",
            "--- DÍA LEGS (Pierna) ---",
            "1. Sentadilla - 4x8",
            "2. Peso Muerto Rumano - 4x10",
            "3. Prensa - 3x12"
        ]
    },
    # AVANZADO (Cluster 2)
    2: {
        "id_rutina": "rut_arnold_01",
        "nombre": "Arnold Split (Élite)",
        "nivel": "Avanzado",
        "objetivo_principal": "Volumen Máximo y Puntos Débiles",
        "frecuencia_sugerida": "6 días por semana",
        "descripcion": "La división de la vieja escuela para máximo volumen. Pecho+Espalda, Hombro+Brazo, Pierna.",
        "ejercicios": [
            "--- DÍA 1: PECHO Y ESPALDA ---",
            "1. Press Banca (Fuerza) - 5x5",
            "2. Dominadas Lastradas - 4x8",
            "3. Superserie: Inclinado + Remo - 4x10",
            "--- DÍA 2: HOMBROS Y BRAZOS ---",
            "1. Press Militar - 4x8",
            "2. Superserie: Curl Barra + Press Francés - 4x10",
            "--- DÍA 3: PIERNA COMPLETA ---",
            "1. Sentadilla Hack - 4x10",
            "2. Peso Muerto - 3x5",
            "3. Zancadas - 3x12"
        ]
    }
}

@app.get("/")
def home():
    return {"mensaje": "Servidor EvoAI (SVM) Activo 🚀"}

@app.post("/asignar_rutina") # Mantenemos tu ruta original
def predecir(usuario: UserData):
    try:
        # --- A. FEATURE ENGINEERING (MATEMÁTICAS DEL SVM) ---
        
        # 1. Calcular BMI (Peso / Talla^2)
        # Normalizamos estatura a metros si viene en cm
        estatura_m = usuario.Estatura / 100.0 if usuario.Estatura > 3 else usuario.Estatura
        bmi = usuario.Peso / (estatura_m ** 2)

        # 2. Codificar Objetivo (Texto -> Número)
        try:
            # Usamos el encoder que guardamos para traducir
            objetivo_n = encoder.transform([usuario.Objetivo])[0]
        except:
            # Si el objetivo es nuevo, usamos 0 por defecto
            objetivo_n = 0

        # 3. Crear DataFrame con LAS MISMAS columnas del entrenamiento
        # Orden: TiempG, Objetivo_n, BMI, Edad
        features = pd.DataFrame([[
            usuario.TiempG, 
            objetivo_n, 
            bmi, 
            usuario.Edad
        ]], columns=['TiempG', 'Objetivo_n', 'BMI', 'Edad'])

        # 4. ESCALAR DATOS (¡Paso Clave del SVM!)
        features_scaled = scaler.transform(features)

        # --- B. PREDICCIÓN ---
        prediccion_num = int(modelo.predict(features_scaled)[0])
        
        # --- C. RESULTADO ---
        rutina_seleccionada = RUTINAS.get(prediccion_num)
        
        # Ajuste de consejo de frecuencia
        dias_sugeridos = 3
        if prediccion_num == 1: dias_sugeridos = 4
        if prediccion_num == 2: dias_sugeridos = 6
        
        mensaje_extra = ""
        if usuario.FrecEntren > dias_sugeridos:
            mensaje_extra = f"\n⚠️ Nota: Quieres ir {usuario.FrecEntren} días, pero esta rutina es de {dias_sugeridos}. ¡Descansa bien!"

        # Copiamos para no alterar el original
        respuesta_final = rutina_seleccionada.copy()
        respuesta_final['descripcion'] += mensaje_extra

        return {
            "cluster_asignado": prediccion_num,
            "rutina": respuesta_final,
            "debug_info": f"BMI: {bmi:.2f} | Modelo: SVM"
        }

    except Exception as e:
        return {"error": str(e), "mensaje": "Hubo un error en el servidor"}

# Configuración para Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)