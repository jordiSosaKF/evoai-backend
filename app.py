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
    # -----------------------------------------------------
    # CLUSTER 0: PRINCIPIANTES -> FULL BODY
    # -----------------------------------------------------
    0: {
        "id_rutina": "rut_fullbody_01",
        "nombre": "Full Body: Cimientos",
        "nivel": "Principiante",
        "objetivo_principal": "Adaptación anatómica y técnica",
        "frecuencia_sugerida": "3 días por semana (dejando 1 día de descanso)",
        "descripcion": "Rutina de cuerpo completo para dominar los patrones de movimiento básicos. Ideal para ganar fuerza inicial.",
        "ejercicios": [
            "1. Sentadilla Globet (Copa) - 3x12 (Cuádriceps)",
            "2. Press de Banca con Mancuernas - 3x12 (Pecho)",
            "3. Jalón al Pecho (Polea Alta) - 3x12 (Espalda)",
            "4. Press Militar Sentado - 3x10 (Hombros)",
            "5. Peso Muerto Rumano con Mancuernas - 3x12 (Isquios)",
            "6. Plancha Abdominal - 3 series al fallo técnico"
        ]
    },
    # -----------------------------------------------------
    # CLUSTER 1: INTERMEDIOS -> PUSH / PULL / LEGS (PPL)
    # -----------------------------------------------------
    1: {
        "id_rutina": "rut_ppl_01",
        "nombre": "Push / Pull / Legs",
        "nivel": "Intermedio",
        "objetivo_principal": "Hipertrofia y Simetría",
        "frecuencia_sugerida": "4 a 6 días por semana (Frecuencia variable)",
        "descripcion": "División clásica moderna. Agrupa músculos que trabajan juntos: Empuje (Pecho/Hombro/Tríceps), Tracción (Espalda/Bíceps) y Pierna.",
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
            "3. Face Pulls - 3x15 (Salud Hombro)",
            "4. Curl de Bíceps con Barra Z - 3x12",
            "5. Curl Martillo - 3x12",
            "--- SESIÓN C: LEGS (Pierna) ---",
            "1. Sentadilla Libre (High Bar) - 4x6-8",
            "2. Prensa de Piernas - 3x12",
            "3. Peso Muerto Rumano - 4x10",
            "4. Extensiones de Cuádriceps - 3x15",
            "5. Elevación de Talones (Pantorrilla) - 4x20"
        ]
    },
    # -----------------------------------------------------
    # CLUSTER 2: AVANZADOS -> ARNOLD SPLIT
    # -----------------------------------------------------
    2: {
        "id_rutina": "rut_arnold_01",
        "nombre": "Arnold Split (Élite)",
        "nivel": "Avanzado",
        "objetivo_principal": "Volumen Máximo y Puntos Débiles",
        "frecuencia_sugerida": "6 días por semana (Alta Intensidad)",
        "descripcion": "La división favorita de Schwarzenegger. Agrupa Pecho con Espalda (antagonistas) y Hombro con Brazos. Permite especializar el torso.",
        "ejercicios": [
            "--- DÍA 1: PECHO Y ESPALDA ---",
            "1. Press Banca Plano (Fuerza) - 5x5",
            "2. Dominadas Lastradas - 4x8",
            "3. Superserie: Press Inclinado + Remo con Mancuerna - 4x10",
            "4. Aperturas (Flyes) con Mancuernas - 3x15",
            "5. Pullover (Serratos/Dorsal) - 3x15",
            "--- DÍA 2: HOMBROS Y BRAZOS ---",
            "1. Press Militar sentado - 4x8",
            "2. Elevaciones Laterales - 4x15",
            "3. Superserie: Curl con Barra + Press Francés - 4x10",
            "4. Superserie: Curl Predicador + Extensión Copa - 3x12",
            "5. Curl de Muñeca (Antebrazo) - 3x20",
            "--- DÍA 3: PIERNA COMPLETA ---",
            "1. Sentadilla Frontal o Hack - 4x10",
            "2. Peso Muerto Convencional - 3x5 (Pesado)",
            "3. Zancadas Búlgaras - 3x12 por pierna",
            "4. Curl Femoral Tumbado - 4x15",
            "5. Gemelo en máquina Costurera - 4x15"
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
