# Firebase Real-time Object Tracker

Sistema de tracking de objetos controlado por Firebase Realtime Database.

## 🎯 Funcionamiento

1. **Escucha Firebase** - Monitorea el nodo `instruction` en tiempo real
2. **Captura Pantalla** - Toma screenshot cuando detecta cambio
3. **Genera Training Set** - Usa Gemini para detectar objetos y crear dataset aumentado
4. **Tracking Selectivo** - Rastrea solo los objetos que coinciden con el training set

## 🔧 Configuración Firebase

### 1. Crear Proyecto Firebase

1. Ve a https://console.firebase.google.com/
2. Crea un nuevo proyecto
3. Activa **Realtime Database**

### 2. Obtener Credenciales

1. En Firebase Console → ⚙️ Settings → Service Accounts
2. Click "Generate new private key"
3. Guarda el archivo JSON como `firebase_credentials.json` en la raíz del proyecto

### 3. Configurar `.env`

Agrega estas variables a tu archivo `.env`:

```env
FIREBASE_DB_URL=https://your-project-id.firebaseio.com
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Estructura de Base de Datos

Tu Firebase Realtime Database debe tener esta estructura:

```json
{
  "instruction": "woman with curly hair"
}
```

## 🚀 Uso

### Iniciar el Listener:

```bash
python src/firebase_tracker_controller.py
```

### Cambiar Instrucción desde Firebase Console:

1. Ve a tu proyecto en Firebase Console
2. Abre Realtime Database
3. Edita el nodo `instruction` con una nueva instrucción
4. Ejemplo: `"person wearing red shirt"`

### El sistema automáticamente:

1. ✅ Captura screenshot
2. ✅ Detecta objetos con Gemini
3. ✅ Genera 16 imágenes aumentadas por detección
4. ✅ Inicia tracking selectivo en webcam

## 🔄 Cambio de Instrucción en Tiempo Real

Puedes cambiar la instrucción en cualquier momento:

```json
// Antes
{
  "instruction": "woman with curly hair"
}

// Después
{
  "instruction": "person with glasses"
}
```

El sistema:
- ⏸️ Detiene el tracking actual
- 📸 Captura nueva screenshot
- 🔄 Regenera training set
- ▶️ Reinicia tracking con nuevo objetivo

## 🧵 Threading

El sistema usa **3 hilos**:

1. **Main Thread** - Listener de Firebase
2. **Process Thread** - Procesa instrucciones (screenshot + training set)
3. **Tracking Thread** - Ejecuta tracking de video sin bloquear

Esto asegura que:
- ✅ La cámara nunca se bloquea
- ✅ Firebase responde instantáneamente
- ✅ Múltiples procesos ejecutan en paralelo

## 📁 Archivos Generados

```
temp_screenshots/           # Screenshots capturados
  └── screenshot_*.png
training_set/output/        # Training set generado
  └── [label]/
      └── det0_*.png (16 variantes)
results/                    # Logs de tracking
  └── *.txt
```

## 🎮 Controles

- **Tracking Window**: Presiona `q` para cerrar
- **Listener**: Presiona `Ctrl+C` para detener

## 📝 Ejemplo de Uso

1. **Iniciar sistema:**
   ```bash
   python src/firebase_tracker_controller.py
   ```

2. **En Firebase Console, crear nodo:**
   ```json
   {
     "instruction": "person wearing blue shirt"
   }
   ```

3. **El sistema:**
   - Captura la pantalla
   - Detecta personas con camisa azul
   - Genera 16 variantes de entrenamiento
   - Inicia tracking en webcam

4. **Cambiar objetivo en tiempo real:**
   ```json
   {
     "instruction": "person with backpack"
   }
   ```

## ⚠️ Requisitos

- Python 3.11+
- Webcam activa
- Cuenta Firebase con Realtime Database
- API Key de Gemini
- Dependencias instaladas (ver requirements.txt)

## 🔒 Seguridad

**IMPORTANTE**: 
- NO subas `firebase_credentials.json` a Git
- NO compartas tu FIREBASE_DB_URL públicamente
- Agrega reglas de seguridad en Firebase:

```json
{
  "rules": {
    "instruction": {
      ".read": "auth != null",
      ".write": "auth != null"
    }
  }
}
```

## 🐛 Troubleshooting

### "Firebase credentials not found"
- Descarga las credenciales de Firebase Console
- Guárdalas como `firebase_credentials.json` en la raíz

### "FIREBASE_DB_URL not set"
- Agrega `FIREBASE_DB_URL` a tu `.env`
- Formato: `https://your-project-id.firebaseio.com`

### "No detections found"
- La instrucción debe ser clara y específica
- Asegúrate que el objeto esté visible en pantalla

### Tracking no inicia
- Verifica que la webcam esté disponible
- Cambia `device='cuda:0'` a `device='cpu'` si no tienes GPU
