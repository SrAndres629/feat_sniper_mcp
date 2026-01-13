# PLAN DE ACCIÓN: MOTOR DE TIEMPO (TIME ENGINE)

## 1. Misión del Motor
El `TimeEngine` no es solo un reloj; es el **Sincronizador Institucional**. Su trabajo es decir *CUÁNDO* altas probabilidades de éxito chocan con la estructura (*DÓNDE*) y la aceleración (*CÓMO*).

## 2. Lógica "Kill Zone" (Vectorizada)
Implementaremos una detección matemática de ventanas de alta liquidez. No usaremos `if/else` simples bucles lentos, sino máscaras vectoriales de Pandas para velocidad extrema (como hicimos con `StructureEngine`).

### Ventanas Definidas (UTC Estándar):
1.  **London Kill Zone (LKZ):** 07:00 - 10:00 UTC. (Puntaje: 1.0)
    *   *Misión:* Capturar el "London Breakout" o inducir el "Judas Swing".
2.  **New York Kill Zone (NYKZ):** 12:00 - 15:00 UTC. (Puntaje: 1.0)
    *   *Misión:* Volatilidad máxima, cruce con Londres (London Close).
3.  **Asia Range (Dead Zone):** 20:00 - 06:00 UTC. (Puntaje: 0.1)
    *   *Misión:* Definir rango de acumulación para ser manipulado luego.

## 3. Arquitectura del Código (`nexus_core/time_engine.py`)

```python
class TimeEngine:
    def __init__(self):
        # Configuración rígida institucional
        self.killzones = {
            "LONDON": (7, 10),
            "NY": (12, 15),
            "ASIA_ACCUMULATION": (23, 6) # Rango previo
        }

    def score_phase_time(self, df: pd.DataFrame) -> pd.Series:
        """
        Retorna `feat_time` (0.0 a 1.0)
        """
        # 1. Extraer hora UTC vectorizada
        hours = df.index.hour
        
        # 2. Calcular proximidad al "Centro de Poder" (Media de la Killzone)
        # Usaremos una curva Gaussiana: 1.0 en el centro de la KZ, bajando a 0.5 en bordes.
        
        # 3. Filtro de Viernes/Lunes (Evitar trampas de liquidez baja)
        # weekday: 0=Mon, 4=Fri. Penalizar cierre de viernes.
        
        return time_score
```

## 4. Integración Macro (El "Guardian" de Noticias)
Planeo dejar un "Slot" en la clase para conectar un futuro `NewsFilter`.
*   *Idea:* Si hay `NFP` a las 13:30, el `TimeEngine` debe forzar `feat_time = 0` (Veto total) 30 minutos antes.

## 5. Salida del Motor (Feature T)
El motor entregará al `feat_processor`:
*   `feat_time_score`: (float) Probabilidad temporal.
*   `session_label`: (int/category) 1=London, 2=NY, 3=Asia.
*   `is_macro_event`: (bool) Flag de peligro.

---
**¿Te parece sólida esta estrategia de implementación?** Pone el foco en "Calidad de Tiempo" más que simplemente "Hora del día".
