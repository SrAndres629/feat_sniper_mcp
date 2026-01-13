# 🗺️ FEAT Sniper: Mapa Mental del Proyecto

Este documento proporciona una visión jerárquica y detallada de la arquitectura de FEAT Sniper MCP, analizando cada módulo, carpeta y archivo clave.

## 📊 Visualización de Arquitectura (Mermaid)

```mermaid
mindmap
  root((FEAT Sniper))
    Entrada Compartida
      mcp_server.py[Servidor MCP Principal]
      nexus_control.py[Controlador de Sistemas]
      nexus_auditor.py[Auditor de Salud]
    app/core
      Logger[Gestión de logs y trazas]
      Config[Configuración centralizada]
      MT5 Connection[Puente con MetaTrader 5]
      ZMQ Bridge[Comunicación asíncrona]
    app/skills (Trading Ops)
      Capas Fractales
        Tiempo[Ciclos y Duración]
        Forma[Estructura y Patrones]
        Espacio[Rangos y Expansión]
        Aceleración[Velocidad y Momentum]
      Liquidity Detector[Zonas de liquidez HFT]
      Execution Engine[Gestión de Órdenes]
      Trade Management[SL/TP/Breakeven]
    app/ml (Inteligencia)
      Motores de Inferencia[GBM + LSTM]
      Aprendizaje MTF[Análisis Multi-Temporal]
      Feature Engineering[Procesamiento de Datos]
    brain_core (Memoria RAG)
      Infinite Memory[RAG con ChromaDB]
      db_engine[Gestión de Narrativas]
      Drift Monitor[Detección de degradación]
    n8n_workflows
      Automatización[Workflows de CI/CD]
      Triggers[Eventos del Sistema]
    dashboard
      Frontend[Dashboard Tiempo Real]
      Visualización[Métricas de Performance]
    tools
      gemini_client[IA Peer-to-Peer]
      map_project[Cartógrafo de Dependencias]
```

---

## 📂 Desglose Detallado de Módulos

### 🚀 Entrada y Control (Root)
| Archivo | Propósito |
| :--- | :--- |
| `mcp_server.py` | Punto de entrada para el protocolo MCP, expone las 10 Master Tools. |
| `nexus_control.py` | Coordinador de procesos, gestiona el ciclo de vida de los servicios. |
| `nexus_auditor.py` | Verifica dependencias, puertos, salud de Docker y MT5. |
| `nexus.bat` | Script de arranque unificado del sistema. |

### 🛠️ Núcleo del Sistema (`app/core`)
| Archivo | Propósito |
| :--- | :--- |
| `logger.py` | Implementa filtrado de recursividad y logging estructurado. |
| `config.py` | Gestiona variables de entorno y constantes del sistema. |
| `mt5_conn.py` | Handler de la conexión nativa con MetaTrader 5. |
| `zmq_bridge.py` | Bridge para comunicación inter-proceso de alta velocidad. |

### 🎯 Habilidades de Trading (`app/skills`)
| Archivo | Propósito |
| :--- | :--- |
| `feat_tiempo.py` | **Capa 1**: Analiza la duración de los movimientos y ciclos de mercado. |
| `feat_forma.py` | **Capa 2**: Identifica patrones geométricos y quiebres de estructura. |
| `feat_espacio.py` | **Capa 3**: Mide la expansión del precio y objetivos de Fibonacci. |
| `feat_aceleracion.py` | **Capa 4**: Detecta clímax y cambios bruscos en la volatilidad. |
| `liquidity_detector.py` | Identifica "Value Areas" y zonas donde reside la liquidez institucional. |
| `execution.py`| Realiza el envío físico de órdenes y validación de lotaje. |
| `trade_mgmt.py` | Gestiona el trailing stop y el riesgo de posiciones abiertas. |

### 🧠 Inteligencia Artificial (`app/ml`)
| Archivo | Propósito |
| :--- | :--- |
| `ml_engine.py` | Coordina las predicciones de los modelos híbridos. |
| `multi_time_learning.py` | Sincroniza predicciones entre diferentes marcos temporales (H1, M15, M1). |
| `feat_processor.py` | Transforma datos brutos de MT5 en tensores para los modelos. |
| `train_models.py` | Pipeline de entrenamiento automático basado en datos históricos. |

### 📂 Gestión de Memoria (`brain_core`)
| Archivo | Propósito |
| :--- | :--- |
| `feat_memory_mcp.py` | Implementa el almacenamiento RAG para "recuerdo" de trades pasados. |
| `db_engine.py` | Gestiona la persistencia de datos en bases de datos relacionales y vectoriales. |
| `drift_monitor.py` | Alerta si el modelo deja de ser efectivo debido a cambios en el mercado. |

---

> [!NOTE]
> Este mapa se genera dinámicamente analizando la arquitectura actual del proyecto FEAT Sniper MCP.