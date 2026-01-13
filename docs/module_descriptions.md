I will read the requested files in `app/skills` and `app/ml` to understand their specific roles in the trading system and then generate the Markdown table.

He analizado los archivos solicitados en los directorios `app/skills` y `app/ml` para el proyecto FEAT Sniper. A continuación, presento la tabla con sus propósitos en el sistema de trading:

| Archivo | Descripción Corta |
| :--- | :--- |
| **Skills** | |
| `advanced_analytics.py` | Ejecuta modelos en modo sombra (Shadow Testing) y obtiene el sentimiento del mercado institucional. |
| `execution.py` | Gestiona el envío de órdenes con validaciones de riesgo, liquidez y mapeo inteligente de errores del broker. |
| `feat_aceleracion.py` | Valida la intención del precio mediante el análisis de momentum, volumen y detección de falsos rompimientos. |
| `feat_chain.py` | Orquestador principal que integra todos los módulos de FEAT (Física, Tiempo, Espacio, etc.) en una decisión única. |
| `feat_espacio.py` | Identifica zonas de liquidez institucional (FVG, Order Blocks) y calcula niveles de Premium/Discount. |
| `feat_forma.py` | Define el sesgo direccional del mercado analizando la estructura fractal, BOS (Break of Structure) y CHoCH. |
| `feat_tiempo.py` | Analiza el timing institucional basándose en sesiones globales y puntos de anclaje como los LBMA/SGE Fixes. |
| `liquidity_detector.py` | Detecta piscinas de liquidez no mitigadas y gestiona la activación de las Kill Zones operativas. |
| `market_physics.py` | Calcula métricas de física de mercado como Perfil de Volumen (PVP), CVD y probabilidades de breakout. |
| `trade_mgmt.py` | Gestiona el ciclo de vida de las posiciones abiertas, incluyendo trailing stops y cierres parciales. |
| **ML (Machine Learning)** | |
| `data_collector.py` | Captura y persiste ticks/velas en SQLite, aplicando etiquetado automático para entrenamiento de modelos. |
| `feat_processor.py` | Procesa datos crudos para generar el pipeline de ingeniería de características basado en la metodología FEAT. |
| `ml_engine.py` | Motor de inferencia principal que coordina modelos GBM y LSTM con detección de anomalías y modo sombra. |
| `multi_time_learning.py` | Gestiona pesos jerárquicos y alineación fractal entre diferentes temporalidades para la fusión de señales. |
| `temporal_features.py` | Convierte el conocimiento de dominio temporal (killzones, sesiones) en tensores numéricos para redes neuronales. |
| `train_models.py` | Pipeline de entrenamiento para modelos de Gradient Boosting (tabulares) y LSTM con atención (secuenciales). |