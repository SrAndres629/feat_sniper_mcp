---
trigger: always_on
---

A partir de ahora, activa el Protocolo de Consistencia en Cascada.

Cada vez que yo te pida modificar un archivo central (como nexus.bat, docker-compose.yml o scripts principales de Python), tu trabajo NO termina al editar ese archivo. Debes actuar proactivamente y verificar toda la cadena de dependencias:

Scan de Dependencias: Si cambias nexus.bat, verifica inmediatamente los scripts que este llama (ej. nexus_auditor.py, nexus_control.py) para asegurar que reciban los argumentos correctos.

Librerías: Si añades código nuevo, verifica automáticamente si faltan imports o si hay que actualizar requirements.txt.

Variables: Si añades una variable nueva (ej. BTC_SYMBOL), verifica que esté definida en .env y en config.py.

Tu nueva directiva: No esperes a que yo te pida arreglar los errores secundarios. Si tocas una pieza del motor, asume que debes recalibrar las piezas conectadas. Entrégame siempre la solución completa: el archivo principal modificado Y los archivos satélites actualizados para que todo funcione a la primera."