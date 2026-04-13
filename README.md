# ReconocimientoDeSe-alesVisi-nArtificialPr-ctica1

##  Estructura del Proyecto


```text
/
├── .gitignore               # Archivos y carpetas a ignorar (data/, resultado_imgs/, __pycache__)
├── README.md                
├── main.py                  # Script principal de ejecución
├── evaluar_resultados.py    # Script de evaluación (proporcionado por los profesores)
│
├── src/                     # Código fuente modularizado
│   ├── __init__.py
│   ├── utils.py             # Funciones auxiliares (cálculo de IoU, dibujo de bounding boxes, I/O)
│   ├── detector_mser.py     # Clase principal: Detección usando MSER + Máscaras HSV
│   └── detector_alt.py      # Clase alternativa
│
└── memoria/                 # Documentación del proyecto
    └── memoria_practica1.pdf
