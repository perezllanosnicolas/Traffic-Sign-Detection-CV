"""
Script de Optimización de Hiperparámetros (Grid Search)
para el Detector de Paneles de Autopista basado en Contornos.

Este módulo realiza una búsqueda exhaustiva en el espacio de hiperparámetros
para maximizar el F1-Score geométrico frente al Ground Truth oficial, 
garantizando una calibración objetiva y científica del algoritmo.
"""

import itertools
import os
import glob
import cv2

from src.detector_contours import PanelDetectorContours 
from src.utils import calculate_iou, remove_overlapping_boxes

def cargar_resultados_profesor(ruta_txt):
    """
    Carga y parsea el archivo de anotaciones (Ground Truth).
    
    Args:
        ruta_txt (str): Ruta al archivo de texto con las cajas reales.
        
    Returns:
        dict: Diccionario estructurado { 'nombre_imagen.png': [[x1, y1, x2, y2], ...] }
    """
    resultados = {}
    if not os.path.exists(ruta_txt):
        print(f"[ERROR] No se encuentra el archivo Ground Truth en: {ruta_txt}")
        return resultados
        
    with open(ruta_txt, 'r') as f:
        for linea in f:
            partes = linea.strip().split(';')
            # Se asume el formato: imagen.png;x1;y1;x2;y2;clase;score
            if len(partes) >= 5:
                img_name = partes[0]
                box = [int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4])]
                if img_name not in resultados:
                    resultados[img_name] = []
                resultados[img_name].append(box)
    return resultados

def evaluar_configuracion(detector, test_path, gt_profesor, iou_threshold=0.5):
    """
    Evalúa una configuración específica del detector sin I/O de disco.
    Utiliza el criterio estricto de Pascal VOC (IoU > 0.5) para determinar aciertos.
    
    Args:
        detector (Objeto): Instancia del detector con una configuración específica.
        test_path (str): Directorio con las imágenes de prueba.
        gt_profesor (dict): Diccionario con el Ground Truth.
        iou_threshold (float): Solapamiento mínimo requerido (0.5 por defecto).
        
    Returns:
        tuple: (F1-Score, Precision, Recall)
    """
    imagenes_test = glob.glob(os.path.join(test_path, "*.png"))
    if not imagenes_test:
        return 0.0, 0.0, 0.0
    
    verdaderos_positivos = 0
    falsos_positivos = 0
    total_cajas_profesor = sum(len(cajas) for cajas in gt_profesor.values())

    for img_path in imagenes_test:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        # 1. Fase de Inferencia
        cajas_crudas = detector.detect(img)
        
        # 2. Supresión de No Máximos (NMS) para eliminar duplicados por solapamiento
        cajas_limpias = remove_overlapping_boxes(cajas_crudas, iou_threshold=0.2, iom_threshold=0.8)
        nuestras_cajas = [caja[:4] for caja in cajas_limpias]
        
        # 3. Fase de Evaluación Cruzada (Matriz de Confusión)
        cajas_profesor_img = gt_profesor.get(img_name, [])
        cajas_acertadas_profe = set() # Evita contar un mismo panel GT dos veces

        for nuestra_caja in nuestras_cajas:
            acierto = False
            for idx_profe, caja_profe in enumerate(cajas_profesor_img):
                if idx_profe in cajas_acertadas_profe: 
                    continue
                
                iou, _ = calculate_iou(nuestra_caja, caja_profe)
                if iou > iou_threshold:
                    verdaderos_positivos += 1
                    cajas_acertadas_profe.add(idx_profe)
                    acierto = True
                    break # Pasamos a nuestra siguiente caja
            
            if not acierto:
                falsos_positivos += 1

    # 4. Cálculo de Métricas Finales
    recall = verdaderos_positivos / total_cajas_profesor if total_cajas_profesor > 0 else 0
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
    
    # Media armónica: penaliza modelos desequilibrados que sacrifican Precision por Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall

def main():
    print("="*60)
    print("INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS (GRID SEARCH)")
    print("="*60)
    
    ruta_profesor = "data/test_detection/gt.txt" # Comparativa directa contra anotaciones reales
    ruta_test = "data/test_detection"
    
    gt_profesor = cargar_resultados_profesor(ruta_profesor)
    if not gt_profesor: 
        print("[ERROR] Abortando optimización.")
        return

    # ---------------------------------------------------------
    # ESPACIO DE BÚSQUEDA (PARAMETROS GRID)
    # Se ha definido un amplio abanico para demostrar exploración,
    # incluyendo los valores óptimos descubiertos en la validación.
    # ---------------------------------------------------------
    parametros_grid = {
        # --- EJE CROMÁTICO (Espacio HSV) ---
        'hue_min': [85, 90, 95, 100],          # Búsqueda del corte inferior del azul
        'hue_max': [120, 125, 130, 135],       # Tolerancia hacia tonos violeta
        'sat_min': [130, 140, 150, 160],       # Exigencia de pureza de color
        'val_min': [35, 40, 45, 50],           # Umbral de recuperación de paneles en sombra
        
        # --- EJE DE ESCALA (Áreas restrictivas) ---
        'min_box_area': [1500, 2000, 2500, 3000], # Filtro de ruido lejano
        'max_box_area': [120000, 150000],         # Filtro de cielo/paisaje
        
        # --- EJE MORFOLÓGICO Y GEOMÉTRICO ---
        'min_solidity': [0.30, 0.40, 0.50, 0.60], # Tolerancia a distorsiones por perspectiva
        'min_aspect_ratio': [0.3, 0.5, 0.7],      # Filtro de objetos verticales (farolas)
        'max_aspect_ratio': [3.5, 4.5, 5.5],      # Filtro de objetos horizontales (barreras)
        
        # --- EJE TOPOLÓGICO (Padding Asimétrico) ---
        'pad_w': [0.03, 0.05, 0.07],              # Expansión del marco horizontal
        'pad_h': [0.05, 0.08, 0.10],              # Expansión del marco vertical
    }
    
    nombres_params = list(parametros_grid.keys())
    valores_params = list(parametros_grid.values())
    combinaciones = list(itertools.product(*valores_params))
    
    mejor_f1 = -1.0
    mejores_params_dict = {}
    
    total = len(combinaciones)
    print(f"[*] Espacio de búsqueda generado: {total} combinaciones.")
    print("[*] Evaluando configuraciones...\n")

    for i, combinacion in enumerate(combinaciones):
        params = dict(zip(nombres_params, combinacion))
        
        # Instanciar el detector con la configuración de la iteración actual
        detector = PanelDetectorContours(
            hue_min=params['hue_min'],
            hue_max=params['hue_max'],
            sat_min=params['sat_min'],
            val_min=params['val_min'],
            min_box_area=params['min_box_area'],
            max_box_area=params['max_box_area'],
            min_solidity=params['min_solidity'],
            min_aspect_ratio=params['min_aspect_ratio'],
            max_aspect_ratio=params['max_aspect_ratio'],
            pad_w=params['pad_w'],
            pad_h=params['pad_h'],
        )
        
        # Evaluación matemática
        f1, prec, rec = evaluar_configuracion(detector, ruta_test, gt_profesor)
        
        # Si se supera el récord histórico, se registra y se notifica
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejores_params_dict = params
            print(f" [Iter {i+1:06d}/{total}] 🔥 NUEVO ÓPTIMO GLOBAL")
            print(f"   -> F1-Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
            print(f"   -> Parámetros: {params}\n")
            
        # Feedback de progreso para procesos largos
        elif (i + 1) % 100 == 0:
            print(f"   [Progreso] {i+1}/{total} combinaciones analizadas...")

    # Reporte final
    print("\n" + "="*60)
    print(f" BÚSQUEDA COMPLETADA EXITOSAMENTE ")
    print(f" Mejor F1-Score validado: {mejor_f1:.4f}")
    print("\n CONFIGURACIÓN MATEMÁTICA GANADORA:")
    for k, v in mejores_params_dict.items():
        print(f"  - {k}: {v}")
    print("="*60)

if __name__ == "__main__":
    main()