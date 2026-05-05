import itertools
import os
import glob
import cv2
from src.detector_mser import PanelDetectorMSER
from src.utils import calculate_iou, remove_overlapping_boxes

def cargar_resultados_profesor(ruta_txt):
    """Carga las cajas de referencia del archivo del profesor."""
    resultados = {}
    if not os.path.exists(ruta_txt):
        print(f"Error: No se encuentra el archivo {ruta_txt}")
        return resultados
    with open(ruta_txt, 'r') as f:
        for linea in f:
            partes = linea.strip().split(';')
            if len(partes) >= 5:
                img_name = partes[0]
                box = [int(partes[1]), int(partes[2]), int(partes[3]), int(partes[4])]
                if img_name not in resultados:
                    resultados[img_name] = []
                resultados[img_name].append(box)
    return resultados

def evaluar_configuracion(detector, test_path, gt_profesor, iou_threshold=0.5):
    """Calcula métricas sin guardar imágenes en disco."""
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
            
        # 1. Detectar cajas con la configuración actual
        cajas_crudas = detector.detect(img)
        
        # 2. Filtrar solapamientos (Simulando el comportamiento de main.py)
        cajas_limpias = remove_overlapping_boxes(cajas_crudas, iou_threshold=0.2, iom_threshold=0.8)
        
        nuestras_cajas = [caja[:4] for caja in cajas_limpias]
        cajas_profesor_img = gt_profesor.get(img_name, [])
        cajas_acertadas_profe = set()

        for nuestra_caja in nuestras_cajas:
            acierto = False
            for idx_profe, caja_profe in enumerate(cajas_profesor_img):
                if idx_profe in cajas_acertadas_profe: 
                    continue
                
                # Usamos la función de utilidad para calcular el IoU
                iou, _ = calculate_iou(nuestra_caja, caja_profe)
                
                if iou > iou_threshold:
                    verdaderos_positivos += 1
                    cajas_acertadas_profe.add(idx_profe)
                    acierto = True
                    break
            
            if not acierto:
                falsos_positivos += 1

    # Cálculo de métricas finales
    recall = verdaderos_positivos / total_cajas_profesor if total_cajas_profesor > 0 else 0
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall

def main():
    print("Iniciando Optimización de Hiperparámetros (Solo métricas)...")
    
    # Configuración de rutas
    ruta_profesor = "data/resultado_jmbuena_road_panels.txt"
    ruta_test = "data/test_detection"
    
    gt_profesor = cargar_resultados_profesor(ruta_profesor)
    if not gt_profesor: 
        print("No se pudo cargar el Ground Truth. Abortando.")
        return

    # Definición del espacio de búsqueda
    parametros_grid = {
        'mser_min_area': [ 500],
        'hsv_hue_min': [90],
        'hsv_hue_max': [130],
        'hsv_sat_min': [150],
        'score_threshold': [0.2],
        'delta': [2],
        'mask_height': [40],
        'mask_width': [80],
        'min_box_area': [ 1000],
        'min_box_width': [18],
        'min_box_height': [18],
        'min_aspect_ratio': [0.6 ],
        'max_aspect_ratio': [4]
        
    }
    
    nombres_params = list(parametros_grid.keys())
    valores_params = list(parametros_grid.values())
    combinaciones = list(itertools.product(*valores_params))
    
    mejor_f1 = -1.0
    mejores_params_dict = {}
    
    total = len(combinaciones)
    print(f"Se van a probar {total} combinaciones distintas.\n")

    for i, combinacion in enumerate(combinaciones):
        params = dict(zip(nombres_params, combinacion))
        
        # Instanciar detector con los parámetros de la iteración
        detector = PanelDetectorMSER(
            min_area=params['mser_min_area'],
            hue_min=params['hsv_hue_min'],
            hue_max=params['hsv_hue_max'],
            sat_min=params['hsv_sat_min'],
            score_threshold=params['score_threshold'],
            delta=params['delta'],
            mask_height=params['mask_height'],
            mask_width=params['mask_width'],
            min_box_area=params['min_box_area'],
            min_box_width=params['min_box_width'],
            min_box_height=params['min_box_height'],
            min_aspect_ratio=params['min_aspect_ratio'],
            max_aspect_ratio=params['max_aspect_ratio']
        )
        
        f1, prec, rec = evaluar_configuracion(detector, ruta_test, gt_profesor)
        
        # Solo imprimimos si hay un nuevo récord para mantener la consola limpia
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejores_params_dict = params
            print(f" [{i+1}/{total}] NUEVO RÉCORD -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
            print(f"   Parámetros: {params}\n")
        elif (i + 1) % 10 == 0:
            # Feedback cada 10 iteraciones para saber que sigue vivo
            print(f"   Progreso: {i+1}/{total} combinaciones analizadas...")

    print("\n" + "="*50)
    print(f" OPTIMIZACIÓN COMPLETA ")
    print(f"Mejor F1-Score: {mejor_f1:.4f}")
    print("\nConfiguración recomendada:")
    for k, v in mejores_params_dict.items():
        print(f"  {k} = {v}")
    print("="*50)

if __name__ == "__main__":
    main()