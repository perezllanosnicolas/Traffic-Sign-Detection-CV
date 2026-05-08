import cv2
import numpy as np

class PanelDetectorContours: 
    """
    Detector alternativo de paneles de información de autopista basado en 
    segmentación cromática (HSV), análisis de componentes conexas y filtrado geométrico.
    """
    
    def __init__(self, hue_min=95, hue_max=125, sat_min=150, val_min=45, score_threshold=0.2, 
                 mask_height=40, mask_width=80, min_box_area=2500, max_box_area=150000, 
                 min_aspect_ratio=0.5, max_aspect_ratio=4.5,
                 min_solidity=0.40, pad_w=0.05, pad_h=0.08, edge_penalty=-2.0):
        
        # --- Parámetros de Color (Espacio HSV) ---
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min
        
        # --- Parámetros Geométricos (Dimensiones y Formas) ---
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        
        # --- Parámetros de Topología y Puntuación (Score) ---
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.score_threshold = score_threshold
        self.mask_height = mask_height
        self.mask_width = mask_width
        
        # Matriz de máscara ideal: Premia el núcleo (+1), penaliza el perímetro perimetral paramétricamente
        self.mascara_ideal = np.ones((self.mask_height, self.mask_width), dtype=np.float32)
        self.mascara_ideal[0:4, :] = edge_penalty
        self.mascara_ideal[-4:, :] = edge_penalty
        self.mascara_ideal[:, 0:4] = edge_penalty
        self.mascara_ideal[:, -4:] = edge_penalty

    def detect(self, image): 
        """
        Procesa una imagen BGR y devuelve una lista de bounding boxes detectados.
        Formato de salida: [[x1, y1, x2, y2, score], ...]
        """
        boxes = []
        alto_img, ancho_img = image.shape[:2]
        
        # ---------------------------------------------------------
        # FASE 1: SEGMENTACIÓN CROMÁTICA
        # ---------------------------------------------------------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_blue = np.array([self.hue_max, 255, 255])
        
        # Aislamos los píxeles que coinciden con la firma lumínica de los paneles
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # ---------------------------------------------------------
        # FASE 2: EXTRACCIÓN DE COMPONENTES CONEXAS
        # ---------------------------------------------------------
        # Se extraen exclusivamente las siluetas exteriores (RETR_EXTERNAL) 
        # para ignorar posibles huecos internos generados por la iconografía.
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours: 
            # -----------------------------------------------------
            # FASE 3: FILTRADO GEOMÉTRICO ESTRICTO
            # -----------------------------------------------------
            area_contorno = cv2.contourArea(c)
            
            # Descarte de ruido por tamaño absoluto
            if area_contorno < (self.min_box_area * 0.4) or area_contorno > self.max_box_area:
                continue
                
            x, y, w, h = cv2.boundingRect(c)

            # Filtro de dimensiones mínimas operativas
            if (w * h) < self.min_box_area or (w * h) > self.max_box_area:
                continue
            
            # Filtro de Proporción (Aspect Ratio): Excluye farolas y líneas del horizonte
            aspect_ratio = w / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
                
            # Filtro de Solidez: Erradica detecciones de objetos irregulares (ej. vehículos)
            solidity = area_contorno / float(w * h)
            if solidity < self.min_solidity:
                continue
            
            # -----------------------------------------------------
            # FASE 4: ALINEACIÓN TOPOLÓGICA (PADDING ASIMÉTRICO)
            # -----------------------------------------------------
            # Expansión asimétrica de la caja proporcional a la máscara ideal (80x40).
            # Centra la masa azul evitando solapamientos erróneos con la zona de castigo.
            exp_w = int(w * self.pad_w)
            exp_h = int(h * self.pad_h)
            
            x1 = max(0, x - exp_w)
            y1 = max(0, y - exp_h)
            x2 = min(ancho_img, x + w + exp_w)
            y2 = min(alto_img, y + h + exp_h)

            roi = image[y1:y2, x1:x2]
            
            # Mecanismo de seguridad ante recortes fuera de los límites de la imagen original
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
            
            # -----------------------------------------------------
            # FASE 5: EVALUACIÓN HEURÍSTICA DEL SCORE
            # -----------------------------------------------------
            # Redimensionado al tamaño de evaluación y extracción del azul
            roi_resized = cv2.resize(roi, (self.mask_width, self.mask_height))
            roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            mask_roi_blue = cv2.inRange(roi_hsv, lower_blue, upper_blue)
            
            # Cierre morfológico interno (Text Eraser): Maximiza la densidad azul 
            # fagocitando los huecos generados por las letras blancas.
            kernel_text = np.ones((5, 5), np.uint8)
            mask_roi_blue = cv2.morphologyEx(mask_roi_blue, cv2.MORPH_CLOSE, kernel_text)

            # Normalización binarizada [0, 1]
            mask_blue_norm = mask_roi_blue.astype(np.float32) / 255.0
            
            # Correlación matemática frente a la máscara ideal restrictiva
            matriz_correlacion = mask_blue_norm * self.mascara_ideal
            correlacion = np.sum(matriz_correlacion)
            
            max_posible = np.sum(self.mascara_ideal[self.mascara_ideal > 0])
            score = correlacion / max_posible
            
            # Acotamiento del score al rango de confianza [0, 1]
            score = max(0.0, min(1.0, score))
            
            # Inclusión final si supera el umbral de confianza métrica
            if score > self.score_threshold:
                boxes.append([x1, y1, x2, y2, score])
                    
        return boxes