import cv2
import numpy as np

class PanelDetectorMSER: 
    def __init__(self):
        #Inciar el detecto MSER
        #Los parámetros (delta, min_area, max_area...) tengo que ajustarlos para mejorar el resultado. 
        self.mser = cv2.MSER_create(delta=6, min_area=750, max_area=205000)
        self.blue_score_threshold = 0.4
        self.mask_height = 40
        self.mask_width = 80
        self.blue_lower = np.array([100, 200, 70], dtype=np.uint8)
        self.blue_upper = np.array([128, 255, 255], dtype=np.uint8)
        self.ideal_blue_mask = self._build_ideal_blue_mask()
        self.min_box_area = 600
        self.min_box_width = 18
        self.min_box_height = 18
        self.min_aspect_ratio = 0.5
        self.max_aspect_ratio = 4.0

    def _build_ideal_blue_mask(self):
        ideal_mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

        # Modelo más estricto de panel: interior azul y borde exterior no azul.
        border_y = int(self.mask_height * 0.20)
        border_x = int(self.mask_width * 0.16)
        ideal_mask[border_y:self.mask_height - border_y, border_x:self.mask_width - border_x] = 1
        return ideal_mask

    def _build_saturated_blue_mask(self, image, x1, y1, x2, y2):
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

        resized = cv2.resize(roi, (self.mask_width, self.mask_height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Convertir a valores {0, 1} como máscara binaria de azul saturado.
        return (mask > 0).astype(np.uint8)

    def _blue_ratio(self, image, x1, y1, x2, y2):
        blue_mask = self._build_saturated_blue_mask(image, x1, y1, x2, y2)
        return self._correlate_with_ideal_mask(blue_mask)

    def _correlate_with_ideal_mask(self, mask):
        if mask.shape != self.ideal_blue_mask.shape:
            return 0.0

        ideal = self.ideal_blue_mask
        tp = float(np.sum((mask == 1) & (ideal == 1)))
        fp = float(np.sum((mask == 1) & (ideal == 0)))
        tn = float(np.sum((mask == 0) & (ideal == 0)))
        positives = float(np.sum(ideal == 1))
        negatives = float(np.sum(ideal == 0))
        if positives <= 0.0:
            return 0.0

        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        specificity = tn / negatives if negatives > 0.0 else 0.0

        # La media armónica favorece regiones con azul bien colocado y penaliza el azul fuera de sitio.
        if (precision + recall) <= 0.0:
            return 0.0

        f1 = 2.0 * precision * recall / (precision + recall)
        return 0.55 * f1 + 0.45 * specificity
        
    def detect(self, image): 
        """
        Recibe imágen BGR y devuelve lista de boundig boxes : [x1, y1, x2, y2, score]
        El score es una similitud [0,1] basada en precision/recall frente a una máscara ideal.
        """
        
        boxes = []
        
        #1. Pasar la imagen a niveles de gris
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Mejorar contrastes si MSER no lo detecta bien 
        #gray = cv2.equalizeHist(gray)
        
        #2. Detectar regiones MSER
        regions, _ = self.mser.detectRegions(gray)
        
        alto_img, ancho_img = image.shape[:2]
        
        for p in regions: 
            #3. Pasar los píxeles de la región a un rectángulo
            x, y, w, h = cv2.boundingRect(p)

            if w < self.min_box_width or h < self.min_box_height:
                continue

            if w * h < self.min_box_area:
                continue
            
            #4. Filtrar por aspecto y tamaño
            aspect_ratio = w / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            #5. Agrandar un poco el rectángulo
            exp_w = int(w * 0.10)
            exp_h = int(h * 0.10)
            
            x1 = max(0, x - exp_w)
            y1 = max(0, y - exp_h)
            x2 = min(ancho_img , x + w + exp_w)
            y2 = min(alto_img, y + h + exp_h)

            #6. Filtrar regiones con suficiente contenido azul
            blue_corr = self._blue_ratio(image, x1, y1, x2, y2)
            if blue_corr < self.blue_score_threshold:
                continue
            
            #Añadir la caja a la lista.
            boxes.append([x1, y1, x2, y2, blue_corr])
    
        return boxes