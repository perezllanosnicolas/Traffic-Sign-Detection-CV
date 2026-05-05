def calculate_iou (boxA, boxB):
    """ 
    Calcula la Intersección sobre la Unión (IoU) de dos bounding boxes.
    box= [x1, y1, x2, y2]
    """
    
    #Coordenadas de la intesección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    #Área de intersección
    interArea = max(0, xB - xA) * max (0, yB- yA)
    
    #Área de ambas cajas
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
     #Calcular IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    minArea = min(boxAArea, boxBArea)
    iom = interArea / float(minArea) if minArea > 0 else 0.0
    return iou, iom


def remove_overlapping_boxes( boxes, iou_threshold=0.2, iom_threshold=0.8):
    
    """
    Elimina cajas repetidas basándose en el solapamiento.
    boxes = lista de [x1, y1, x2, y2, score]
    """
    
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True) #Ordenar por score

    kept_boxes = []
    
    for box in boxes:
        overlap = False
        for kept_box in kept_boxes:
            #Calcular el solapamiento con las ya guardadas
            iou, iom = calculate_iou(box[:4], kept_box[:4])
            if iou > iou_threshold or iom > iom_threshold:
                    overlap = True
                    break
            
        if not overlap:
            kept_boxes.append(box)
        
    return kept_boxes
            


 