import argparse
import os
import glob
import cv2

from src.detector_mser import PanelDetectorMSER
from src.detector_alt import PanelDetectorAlt
from src.detector_hough_primary import PanelDetectorHoughPrimary
from src.detector_hybrid import PanelDetectorHybrid
from src.utils import remove_overlapping_boxes

def parse_args():
    parser = argparse.ArgumentParser(description="Detector de Paneles de Autopista")
    parser.add_argument("--train_path", type=str, required=True, help="Ruta al directorio de entrenamiento")
    parser.add_argument("--test_path", type=str, required=True, help="Ruta al directorio de test")
    parser.add_argument("--detector", type=str, required=True, choices=['mser', 'hough', 'hough_primary', 'hybrid'], help= "Nombre del detector a usar")
    return parser.parse_args()


def main():
        args = parse_args()
        
        #1. Crear directorio de resultados si no existe
        output_dir = "resultado_imgs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        #2. Instanciar el detector elegido
        if args.detector == 'mser':
            print("Usando el detector MSER")
            detector = PanelDetectorMSER()
        elif args.detector == 'hough':
            print("Usando el detector Hough (color primero)")
            detector = PanelDetectorAlt()
        elif args.detector == 'hough_primary':
            print("Usando el detector Hough (Hough primero)")
            detector = PanelDetectorHoughPrimary()
        elif args.detector == 'hybrid':
            print("Usando el detector Híbrido (color + Hough refinamiento)")
            detector = PanelDetectorHybrid()
            
        #3. Preparar el archivo de salida de texto
        output_txt_path = "resultado.txt"
        
        #4. Procesar imágenes del test
        test_images = glob.glob(os.path.join(args.test_path, "*.png"))
        
        with open(output_txt_path, 'w') as f_out : 
            for img_path in test_images:
                img_name= os.path.basename(img_path)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Error al cargar la imagen {img_name}")
                    continue
                
                #Llamar al detector 
                bbox_list = detector.detect(img)
                
                #Filtrar repetidos
                bbox_list = remove_overlapping_boxes(bbox_list, iou_threshold=0.1)
                
                #Dibujar y guardar el resultado 
                img_result = img.copy()
                for bbox in bbox_list:
                    x1, y1, x2, y2, score = bbox
                    
                    #FORMATO DE SALIDA : <nombre_fichero>;<x1>;<y1>;<x2>;<y2>;<tipo>;<score>
                    #El tipo siempre es 1
                    f_out.write(f"{img_name};{x1};{y1};{x2};{y2};1;{score:.3f}\n")
                    
                    #Dibujar el rectángulo rojo y texto amarillo: 
                    cv2.rectangle(img_result, (x1, y1,), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_result, f"{score:.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                #Guardar la imagen resultante
                cv2.imwrite(os.path.join(output_dir, img_name), img_result)
                
        print(f"Proceso finalizado. Resultados guardados en '{output_dir}/' y '{output_txt_path}' .")

if __name__ == "__main__":
    main()