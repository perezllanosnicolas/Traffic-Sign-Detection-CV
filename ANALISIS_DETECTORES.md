# Análisis Comparativo de Detectores de Paneles

## Resumen de Rendimiento

### Métricas cuantitativas

| Detector | AP@0.5 | AP@0.7 | Estrategia |
|----------|--------|--------|-----------|
| **MSER** (baseline) | 68.1% | - | Regiones maximalmente estables |
| **Color-First (detector_alt)** | 27.61% | 8.65% | Máscara HSV azul → validación Hough |
| **Hough Primary** | 1.1% | 0.06% | HoughLinesP global → intersecciones H×V |
| **Hybrid** | 24.02% | 10.10% | Máscara HSV azul → refinamiento Hough local |

---

## 1. MSER (Baseline)

**Rendimiento:** AP@0.5 = 68.1%

**Estrategia:**
- Detecta regiones maximalmente estables (máxima variación de umbral binario)
- Agrupa píxeles con propiedades de contraste estables
- Genera múltiples regiones por imagen (no específico a paneles)

**Ventajas:**
- Excelente rendimiento general
- No requiere sintonización de umbrales de color
- Robusto a variaciones de iluminación

**Desventajas:**
- Técnica de ML clásica, no permitida por especificación de la tarea
- Computacionalmente costosa
- No explota la característica distintiva de color azul

---

## 2. Color-First (detector_alt)

**Rendimiento:** 
- AP@0.5 = 27.61% 
- AP@0.7 = 8.65%

**Pipeline:**
```
1. Máscara HSV azul (H: 100-128, S: 200-255, V: 70-255)
2. Morfología: OPEN(3×3) + CLOSE(7×7, 2 iter)
3. Extracción de blobs + filtro geométrico
4. Expansión conservadora de bbox (15% ratio, max 25px)
5. Validación Hough en ROI local (H/V line detection)
6. Score por correlación con máscara ideal 80×40
```

**Características clave:**
- **Detector primario:** Máscara azul (muy específica a paneles)
- **Refinador:** Hough (valida pero no redefine coordenadas)
- **Expansión:** Pequeña (max 25px) para evitar sobreajuste

**Ventajas:**
- Máscara azul es muy específica a paneles de autopista
- Bajo ruido de falsas positivas (raro azul saturado fuera de paneles)
- Rápido: Hough solo en ROI local, no en imagen completa

**Limitaciones:**
- Bajo AP@0.7 (8.65%) → cajas imprecisas para IoU≥0.7
- Confiado en expansión fija, no se adapta a bordes blancos reales
- Hough usado para validación, no para refinamiento de coordenadas

---

## 3. Hough Primary (Análisis de por qué falla)

**Rendimiento:** AP@0.5 = 1.1%, AP@0.7 = 0.06%

**Pipeline:**
```
1. Preproceso: CLAHE + GaussianBlur + Canny (full image)
2. HoughLinesP en imagen completa
3. Clasificación H/V y agrupación de paralelas
4. Formar candidatos rectangulares (intersecciones H×V)
5. Validación HSV azul (filtro rápido)
6. Score por correlación de máscara
```

**Por qué falla:**

Debug output para imagen 00006.png (1920×1080, 1 panel real):

```
Canny edges:     50,204 píxeles de borde
HoughLinesP raw: 522 líneas detectadas
Clasificación:   421 H (horizontales), 47 V (verticales)
                 Ratio H/V = 9:1 ← PROBLEMA: carretera domina

Agrupación:      7 líneas H finales, 8 líneas V finales
Candidatos:      115 rectángulos (C(7,2)×C(8,2))
Azul filtering:  Solo 17/115 tienen azul (98 = 0% azul)
Resultado:       17 "detecciones" para 1 panel real
                 → sobre-segmentación (sub-regiones del mismo panel)
```

**Conclusión:** Las líneas Hough globales incluyen:
- Líneas de carretera (muy abundantes en horizontal)
- Bordes de edificios y estructuras
- Los paneles NO alinean perfectamente con estas líneas globales
- Genera múltiples intersecciones parciales del mismo panel

---

## 4. Hybrid (Color + Hough Refinamiento)

**Rendimiento:** 
- AP@0.5 = 24.02%
- AP@0.7 = 10.10%

**Pipeline:**
```
1. Máscara azul HSV (detector primario)
2. Extracción blobs + filtro geométrico
3. Expansión de zona de búsqueda: ±35px (para capturar bordos blancos)
4. HoughLinesP en zona de búsqueda local (no imagen completa)
5. Refinamiento: 
   - Líneas H arriba/abajo del blob → ajustar y1, y2
   - Líneas V izq/dcha del blob → ajustar x1, x2
   - Fallback: ±10px si pocas líneas
6. Score por correlación de máscara ideal
```

**Características clave:**
- **Detector primario:** Máscara azul (muy preciso)
- **Refinador:** Hough local (encuentra bordes blancos reales)
- **Expansión búsqueda:** 35px (suficiente para capturar borde blanco ~20px)
- **Refinamiento adaptativo:** Usa líneas detectadas o fallback conservador

**Ventajas:**
- Mejor AP@0.7 (10.10% vs 8.65% de color-first)
- Refina cajas basándose en bordes blancos reales, no expansión fija
- Hough ejecutado localmente → menos ruido de líneas globales
- Fallback conservador evita sobreajustes

**Trade-off respecto color-first:**
- Pierde 3.6% AP@0.5 (27.61% → 24.02%)
- Gana 1.5% AP@0.7 (8.65% → 10.10%)

**Interpretación del trade-off:**
- Color-first: Encuentra más paneles (mayor recall) pero con cajas menos precisas
- Hybrid: Encuentra fewer paneles pero con mejor localización exacta
- A IoU≥0.7, la precisión de localización es más importante que la cantidad de detecciones

Debug visual para imagen 00006.png:
```
Blob detectado:     (636,54)-(932,333) = 296×279 píxeles
Zona búsqueda:      (601,19)-(967,368) = ±35px expandida
Líneas Hough:       47 H, 17 V (locales a la zona)
Bbox refinado:      (626,48)-(940,341) = 314×293 píxeles
Refinamiento:       Δx1=-10, Δy1=-6, Δx2=+8, Δy2=+8 (pequeños ajustes)
Score:              0.576 (aceptado)
```

---

## Recomendaciones

### Para máximo AP@0.5 (recall):
**Usar: detector_alt (color-first)**
- 27.61% AP@0.5 es mejor que 24.02% del hybrid
- Más apropiado si importa encontrar el máximo de paneles

### Para máximo AP@0.7 (precision):
**Usar: detector_hybrid**
- 10.10% AP@0.7 es mejor que 8.65% del color-first
- Más apropiado si importa la localización exacta

### Conclusión general:
**No se alcanzan los 68.1% del MSER (baseline)** con técnicas puras de imagen.

El gap refleja que:
1. MSER captura patrones de estabilidad regional más robustos
2. El azul saturado y Hough son características débiles individualmente
3. La combinación híbrida es el mejor balance sin ML, pero fundamentalmente limitada

---

## Detalles técnicos finales

### Parámetros óptimos encontrados

**Blue mask (HSV):**
- H: [100-128] ← saturated blue only
- S: [200-255] ← CRÍTICO: S≥200 vs S≥150 cambia AP de 27.6% a 11%
- V: [70-255] ← permite sombras

**Morfología:**
- OPEN(3×3): Elimina ruido pequeño
- CLOSE(7×7, 2 iter): Conecta regiones fragmentadas

**Hough (local a ROI):**
- rho=1, theta=π/180
- threshold=15 (low tolerance para detectar líneas débiles)
- minLineLength=18% del lado menor del ROI
- maxLineGap=8% del lado menor del ROI

**Score:**
- 0.55×F1-score + 0.45×specificity
- threshold=0.25 para aceptar detección
- Máscara ideal: 80×40, interior azul, borde blanco

**Expansión (hybrid):**
- Zona búsqueda: ±35px
- Fallback refinement: ±10px
