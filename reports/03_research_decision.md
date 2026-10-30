# Decisión de investigación y protocolo congelado

Fecha: 2026-10-30. **Escrito y versionado antes de ejecutar cualquier experimento de evaluación.**
Decisión del equipo: alternativa A. Sin "digital twin" como contribución y sin conmutación en ruta.

## Pregunta

¿Cómo debe entrar la evidencia propioceptiva a un mapa de riesgo para reducir el peligro acumulado
en una misión, y cuánto depende de los errores de percepción y de la configuración (patas o ruedas,
fija antes de la misión)?

## Hipótesis (primaria primero)

- **H1 (primaria).** Propagar evidencia por apariencia (M2) aumenta la probabilidad de supervivencia
  de la misión frente a actualizar solo lo atravesado (M1) y frente a solo geometría (M0).
- **H2.** M1 mejora a M0.
- **H3.** La fusión con incertidumbre (M3) mejora a M2 cuando hay errores de clasificación o
  apariencia ambigua.
- **H4 (física).** En MuJoCo, un clasificador ligero distingue terrenos parametrizados en episodios no
  vistos, y la fusión IMU + articulaciones supera a cada canal por separado, en F1 macro.
- **H5.** La utilidad de la propagación difiere entre patas y ruedas **sin** asumir tablas de fallo
  distintas: solo por la matriz de confusión medida.

**Qué las refuta.** H1: el IC 95 % de la diferencia pareada de M2−M1 incluye o queda bajo 0.
H3: lo mismo para M3−M2. H4: F1 macro de fusión sin mejora frente al mejor canal en IC pareado.
H5: IC de la interacción configuración × método que incluye 0. Los resultados negativos se publican.

## Métricas

- **Primaria:** supervivencia analítica por misión, `P_s = exp(−H)`, con `H` la integral de peligro
  verdadero sobre la trayectoria ejecutada; 0 si no alcanza la meta dentro del presupuesto.
- **Secundarias:** éxito realizado (umbral exponencial común a todos los métodos), exposición en metros
  a terreno no nominal, longitud, tiempo de planificación, replanificaciones, reducción de riesgo
  frente a M0.

## Unidad estadística

El mapa. Las réplicas de un mapa se promedian antes de comparar. IC 95 % por bootstrap pareado de
mapas (10 000 remuestreos), Wilcoxon de rangos con signo por mapa, corrección de Holm dentro de cada
familia, tamaño del efecto dz y correlación biserial de rangos. Wilson para proporciones cuando las
observaciones son independientes (una réplica por mapa).

## Conjuntos

- **Desarrollo:** mapas 100000–100049. Solo para ajustar λ, σ_k, τ, ρ, κ.
- **Evaluación:** mapas 0–199. No se usan para ajustar nada.
- **Clasificador:** partición por episodio (entrenamiento / calibración / prueba), nunca por ventana.
  La matriz con la que el método razona viene de calibración y las observaciones simuladas se sortean
  de prueba.

## Clases de evidencia

- Monte Carlo: ALGORITHMIC-MONTE-CARLO. Las tablas de peligro son ASSUMED y se barren.
- MuJoCo: MEASURED-SIMULATION, con controlador de ruedas propio y fricción parametrizada.
- El Monte Carlo es **independiente** del documento A; no lo reproduce, porque su script no existe.
