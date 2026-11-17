# Perfiles de micronegocios en el Valle del Cauca · ACP y K-Means

Taller individual de **Machine Learning aplicado a Economía y Finanzas** (Pontificia Universidad Javeriana Cali).
Aplicación en Streamlit que recorre un análisis de aprendizaje no supervisado sobre la Encuesta de Micronegocios
(EMICRON) 2024 del DANE.

- **Aplicación desplegada:** _pegar aquí el enlace de Streamlit Community Cloud_
- **Video de sustentación:** _pegar aquí el enlace_

## Contexto y pregunta

Los micronegocios (hasta nueve personas ocupadas) suelen tratarse como un grupo homogéneo. La pregunta del
proyecto es si los micronegocios del Valle del Cauca se diferencian por unas pocas dimensiones de fondo
(escala, trayectoria, adopción digital) y si, a partir de ellas, pueden reconocerse perfiles que requieran
apoyos distintos de política pública.

- **ACP** es pertinente porque las variables vienen en bloques correlacionados (ventas–personal,
  edad–antigüedad, dispositivos–uso de internet) y conviene comprobar cuántas dimensiones hay realmente.
- **K-Means** es pertinente porque no existe una tipología oficial que combine tamaño, trayectoria y
  tecnología: es un problema sin etiquetas.

## Datos

| | |
|---|---|
| Fuente | DANE, Encuesta de Micronegocios (EMICRON) 2024, microdatos públicos anonimizados: <https://microdatos.dane.gov.co/index.php/catalog/875> |
| Unidad de observación | Un micronegocio (llave `DIRECTORIO + SECUENCIA_P + SECUENCIA_ENCUESTA`) |
| Cobertura | Valle del Cauca (departamento 76), enero a diciembre de 2024 |
| Tamaño | 4.228 micronegocios encuestados; 3.989 entran al análisis |
| Módulos usados | Identificación, ventas o ingresos, emprendimiento y TIC |

El archivo `data/micronegocios_valle.csv` ya está en el repositorio. Para regenerarlo desde los microdatos
originales, descarga los cuatro módulos en CSV en una carpeta y ejecuta:

```bash
python scripts/construir_base.py ruta/a/la/carpeta
```

### Variables del modelo

| Variable | Definición | Origen |
|---|---|---|
| `ventas_mes` | Ventas del mes anterior (pesos) | `VENTAS_MES_ANTERIOR` |
| `margen` | Ingreso mixto / ventas × 100 | `INGRESO_MIXTO`, `VENTAS_MES_ANTERIOR` |
| `personal_ocupado` | Propietario + trabajadores pagos + socios + no remunerados | `P3032_1`, `P3032_2`, `P3032_3` |
| `antiguedad_meses` | Meses que el propietario lleva en el negocio | `P3034` |
| `edad_propietario` | Edad en años | `P241` |
| `dispositivos_tic` | Computadores de escritorio + portátiles + tabletas + teléfonos inteligentes | `P1087`, `P1088`, `P977`, `P978` |
| `personas_internet` | Personas del negocio que usan internet | `P980` |

Variables descriptivas que **no** entran al modelo y se usan para caracterizar los grupos: sector, zona, sexo
del propietario, motivo de creación, uso de internet, redes sociales, correo y nombre comercial. También se
conservan `costos_mes` e `ingreso_mixto` en pesos.

### Limitaciones de la fuente

- Ventas y costos son declarados por el propietario y corresponden a un solo mes.
- No se usan los factores de expansión: los resultados describen la muestra, no el total del departamento.
- Personal, dispositivos y personas que usan internet son conteos con muy pocos valores distintos.

## Metodología

1. **Exploración:** estructura, tipos, faltantes, duplicados, descriptivas, distribuciones y correlaciones.
2. **Preparación** (`analisis.preparar`):
   - se excluyen identificadores y factor de expansión del modelo;
   - se excluyen 239 negocios sin ventas en el mes (su margen no existe y con logaritmos formarían un grupo artificial);
   - logaritmo de las ventas, winsorización en percentiles 1 y 99 y estandarización (media 0, desviación 1).
3. **ACP** (`analisis.ajustar_acp`): varianza por componente y acumulada, gráfico de sedimentación, cargas e
   interpretación. El número de componentes se decide con los resultados (Kaiser, 80 % acumulado o manual).
4. **K-Means** (`analisis.evaluar_k`, `analisis.ajustar_kmeans`): se aplica sobre los componentes
   conservados; K se compara con el método del codo y la silueta; los grupos se describen con las variables
   originales y se validan con variables externas.

Todo se calcula en vivo: la aplicación no muestra imágenes ni resultados fijos.

## Resultados principales (configuración por defecto)

- **Tres componentes** con valor propio mayor que 1 retienen el **69,7 %** de la varianza (para llegar al 80 %
  harían falta cuatro). Ninguno domina: CP1 explica 31,7 %.
  - CP1, modernización y tamaño: dispositivos, uso de internet y ventas.
  - CP2, trayectoria: antigüedad y edad del propietario.
  - CP3, rentabilidad: margen.
- **K = 4** tiene la mayor silueta (0,346), muy cerca de K = 3 (0,338). El codo no marca un quiebre único.
  La estructura es débil a moderada: son perfiles típicos, no categorías cerradas.

| Perfil | % muestra | Ventas (mediana) | Margen | Personal | Antigüedad | Edad | Usa internet |
|---|---|---|---|---|---|---|---|
| Unipersonales jóvenes y conectados | 47 % | $1.500.000 | 50 % | 1 | 4 años | 39 | 99 % |
| Larga trayectoria | 19 % | $1.700.000 | 59 % | 1 | 27 años | 62 | 68 % |
| Mayor escala | 10 % | $6.196.500 | 20 % | 2 | 10 años | 46 | 92 % |
| Tradicionales desconectados | 24 % | $800.000 | 44 % | 1 | 5 años | 54 | 2 % |

Los grupos definidos por lo digital son los más nítidos (silueta 0,44 y 0,34); los de larga trayectoria y
mayor escala son difusos (0,22 y 0,20). El 11 % de los micronegocios queda en zona de frontera.

## Cómo ejecutar

```bash
git clone <url-del-repositorio>
cd <carpeta>
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Probado con Python 3.13.

## Estructura

```
app.py                      Aplicación Streamlit (interfaz y gráficos)
analisis.py                 Preparación, ACP y K-Means (pandas, numpy, scikit-learn)
scripts/construir_base.py   Construye la base desde los microdatos del DANE
data/micronegocios_valle.csv
requirements.txt
.streamlit/config.toml      Identidad visual
```

## Uso de inteligencia artificial

Se usó un asistente de IA como apoyo en la programación y la redacción. Las decisiones metodológicas y la
interpretación fueron revisadas y son sustentadas por el autor.

## Autor

Santiago Loaiza · Pontificia Universidad Javeriana Cali
