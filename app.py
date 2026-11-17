"""
Perfiles de micronegocios en el Valle del Cauca (EMICRON 2024)
ACP y K-Means · Taller individual de Machine Learning aplicado a Economía y Finanzas

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analisis as an

st.set_page_config(page_title="Micronegocios del Valle · ACP y K-Means", page_icon="🏪", layout="wide")

# ---------------------------------------------------------------- identidad visual
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGENTE = [[0, "#c0392b"], [0.5, "#f0efec"], [1, "#1c5cab"]]
TINTA, GRIS = "#1f2933", "#7b8794"
SEMILLA_DEF, K_MAX = 42, 8


def color_grupos(grupos):
    return {g: COLORES[ord(g) - 65] for g in sorted(set(grupos))}


def estilo(fig, alto=420):
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=60, b=10), font=dict(color=TINTA, size=13),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb", title_font_size=16,
        legend=dict(orientation="h", y=-0.18, title_text=""),
    )
    fig.update_xaxes(gridcolor="#e4e7eb", zerolinecolor="#cbd2d9")
    fig.update_yaxes(gridcolor="#e4e7eb", zerolinecolor="#cbd2d9")
    return fig


def pesos(x):
    return f"${x:,.0f}".replace(",", ".")


def formato_perfil(tabla):
    """Da formato legible a una tabla de perfiles (filas = grupos)."""
    t = tabla.copy()
    for c in t.columns:
        if c in ("ventas_mes", "costos_mes", "ingreso_mixto"):
            t[c] = t[c].map(pesos)
        elif c == "margen":
            t[c] = t[c].map(lambda v: f"{v:.0f} %")
        elif c == "antiguedad_meses":
            t[c] = t[c].map(lambda v: f"{v:.0f} meses ({v / 12:.0f} años)")
        elif c not in ("Micronegocios", "% de la muestra"):
            t[c] = t[c].map(lambda v: f"{v:.0f}")
    return t.rename(columns={**an.ETIQUETAS, **an.APOYO})


# ---------------------------------------------------------------- cálculo (en caché)
@st.cache_data
def datos():
    return an.cargar_datos()


@st.cache_data
def paso_preparacion(variables, transformar, winsorizar):
    d, original, bit = an.preparar(datos(), list(variables), transformar, winsorizar)
    z = an.estandarizar(d, list(variables))
    return d, original, bit, z


@st.cache_data
def paso_acp(variables, transformar, winsorizar):
    return an.ajustar_acp(paso_preparacion(variables, transformar, winsorizar)[3])


def matriz_kmeans(variables, transformar, winsorizar, base, n_comp):
    z = paso_preparacion(variables, transformar, winsorizar)[3]
    if base == "Componentes retenidos":
        return paso_acp(variables, transformar, winsorizar)[0].iloc[:, :n_comp]
    return z


@st.cache_data
def paso_evaluar_k(variables, transformar, winsorizar, base, n_comp, semilla):
    return an.evaluar_k(matriz_kmeans(variables, transformar, winsorizar, base, n_comp), K_MAX, semilla)


@st.cache_data
def paso_kmeans(variables, transformar, winsorizar, base, n_comp, k, semilla):
    return an.ajustar_kmeans(matriz_kmeans(variables, transformar, winsorizar, base, n_comp), k, semilla)


# ---------------------------------------------------------------- panel lateral
st.sidebar.title("🏪 Micronegocios del Valle")
st.sidebar.caption("ACP y K-Means · EMICRON 2024 (DANE)")
SECCIONES = ["1. Contexto", "2. Datos", "3. Preparación", "4. ACP", "5. K-Means", "6. Conclusiones"]
seccion = st.sidebar.radio("Recorrido", SECCIONES)

st.sidebar.divider()
st.sidebar.subheader("Parámetros del análisis")
TODAS = list(an.VARIABLES)
variables = st.sidebar.multiselect("Variables del modelo", TODAS, default=TODAS, format_func=an.ETIQUETAS.get)
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos cuatro variables.")
    st.stop()
variables = tuple(v for v in TODAS if v in variables)
transformar = st.sidebar.checkbox("Logaritmo de las ventas", True, help="Reduce la asimetría de las ventas.")
winsorizar = st.sidebar.checkbox("Winsorizar (percentiles 1 y 99)", True, help="Recorta valores extremos sin eliminar filas.")

d, original, bit, z = paso_preparacion(variables, transformar, winsorizar)
puntajes, varianza, cargas = paso_acp(variables, transformar, winsorizar)
sug = an.sugerir_componentes(varianza)

criterio = st.sidebar.radio(
    "Componentes a conservar", ["Kaiser (valor propio > 1)", "Varianza acumulada ≥ 80 %", "Manual"],
    help="La sugerencia se recalcula con los datos; no es una cifra fija.",
)
if criterio.startswith("Kaiser"):
    n_comp = max(sug["kaiser"], 2)
elif criterio.startswith("Varianza"):
    n_comp = max(sug["acumulada"], 2)
else:
    n_comp = st.sidebar.slider("Número de componentes", 2, len(variables), max(sug["kaiser"], 2))
st.sidebar.caption(f"Se conservan **{n_comp}** componentes ({varianza['Acumulada (%)'][n_comp - 1]:.1f} % de la varianza).")

base = st.sidebar.radio("K-Means se aplica sobre", ["Componentes retenidos", "Variables estandarizadas"])
semilla = st.sidebar.number_input("Semilla aleatoria", 0, 9999, SEMILLA_DEF)
eval_k = paso_evaluar_k(variables, transformar, winsorizar, base, n_comp, semilla)
k_mejor = int(eval_k.loc[eval_k["Silueta"].idxmax(), "K"])
k = st.sidebar.slider("Número de grupos (K)", 2, K_MAX, k_mejor, help="Por defecto, el K con mayor silueta promedio.")

grupos, siluetas = paso_kmeans(variables, transformar, winsorizar, base, n_comp, k, semilla)
paleta = color_grupos(grupos)
perfil_z = an.perfil_estandarizado(z, grupos)
sil_media = float(siluetas.mean())
ES_DEFECTO = (
    variables == tuple(TODAS) and transformar and winsorizar and criterio.startswith("Kaiser")
    and base == "Componentes retenidos" and semilla == SEMILLA_DEF and k == k_mejor
)


def rasgos(grupo, umbral=0.4, maximo=3):
    """Variables en las que el grupo se aleja más del promedio de la muestra."""
    fila = perfil_z.loc[grupo]
    orden = fila.abs().sort_values(ascending=False)
    partes = [f"{an.ETIQUETAS[v].lower()} {'alta' if fila[v] > 0 else 'baja'}" for v in orden.index[:maximo] if abs(fila[v]) >= umbral]
    return ", ".join(partes) if partes else "cerca del promedio en todas las variables"


def lectura_componente(cp, umbral=0.5):
    col = cargas[cp]
    pos = [an.ETIQUETAS[v] for v in col.index if col[v] >= umbral]
    neg = [an.ETIQUETAS[v] for v in col.index if col[v] <= -umbral]
    texto = []
    if pos:
        texto.append("crece con **" + ", ".join(pos) + "**")
    if neg:
        texto.append("disminuye con **" + ", ".join(neg) + "**")
    return " y ".join(texto) if texto else "no tiene cargas superiores a 0,5 en valor absoluto"


# ================================================================ 1. CONTEXTO
if seccion == SECCIONES[0]:
    st.title("¿Qué perfiles de micronegocios existen en el Valle del Cauca?")
    st.markdown(
        "**Taller individual · ACP y K-Means** — Machine Learning aplicado a Economía y Finanzas, "
        "Pontificia Universidad Javeriana Cali."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Micronegocios encuestados", f"{bit['inicial']:,}".replace(",", "."))
    c2.metric("Entran al análisis", f"{bit['final']:,}".replace(",", "."))
    c3.metric("Variables cuantitativas", len(TODAS))

    st.header("El problema")
    st.markdown(
        """
Los micronegocios (unidades de hasta nueve personas ocupadas) son la forma más común de generar ingresos en
Colombia, pero suelen tratarse como un bloque homogéneo: "el tendero", "el informal". Una política de formalización,
crédito o digitalización diseñada para un micronegocio "promedio" puede no servirle a casi ninguno.

**Pregunta que orienta el ejercicio:** ¿los micronegocios del Valle del Cauca se diferencian por unas pocas
dimensiones de fondo (escala, trayectoria, adopción digital) y, a partir de ellas, pueden reconocerse perfiles
que requieran apoyos distintos?
"""
    )
    st.header("Los datos")
    st.markdown(
        """
- **Fuente:** Encuesta de Micronegocios (EMICRON) 2024 del DANE, microdatos públicos y anonimizados.
- **Unidad de observación:** cada fila es un micronegocio y su propietario, identificado por la llave
  `DIRECTORIO + SECUENCIA_P + SECUENCIA_ENCUESTA`.
- **Cobertura:** departamento del Valle del Cauca (código 76), los doce meses de 2024, cabeceras y zona rural.
- **Módulos utilizados:** identificación, ventas o ingresos, emprendimiento y TIC, unidos por la llave anterior.
"""
    )
    st.dataframe(
        pd.DataFrame(
            [{"Variable": an.VARIABLES[v][0], "Qué mide": an.VARIABLES[v][1]} for v in TODAS]
        ),
        hide_index=True, width="stretch",
    )
    st.header("Por qué ACP y K-Means")
    st.markdown(
        """
- **ACP.** Las siete variables no son independientes: ventas y personal se mueven juntas, lo mismo que edad y
  antigüedad, o dispositivos y personas que usan internet. El ACP permite comprobar si detrás hay pocas dimensiones
  y cuáles son, en lugar de suponerlas.
- **K-Means.** No existe una clasificación oficial de "tipos" de micronegocio que combine tamaño, trayectoria y
  tecnología. Es un problema sin etiqueta previa: se busca si los datos sugieren grupos y qué tan nítidos son.
- **Juntos.** Agrupar sobre los componentes evita que bloques de variables redundantes pesen doble en las distancias.
"""
    )
    st.header("Limitaciones de la fuente")
    st.markdown(
        """
- Las cifras de ventas y costos son **declaradas** por el propietario y corresponden a **un solo mes**.
- La encuesta tiene diseño muestral con factores de expansión; aquí cada micronegocio pesa igual, así que los
  resultados describen **la muestra**, no el total expandido del departamento.
- Varias variables son **conteos pequeños** (personal, dispositivos), con poca variación entre negocios.
"""
    )

# ================================================================ 2. DATOS
elif seccion == SECCIONES[1]:
    st.title("Exploración de los datos")
    df = datos()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}".replace(",", "."))
    c2.metric("Columnas", df.shape[1])
    c3.metric("Duplicados por id", int(df.duplicated("id").sum()))
    c4.metric("Sin ventas en el mes", bit["sin_ventas"])

    st.subheader("Estructura de la base")
    roles = {**{v: "Cuantitativa (modelo)" for v in TODAS}, **{v: "Cuantitativa (apoyo)" for v in an.APOYO},
             "id": "Identificador", "factor_expansion": "Diseño muestral"}
    estructura = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": [str(t) for t in df.dtypes],
        "Papel": [roles.get(c, "Descriptiva") for c in df.columns],
        "Faltantes": df.isna().sum().values,
        "Valores distintos": df.nunique().values,
    })
    st.dataframe(estructura, hide_index=True, width="stretch")
    st.caption(
        "El identificador, el factor de expansión y las variables descriptivas (sector, zona, sexo, uso de internet…) "
        "no entran al modelo; se usan después para caracterizar los grupos. Los faltantes del margen corresponden a "
        "negocios sin ventas en el mes: no se puede dividir entre cero."
    )
    with st.expander("Ver una muestra de la base"):
        st.dataframe(df.head(50), hide_index=True, width="stretch")

    st.subheader("Estadísticas descriptivas (unidades originales)")
    desc = df[TODAS + list(an.APOYO)].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.index = [an.ETIQUETAS.get(i, an.APOYO.get(i)) for i in desc.index]
    desc.columns = ["Media", "Desv. estándar", "Mínimo", "P25", "Mediana", "P75", "Máximo"]
    st.dataframe(desc.style.format("{:,.1f}"), width="stretch")
    st.markdown(
        "**Lectura.** En ventas la media duplica a la mediana y el máximo es cientos de veces mayor: unos pocos "
        "negocios grandes estiran la distribución. Tres de cada cuatro micronegocios tienen una sola persona ocupada "
        "y un solo dispositivo. El margen tiene mínimos muy negativos (meses con pérdida)."
    )

    st.subheader("Distribución de cada variable")
    var = st.selectbox("Variable", TODAS, format_func=an.ETIQUETAS.get)
    escala = st.radio("Escala del eje horizontal", ["Original", "Logarítmica"], horizontal=True,
                      disabled=var != "ventas_mes")
    serie = df[var].dropna()
    if var == "ventas_mes" and escala == "Logarítmica":
        serie = np.log10(serie[serie > 0])
        titulo_x = "Ventas del mes (log10 de pesos)"
    else:
        titulo_x = f"{an.VARIABLES[var][0]} ({an.VARIABLES[var][1]})"
    fig = px.histogram(serie, nbins=50, color_discrete_sequence=[COLORES[0]],
                       title=f"Distribución de {an.ETIQUETAS[var].lower()}")
    fig.update_layout(showlegend=False, bargap=0.05)
    fig.update_xaxes(title=titulo_x)
    fig.update_yaxes(title="Número de micronegocios")
    st.plotly_chart(estilo(fig, 380), width="stretch")

    st.subheader("Relación entre variables")
    metodo = st.radio("Coeficiente", ["Spearman (rangos)", "Pearson (lineal)"], horizontal=True,
                      help="Spearman es menos sensible a valores extremos.")
    corr = df[TODAS].corr(method="spearman" if metodo.startswith("Spearman") else "pearson")
    nombres = [an.ETIQUETAS[v] for v in TODAS]
    fig = go.Figure(go.Heatmap(z=corr.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=DIVERGENTE,
                               text=corr.round(2).values, texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(title="Correlación")))
    fig.update_layout(title="Matriz de correlaciones entre las variables del modelo")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(fig, 520), width="stretch")
    st.markdown(
        "**Lectura.** Se ven tres bloques: (1) **ventas y personal**, que hablan del tamaño; (2) **edad y "
        "antigüedad**, que hablan de la trayectoria; y (3) **dispositivos y personas que usan internet**, que hablan "
        "de la adopción digital. La edad se relaciona de forma negativa con lo digital. Esa redundancia por bloques "
        "es justamente lo que el ACP puede resumir."
    )

    st.subheader("Composición de la muestra")
    cat = st.selectbox("Variable descriptiva", ["sector", "zona", "sexo_propietario", "motivo_creacion"],
                       format_func=lambda c: c.replace("_", " ").capitalize())
    cuenta = df[cat].value_counts().sort_values()
    fig = px.bar(x=cuenta.values, y=cuenta.index, orientation="h", color_discrete_sequence=[COLORES[0]],
                 text=cuenta.values, title=f"Micronegocios según {cat.replace('_', ' ')}")
    fig.update_xaxes(title="Número de micronegocios")
    fig.update_yaxes(title="")
    st.plotly_chart(estilo(fig, 150 + 45 * len(cuenta)), width="stretch")

# ================================================================ 3. PREPARACIÓN
elif seccion == SECCIONES[2]:
    st.title("Preparación de los datos")
    st.markdown("Cada decisión responde a un problema observado en la exploración. Los interruptores del panel "
                "lateral permiten ver qué cambia si no se toma.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros iniciales", bit["inicial"])
    c2.metric("Duplicados eliminados", bit["duplicados"])
    c3.metric("Excluidos sin ventas", bit["sin_ventas"])
    c4.metric("Registros finales", bit["final"])

    st.markdown(
        f"""
| # | Problema | Decisión | Justificación |
|---|---|---|---|
| 1 | Columnas que no son características del negocio (`id`, factor de expansión) | No entran al modelo | Un identificador no mide nada; meterlo crearía distancias sin sentido |
| 2 | Preguntas de conteo vacías (ayudantes, equipos) | Se reemplazan por 0 al construir la base | El DANE solo hace la pregunta a quien respondió "sí" en el filtro; vacío significa "no tiene" |
| 3 | {bit['sin_ventas']} negocios sin ventas en el mes de referencia | Se excluyen | Su margen no existe y, con logaritmos, quedarían tan lejos del resto que K-Means los aislaría como un grupo artificial |
| 4 | Ingreso mixto en pesos casi duplica la información de las ventas y tiene valores negativos | Se usa el **margen** (ingreso mixto / ventas) | Separa "cuánto vende" de "cuánto le queda por cada peso vendido" |
| 5 | Ventas muy asimétricas | Logaritmo natural {'✅' if transformar else '❌ (desactivado)'} | Pasar de 1 a 2 millones pesa igual que pasar de 10 a 20: importan las diferencias proporcionales |
| 6 | Valores extremos (márgenes de −2.800 %, 9 ocupados, 10 dispositivos) | Winsorización en percentiles 1 y 99 {'✅' if winsorizar else '❌ (desactivado)'} | Conserva la observación pero limita su influencia en medias, varianzas y centroides |
| 7 | Escalas distintas (pesos, años, meses, conteos) | **Estandarización** a media 0 y desviación 1 | ACP y K-Means se basan en varianzas y distancias; sin esto mandaría la variable con números más grandes |
"""
    )
    st.subheader("Antes y después")
    var = st.selectbox("Variable", list(variables), format_func=an.ETIQUETAS.get)
    c1, c2 = st.columns(2)
    for col, serie, titulo, eje, color in [
        (c1, original[var], "Original", an.VARIABLES[var][1], GRIS),
        (c2, z[var], "Preparada y estandarizada", "desviaciones estándar respecto a la media", COLORES[0]),
    ]:
        fig = px.histogram(serie, nbins=40, color_discrete_sequence=[color], title=f"{an.ETIQUETAS[var]} · {titulo}")
        fig.update_layout(showlegend=False, bargap=0.05)
        fig.update_xaxes(title=eje)
        fig.update_yaxes(title="Micronegocios")
        col.plotly_chart(estilo(fig, 340), width="stretch")

    st.subheader("Comprobación de la estandarización")
    comp = pd.DataFrame({"Media": z.mean(), "Desviación estándar": z.std(ddof=0), "Asimetría": z.skew()})
    comp.index = [an.ETIQUETAS[v] for v in comp.index]
    st.dataframe(comp.style.format("{:.2f}"), width="stretch")
    st.caption("Todas las variables quedan con media 0 y desviación 1. La asimetría que persiste en personal ocupado "
               "se debe a que la gran mayoría de negocios tiene una sola persona: es una limitación del dato, no del "
               "procedimiento.")

# ================================================================ 4. ACP
elif seccion == SECCIONES[3]:
    st.title("Análisis de Componentes Principales")
    acum = varianza["Acumulada (%)"][n_comp - 1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{acum:.1f} %")
    c3.metric("Sugerencia Kaiser / 80 %", f"{sug['kaiser']} / {sug['acumulada']}")

    st.subheader("Varianza explicada")
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(x=varianza["Componente"], y=varianza["Varianza explicada (%)"], name="Por componente",
                marker_color=[COLORES[0] if i < n_comp else "#cbd2d9" for i in range(len(varianza))])
    fig.add_scatter(x=varianza["Componente"], y=varianza["Acumulada (%)"], name="Acumulada",
                    mode="lines+markers", line=dict(color=TINTA, width=2))
    fig.add_hline(y=80, line_dash="dot", line_color=GRIS, annotation_text="80 %")
    fig.update_layout(title="Gráfico de sedimentación: varianza por componente y acumulada")
    fig.update_yaxes(title="% de la varianza total", range=[0, 105])
    fig.update_xaxes(title="Componente principal (en azul, los conservados)")
    c1.plotly_chart(estilo(fig), width="stretch")
    c2.dataframe(varianza.style.format({"Valor propio": "{:.2f}", "Varianza explicada (%)": "{:.1f}",
                                        "Acumulada (%)": "{:.1f}"}), hide_index=True, width="stretch")
    st.markdown(
        f"""
**¿Cuántos conservar?** Los dos criterios no coinciden, y eso es informativo:

- **Kaiser** (valor propio mayor que 1, es decir, componentes que explican más que una variable original): **{sug['kaiser']}**.
- **Varianza acumulada de al menos 80 %**: **{sug['acumulada']}**.

Con la selección actual se conservan **{n_comp} componentes, que resumen el {acum:.1f} %** de la variación de las
{len(variables)} variables; se pierde el {100 - acum:.1f} %. Ningún componente domina: el primero explica apenas
{varianza['Varianza explicada (%)'][0]:.0f} %. Estos datos **no se dejan reducir a dos dimensiones** sin perder
bastante información.
"""
    )

    st.subheader("Cargas: qué significa cada componente")
    sub = cargas.iloc[:, :n_comp]
    fig = go.Figure(go.Heatmap(z=sub.values, x=sub.columns, y=[an.ETIQUETAS[v] for v in sub.index], zmin=-1, zmax=1,
                               colorscale=DIVERGENTE, text=sub.round(2).values, texttemplate="%{text}",
                               xgap=2, ygap=2, colorbar=dict(title="Carga")))
    fig.update_layout(title="Correlación de cada variable con los componentes conservados")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(fig, 430), width="stretch")
    st.markdown("Una carga cercana a 1 o −1 indica que la variable se mueve con el componente; cercana a 0, que no "
                "le aporta. Lectura automática con cargas de al menos 0,5 en valor absoluto:")
    for cp in sub.columns:
        st.markdown(f"- **{cp}** ({varianza.set_index('Componente').loc[cp, 'Varianza explicada (%)']:.1f} %): "
                    f"{lectura_componente(cp)}.")
    if ES_DEFECTO:
        st.info(
            "**Interpretación.** CP1 es un eje de **modernización y tamaño**: negocios más conectados, con más "
            "ventas y personal, y propietarios algo más jóvenes. CP2 es un eje de **trayectoria**: antigüedad y edad, "
            "acompañadas de algo más de escala. CP3 es un eje de **rentabilidad**: margen alto en negocios sin "
            "ayudantes. Que lo digital pese más que las ventas en el primer componente sugiere que la brecha "
            "tecnológica es la principal fuente de diferencias entre micronegocios del Valle."
        )

    st.subheader("Los micronegocios en el espacio de componentes")
    opciones = list(puntajes.columns[:max(n_comp, 2)])
    c1, c2, c3 = st.columns(3)
    eje_x = c1.selectbox("Eje horizontal", opciones, 0)
    eje_y = c2.selectbox("Eje vertical", opciones, 1)
    color_por = c3.selectbox("Colorear por", ["sector", "zona", "sexo_propietario", "usa_internet"],
                             format_func=lambda c: c.replace("_", " ").capitalize())
    pv = varianza.set_index("Componente")["Varianza explicada (%)"]
    plano = puntajes[[eje_x, eje_y]].copy() if eje_x != eje_y else puntajes[[eje_x]].assign(**{eje_y + " ": puntajes[eje_y]})
    plano["color"] = original[color_por].map({0: "No", 1: "Sí"}).fillna(original[color_por]) if color_por == "usa_internet" else original[color_por]
    fig = px.scatter(plano, x=plano.columns[0], y=plano.columns[1], color="color", opacity=0.55,
                     color_discrete_sequence=COLORES,
                     title=f"Micronegocios en el plano {eje_x}–{eje_y}, según {color_por.replace('_', ' ')}")
    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(title=f"{eje_x} ({pv[eje_x]:.1f} % de la varianza)")
    fig.update_yaxes(title=f"{eje_y} ({pv[eje_y]:.1f} % de la varianza)")
    st.plotly_chart(estilo(fig, 520), width="stretch")
    suma = pv[eje_x] + (pv[eje_y] if eje_x != eje_y else 0)
    st.warning(f"Este plano muestra solo el **{suma:.1f} %** de la variación total. Dos puntos que aquí parecen "
               "vecinos pueden estar lejos en las dimensiones que no se ven.")

# ================================================================ 5. K-MEANS
elif seccion == SECCIONES[4]:
    st.title("Segmentación con K-Means")
    st.markdown(
        f"El algoritmo se aplica sobre **{base.lower()}** "
        + (f"({n_comp} componentes, {varianza['Acumulada (%)'][n_comp - 1]:.1f} % de la varianza). Así los bloques de "
           "variables redundantes no cuentan doble y se descarta la variación residual."
           if base == "Componentes retenidos" else f"({len(variables)} variables, sin reducción previa).")
    )
    st.subheader("¿Cuántos grupos?")
    c1, c2 = st.columns(2)
    fig = px.line(eval_k, x="K", y="Inercia", markers=True, color_discrete_sequence=[COLORES[0]],
                  title="Método del codo: inercia según K")
    fig.add_vline(x=k, line_dash="dot", line_color=GRIS)
    fig.update_yaxes(title="Inercia (suma de distancias² al centroide)")
    fig.update_xaxes(title="Número de grupos (K)", dtick=1)
    c1.plotly_chart(estilo(fig, 360), width="stretch")
    fig = px.line(eval_k, x="K", y="Silueta", markers=True, color_discrete_sequence=[COLORES[1]],
                  title="Coeficiente de silueta promedio según K")
    fig.add_vline(x=k, line_dash="dot", line_color=GRIS)
    fig.update_yaxes(title="Silueta promedio (−1 a 1; mayor es mejor)")
    fig.update_xaxes(title="Número de grupos (K)", dtick=1)
    c2.plotly_chart(estilo(fig, 360), width="stretch")

    orden = eval_k.sort_values("Silueta", ascending=False).reset_index(drop=True)
    caida = eval_k.assign(Reduccion=-eval_k["Inercia"].pct_change() * 100)
    st.markdown(
        f"""
- **Codo.** La inercia siempre baja al aumentar K; interesa dónde deja de bajar con fuerza. De K=2 a K=3 cae
  {caida['Reduccion'][1]:.0f} %, de 3 a 4 cae {caida['Reduccion'][2]:.0f} % y de 4 a 5 cae {caida['Reduccion'][3]:.0f} %.
  No hay un quiebre único y evidente.
- **Silueta.** El valor más alto está en **K = {int(orden['K'][0])}** ({orden['Silueta'][0]:.3f}), seguido de
  K = {int(orden['K'][1])} ({orden['Silueta'][1]:.3f}). La diferencia entre los primeros es pequeña.
- **Elección actual: K = {k}**, con silueta promedio de **{sil_media:.3f}**. Valores entre 0,25 y 0,50 suelen leerse
  como estructura débil a moderada: hay grupos, pero con fronteras difusas.
"""
    )
    st.dataframe(eval_k.style.format({"Inercia": "{:,.0f}", "Silueta": "{:.3f}"}), hide_index=True)

    st.subheader("Los grupos en el espacio de componentes")
    pv = varianza.set_index("Componente")["Varianza explicada (%)"]
    vista = puntajes[["CP1", "CP2"]].assign(Grupo=grupos).sort_values("Grupo")
    fig = px.scatter(vista, x="CP1", y="CP2", color="Grupo", color_discrete_map=paleta, opacity=0.6,
                     title=f"Grupos de K-Means (K = {k}) sobre los dos primeros componentes")
    fig.update_traces(marker=dict(size=6))
    cen = vista.groupby("Grupo")[["CP1", "CP2"]].mean()
    fig.add_scatter(x=cen["CP1"], y=cen["CP2"], mode="markers+text", text=cen.index, textposition="top center",
                    textfont=dict(size=16, color=TINTA),
                    marker=dict(symbol="x", size=14, color=TINTA, line=dict(width=2, color="white")), name="Centroide")
    fig.update_xaxes(title=f"CP1 ({pv['CP1']:.1f} % de la varianza)")
    fig.update_yaxes(title=f"CP2 ({pv['CP2']:.1f} % de la varianza)")
    st.plotly_chart(estilo(fig, 520), width="stretch")
    st.caption(f"El plano muestra el {pv['CP1'] + pv['CP2']:.1f} % de la variación. Si dos grupos se superponen aquí, "
               "pueden estar separados por un componente que no se ve.")

    st.subheader("Descripción de los grupos con las variables originales")
    st.markdown("Las letras son solo nombres: **no indican orden, tamaño ni jerarquía**. Con otra semilla el mismo "
                "grupo puede recibir otra letra.")
    perfil = an.perfil_grupos(original, list(variables) + list(an.APOYO), grupos)
    perfil.insert(1, "% de la muestra", (perfil["Micronegocios"] / len(original) * 100).map(lambda v: f"{v:.0f} %"))
    st.markdown("**Mediana de cada variable por grupo** (la mediana resiste mejor los valores extremos que la media)")
    st.dataframe(formato_perfil(perfil), width="stretch")

    fig = go.Figure(go.Heatmap(z=perfil_z.values, x=[an.ETIQUETAS[v] for v in perfil_z.columns],
                               y=[f"Grupo {g}" for g in perfil_z.index], zmin=-2, zmax=2, colorscale=DIVERGENTE,
                               text=perfil_z.round(2).values, texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(title="Desv. est.")))
    fig.update_layout(title="Qué distingue a cada grupo: distancia al promedio de la muestra (en desviaciones estándar)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(fig, 150 + 55 * k), width="stretch")
    for g in perfil_z.index:
        st.markdown(f"- **Grupo {g}** ({(grupos == g).sum()} micronegocios): {rasgos(g)}.")

    st.markdown("**Características que no entraron al modelo** (sirven para validar si los grupos tienen sentido)")
    extra = original.groupby(grupos)[list(an.DESCRIPTIVAS)].mean().mul(100).rename(columns=an.DESCRIPTIVAS)
    extra["Propietaria mujer (%)"] = (original["sexo_propietario"] == "Mujer").groupby(grupos).mean() * 100
    extra["En cabecera municipal (%)"] = (original["zona"] == "Cabecera municipal").groupby(grupos).mean() * 100
    extra.index.name = "Grupo"
    st.dataframe(extra.style.format("{:.0f}"), width="stretch")
    sector = pd.crosstab(grupos, original["sector"], normalize="index").mul(100).reset_index(names="Grupo").melt(
        "Grupo", var_name="Sector", value_name="pct")
    fig = px.bar(sector, y="Grupo", x="pct", color="Sector", orientation="h", color_discrete_sequence=COLORES,
                 text=sector["pct"].map(lambda v: f"{v:.0f}%"), title="Composición sectorial de cada grupo")
    fig.update_traces(marker_line=dict(width=2, color="#fcfcfb"))
    fig.update_xaxes(title="% de los micronegocios del grupo")
    fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(estilo(fig, 150 + 50 * k), width="stretch")

    motivo = pd.crosstab(original["motivo_creacion"], grupos, normalize="columns").mul(100)
    motivo = motivo.loc[motivo.mean(axis=1).sort_values(ascending=False).index]
    motivo.columns = [f"Grupo {g}" for g in motivo.columns]
    motivo.index.name = "Motivo principal para crear el negocio (% del grupo)"
    st.dataframe(motivo.style.format("{:.0f}"), width="stretch")

    st.subheader("¿Qué tan bien separados están?")
    sil = pd.DataFrame({"Grupo": grupos, "Silueta": siluetas}).sort_values(["Grupo", "Silueta"]).reset_index(drop=True)
    sil["pos"] = range(len(sil))
    fig = px.bar(sil, x="Silueta", y="pos", color="Grupo", orientation="h", color_discrete_map=paleta,
                 title="Gráfico de silueta: una barra por micronegocio")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(bargap=0)
    fig.add_vline(x=sil_media, line_dash="dot", line_color=TINTA, annotation_text=f"promedio {sil_media:.2f}")
    fig.update_yaxes(title="Micronegocios ordenados dentro de cada grupo", showticklabels=False)
    fig.update_xaxes(title="Coeficiente de silueta (negativo: más cerca de otro grupo que del propio)")
    st.plotly_chart(estilo(fig, 480), width="stretch")
    resumen = sil.groupby("Grupo")["Silueta"].agg(Promedio="mean", Negativas=lambda s: (s < 0).mean() * 100,
                                                  Dudosas=lambda s: (s < 0.1).mean() * 100)
    st.dataframe(resumen.style.format({"Promedio": "{:.3f}", "Negativas": "{:.1f} %", "Dudosas": "{:.1f} %"}),
                 width="stretch")
    peor, mejor = resumen["Promedio"].idxmin(), resumen["Promedio"].idxmax()
    st.markdown(
        f"El grupo mejor definido es **{mejor}** (silueta {resumen.loc[mejor, 'Promedio']:.2f}) y el más difuso es "
        f"**{peor}** ({resumen.loc[peor, 'Promedio']:.2f}). En total, el **{(siluetas < 0.1).mean() * 100:.0f} %** de "
        "los micronegocios tiene silueta menor que 0,1: son casos de frontera que podrían pertenecer a dos grupos."
    )
    salida = original.assign(grupo=grupos, silueta=siluetas.round(3))
    st.download_button("Descargar la base con el grupo asignado (CSV)", salida.to_csv(index=False).encode("utf-8"),
                       "micronegocios_valle_grupos.csv", "text/csv")

# ================================================================ 6. CONCLUSIONES
else:
    st.title("Discusión de resultados")
    acum = varianza["Acumulada (%)"][n_comp - 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micronegocios analizados", bit["final"])
    c2.metric("Componentes", n_comp, f"{acum:.1f} % de la varianza", delta_color="off")
    c3.metric("Grupos (K)", k)
    c4.metric("Silueta promedio", f"{sil_media:.2f}")
    if not ES_DEFECTO:
        st.warning("Cambiaste los parámetros del panel lateral. Las cifras de esta página se recalculan con tu "
                   "selección; la lectura del autor solo se muestra con la configuración por defecto.")

    st.header("1. ¿Qué variables definen los componentes?")
    for cp in cargas.columns[:n_comp]:
        st.markdown(f"- **{cp}** ({varianza.set_index('Componente').loc[cp, 'Varianza explicada (%)']:.1f} %): "
                    f"{lectura_componente(cp)}.")
    st.header("2. ¿Cuánta variación se conservó?")
    st.markdown(
        f"Se conservaron **{n_comp} de {len(variables)} componentes**, que retienen el **{acum:.1f} %** de la variación "
        f"original. El {100 - acum:.1f} % restante queda fuera del análisis de grupos"
        + (" porque K-Means se aplicó sobre los componentes." if base == "Componentes retenidos"
           else ", aunque en esta configuración K-Means usa las variables estandarizadas completas.")
    )
    st.header("3. ¿Qué diferencias reales aparecen entre los grupos?")
    perfil = an.perfil_grupos(original, list(variables) + list(an.APOYO), grupos)
    st.dataframe(formato_perfil(perfil), width="stretch")
    for g in perfil_z.index:
        st.markdown(f"- **Grupo {g}** ({(grupos == g).mean() * 100:.0f} % de la muestra): {rasgos(g)}.")
    st.header("4. ¿Son grupos claramente distinguibles?")
    dudosos = (siluetas < 0.1).mean() * 100
    nivel = "moderada a fuerte" if sil_media >= 0.5 else "débil a moderada" if sil_media >= 0.25 else "muy débil"
    st.markdown(
        f"La silueta promedio es **{sil_media:.2f}**, que corresponde a una estructura **{nivel}**. El "
        f"**{dudosos:.0f} %** de los micronegocios está en zona de frontera (silueta menor que 0,1) y el "
        f"{(siluetas < 0).mean() * 100:.1f} % queda más cerca de otro grupo que del propio."
    )

    if ES_DEFECTO:
        st.header("5. Lectura del autor")
        st.markdown(
            """
**Lo que aprendí de los datos.** Los micronegocios del Valle no se ordenan en una sola escala de "pequeño a grande".
Se diferencian en tres dimensiones casi independientes: qué tan conectados y grandes son, cuánta trayectoria tienen
y cuánto margen les queda. La primera es la más fuerte, y en ella pesa más la tecnología que las ventas.

**Cuatro perfiles, con fronteras difusas:**

- **Unipersonales jóvenes y conectados** (el grupo más numeroso, cerca de la mitad). Propietarios alrededor de los
  40 años, pocos años de operación, un teléfono inteligente y uso de internet. Ventas medias y margen cercano a la mitad.
- **Negocios de larga trayectoria.** Propietarios en torno a los 60 años, con décadas en la actividad. Venden parecido al
  grupo anterior pero conservan el mayor margen: negocios estables, de bajo costo.
- **Tradicionales desconectados** (cerca de una cuarta parte). Sin dispositivos ni uso de internet, con las ventas
  más bajas. Tienen más presencia rural, agropecuaria y de comercio.
- **Negocios de mayor escala** (uno de cada diez). Venden varias veces más que el resto, tienen ayudantes y más
  equipos, pero su margen es el más bajo: mueven más dinero y también cargan más costos y nómina.

El motivo de creación, que no entró al modelo, respalda la lectura: casi la mitad de los desconectados abrió el
negocio porque no tenía otra alternativa de ingresos, mientras que más de la mitad de los de mayor escala lo hizo
porque identificó una oportunidad.

**Relación con el problema.** La pregunta era si un micronegocio "promedio" representa al conjunto: no lo hace.
Un programa de digitalización tiene su población objetivo en el grupo desconectado, que es además el de menores
ventas; ofrecérselo al grupo conectado sería redundante. El grupo de mayor escala necesita otra cosa: gestión de
costos y formalización laboral. Y el margen alto de los negocios antiguos advierte que vender poco no equivale a
ser inviable.

**Qué tan firme es la conclusión.** Moderadamente. La silueta ronda 0,35 y el codo no marca un quiebre único; con
K = 3 o K = 4 la calidad es casi igual. Los grupos más nítidos son los definidos por lo digital, porque esas
variables son casi binarias (0 o 1 dispositivo). Los de larga trayectoria y mayor escala son los más difusos: sus
fronteras con los demás son graduales. Hablo entonces de **perfiles típicos**, no de categorías cerradas.
"""
        )
    st.header(("6" if ES_DEFECTO else "5") + ". Limitaciones al usar estos resultados")
    st.markdown(
        """
- **Describen la muestra, no la población.** No se usaron los factores de expansión; los porcentajes de cada grupo
  no son estimaciones oficiales para el Valle del Cauca.
- **Variables discretas.** Personal, dispositivos y personas que usan internet toman muy pocos valores. K-Means
  supone variables continuas y grupos aproximadamente esféricos; parte de la separación refleja simplemente
  "tiene / no tiene" teléfono inteligente.
- **Un solo mes y cifras declaradas.** Ventas y margen pueden estar afectados por estacionalidad o subdeclaración.
- **Se excluyeron los negocios sin ventas** en el mes, que pueden ser precisamente los más vulnerables.
- **Sensibilidad.** Cambiar K, la semilla o el número de componentes mueve casos de frontera; conviene comprobarlo
  en el panel lateral antes de tomar decisiones sobre un negocio individual.
- **No hay causalidad.** Que los conectados vendan más no prueba que conectarse aumente las ventas.
"""
    )

st.sidebar.divider()
st.sidebar.caption("Fuente: DANE, EMICRON 2024. Elaboración propia.")
