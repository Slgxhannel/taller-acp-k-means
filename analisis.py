"""Preparación, ACP y K-Means que usa la aplicación (app.py)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# columna -> (etiqueta, unidad / definición)
VARIABLES = {
    "ventas_mes": ("Ventas del mes", "pesos, mes anterior a la encuesta"),
    "margen": ("Margen", "% de las ventas que queda como ingreso mixto del propietario; negativo si hubo pérdida"),
    "personal_ocupado": ("Personal ocupado", "personas, incluido el propietario"),
    "antiguedad_meses": ("Antigüedad", "meses que el propietario lleva en el negocio"),
    "edad_propietario": ("Edad del propietario", "años"),
    "dispositivos_tic": ("Dispositivos TIC", "computadores, portátiles, tabletas y teléfonos inteligentes en uso"),
    "personas_internet": ("Personas que usan internet", "personas del negocio que usan internet para la actividad"),
}
ETIQUETAS = {k: v[0] for k, v in VARIABLES.items()}
# Variables en pesos que se muestran al describir los grupos pero no entran al modelo
APOYO = {"costos_mes": "Costos del mes", "ingreso_mixto": "Ingreso mixto"}
DESCRIPTIVAS = {
    "usa_internet": "Usa internet (%)",
    "usa_redes_sociales": "Usa redes sociales (%)",
    "tiene_correo": "Tiene correo electrónico (%)",
    "nombre_comercial": "Tiene nombre comercial (%)",
}


def cargar_datos(ruta="data/micronegocios_valle.csv"):
    """Lee la base y calcula el margen. Sin ventas el margen no existe (queda vacío)."""
    df = pd.read_csv(ruta)
    df["margen"] = (df["ingreso_mixto"] / df["ventas_mes"].where(df["ventas_mes"] > 0)) * 100
    return df


def preparar(df, variables, transformar=True, winsorizar=True):
    """Aplica las decisiones de calidad. Devuelve (base para el modelo, base original, bitácora).

    1. Elimina duplicados por identificador.
    2. Excluye los micronegocios sin ventas en el mes de referencia: su margen no está
       definido y, al aplicar logaritmos, formarían un grupo artificial.
    3. Elimina filas con faltantes en las variables elegidas.
    4. Opcional: logaritmo de las ventas, cuya distribución es muy asimétrica.
    5. Opcional: winsoriza cada variable en los percentiles 1 y 99.
    """
    bit = {"inicial": len(df)}
    d = df.drop_duplicates("id")
    bit["duplicados"] = bit["inicial"] - len(d)
    bit["sin_ventas"] = int((d["ventas_mes"] <= 0).sum())
    d = d[d["ventas_mes"] > 0]
    bit["faltantes"] = int(d[variables].isna().any(axis=1).sum())
    d = d.dropna(subset=variables).reset_index(drop=True)
    original = d.copy()

    d = d.copy()
    d[variables] = d[variables].astype(float)
    if transformar and "ventas_mes" in variables:
        d["ventas_mes"] = np.log(d["ventas_mes"])
    if winsorizar:
        d[variables] = d[variables].clip(d[variables].quantile(0.01), d[variables].quantile(0.99), axis=1)
    bit["final"] = len(d)
    return d, original, bit


def estandarizar(d, variables):
    """Media 0 y desviación estándar 1 en cada variable."""
    return pd.DataFrame(StandardScaler().fit_transform(d[variables]), columns=variables)


def ajustar_acp(z):
    """ACP completo: puntajes, tabla de varianza y cargas."""
    acp = PCA().fit(z)
    nombres = [f"CP{i + 1}" for i in range(acp.n_components_)]
    puntajes = pd.DataFrame(acp.transform(z), columns=nombres)
    varianza = pd.DataFrame(
        {
            "Componente": nombres,
            "Valor propio": acp.explained_variance_,
            "Varianza explicada (%)": acp.explained_variance_ratio_ * 100,
            "Acumulada (%)": acp.explained_variance_ratio_.cumsum() * 100,
        }
    )
    # Cargas = correlación entre cada variable y cada componente
    cargas = pd.DataFrame(acp.components_.T * np.sqrt(acp.explained_variance_), index=z.columns, columns=nombres)
    return puntajes, varianza, cargas


def sugerir_componentes(varianza, umbral=80.0):
    """Número de componentes según Kaiser (valor propio > 1) y según varianza acumulada."""
    kaiser = max(int((varianza["Valor propio"] > 1).sum()), 1)
    acumulada = int((varianza["Acumulada (%)"] >= umbral).values.argmax()) + 1
    return {"kaiser": kaiser, "acumulada": acumulada}


def evaluar_k(x, k_max=8, semilla=42):
    """Inercia (método del codo) y silueta promedio para K = 2..k_max."""
    filas = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(x)
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": silhouette_score(x, km.labels_)})
    return pd.DataFrame(filas)


def ajustar_kmeans(x, k, semilla=42):
    """Etiquetas (como letras, para no sugerir un orden) y silueta de cada observación."""
    km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(x)
    letras = np.array([chr(65 + i) for i in km.labels_])
    return letras, silhouette_samples(x, km.labels_)


def perfil_grupos(original, variables, grupos):
    """Tamaño y mediana de cada variable original por grupo."""
    d = original[variables].copy()
    d["Grupo"] = grupos
    perfil = d.groupby("Grupo").median()
    perfil.insert(0, "Micronegocios", d.groupby("Grupo").size())
    return perfil


def perfil_estandarizado(z, grupos):
    """Media de cada variable estandarizada por grupo (0 = promedio de la muestra)."""
    d = z.copy()
    d["Grupo"] = grupos
    return d.groupby("Grupo").mean()
