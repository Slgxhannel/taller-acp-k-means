"""
Construye data/micronegocios_valle.csv a partir de los microdatos públicos
de la Encuesta de Micronegocios (EMICRON) 2024 del DANE.

Uso:
    python scripts/construir_base.py <carpeta_con_los_modulos>

Descarga de los módulos (CSV): https://microdatos.dane.gov.co/index.php/catalog/875
Se necesitan en la carpeta:
    Módulo de identificación.csv
    Módulo de ventas o ingresos.csv
    Módulo de emprendimiento.csv
    Módulo de TIC.csv

Cada fila es un micronegocio, identificado por la llave
DIRECTORIO + SECUENCIA_P + SECUENCIA_ENCUESTA. Se conserva solo el
departamento 76 (Valle del Cauca).
"""
import glob
import sys

import pandas as pd

LLAVE = ["DIRECTORIO", "SECUENCIA_P", "SECUENCIA_ENCUESTA"]
DEPTO = "76"

SECTOR = {1: "Agropecuario", 2: "Industria", 3: "Comercio", 4: "Servicios"}
MOTIVO = {
    1: "No tiene otra alternativa de ingresos",
    2: "Lo identificó como una oportunidad de negocio",
    3: "Por tradición familiar o lo heredó",
    4: "Para complementar el ingreso familiar",
    5: "Para ejercer su oficio, carrera o profesión",
    6: "No tenía la experiencia, escolaridad o capacitación requerida para un empleo",
    7: "Otro",
}


def leer(carpeta, patron, **kw):
    ruta = [r for r in glob.glob(f"{carpeta}/*{patron}*.csv") if "Limpio" not in r]
    if not ruta:
        raise FileNotFoundError(f"No se encontró un archivo con '{patron}' en {carpeta}")
    return pd.read_csv(sorted(ruta, key=len)[0], dtype={"COD_DEPTO": str}, **kw)


def main(carpeta, salida="data/micronegocios_valle.csv"):
    ide = leer(carpeta, "identificaci")
    ven = leer(carpeta, "ventas", usecols=LLAVE + ["VENTAS_MES_ANTERIOR", "VALOR_AGREGADO", "INGRESO_MIXTO"])
    emp = leer(carpeta, "emprendimiento", usecols=LLAVE + ["P3051"])
    tic = leer(carpeta, "TIC", usecols=LLAVE + ["P1087", "P1088", "P977", "P978", "P2524", "P1559", "P980"])

    d = ide[ide["COD_DEPTO"] == DEPTO].merge(ven, on=LLAVE).merge(emp, on=LLAVE).merge(tic, on=LLAVE)

    # Las preguntas de conteo solo se hacen a quien respondió "sí" en la pregunta
    # filtro; el vacío restante significa "no tiene", es decir, cero.
    ayudantes = d[["P3032_1", "P3032_2", "P3032_3"]].fillna(0).sum(axis=1)
    # P1087 computadores de escritorio, P1088 portátiles, P977 tabletas, P978 teléfonos inteligentes
    equipos = d[["P1087", "P1088", "P977", "P978"]].fillna(0)

    base = pd.DataFrame(
        {
            "id": d["DIRECTORIO"].astype(str) + "-" + d["SECUENCIA_P"].astype(str) + "-" + d["SECUENCIA_ENCUESTA"].astype(str),
            # --- variables cuantitativas ---
            "ventas_mes": d["VENTAS_MES_ANTERIOR"],
            "costos_mes": d["VENTAS_MES_ANTERIOR"] - d["VALOR_AGREGADO"],
            "ingreso_mixto": d["INGRESO_MIXTO"],
            "personal_ocupado": 1 + ayudantes,
            "antiguedad_meses": d["P3034"],
            "edad_propietario": d["P241"],
            "dispositivos_tic": equipos.sum(axis=1),
            "personas_internet": d["P980"].fillna(0),
            # --- variables descriptivas (no entran al modelo) ---
            "sector": d["GRUPOS4"].astype(int).map(SECTOR),
            "zona": d["CLASE_TE"].map({1: "Cabecera municipal", 2: "Centros poblados y rural disperso"}),
            "sexo_propietario": d["P35"].map({1: "Hombre", 2: "Mujer"}),
            "motivo_creacion": d["P3051"].map(MOTIVO),
            "usa_internet": (d["P2524"] == 1).astype(int),
            "usa_redes_sociales": (d["P1559"] == 1).astype(int),
            "tiene_correo": (d["P3000"] == 1).astype(int),
            "nombre_comercial": (d["P3035"] == 1).astype(int),
            "factor_expansion": d["F_EXP"].round(2),
        }
    )
    for c in ["ventas_mes", "costos_mes", "ingreso_mixto", "personal_ocupado", "dispositivos_tic", "personas_internet"]:
        base[c] = base[c].astype("int64")
    base.to_csv(salida, index=False)
    print(f"{salida}: {base.shape[0]} micronegocios, {base.shape[1]} columnas")


if __name__ == "__main__":
    main(sys.argv[1])
