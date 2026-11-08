#!/usr/bin/env python3
"""
Pronóstico WRF-SMN (4 km) de viento, temperatura, humedad y Delta T
para Río Cuarto y alrededores. KS AgroDrones.

Uso:
    pip install xarray h5netcdf h5py s3fs pyproj numpy
    python pronostico_wrf.py            # próximas 72 h
    python pronostico_wrf.py --horas 72

Para sumar localidades, agregá una línea en LOCALIDADES (nombre, latitud, longitud).
Los datos son abiertos del Servicio Meteorológico Nacional (licencia CC BY 2.5 AR).
"""

import argparse
import csv
import datetime as dt
import html
import math
import os
import pathlib
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from pyproj import Transformer

# ---------------------------------------------------------------------------
# LOCALIDADES — agregá o quitá líneas acá
# ---------------------------------------------------------------------------
LOCALIDADES = [
    ("Río Cuarto", -33.123, -64.349),
    ("Las Higueras", -33.093, -64.287),
    ("Holmberg", -33.205, -64.432),
    ("Alcira Gigena", -32.756, -64.338),
    ("Sampacho", -33.384, -64.722),
    ("Coronel Moldes", -33.623, -64.597),
    ("Adelia María", -33.632, -64.021),
    ("Vicuña Mackenna", -33.917, -64.390),
    ("General Cabrera", -32.813, -63.873),
]

# Criterios de aplicación (ajustables)
VIENTO_MIN = 3.0     # km/h — por debajo: riesgo de inversión térmica
VIENTO_MAX = 15.0    # km/h — por encima: riesgo de deriva
DELTAT_MIN = 2.0     # °C
DELTAT_IDEAL = 8.0   # °C
DELTAT_MAX = 10.0    # °C
TEMP_MAX = 30.0      # °C

BUCKET = "smn-ar-wrf"
VARIABLES = ["magViento10", "dirViento10", "T2", "HR2"]
TZ_LOCAL = dt.timezone(dt.timedelta(hours=-3))
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


# ---------------------------------------------------------------------------
# Acceso a datos
# ---------------------------------------------------------------------------
class Fuente:
    """Lee archivos del bucket S3 público del SMN, o de una carpeta local."""

    def __init__(self, carpeta_local=None):
        self.local = carpeta_local
        if not self.local:
            import s3fs
            self.fs = s3fs.S3FileSystem(anon=True)

    def _ruta(self, init, fhr):
        nombre = f"WRFDETAR_01H_{init:%Y%m%d_%H}_{fhr:03d}.nc"
        if self.local:
            return os.path.join(self.local, nombre)
        return f"{BUCKET}/DATA/WRF/DET/{init:%Y/%m/%d/%H}/{nombre}"

    def existe(self, init, fhr):
        ruta = self._ruta(init, fhr)
        return os.path.exists(ruta) if self.local else self.fs.exists(ruta)

    def abrir(self, init, fhr):
        ruta = self._ruta(init, fhr)
        f = open(ruta, "rb") if self.local else self.fs.open(ruta, "rb")
        return xr.open_dataset(f, engine="h5netcdf", decode_coords="all")


def ultima_corrida(fuente, horas):
    """Busca la corrida más reciente con todas las horas pedidas ya publicadas."""
    ahora = dt.datetime.now(dt.timezone.utc)
    base = ahora.replace(minute=0, second=0, microsecond=0)
    base = base.replace(hour=base.hour - base.hour % 6)
    for i in range(8):
        init = base - dt.timedelta(hours=6 * i)
        if fuente.existe(init, horas):
            return init
    raise SystemExit("No encontré corridas recientes del WRF. Revisá la conexión.")


def transformador(ds):
    a = ds["Lambert_Conformal"].attrs
    sp = np.atleast_1d(a["standard_parallel"])
    lat1, lat2 = float(sp[0]), float(sp[-1])
    proj = (f"+proj=lcc +lat_1={lat1} +lat_2={lat2} "
            f"+lat_0={float(a['latitude_of_projection_origin'])} "
            f"+lon_0={float(a['longitude_of_central_meridian'])} "
            f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m")
    return Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def leer_hora(fuente, init, fhr, puntos_xy):
    for intento in range(3):
        try:
            with fuente.abrir(init, fhr) as ds:
                sel = ds[VARIABLES].sel(
                    x=xr.DataArray([p[0] for p in puntos_xy], dims="loc"),
                    y=xr.DataArray([p[1] for p in puntos_xy], dims="loc"),
                    method="nearest",
                ).load()
                tiempo = init + dt.timedelta(hours=fhr)
                datos = {v: np.ravel(sel[v].values) for v in VARIABLES}
                return fhr, tiempo, datos
        except Exception as e:  # noqa: BLE001
            err = e
    print(f"  ! hora +{fhr:03d} no se pudo leer ({err})", file=sys.stderr)
    return fhr, None, None


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
def bulbo_humedo(t, hr):
    """Temperatura de bulbo húmedo (Stull 2011), °C."""
    return (t * math.atan(0.151977 * math.sqrt(hr + 8.313659))
            + math.atan(t + hr) - math.atan(hr - 1.676331)
            + 0.00391838 * hr ** 1.5 * math.atan(0.023101 * hr)
            - 4.686035)


def evaluar(viento, dT, temp):
    """Devuelve 'apto', 'precaucion' o 'no'."""
    if viento > VIENTO_MAX or dT > DELTAT_MAX or temp > TEMP_MAX:
        return "no"
    if viento < VIENTO_MIN or dT < DELTAT_MIN or dT > DELTAT_IDEAL:
        return "precaucion"
    return "apto"


def punto_cardinal(grados):
    return DIRS[int((grados % 360) / 22.5 + 0.5) % 16]


# ---------------------------------------------------------------------------
# Informe HTML
# ---------------------------------------------------------------------------
def ventanas(filas):
    """Agrupa horas aptas consecutivas en ventanas."""
    res, actual = [], None
    for f in filas:
        if f["estado"] == "apto":
            if actual and f["hora"] - actual[1] == dt.timedelta(hours=1):
                actual[1] = f["hora"]
            else:
                actual = [f["hora"], f["hora"]]
                res.append(actual)
        else:
            actual = None
    return res


def html_informe(init, resultados):
    gen = dt.datetime.now(TZ_LOCAL)
    init_loc = init.astimezone(TZ_LOCAL)
    secciones, nav = [], []
    for i, (nombre, filas) in enumerate(resultados):
        vent = ventanas(filas)
        if vent:
            txt = " · ".join(
                f"{DIAS[a.weekday()]} {a:%d} {a:%H}h–{(b + dt.timedelta(hours=1)):%H}h" for a, b in vent)
        else:
            txt = "Sin ventanas aptas en el período"
        nav.append(f'<a href="#l{i}">{html.escape(nombre)}</a>')
        dias, trs, dia_prev = {}, [], None
        for f in filas:
            d = f"{DIAS[f['hora'].weekday()]} {f['hora']:%d/%m}"
            if d != dia_prev:
                trs.append(f'<tr class="dia"><td colspan="6">{d}</td></tr>')
                dia_prev = d
            flecha = (f'<span class="fl" style="transform:rotate({(f["dir"] + 180) % 360:.0f}deg)">↑</span>')
            trs.append(
                f'<tr class="{f["estado"]}"><td>{f["hora"]:%H}h</td>'
                f'<td class="num">{f["viento"]:.0f}</td>'
                f'<td>{flecha} {punto_cardinal(f["dir"])}</td>'
                f'<td class="num">{f["temp"]:.0f}°</td>'
                f'<td class="num">{f["hr"]:.0f}%</td>'
                f'<td class="num">{f["dT"]:.1f}</td></tr>')
        secciones.append(f"""
<section id="l{i}">
  <h2>{html.escape(nombre)}</h2>
  <p class="vent"><b>Ventanas aptas:</b> {html.escape(txt)}</p>
  <table>
    <thead><tr><th>Hora</th><th>Viento<br>km/h</th><th>Dir.</th><th>Temp</th><th>HR</th><th>ΔT</th></tr></thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</section>""")

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pronóstico WRF</title>
<style>
:root{{--bg:#f6f5f1;--card:#fff;--tx:#1d2420;--mut:#66706a;--ok:#d8efd9;--wa:#fbefcc;--no:#f7d6d2;--ln:#e3e1da;--ac:#2f6b3f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151816;--card:#1e2320;--tx:#e8ebe7;--mut:#9aa39d;--ok:#21402a;--wa:#4a3f1c;--no:#4d2622;--ln:#2e3430;--ac:#7cc48e}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:20px 16px 8px;max-width:760px;margin:auto}}h1{{margin:0 0 4px;font-size:22px}}
.meta{{color:var(--mut);font-size:13px}}nav{{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px;max-width:760px;margin:auto}}
nav a{{color:var(--ac);text-decoration:none;border:1px solid var(--ln);border-radius:20px;padding:3px 10px;font-size:13px;background:var(--card)}}
main{{max-width:760px;margin:auto;padding:0 16px 40px}}section{{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:14px;margin:14px 0}}
h2{{margin:0 0 6px;font-size:18px}}.vent{{margin:0 0 10px;font-size:14px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th{{text-align:left;color:var(--mut);font-weight:600;font-size:12px;border-bottom:1px solid var(--ln);padding:4px}}
td{{padding:4px;border-bottom:1px solid var(--ln)}}.num{{font-variant-numeric:tabular-nums}}
tr.dia td{{background:transparent;font-weight:700;padding-top:12px;border-bottom:2px solid var(--ln)}}
tr.apto td{{background:var(--ok)}}tr.precaucion td{{background:var(--wa)}}tr.no td{{background:var(--no)}}
.fl{{display:inline-block}}.ley{{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin-top:6px}}
.ley span::before{{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
.ley .a::before{{background:var(--ok)}}.ley .p::before{{background:var(--wa)}}.ley .n::before{{background:var(--no)}}
footer{{color:var(--mut);font-size:12px;max-width:760px;margin:auto;padding:0 16px 30px}}
</style></head><body>
<header>
  <h1>Pronóstico para aplicaciones</h1>
  <div class="meta">Modelo WRF-SMN 4 km · corrida {init_loc:%d/%m %H}h (hora local) · generado {gen:%d/%m %H:%M}</div>
  <div class="ley"><span class="a">Apto: viento {VIENTO_MIN:.0f}–{VIENTO_MAX:.0f} km/h y ΔT {DELTAT_MIN:.0f}–{DELTAT_IDEAL:.0f}</span>
  <span class="p">Precaución: viento bajo (inversión) o ΔT {DELTAT_IDEAL:.0f}–{DELTAT_MAX:.0f}</span>
  <span class="n">No aplicar</span></div>
</header>
<nav>{''.join(nav)}</nav>
<main>{''.join(secciones)}</main>
<footer>Viento medio a 10 m (corregido por el SMN). El modelo no incluye ráfagas: confirmá siempre con anemómetro en el lote.
Horas en hora local (UTC−3). Datos: Servicio Meteorológico Nacional, CC BY 2.5 AR.</footer>
</body></html>"""


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horas", type=int, default=72, help="horas de pronóstico (máx. 72)")
    ap.add_argument("--salida", default="pronostico_wrf.html")
    ap.add_argument("--local", help="carpeta con archivos .nc ya descargados (opcional)")
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el navegador")
    args = ap.parse_args()
    horas = max(1, min(args.horas, 72))

    fuente = Fuente(args.local)
    print("Buscando la última corrida del WRF-SMN…")
    init = ultima_corrida(fuente, horas)
    print(f"Corrida {init:%Y-%m-%d %H} UTC")

    ahora = dt.datetime.now(dt.timezone.utc)
    desde = max(0, int((ahora - init).total_seconds() // 3600))
    fhrs = list(range(desde, horas + 1))

    with fuente.abrir(init, fhrs[0]) as ds0:
        tr = transformador(ds0)
    puntos_xy = [tr.transform(lon, lat) for _, lat, lon in LOCALIDADES]

    print(f"Leyendo {len(fhrs)} horas para {len(LOCALIDADES)} localidades…")
    leidos = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, r in enumerate(ex.map(lambda h: leer_hora(fuente, init, h, puntos_xy), fhrs), 1):
            leidos.append(r)
            print(f"\r  {n}/{len(fhrs)}", end="", flush=True)
    print()
    leidos = sorted((r for r in leidos if r[1] is not None), key=lambda r: r[0])

    resultados, csv_filas = [], []
    for i, (nombre, lat, lon) in enumerate(LOCALIDADES):
        filas = []
        for _, tiempo, d in leidos:
            viento = float(d["magViento10"][i]) * 3.6
            temp = float(d["T2"][i])
            hr = float(min(max(d["HR2"][i], 1), 100))
            dT = temp - bulbo_humedo(temp, hr)
            f = {"hora": tiempo.astimezone(TZ_LOCAL), "viento": viento,
                 "dir": float(d["dirViento10"][i]), "temp": temp, "hr": hr,
                 "dT": dT, "estado": evaluar(viento, dT, temp)}
            filas.append(f)
            csv_filas.append([nombre, lat, lon, f["hora"].strftime("%Y-%m-%d %H:%M"),
                              round(viento, 1), round(f["dir"]), round(temp, 1),
                              round(hr), round(dT, 1), f["estado"]])
        resultados.append((nombre, filas))

    with open(args.salida, "w", encoding="utf-8") as fh:
        fh.write(html_informe(init, resultados))
    csv_path = os.path.splitext(args.salida)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["localidad", "lat", "lon", "hora_local", "viento_kmh", "dir_grados",
                    "temp_c", "hr_pct", "delta_t", "estado"])
        w.writerows(csv_filas)

    print(f"Listo: {os.path.abspath(args.salida)}")
    print(f"       {os.path.abspath(csv_path)}")
    if not args.no_abrir:
        webbrowser.open(pathlib.Path(args.salida).resolve().as_uri())


if __name__ == "__main__":
    main()
