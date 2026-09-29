#!/usr/bin/env python3
"""
soppkart_as.py v2: mushroom map around Ås that learns from your own finds.

Data
  - GBIF/Artskart observations (kantarell, traktkantarell, steinsopp, svart trompetsopp)
  - Your own log in mine_funn.csv (finds AND blank searches)
  - Terrain from Kartverket (elevation, slope, north-facing) and forest from
    NIBIO SR16 (tree species, site index, age) sampled on a grid
  - A gradient-boosting habitat model per species, trained on GBIF + your log
    (your rows weighted 3x), predicted over the grid as a probability overlay

Workflow
  pip install folium requests numpy scikit-learn pillow
  python soppkart_as.py --probe            # check that the WMS/elevation APIs answer
  python soppkart_as.py                    # build map, first run fetches covariates (cached)
  ... go to the forest, click the map where you find (or do not find) anything,
      paste the line the popup gives you into mine_funn.csv, re-run ...

mine_funn.csv columns
  art,lat,lon,dato,funnet,mengde,notat
  art     Kantarell | Traktkantarell | Steinsopp | Svart trompetsopp
  funnet  1 = found, 0 = searched here and found nothing (these matter!)
"""
import argparse, csv, json, math, os, random, re, sys, time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import folium
from folium.plugins import HeatMap, LocateControl

AS_LAT, AS_LON = 59.665, 10.795
GBIF = "https://api.gbif.org/v1"
HOYDE = "https://ws.geonorge.no/hoydedata/v1/punkt"
SR16_WMS = "https://wms.nibio.no/cgi-bin/sr16"
AR50_WMS = "https://wms.nibio.no/cgi-bin/ar50_2"
KARTVERKET_TOPO = "https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png"
HEADERS = {"User-Agent": "soppkart-as/2.0 (personal use)"}

# norsk navn -> (vitenskapelig navn, farge, sesongmaaneder)
SPECIES = {
    "Steinsopp":          ("Boletus edulis",              "#6b3e26", (8, 10)),
    "Kantarell":          ("Cantharellus cibarius",       "#f2b705", (7, 10)),
    "Svart trompetsopp":  ("Craterellus cornucopioides",  "#222222", (8, 10)),
    "Traktkantarell":     ("Craterellus tubaeformis",     "#c26a1b", (8, 11)),
    "Blek piggsopp":      ("Hydnum repandum",             "#e8c39e", (7, 10)),
    "Morkel":             ("Morchella esculenta",         "#7d6b4f", (4, 6)),
    "Brunskrubb":         ("Leccinum scabrum",            "#8a6b4a", (7, 10)),
    "Gul trompetsopp":    ("Craterellus lutescens",       "#d9a441", (8, 11)),
}
DEFAULT_SPECIES = ["Steinsopp", "Kantarell", "Svart trompetsopp", "Traktkantarell"]
# felt -> SR16 rasterlag. Laget SRRALDER finnes ikke, skoghoyde (dm) er nærmeste stedfortreder.
SR16_LAYERS = {"treslag": "SRRTRESLAG", "bonitet": "SRRBONITET", "kronedek": "SRRKRONEDEK", "skoghoyde": "SRRHOYDEM"}
SR16_FIELDS = list(SR16_LAYERS)
OWN_WEIGHT = 3.0
# Egne funn lagres i Supabase (tabell soppfunn, RLS: bare egne rader). Noekkelen er
# publiserbar og ment for nettleseren; tilgang styres av innlogging, ikke av noekkelen.
SUPABASE_URL = "https://xishtaqioetncnczznuv.supabase.co"
SUPABASE_KEY = "sb_publishable_uH-lmVlIlxiW-GQeXEH-3A_gIUEV_vW"

# ---------------------------------------------------------------- geometry
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))

def merc(lat, lon):
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi
    return x, y

def make_grid(lat, lon, radius_km, grid_m):
    dlat = grid_m / 111320
    dlon = grid_m / (111320 * math.cos(math.radians(lat)))
    n = int(math.ceil(radius_km * 1000 / grid_m))
    lats = lat + dlat * np.arange(-n, n + 1)
    lons = lon + dlon * np.arange(-n, n + 1)
    return lats[::-1], lons  # lats descending = image row order

def cell_index(lats, lons, la, lo):
    i = int(np.argmin(np.abs(lats - la)))
    j = int(np.argmin(np.abs(lons - lo)))
    return i, j

# ---------------------------------------------------------------- GBIF
def taxon_key(name):
    js = requests.get(f"{GBIF}/species/match", params={"name": name}, headers=HEADERS, timeout=30).json()
    return js.get("acceptedUsageKey") or js.get("usageKey")

def fetch_gbif(sci, lat, lon, radius_km, year_min, max_unc, months=None):
    key = taxon_key(sci)
    dlat = radius_km / 111.32; dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    p = {"taxonKey": key, "country": "NO", "hasCoordinate": "true", "hasGeospatialIssue": "false",
         "occurrenceStatus": "PRESENT", "decimalLatitude": f"{lat-dlat:.5f},{lat+dlat:.5f}",
         "decimalLongitude": f"{lon-dlon:.5f},{lon+dlon:.5f}", "year": f"{year_min},2100", "limit": 300, "offset": 0}
    rows = []
    while True:
        js = requests.get(f"{GBIF}/occurrence/search", params=p, headers=HEADERS, timeout=60).json()
        for o in js.get("results", []):
            la, lo = o.get("decimalLatitude"), o.get("decimalLongitude")
            unc = o.get("coordinateUncertaintyInMeters")
            if la is None or (unc is not None and unc > max_unc) or haversine_km(lat, lon, la, lo) > radius_km:
                continue
            mo = o.get("month")
            if months and mo is not None and not (months[0] <= mo <= months[1]):
                continue
            rows.append({"lat": la, "lon": lo, "date": (o.get("eventDate") or "")[:10], "month": mo,
                         "uncertainty_m": unc, "dataset": o.get("datasetName") or "", "gbif_key": o.get("key")})
        if js.get("endOfRecords", True):
            break
        p["offset"] += p["limit"]
    return rows

# ---------------------------------------------------------------- own log
def load_env(path=".env"):
    """Enkel .env-leser (KEY=verdi per linje), setter bare variabler som mangler."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def load_supabase():
    """Henter egne funn fra Supabase med SOPPKART_EMAIL/SOPPKART_PASSWORD fra .env."""
    email, pw = os.environ.get("SOPPKART_EMAIL"), os.environ.get("SOPPKART_PASSWORD")
    if not (email and pw):
        print("Supabase: SOPPKART_EMAIL/SOPPKART_PASSWORD mangler i .env, hopper over", file=sys.stderr)
        return []
    h = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    try:
        tok = requests.post(f"{SUPABASE_URL}/auth/v1/token", params={"grant_type": "password"}, headers=h,
                            json={"email": email, "password": pw}, timeout=30)
        tok.raise_for_status()
        h["Authorization"] = f"Bearer {tok.json()['access_token']}"
        rows = requests.get(f"{SUPABASE_URL}/rest/v1/soppfunn", headers=h, timeout=30,
                            params={"select": "art,lat,lon,dato,funnet,mengde,notat", "order": "dato"})
        rows.raise_for_status()
        rows = rows.json()
    except Exception as e:
        print(f"Supabase failed: {e}", file=sys.stderr)
        return []
    out = [{"art": r["art"], "lat": float(r["lat"]), "lon": float(r["lon"]), "dato": r["dato"] or "",
            "funnet": 1 if r["funnet"] else 0, "mengde": r.get("mengde") or "", "notat": r.get("notat") or ""}
           for r in rows]
    print(f"Supabase: {len(out)} egne registreringer")
    return out

def load_own(path):
    """Egne funn: Supabase (fra mobilen) pluss eventuell lokal mine_funn.csv."""
    out = load_supabase()
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append({"art": r["art"].strip(), "lat": float(r["lat"]), "lon": float(r["lon"]),
                            "dato": r.get("dato", ""), "funnet": int(r.get("funnet") or 1),
                            "mengde": r.get("mengde", ""), "notat": r.get("notat", "")})
            except (ValueError, KeyError) as e:
                print(f"Skipping bad row in {path}: {r} ({e})", file=sys.stderr)
    return out

# ---------------------------------------------------------------- covariates
def elevation_batch(points):
    """points: list of (lat, lon). Returns list of z (or nan)."""
    out = []
    for k in range(0, len(points), 50):
        chunk = points[k:k + 50]
        pts = json.dumps([[lo, la] for la, lo in chunk])
        try:
            js = requests.get(HOYDE, params={"koordsys": 4258, "punkter": pts, "geojson": "false"},
                              headers=HEADERS, timeout=60).json()
            zs = [p.get("z") for p in js.get("punkter", [])]
            zs = [float(z) if z is not None else float("nan") for z in zs]
            if len(zs) != len(chunk):
                zs = [float("nan")] * len(chunk)
        except Exception as e:
            print(f"elevation failed: {e}", file=sys.stderr)
            zs = [float("nan")] * len(chunk)
        out.extend(zs)
    return out

def sr16_layers():
    r = requests.get(SR16_WMS, params={"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetCapabilities"},
                     headers=HEADERS, timeout=60)
    root = ET.fromstring(r.content)
    names = []
    for el in root.iter():
        if el.tag.split("}")[-1] == "Layer":
            nm = [c.text for c in el if c.tag.split("}")[-1] == "Name" and c.text]
            if nm:
                names.append(nm[0].strip())
    return names

def sr16_featureinfo(lat, lon, raw=False):
    """Ett kall per rasterlag; HTML-svaret har verdien i siste rad (foerste er 'ikke relevant')."""
    x, y = merc(lat, lon)
    half = 50 * 16  # 101 px window at 16 m/px
    vals, raws = {}, []
    for field, layer in SR16_LAYERS.items():
        p = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo", "CRS": "EPSG:3857",
             "BBOX": f"{x-half},{y-half},{x+half},{y+half}", "WIDTH": 101, "HEIGHT": 101, "I": 50, "J": 50,
             "LAYERS": layer, "QUERY_LAYERS": layer, "INFO_FORMAT": "text/html"}
        r = requests.get(SR16_WMS, params=p, headers=HEADERS, timeout=60)
        r.encoding = "utf-8"
        raws.append(f"{layer}: {r.text}")
        rows = re.findall(r'write\("<tr><td>[^<]*</td><td>(-?\d+(?:\.\d+)?)</td>', r.text)
        if rows and float(rows[0]) > -9000:
            vals[field] = float(rows[0])
    return "\n".join(raws) if raw else vals

def grid_covariates(lats, lons, cache_path, layers):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    keys = [(i, j, f"{lats[i]:.5f},{lons[j]:.5f}") for i in range(len(lats)) for j in range(len(lons))]

    todo = [(i, j, k) for i, j, k in keys if k not in cache or "z" not in cache[k]]
    if todo:
        print(f"Fetching elevation for {len(todo)} cells ...")
        zs = elevation_batch([(lats[i], lons[j]) for i, j, _ in todo])
        for (i, j, k), z in zip(todo, zs):
            cache.setdefault(k, {})["z"] = z
        json.dump(cache, open(cache_path, "w"))

    todo = [(i, j, k) for i, j, k in keys if "sr16" not in cache[k]]
    if todo and layers:
        print(f"Fetching SR16 forest info for {len(todo)} cells (cached afterwards) ...")
        def one(t):
            i, j, k = t
            for attempt in range(3):
                try:
                    return k, sr16_featureinfo(lats[i], lons[j])
                except Exception as e:
                    err = e
                    time.sleep(1 + attempt)
            print(f"SR16 failed for {k}: {err}", file=sys.stderr)
            return k, None
        with ThreadPoolExecutor(max_workers=12) as ex:
            for n, (k, v) in enumerate(ex.map(one, todo), 1):
                if v is not None:
                    cache[k]["sr16"] = v
                if n % 200 == 0:
                    print(f"  {n}/{len(todo)}", flush=True); json.dump(cache, open(cache_path, "w"))
        json.dump(cache, open(cache_path, "w"))

    ni, nj = len(lats), len(lons)
    Z = np.full((ni, nj), np.nan)
    F = {f: np.full((ni, nj), np.nan) for f in SR16_FIELDS}
    for i, j, k in keys:
        Z[i, j] = cache[k].get("z", np.nan)
        for f in SR16_FIELDS:
            F[f][i, j] = cache[k].get("sr16", {}).get(f, np.nan)

    cell_m = haversine_km(lats[0], lons[0], lats[1], lons[0]) * 1000
    gy, gx = np.gradient(Z, cell_m)          # gy: north->south rows (lats descending)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)             # 0 = north-facing
    northness = np.cos(aspect)
    feats = {"hoyde": Z, "helning": slope, "nordvendt": northness}
    feats.update(F)
    names = list(feats.keys())
    X = np.stack([feats[n] for n in names], axis=-1)  # (ni, nj, p)
    return X, names

# ---------------------------------------------------------------- model
def train_predict(X, names, lats, lons, presence_pts, absence_pts, seed=0):
    """presence_pts/absence_pts: list of (lat, lon, weight). Returns prob grid, info dict."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    ni, nj, p = X.shape
    Xf = X.reshape(-1, p)
    valid = ~np.isnan(Xf[:, 0])

    def cells(pts):
        d = {}
        for la, lo, w in pts:
            i, j = cell_index(lats, lons, la, lo)
            idx = i * nj + j
            d[idx] = d.get(idx, 0) + w
        return d

    pres = cells(presence_pts)
    absn = cells(absence_pts)
    if len(pres) < 5:
        return None, {"error": f"only {len(pres)} presence cells, need >= 5"}

    rng = np.random.default_rng(seed)
    pool = np.array([k for k in np.flatnonzero(valid) if k not in pres and k not in absn])
    nbg = min(len(pool), max(50, 3 * len(pres)))
    bg = rng.choice(pool, nbg, replace=False)

    idx = list(pres) + list(absn) + list(bg)
    y = np.array([1] * len(pres) + [0] * len(absn) + [0] * len(bg))
    w = np.array([min(pres[k], 3 * OWN_WEIGHT) for k in pres] + [absn[k] for k in absn] + [1.0] * len(bg))
    # balance classes
    w[y == 0] *= w[y == 1].sum() / max(w[y == 0].sum(), 1e-9)
    Xt = Xf[idx]

    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0,
                                         min_samples_leaf=5, random_state=seed)
    aucs = []
    if len(pres) >= 10:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xt, y):
            clf.fit(Xt[tr], y[tr], sample_weight=w[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(Xt[te])[:, 1], sample_weight=w[te]))
    clf.fit(Xt, y, sample_weight=w)

    prob = np.full(ni * nj, np.nan)
    prob[valid] = clf.predict_proba(Xf[valid])[:, 1]
    prob = prob.reshape(ni, nj)

    from sklearn.inspection import permutation_importance
    pi = permutation_importance(clf, Xt, y, sample_weight=w, n_repeats=5, random_state=seed)
    imp = sorted(zip(names, pi.importances_mean), key=lambda t: -t[1])
    return prob, {"n_presence_cells": len(pres), "n_own_absence_cells": len(absn), "n_background": int(nbg),
                  "cv_auc": float(np.mean(aucs)) if aucs else None, "importance": imp}

def prob_overlay(prob, lats, lons, color_hex):
    from PIL import Image
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    a = np.nan_to_num(prob, nan=0.0)
    a = np.clip((a - 0.2) / 0.8, 0, 1)      # fade in above 0.2
    img = np.zeros((*a.shape, 4), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = r, g, b
    img[..., 3] = (a * 220).astype(np.uint8)
    im = Image.fromarray(img, "RGBA").resize((a.shape[1] * 4, a.shape[0] * 4), Image.BILINEAR)
    path = "/tmp/_overlay.png"
    im.save(path)
    bounds = [[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]]
    return path, bounds

# ---------------------------------------------------------------- map
# Loggeverktoey i kartet. Lagrer funn i telefonens nettleser (localStorage) og
# lar deg dele/laste ned CSV-linjer for mine_funn.csv. %%MAP%% og %%SPECIES%% fylles inn.
MOBILE_CSS = """
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
<script>
  // Posisjon virker bare over https; send http-besoek videre (ikke localhost/fil).
  if (location.protocol === 'http:' && !/^(localhost|127\.0\.0\.1)$/.test(location.hostname))
    location.replace('https://' + location.host + location.pathname + location.search + location.hash);
</script>
<style>
  html, body { height: 100%; margin: 0; overscroll-behavior: none; }
  .leaflet-touch .leaflet-bar a { width: 44px; height: 44px; line-height: 44px; font-size: 20px; }
  .leaflet-control-layers-toggle { width: 44px !important; height: 44px !important; }
  .leaflet-control-layers-expanded { max-height: 60vh; overflow-y: auto; font-size: 15px; }
  .leaflet-control-layers label { padding: 4px 0; }
  .sk-pop { font-size: 15px; min-width: 220px; }
  .sk-pop select, .sk-pop input { width: 100%; font-size: 16px; padding: 8px; margin: 6px 0; box-sizing: border-box; }
  .sk-row { display: flex; gap: 8px; margin-top: 6px; }
  .sk-btn { flex: 1; font-size: 16px; padding: 12px 8px; border: 0; border-radius: 8px; color: #fff; cursor: pointer; }
  .sk-yes { background: #2e7d32; } .sk-no { background: #6d6d6d; }
  .sk-bar { position: absolute; left: 50%; transform: translateX(-50%); z-index: 1000;
            bottom: calc(24px + env(safe-area-inset-bottom)); display: flex; gap: 8px; }
  .sk-bar button { font-size: 16px; padding: 12px 16px; border: 0; border-radius: 22px; background: #fff;
                   box-shadow: 0 1px 6px rgba(0,0,0,.35); cursor: pointer; white-space: nowrap; }
  .sk-list { position: absolute; inset: auto 8px calc(84px + env(safe-area-inset-bottom)) 8px; z-index: 1001;
             max-height: 55vh; overflow-y: auto; background: #fff; border-radius: 12px; padding: 12px;
             box-shadow: 0 2px 12px rgba(0,0,0,.4); font-size: 14px; display: none; }
  .sk-tbl { max-height: 30vh; overflow-y: auto; font-size: 13px; margin: 8px 0; line-height: 1.6; }
  .sk-del { margin-top: 6px; padding: 6px 12px; border: 1px solid #c62828; color: #c62828; background: #fff; border-radius: 6px; }
  .sk-toast { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); z-index: 1002; background: #222;
              color: #fff; padding: 10px 16px; border-radius: 20px; font-size: 15px; display: none; }
</style>
"""

CLICK_JS = """
// Egne funn: lagres i Supabase (bare synlige for innlogget eier). Uten nett legges de
// i en kø i nettleseren og sendes når nettet er tilbake.
var SK_SPECIES = %%SPECIES%%, SK_Q = 'soppkart_koe', SK_ART = 'soppkart_art';
var sb = window.supabase.createClient('%%SB_URL%%', '%%SB_KEY%%');
var skUser = null, skRows = [], skLayer = null;

function skGet(k, d) { try { return JSON.parse(localStorage.getItem(k)) || d; } catch (e) { return d; } }
function skPut(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} }
function skEsc(t) { return String(t == null ? '' : t).replace(/[&<>"']/g, function (c) {
  return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]; }); }
function skUuid() { return (crypto.randomUUID ? crypto.randomUUID() :
  'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    var r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 3 | 8)).toString(16); })); }
function skCsv(r) { return [r.art, (+r.lat).toFixed(5), (+r.lon).toFixed(5), r.dato, r.funnet ? 1 : 0,
  r.mengde || '', (r.notat || '').replace(/[,\\n]/g, ';')].join(','); }
function skToast(t) { var el = document.getElementById('sk-toast'); el.textContent = t; el.style.display = 'block';
  clearTimeout(el._t); el._t = setTimeout(function () { el.style.display = 'none'; }, 2200); }

function onMapClick(e) {
  var map = %%MAP%%;
  if (!skUser) { skShowLogin(); return; }
  var la = e.latlng.lat, lo = e.latlng.lng, acc = e.accuracy ? Math.round(e.accuracy) : '';
  var last = skGet(SK_ART, SK_SPECIES[0]);
  var opts = SK_SPECIES.map(function (s) { return '<option' + (s === last ? ' selected' : '') + '>' + s + '</option>'; }).join('');
  var html = '<div class="sk-pop"><b>Logg her</b> <small>' + la.toFixed(5) + ', ' + lo.toFixed(5) +
    (acc ? ' (±' + acc + ' m)' : '') + '</small>' +
    '<select id="sk-art">' + opts + '</select>' +
    '<input id="sk-mengde" placeholder="Mengde (valgfritt)">' +
    '<input id="sk-notat" placeholder="Notat (valgfritt)">' +
    '<div class="sk-row"><button class="sk-btn sk-yes" onclick="skAdd(' + la + ',' + lo + ',true,\\'' + acc + '\\')">Funnet</button>' +
    '<button class="sk-btn sk-no" onclick="skAdd(' + la + ',' + lo + ',false,\\'' + acc + '\\')">Ingen funn</button></div></div>';
  L.popup({maxWidth: Math.min(320, window.innerWidth - 40), autoPanPadding: [20, 80]})
    .setLatLng(e.latlng).setContent(html).openOn(map);
}

function skAdd(la, lo, funnet, acc) {
  var art = document.getElementById('sk-art').value;
  skPut(SK_ART, art);
  var d = new Date(), dato = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  var row = {client_id: skUuid(), art: art, lat: +la.toFixed(6), lon: +lo.toFixed(6), dato: dato, funnet: funnet,
             mengde: document.getElementById('sk-mengde').value.trim() || null,
             notat: document.getElementById('sk-notat').value.trim() || null,
             noyaktighet_m: acc ? +acc : null};
  var q = skGet(SK_Q, []); q.push(row); skPut(SK_Q, q);
  %%MAP%%.closePopup(); skDraw();
  skFlush().then(function (ok) { skToast(ok ? (funnet ? 'Funn lagret' : 'Blankt søk lagret') : 'Lagret lokalt, sendes når du har nett'); });
}

// Send køen. client_id er unik, så gjentatte forsøk lager ikke duplikater.
function skFlush() {
  var q = skGet(SK_Q, []);
  if (!skUser || !q.length) return Promise.resolve(!q.length);
  return sb.from('soppfunn').upsert(q, {onConflict: 'client_id', ignoreDuplicates: true}).then(function (r) {
    if (r.error) { console.warn('soppfunn', r.error.message); return false; }
    var sent = q.map(function (x) { return x.client_id; });
    skPut(SK_Q, skGet(SK_Q, []).filter(function (x) { return sent.indexOf(x.client_id) < 0; }));
    return skLoad().then(function () { return true; });
  }, function () { return false; });
}

function skLoad() {
  if (!skUser) { skRows = []; skDraw(); return Promise.resolve(); }
  return sb.from('soppfunn').select('id,client_id,art,lat,lon,dato,funnet,mengde,notat,noyaktighet_m')
    .order('dato', {ascending: false}).then(function (r) {
      if (r.error) { skToast('Kunne ikke hente funn: ' + r.error.message); return; }
      skRows = r.data; skDraw();
    });
}

function skDraw() {
  var map = %%MAP%%, q = skGet(SK_Q, []), all = skRows.concat(q.map(function (x) { x._ko = true; return x; }));
  if (skLayer) map.removeLayer(skLayer);
  skLayer = L.layerGroup(all.map(function (r) {
    var mk = L.circleMarker([+r.lat, +r.lon], {radius: 8, weight: 2, color: r._ko ? '#f9a825' : '#fff',
      fillColor: r.funnet ? '#2e7d32' : '#9e9e9e', fillOpacity: 0.95});
    mk.bindPopup('<b>' + skEsc(r.art) + '</b><br>' + skEsc(r.dato) + (r.funnet ? ' funnet' : ' ingen funn') +
      (r.mengde ? '<br>' + skEsc(r.mengde) : '') + (r.notat ? '<br>' + skEsc(r.notat) : '') +
      (r._ko ? '<br><i>venter på nett</i>' : '<br><button class="sk-del" onclick="skDelete(' + r.id + ')">Slett</button>'));
    return mk;
  })).addTo(map);
  document.getElementById('sk-count').textContent = all.length;
}

function skDelete(id) {
  if (!confirm('Slette denne registreringen?')) return;
  sb.from('soppfunn').delete().eq('id', id).then(function (r) {
    if (r.error) { skToast('Kunne ikke slette: ' + r.error.message); return; }
    %%MAP%%.closePopup(); skLoad(); skToast('Slettet');
  });
}

function skShowList() {
  var el = document.getElementById('sk-list');
  if (el.style.display === 'block') { el.style.display = 'none'; return; }
  if (!skUser) { skShowLogin(); return; }
  var q = skGet(SK_Q, []), n1 = skRows.filter(function (r) { return r.funnet; }).length;
  el.innerHTML = '<b>Mine registreringer</b><br><small>' + n1 + ' funn, ' + (skRows.length - n1) + ' blanke søk' +
    (q.length ? ', ' + q.length + ' venter på nett' : '') + '. Lagret i Supabase, og modellen henter dem ved neste bygging.</small>' +
    '<div class="sk-tbl">' + (skRows.length ? skRows.slice(0, 50).map(function (r) {
      return '<div>' + skEsc(r.dato) + ' · ' + skEsc(r.art) + ' · ' + (r.funnet ? 'funnet' : 'ingen') + '</div>'; }).join('')
      : 'Ingen registreringer ennå.') + '</div>' +
    '<div class="sk-row"><button class="sk-btn sk-yes" onclick="skExport()">Last ned CSV</button>' +
    '<button class="sk-btn sk-no" onclick="skLogout()">Logg ut</button></div>' +
    '<small>Innlogget som ' + skEsc(skUser.email) + '</small>';
  el.style.display = 'block';
}

function skExport() {
  var csv = 'art,lat,lon,dato,funnet,mengde,notat\\n' + skRows.map(skCsv).join('\\n') + '\\n';
  var u = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})), a = document.createElement('a');
  a.href = u; a.download = 'mine_funn.csv'; document.body.appendChild(a); a.click(); a.remove();
}

function skShowLogin() {
  var el = document.getElementById('sk-list');
  el.innerHTML = '<b>Logg inn for å registrere funn</b><br><small>Funnene dine er private og lagres i Supabase.</small>' +
    '<form id="sk-login" class="sk-pop" style="min-width:0">' +
    '<input id="sk-email" type="email" autocomplete="username" placeholder="E-post" required>' +
    '<input id="sk-pw" type="password" autocomplete="current-password" placeholder="Passord (minst 6 tegn)" required minlength="6">' +
    '<div class="sk-row"><button class="sk-btn sk-yes" type="submit">Logg inn</button>' +
    '<button class="sk-btn sk-no" type="button" onclick="skSignup()">Opprett konto</button></div></form>';
  el.style.display = 'block';
  document.getElementById('sk-login').onsubmit = function (ev) {
    ev.preventDefault();
    sb.auth.signInWithPassword({email: document.getElementById('sk-email').value.trim(),
                                password: document.getElementById('sk-pw').value}).then(function (r) {
      if (r.error) { skToast(r.error.message === 'Email not confirmed' ? 'Bekreft e-posten din først' : 'Feil e-post eller passord'); return; }
      el.style.display = 'none'; skToast('Logget inn');
    });
  };
}

function skSignup() {
  var email = document.getElementById('sk-email').value.trim(), pw = document.getElementById('sk-pw').value;
  if (!email || pw.length < 6) { skToast('Fyll inn e-post og passord (minst 6 tegn)'); return; }
  sb.auth.signUp({email: email, password: pw, options: {emailRedirectTo: location.origin + location.pathname}})
    .then(function (r) {
      if (r.error) { skToast(r.error.message); return; }
      skToast(r.data.session ? 'Konto opprettet' : 'Sjekk e-posten for å bekrefte kontoen');
    });
}

function skLogout() { sb.auth.signOut(); document.getElementById('sk-list').style.display = 'none'; }

window.addEventListener('load', function () {
  var map = %%MAP%%, shown = false, here = null, box = map.getContainer();
  box.insertAdjacentHTML('beforeend',
    '<div class="sk-bar"><button id="sk-here">Logg her</button><button id="sk-mine">Mine (<span id="sk-count">0</span>)</button></div>' +
    '<div class="sk-list" id="sk-list"></div><div class="sk-toast" id="sk-toast"></div>');
  ['sk-here', 'sk-mine', 'sk-list'].forEach(function (id) {
    var el = document.getElementById(id); L.DomEvent.disableClickPropagation(el); L.DomEvent.disableScrollPropagation(el); });
  document.getElementById('sk-here').onclick = function () {
    if (!skUser) { skShowLogin(); return; }
    if (here) { map.setView(here.latlng, Math.max(map.getZoom(), 16)); onMapClick(here); }
    else { skToast('Finner posisjon …'); shown = false; map.locate({setView: true, maxZoom: 16, enableHighAccuracy: true}); }
  };
  document.getElementById('sk-mine').onclick = skShowList;
  map.on('click', function (e) { document.getElementById('sk-list').style.display = 'none'; onMapClick(e); });
  // Første gang posisjonen er funnet: åpne loggeruta der du står.
  map.on('locationfound', function (e) { here = e; if (!shown) { shown = true; onMapClick(e); } });
  map.on('locationerror', function (e) { skToast('Fant ikke posisjonen: ' + e.message); });
  window.addEventListener('online', skFlush);
  sb.auth.onAuthStateChange(function (ev, session) {
    skUser = session ? session.user : null;
    setTimeout(function () { skLoad().then(skFlush); }, 0);
  });
  skDraw();
});
"""

def build_map(args):
    m = folium.Map(location=[args.lat, args.lon], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer(KARTVERKET_TOPO, name="Kartverket topo", attr="&copy; Kartverket").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    folium.raster_layers.WmsTileLayer(url=AR50_WMS, layers="Treslag", fmt="image/png", transparent=True,
                                      version="1.3.0", name="AR50 treslag (NIBIO)", attr="NIBIO", show=False,
                                      opacity=0.6).add_to(m)

    own = load_own(args.log)
    lats, lons = make_grid(args.lat, args.lon, args.radius_km, args.grid_m)
    try:
        layers = [l for l in sr16_layers() if any(f in l.lower() for f in ("treslag", "bonitet", "alder", "hogst"))] or sr16_layers()[:3]
    except Exception as e:
        print(f"SR16 capabilities failed: {e}", file=sys.stderr); layers = []
    X, names = grid_covariates(lats, lons, args.cache, layers)

    for label in args.species:
        sci, color, months = SPECIES[label]
        try:
            gb = fetch_gbif(sci, args.lat, args.lon, args.radius_km, args.year_min,
                            args.max_uncertainty_m, months if args.season_filter else None)
        except Exception as e:
            print(f"GBIF failed for {sci}: {e}", file=sys.stderr); gb = []
        mine = [r for r in own if r["art"].lower() == label.lower()]
        found = [r for r in mine if r["funnet"] == 1]
        blank = [r for r in mine if r["funnet"] == 0]
        print(f"\n{label}: GBIF {len(gb)}, egne funn {len(found)}, egne blanke {len(blank)}")

        fg = folium.FeatureGroup(name=f"{label}: GBIF-funn ({len(gb)})", show=(label == args.species[0]))
        for r in gb:
            folium.CircleMarker([r["lat"], r["lon"]], radius=4, color="#fff", weight=1, fill=True, fill_color=color,
                                fill_opacity=0.9, popup=f"{label}<br>{r['date']}<br>{r['dataset']}").add_to(fg)
        fg.add_to(m)


        pres = [(r["lat"], r["lon"], 1.0) for r in gb] + [(r["lat"], r["lon"], OWN_WEIGHT) for r in found]
        absn = [(r["lat"], r["lon"], OWN_WEIGHT) for r in blank]
        prob, info = train_predict(X, names, lats, lons, pres, absn)
        if prob is None:
            print(f"  model skipped: {info['error']}"); continue
        print(f"  presence cells {info['n_presence_cells']}, own absence {info['n_own_absence_cells']}, "
              f"background {info['n_background']}, CV AUC {info['cv_auc']}")
        print("  importance: " + ", ".join(f"{n}={v:.3f}" for n, v in info["importance"][:6]))
        png, bounds = prob_overlay(prob, lats, lons, color)
        import base64
        uri = "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
        folium.raster_layers.ImageOverlay(uri, bounds=bounds, opacity=0.65, name=f"Modell: {label}",
                                          show=(label == args.species[0])).add_to(m)
        os.makedirs("modell", exist_ok=True)
        np.save(os.path.join("modell", f"{label.replace(' ', '_')}_prob.npy"), prob)

    # Vis egen posisjon (krever https eller localhost). Klikk den blaa prikken for aa logge funn der.
    LocateControl(position="topleft", flyTo=True, keepCurrentZoomLevel=False, initialZoomLevel=16,
                  showCompass=True, strings={"title": "Vis min posisjon"},
                  locateOptions={"enableHighAccuracy": True, "maxZoom": 16}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    v = m.get_name()
    m.get_root().header.add_child(folium.Element(MOBILE_CSS))
    js = CLICK_JS.replace("%%MAP%%", v).replace("%%SB_URL%%", SUPABASE_URL).replace("%%SB_KEY%%", SUPABASE_KEY).replace("%%SPECIES%%", json.dumps(list(SPECIES), ensure_ascii=False))
    m.get_root().script.add_child(folium.Element(js))
    return m

def probe(args):
    print("Elevation:", elevation_batch([(args.lat, args.lon)]))
    ls = sr16_layers(); print("SR16 layers:", ls)
    sel = [l for l in ls if any(f in l.lower() for f in ("treslag", "bonitet", "alder", "hogst"))] or ls[:3]
    print("Using:", sel)
    print("Raw GetFeatureInfo:\n", sr16_featureinfo(args.lat, args.lon, raw=True)[:2000])
    print("Parsed:", sr16_featureinfo(args.lat, args.lon))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, default=AS_LAT); ap.add_argument("--lon", type=float, default=AS_LON)
    ap.add_argument("--radius-km", type=float, default=8)
    ap.add_argument("--grid-m", type=int, default=250, help="cell size for covariates and prediction")
    ap.add_argument("--year-min", type=int, default=2000)
    ap.add_argument("--max-uncertainty-m", type=float, default=300)
    ap.add_argument("--log", default="mine_funn.csv"); ap.add_argument("--cache", default="covariates_cache.json")
    ap.add_argument("--out", default="docs/index.html", help="docs/ publiseres av GitHub Pages")
    ap.add_argument("--species", nargs="+", default=DEFAULT_SPECIES, choices=list(SPECIES),
                    help="arter aa modellere, standard er de fire hoestartene")
    ap.add_argument("--all-species", action="store_true", help="bruk alle 8 artene")
    ap.add_argument("--no-season-filter", dest="season_filter", action="store_false",
                    help="ikke filtrer GBIF-funn paa artens sesongmaaneder")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    load_env()
    if args.all_species:
        args.species = list(SPECIES)
    if args.probe:
        return probe(args)
    m = build_map(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.save(args.out)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
