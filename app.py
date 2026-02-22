import json
from pathlib import Path
from random import Random

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIG UI
# =========================================================
st.set_page_config(
    page_title="Observatoire Transition Écologique — Orléans Métropole",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
      h1 { letter-spacing: -0.02em; margin-bottom: 0.25rem; }
      .subtle { color: var(--text-color); opacity: 0.78; }
      .card {
        border: 1px solid rgba(49,51,63,0.20);
        border-radius: 16px;
        padding: 14px 16px;
        background: var(--secondary-background-color);
        color: var(--text-color);
      }
      .card-title { font-size: 0.85rem; opacity: 0.78; }
      .card-value { font-size: 1.35rem; font-weight: 750; margin-top: 6px; }
      .card-note  { font-size: 0.85rem; opacity: 0.78; margin-top: 6px; }
      button[data-baseweb="tab"] { font-weight: 600; }
      .divider { height: 10px; }
      .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        font-size: 0.8rem;
        margin-left: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

FILES = {
    "conso": DATA_DIR / "consommation-epci-annuelle-2012-a-2017.csv",
    "topics": DATA_DIR / "om-environnement-transitionecologique-topics.csv",
    "solutions": DATA_DIR / "om-environnement-transitionecologique-repertoiresolutions.csv",
    "geojson": DATA_DIR / "administratif_adm_com_agglo.geojson",
}


# =========================================================
# COMMUNES DATA (TES DONNÉES) — sans gentilé
# =========================================================
@st.cache_data(show_spinner=False)
def load_communes_stats() -> pd.DataFrame:
    data = [
        {"nom": "Orléans", "insee": "45234", "superficie_km2": 27.48, "population_2023": 116357, "densite_hab_km2": 4234},
        {"nom": "Boigny-sur-Bionne", "insee": "45034", "superficie_km2": 7.53, "population_2023": 2194, "densite_hab_km2": 291},
        {"nom": "Bou", "insee": "45043", "superficie_km2": 6.29, "population_2023": 1036, "densite_hab_km2": 165},
        {"nom": "Chanteau", "insee": "45072", "superficie_km2": 28.85, "population_2023": 1590, "densite_hab_km2": 55},
        {"nom": "La Chapelle-Saint-Mesmin", "insee": "45075", "superficie_km2": 8.96, "population_2023": 11017, "densite_hab_km2": 1230},
        {"nom": "Chécy", "insee": "45089", "superficie_km2": 15.47, "population_2023": 9083, "densite_hab_km2": 587},
        {"nom": "Combleux", "insee": "45100", "superficie_km2": 1.10, "population_2023": 547, "densite_hab_km2": 497},
        {"nom": "Fleury-les-Aubrais", "insee": "45147", "superficie_km2": 10.12, "population_2023": 21804, "densite_hab_km2": 2155},
        {"nom": "Ingré", "insee": "45169", "superficie_km2": 20.82, "population_2023": 10062, "densite_hab_km2": 483},
        {"nom": "Mardié", "insee": "45194", "superficie_km2": 17.28, "population_2023": 3078, "densite_hab_km2": 178},
        {"nom": "Marigny-les-Usages", "insee": "45197", "superficie_km2": 9.66, "population_2023": 1887, "densite_hab_km2": 195},
        {"nom": "Olivet", "insee": "45232", "superficie_km2": 23.39, "population_2023": 23507, "densite_hab_km2": 1005},
        {"nom": "Ormes", "insee": "45235", "superficie_km2": 18.15, "population_2023": 4376, "densite_hab_km2": 241},
        {"nom": "Saint-Cyr-en-Val", "insee": "45272", "superficie_km2": 44.23, "population_2023": 3467, "densite_hab_km2": 78},
        {"nom": "Saint-Denis-en-Val", "insee": "45274", "superficie_km2": 17.11, "population_2023": 7766, "densite_hab_km2": 454},
        {"nom": "Saint-Hilaire-Saint-Mesmin", "insee": "45282", "superficie_km2": 14.12, "population_2023": 3261, "densite_hab_km2": 231},
        {"nom": "Saint-Jean-de-Braye", "insee": "45284", "superficie_km2": 13.70, "population_2023": 23147, "densite_hab_km2": 1690},
        {"nom": "Saint-Jean-de-la-Ruelle", "insee": "45285", "superficie_km2": 6.10, "population_2023": 16768, "densite_hab_km2": 2749},
        {"nom": "Saint-Jean-le-Blanc", "insee": "45286", "superficie_km2": 7.66, "population_2023": 9562, "densite_hab_km2": 1248},
        {"nom": "Saint-Pryvé-Saint-Mesmin", "insee": "45298", "superficie_km2": 8.87, "population_2023": 6256, "densite_hab_km2": 705},
        {"nom": "Saran", "insee": "45302", "superficie_km2": 19.65, "population_2023": 17316, "densite_hab_km2": 881},
        {"nom": "Semoy", "insee": "45308", "superficie_km2": 7.78, "population_2023": 3269, "densite_hab_km2": 420},
    ]
    df = pd.DataFrame(data)
    df["population_share"] = df["population_2023"] / df["population_2023"].sum()
    return df


# =========================================================
# HELPERS
# =========================================================
def kpi_card(label: str, value: str, note: str | None = None):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{label}</div>
          <div class="card-value">{value}</div>
          {f'<div class="card-note">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("’", "'", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_col_contains(df: pd.DataFrame, substrings: list[str]) -> str | None:
    for col in df.columns:
        c = str(col).lower()
        if any(s in c for s in substrings):
            return col
    return None


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner=False)
def read_csv_fr_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=";",
                engine="python",
                encoding=enc,
                quotechar='"',
                doublequote=True,
                on_bad_lines="skip",
            )
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError(f"Impossible de parser {path}")


@st.cache_data(show_spinner=False)
def load_data():
    missing = [k for k, p in FILES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers manquants dans /data : " + ", ".join(missing) + "\n"
            + "\n".join([f"- {k}: {FILES[k]}" for k in missing])
        )

    conso = _norm_cols(read_csv_fr_robust(FILES["conso"]))
    topics = _norm_cols(read_csv_fr_robust(FILES["topics"]))
    sol = _norm_cols(read_csv_fr_robust(FILES["solutions"]))

    with open(FILES["geojson"], "r", encoding="utf-8") as f:
        geo = json.load(f)

    return conso, topics, sol, geo


def detect_geojson_name_key(geo: dict):
    features = geo.get("features", [])
    if not features:
        return None
    props = features[0].get("properties", {}) or {}
    for k in ["nom", "name", "commune", "nom_commune", "libelle", "libellé"]:
        if k in props:
            return k
    # fallback: first string property
    for k, v in props.items():
        if isinstance(v, str) and len(v) > 1:
            return k
    return None


def get_communes_from_geojson(geo: dict):
    name_key = detect_geojson_name_key(geo)
    if not name_key:
        return [], None
    names = []
    for ft in geo.get("features", []):
        props = ft.get("properties", {}) or {}
        if name_key in props:
            names.append(str(props[name_key]))
    names = sorted(list({n for n in names if n}))
    feature_id_key = f"properties.{name_key}"
    return names, feature_id_key


def build_demo_conso():
    rng = Random(42)
    years = list(range(2012, 2018))
    base = 1000.0
    v = base
    vals = []
    for _ in years:
        v = v * (1 + rng.uniform(-0.03, 0.02))
        vals.append(v)
    return pd.DataFrame({"annee": years, "valeur": vals})


# =========================================================
# LOAD
# =========================================================
try:
    conso_df, topics_df, sol_df, geojson = load_data()
except Exception as e:
    st.error("Impossible de charger les données. Vérifie le dossier /data et les noms des fichiers.")
    st.exception(e)
    st.stop()

communes_stats = load_communes_stats()
communes_geo, geo_feature_id_key = get_communes_from_geojson(geojson)

# =========================================================
# DETECTION COLONNES (conso)
# =========================================================
year_col = pick_col(conso_df, ["annee", "année", "year"]) or pick_col_contains(conso_df, ["annee", "année", "year"])
value_col = pick_col(conso_df, ["valeur", "value", "consommation", "conso", "conso_totale", "consommation_totale"]) or pick_col_contains(conso_df, ["valeur", "conso", "consommation", "mwh", "kwh"])

indicator_col = (
    pick_col(conso_df, ["indicateur", "energie", "énergie", "type", "vecteur_energetique"])
    or pick_col_contains(conso_df, ["ener", "gaz", "elec", "élec", "fioul", "carbur"])
)

if year_col:
    conso_df[year_col] = safe_numeric(conso_df[year_col]).astype("Int64")
if value_col:
    conso_df[value_col] = safe_numeric(conso_df[value_col])

# solutions
theme_col = pick_col(sol_df, ["nom_du_thème", "nom_du_theme", "theme", "thème"]) or pick_col_contains(sol_df, ["nom_du_th", "theme", "thème"])
totem_col = pick_col(sol_df, ["solution_totem", "totem"])
cost_col = pick_col(sol_df, ["valeur_cout", "cout", "coût"])
climate_col = pick_col(sol_df, ["valeur_climat", "impact_climat", "climat"])
if cost_col:
    sol_df[cost_col] = safe_numeric(sol_df[cost_col])
if climate_col:
    sol_df[climate_col] = safe_numeric(sol_df[climate_col])


# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.title("Filtres")

# Liste commune = ta table (référence)
communes_list = communes_stats["nom"].tolist()
selected_commune = st.sidebar.selectbox("Commune", ["(Toutes)"] + communes_list, index=0)

# Période
if year_col and conso_df[year_col].notna().any():
    years = sorted([int(y) for y in conso_df[year_col].dropna().unique().tolist()])
else:
    years = list(range(2012, 2018))
period = st.sidebar.slider("Période", min(years), max(years), (min(years), max(years)))

st.sidebar.divider()

selected_theme = "(Tous)"
if theme_col:
    themes = sorted(sol_df[theme_col].dropna().astype(str).unique().tolist())
    selected_theme = st.sidebar.selectbox("Thème (solutions)", ["(Tous)"] + themes, index=0)

selected_totem = "(Tous)"
if totem_col:
    selected_totem = st.sidebar.selectbox("Solution Totem", ["(Tous)", "Oui", "Non"], index=0)

with st.sidebar.expander("🔎 Debug", expanded=False):
    st.write({"year_col": year_col, "value_col": value_col, "indicator_col": indicator_col})
    st.write({"theme_col": theme_col, "cost_col": cost_col, "climate_col": climate_col, "totem_col": totem_col})


def apply_filters_solutions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if theme_col and selected_theme != "(Tous)":
        d = d[d[theme_col].astype(str) == selected_theme]
    if totem_col and selected_totem != "(Tous)":
        is_yes = d[totem_col].astype(str).str.lower().isin(["1", "true", "oui", "yes"])
        d = d[is_yes] if selected_totem == "Oui" else d[~is_yes]
    return d


# =========================================================
# BUILD CONSO (EPCI) + ALLOCATION COMMUNE (population share)
# =========================================================
use_demo_conso = not (year_col and value_col and conso_df[value_col].notna().any())

if use_demo_conso:
    base_trend = build_demo_conso()
    base_trend = base_trend[base_trend["annee"].between(period[0], period[1])].copy()
    year_used, value_used = "annee", "valeur"

    # énergie démo (si besoin)
    base_energy = pd.DataFrame()
else:
    dfc = conso_df.copy()
    dfc = dfc[dfc[year_col].between(period[0], period[1])].copy()
    year_used, value_used = year_col, value_col

    base_trend = dfc.groupby(year_col, as_index=False)[value_col].sum().sort_values(year_col)

    # Répartition énergie/indicateur (si dispo)
    if indicator_col:
        base_energy = dfc.groupby([year_col, indicator_col], as_index=False)[value_col].sum()
    else:
        base_energy = pd.DataFrame()

# Allocation commune (structurante) : conso_commune = conso_epci * part_pop
allocation_note = None
if selected_commune != "(Toutes)":
    share = float(communes_stats.loc[communes_stats["nom"] == selected_commune, "population_share"].iloc[0])
    allocation_note = "Estimation structurante : répartition EPCI au prorata de la population (pas une mesure communale)."

    trend_commune = base_trend.copy()
    trend_commune[value_used] = trend_commune[value_used] * share
    conso_trend = trend_commune

    if not base_energy.empty:
        energy_commune = base_energy.copy()
        energy_commune[value_used] = energy_commune[value_used] * share
        conso_energy = energy_commune
    else:
        conso_energy = pd.DataFrame()
else:
    conso_trend = base_trend
    conso_energy = base_energy

# Solutions filtrées
sol_f = apply_filters_solutions(sol_df)


# =========================================================
# HEADER
# =========================================================
st.title("🌿 Observatoire Transition Écologique — Orléans Métropole")
st.markdown(
    '<div class="subtle">Évaluation des politiques publiques • Indicateurs énergie (EPCI) + lecture territoriale (communes)</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================
tabs = st.tabs(
    [
        "🏁 Synthèse exécutive",
        "⚡ Évaluation énergétique",
        "🗺 Carte des 22 communes",
        "🎯 Solutions & politiques publiques",
        "📊 Priorisation stratégique",
    ]
)

# ---------------------------------------------------------
# TAB 1 — Synthèse + filtre commune appliqué
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Synthèse exécutive")

    latest_year = int(conso_trend.iloc[-1][year_used])
    latest_value = float(conso_trend.iloc[-1][value_used])
    first_value = float(conso_trend.iloc[0][value_used])
    var_pct = ((latest_value - first_value) / first_value * 100) if first_value else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        label = "Périmètre"
        val = "Orléans Métropole"
        note = f"Commune : {selected_commune}"
        kpi_card(label, val, note)
    with c2:
        kpi_card("Dernière année", f"{latest_year}")
    with c3:
        kpi_card("Consommation (agrégée)", f"{latest_value:,.0f}".replace(",", " "),
                 "Données démo" if use_demo_conso else "Données source")
    with c4:
        kpi_card("Évolution sur période", f"{var_pct:+.1f}%", f"{period[0]}–{period[1]}")

    if allocation_note:
        st.markdown(f"<span class='badge'>Commune = estimation</span> <span class='subtle'>{allocation_note}</span>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        fig = px.line(conso_trend, x=year_used, y=value_used, markers=True, title="Évolution annuelle (filtre commune appliqué)")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if conso_energy.empty:
            st.info("Répartition par énergie non disponible (colonne indicateur/énergie non détectée dans le CSV conso).")
        else:
            # dernière année : treemap énergie
            sec = conso_energy[conso_energy[year_used] == latest_year].groupby(indicator_col, as_index=False)[value_used].sum()
            fig2 = px.treemap(sec, path=[indicator_col], values=value_used, title=f"Répartition par énergie / indicateur — {latest_year}")
            fig2.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — Évaluation énergétique (filtre commune appliqué)
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Évaluation énergétique")

    df_eval = conso_trend.copy()
    df_eval["variation_%"] = df_eval[value_used].pct_change() * 100

    a, b = st.columns([1.2, 1])
    with a:
        fig = px.bar(df_eval, x=year_used, y="variation_%", title="Variation annuelle (%) — filtre commune appliqué")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with b:
        avg = df_eval["variation_%"].dropna().mean()
        mn = df_eval["variation_%"].dropna().min()
        mx = df_eval["variation_%"].dropna().max()

        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_card("Moyenne variation", f"{avg:+.2f}%")
        with k2:
            kpi_card("Min variation", f"{mn:+.2f}%")
        with k3:
            kpi_card("Max variation", f"{mx:+.2f}%")

        if allocation_note:
            st.caption(allocation_note)

# ---------------------------------------------------------
# TAB 3 — Carte + fiche commune (tes données)
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Carte des 22 communes & fiche (population / superficie / densité)")

    # Carte
    if communes_geo and geo_feature_id_key:
        map_df = pd.DataFrame({"commune": communes_geo})
        map_df["z"] = 1
        if selected_commune != "(Toutes)" and selected_commune in set(communes_geo):
            map_df.loc[map_df["commune"] == selected_commune, "z"] = 2

        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="commune",
            featureidkey=geo_feature_id_key,
            color="z",
            mapbox_style="open-street-map",
            zoom=9.2,
            center={"lat": 47.9, "lon": 1.9},
            opacity=0.65,
        )
        fig.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("GeoJSON non exploitable pour la carte (clé nom commune introuvable).")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if selected_commune == "(Toutes)":
        st.info("Sélectionne une commune (sidebar) pour afficher sa fiche.")
    else:
        row = communes_stats[communes_stats["nom"] == selected_commune].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Commune", row["nom"], f"INSEE : {row['insee']}")
        with c2:
            kpi_card("Population (2023)", f"{int(row['population_2023']):,}".replace(",", " "))
        with c3:
            kpi_card("Superficie", f"{float(row['superficie_km2']):.2f} km²")
        with c4:
            kpi_card("Densité", f"{int(row['densite_hab_km2']):,} hab/km²".replace(",", " "))

        st.markdown("**Typologie & priorités (heuristique)**")
        dens = float(row["densite_hab_km2"])
        if dens >= 2000:
            st.write("- **Urbain dense** : priorités mobilité douce/TC, rénovation tertiaire et logements collectifs.")
        elif dens >= 600:
            st.write("- **Périurbain** : intermodalité, rénovation pavillonnaire, EnR locales, sobriété.")
        else:
            st.write("- **Peu dense** : rénovation, solutions bas-carbone, optimisation des déplacements, mutualisation.")

        st.caption("Fiche commune basée sur les valeurs wikipédia (Population 2023 / superficie / densité).")

# ---------------------------------------------------------
# TAB 4 — Solutions
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Solutions & politiques publiques")

    if not theme_col:
        st.warning("Colonne thème non détectée dans le répertoire solutions.")
    else:
        left, right = st.columns([1.2, 1])

        with left:
            cnt = sol_f.groupby(theme_col, as_index=False).size().rename(columns={"size": "nb_actions"})
            cnt = cnt.sort_values("nb_actions", ascending=False)
            fig = px.bar(cnt, x="nb_actions", y=theme_col, orientation="h", title="Nombre d’actions par thème")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            if totem_col:
                tmp = sol_f.copy()
                tmp["_totem"] = tmp[totem_col].astype(str).str.lower().isin(["1", "true", "oui", "yes"])
                pie = tmp["_totem"].value_counts().reset_index()
                pie.columns = ["totem", "nb"]
                pie["totem"] = pie["totem"].map({True: "Totem", False: "Non Totem"})
                fig2 = px.pie(pie, names="totem", values="nb", title="Part des solutions Totem")
                fig2.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Colonne Totem non disponible.")

# ---------------------------------------------------------
# TAB 5 — Priorisation
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Priorisation stratégique (coût vs impact climat)")

    if not (cost_col and climate_col):
        st.warning("Colonnes coût / impact climat non détectées dans le répertoire solutions.")
    else:
        d = sol_f.dropna(subset=[cost_col, climate_col]).copy()
        if len(d) == 0:
            st.info("Aucune donnée exploitable après filtres.")
        else:
            fig = px.scatter(
                d,
                x=cost_col,
                y=climate_col,
                color=theme_col if theme_col else None,
                hover_data=[totem_col] if totem_col else None,
                title="Matrice coût / impact",
            )
            x_med = d[cost_col].median()
            y_med = d[climate_col].median()
            fig.add_vline(x=x_med, line_dash="dot")
            fig.add_hline(y=y_med, line_dash="dot")
            fig.update_layout(height=540, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Lecture : en haut à gauche = **fort impact / coût faible** (quick wins).")