# app.py
"""
Merged Streamlit app for LAMATA emission tool.
- Combines the Streamlit scaffold UI with an EF-based GHG calculation and
  optional hooks to use local copert.py and osm_network.py if present.
- Save this file next to copert.py and osm_network.py (if available).
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import os
from typing import Dict, Tuple, Optional

st.set_page_config(page_title="Emission Tool (LAMATA)", layout="wide")

# ---------------------------
# Try importing your local modules (copert and osm_network)
# ---------------------------
_HAS_COPERT = False
_HAS_OSM_NETWORK = False
copert = None
osm_network = None
try:
    import copert as copert_impl  # your uploaded copert.py (note filename)
    copert = copert_impl
    _HAS_COPERT = True
except Exception:
    _HAS_COPERT = False

try:
    import osm_network as osm_network_impl
    osm_network = osm_network_impl
    _HAS_OSM_NETWORK = True
except Exception:
    _HAS_OSM_NETWORK = False

# ---------------------------
# Templates & defaults
# ---------------------------
LINK_TEMPLATE = """OSM_ID,Length_km,Flow,Speed,Gasoline_Prop,PC_Prop,4Stroke_Prop
12345,0.42,500,35,0.60,0.45,0.8
23456,1.10,1200,25,0.30,0.60,0.6
"""
TRIP_TEMPLATE = """trip_id,start_lon,start_lat,end_lon,end_lat,vehicle_type,distance_km,fuel_l
T001,3.4001,6.4512,3.4100,6.4590,BRT Bus,2.1,5.8
"""
DEFAULT_EF_IPCC = """vehicle_type,pollutant,EF_g_km,notes
BRT Bus,CO2,1000,IPCC-sample
BRT Bus,CH4,0.05,IPCC-sample
BRT Bus,N2O,0.01,IPCC-sample
Standard Bus,CO2,800,IPCC-sample
Standard Bus,CH4,0.03,IPCC-sample
Standard Bus,N2O,0.008,IPCC-sample
"""

def download_button_string(name: str, data_str: str, mime="text/csv"):
    b = data_str.encode("utf-8")
    st.download_button(label=f"⬇️ Download {name}", data=b, file_name=name, mime=mime)

# ---------------------------
# Validation helpers
# ---------------------------
REQ_LINK_COLS = {"OSM_ID", "Length_km", "Flow", "Speed"}
REQ_EF_COLS = {"vehicle_type", "pollutant", "EF_g_km"}

def read_csv_file(file):
    file.seek(0)
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=r'\s+', header=None, engine='python')
            return df, None
        except Exception as e2:
            return None, f"Unable to parse CSV: {e}"

def validate_links_df(df: pd.DataFrame):
    missing = REQ_LINK_COLS - set(df.columns)
    problems = []
    if missing:
        problems.append(f"Missing columns: {', '.join(missing)}")
    for col in ["Length_km","Flow","Speed"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column {col} must be numeric")
    if "Length_km" in df.columns and (df["Length_km"] <= 0).any():
        problems.append("Some rows have Length_km <= 0")
    if "Flow" in df.columns and (df["Flow"] < 0).any():
        problems.append("Some rows have negative Flow")
    return problems

# ---------------------------
# GHG calculation functions (EF-table path + COPERT hook)
# ---------------------------
DEFAULT_GWP = {"CH4": 28.0, "N2O": 265.0}

def ef_lookup(ef_df: pd.DataFrame, vehicle_type: str, pollutant: str, fallback: float = 0.0) -> float:
    if ef_df is None or ef_df.empty:
        return fallback
    mask = (ef_df["vehicle_type"] == vehicle_type) & (ef_df["pollutant"].str.upper() == pollutant.upper())
    if not mask.any():
        return fallback
    val = ef_df.loc[mask, "EF_g_km"].astype(float).mean()
    return float(val)

def calculate_from_ef_table(
    links_df: pd.DataFrame,
    ef_df: pd.DataFrame,
    vehicle_distribution: Dict[str, float],
    gwp: Dict[str, float] = DEFAULT_GWP
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not REQ_LINK_COLS.issubset(set(links_df.columns)):
        missing = REQ_LINK_COLS - set(links_df.columns)
        raise ValueError(f"links_df missing columns: {missing}")
    if not REQ_EF_COLS.issubset(set(ef_df.columns)):
        missing = REQ_EF_COLS - set(ef_df.columns)
        raise ValueError(f"ef_df missing columns: {missing}")

    total_share = sum(vehicle_distribution.values()) if vehicle_distribution else 0.0
    if total_share <= 0:
        raise ValueError("vehicle_distribution sums to zero; provide positive shares.")
    norm_dist = {k: v / total_share for k, v in vehicle_distribution.items()}

    results = []
    for _, row in links_df.iterrows():
        osm = int(row["OSM_ID"])
        L = float(row["Length_km"])
        flow = float(row["Flow"])
        total_co2 = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        for vtype, share in norm_dist.items():
            ef_co2 = ef_lookup(ef_df, vtype, "CO2", fallback=0.0)
            ef_ch4 = ef_lookup(ef_df, vtype, "CH4", fallback=0.0)
            ef_n2o = ef_lookup(ef_df, vtype, "N2O", fallback=0.0)
            vkm = flow * L * share
            total_co2 += ef_co2 * vkm
            total_ch4 += ef_ch4 * vkm
            total_n2o += ef_n2o * vkm
        ch4_co2e = total_ch4 * gwp.get("CH4", DEFAULT_GWP["CH4"])
        n2o_co2e = total_n2o * gwp.get("N2O", DEFAULT_GWP["N2O"])
        co2e = total_co2 + ch4_co2e + n2o_co2e
        results.append({
            "OSM_ID": osm,
            "Length_km": L,
            "Flow": flow,
            "CO2_g": total_co2,
            "CH4_g": total_ch4,
            "N2O_g": total_n2o,
            "CO2e_g": co2e
        })
    res_df = pd.DataFrame(results)
    summary = {
        "Total_CO2_g": res_df["CO2_g"].sum(),
        "Total_CH4_g": res_df["CH4_g"].sum(),
        "Total_N2O_g": res_df["N2O_g"].sum(),
        "Total_CO2e_g": res_df["CO2e_g"].sum()
    }
    return res_df, summary

def calculate_with_copert_stub(links_df: pd.DataFrame, copert_obj, vehicle_mix_params: dict):
    """
    Hook function: placeholder to call your copert.Copert-based logic.
    I intentionally keep this conservative — paste your original loops here to keep exact behavior.
    """
    if copert_obj is None:
        raise RuntimeError("copert object required for COPERT path.")
    # Minimal example returning empty structure — replace with your original loops to preserve authenticity.
    results = []
    for _, row in links_df.iterrows():
        results.append({
            "OSM_ID": int(row["OSM_ID"]),
            "Length_km": float(row["Length_km"]),
            "Flow": float(row["Flow"]),
            "CO_g": 0.0,
            "NOx_g": 0.0,
            "PM_g": 0.0
        })
    return pd.DataFrame(results), {"Total_CO_g": 0.0, "Total_NOx_g": 0.0, "Total_PM_g": 0.0}

# ---------------------------
# PM stub (simple EF approach)
# ---------------------------
def pm_calculate_from_ef(links_df: pd.DataFrame, ef_pm_df: pd.DataFrame, vehicle_distribution: Dict[str,float]):
    if not REQ_LINK_COLS.issubset(set(links_df.columns)):
        raise ValueError("links_df missing required columns")
    total_share = sum(vehicle_distribution.values()) if vehicle_distribution else 0.0
    if total_share <= 0:
        raise ValueError("vehicle_distribution sums to zero")
    norm_dist = {k: v / total_share for k, v in vehicle_distribution.items()}
    results = []
    for _, row in links_df.iterrows():
        osm = int(row["OSM_ID"]); L = float(row["Length_km"]); flow = float(row["Flow"])
        pm25 = 0.0
        for vtype, share in norm_dist.items():
            mask = (ef_pm_df["vehicle_type"]==vtype) & (ef_pm_df["pollutant"].str.upper()=="PM2.5")
            ef = float(ef_pm_df.loc[mask,"EF_g_km"].mean()) if mask.any() else 0.0
            pm25 += ef * L * flow * share
        results.append({"OSM_ID": osm, "Length_km": L, "PM2.5_g": pm25})
    return pd.DataFrame(results)

# ---------------------------
# LAMATA trip function
# ---------------------------
def lamata_trip_estimate(trips_df: pd.DataFrame, ef_df: pd.DataFrame):
    rows = []
    for _, r in trips_df.iterrows():
        trip = r.get("trip_id","")
        vtype = r.get("vehicle_type","Standard Bus")
        dist = float(r.get("distance_km", np.nan)) if not pd.isna(r.get("distance_km", np.nan)) else np.nan
        fuel = r.get("fuel_l", np.nan)
        mask = (ef_df["vehicle_type"]==vtype) & (ef_df["pollutant"].str.upper()=="CO2")
        ef_co2 = float(ef_df.loc[mask,"EF_g_km"].mean()) if mask.any() else 0.0
        if not np.isnan(dist):
            co2 = ef_co2 * dist
        elif not pd.isna(fuel):
            co2 = float(fuel) * 2650.0
        else:
            co2 = np.nan
        rows.append({"trip_id": trip,"vehicle_type":vtype,"distance_km":dist,"CO2_g":co2})
    return pd.DataFrame(rows)

# ---------------------------
# Session state
# ---------------------------
for key in ["links_df","ef_df","pm_ef_df","trips_df","ghg_results","pm_results","lamata_results","map_fig"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# UI layout and tabs
# ---------------------------
st.title("🚍 Emission Calculation Tool — LAMATA (Merged app.py)")
st.caption("This single-file app uses EF-based GHG calculations and safely hooks to local copert/osm modules if present.")

tab_setup, tab_ghg, tab_pm, tab_lamata, tab_results = st.tabs(
    ["Project Setup", "GHG Model", "PM Model", "LAMATA Fleet", "Results"]
)

with tab_setup:
    st.header("Project Setup")
    col1, col2 = st.columns([2,1])
    with col1:
        osm_file = st.file_uploader("OSM network (.osm) - optional (used if available)", type=["osm","xml"])
        link_file = st.file_uploader("Link/Segment CSV (required)", type=["csv","txt"])
        ef_file = st.file_uploader("Emission Factor CSV (vehicle_type,pollutant,EF_g_km) - optional", type=["csv","txt"])
        pm_ef_file = st.file_uploader("PM EF CSV (vehicle_type,pollutant,EF_g_km) - optional", type=["csv","txt"])
        trip_file = st.file_uploader("Trips CSV (optional)", type=["csv","txt"])
    with col2:
        st.subheader("Templates")
        download_button_string("link_template.csv", LINK_TEMPLATE)
        download_button_string("trip_template.csv", TRIP_TEMPLATE)
        download_button_string("default_ef_ipcc.csv", DEFAULT_EF_IPCC)
        st.write("Module availability:")
        st.write(f"copert.py present: {_HAS_COPERT}")
        st.write(f"osm_network.py present: {_HAS_OSM_NETWORK}")

    st.markdown("---")
    st.subheader("Global settings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        x_center = st.number_input("Map center Longitude", value=3.41000, format="%.6f")
        y_center = st.number_input("Map center Latitude", value=6.45000, format="%.6f")
    with col_b:
        tolerance = st.number_input("OSM parse tolerance", value=0.005, format="%.5f")
        ncore = st.number_input("Parsing cores (unused)", value=4, min_value=1, max_value=16)
    with col_c:
        gwp_choice = st.selectbox("GWP baseline", ["IPCC AR5 (100yr) (default)", "Custom"])
        if gwp_choice.startswith("IPCC"):
            gwp = {"CH4":28.0, "N2O":265.0}
        else:
            ch4 = st.number_input("GWP CH4", value=28.0)
            n2o = st.number_input("GWP N2O", value=265.0)
            gwp = {"CH4":ch4, "N2O":n2o}
        st.write("GWP in use:", gwp)

    st.markdown("---")
    if link_file is not None:
        df_link, err = read_csv_file(link_file)
        if err:
            st.error(err)
        else:
            st.session_state.links_df = df_link
            st.write("Link file preview")
            st.dataframe(df_link.head(10))
            issues = validate_links_df(df_link)
            if issues:
                st.warning("Validation issues:")
                for it in issues:
                    st.warning(it)
            else:
                st.success("Link file validation OK")
    else:
        st.info("Upload Link/Segment CSV to validate")

    if ef_file is not None:
        df_ef, err = read_csv_file(ef_file)
        if err:
            st.error(err)
        else:
            st.session_state.ef_df = df_ef
            if not REQ_EF_COLS.issubset(set(df_ef.columns)):
                st.warning("EF file missing expected columns (vehicle_type,pollutant,EF_g_km). You can still use it but check column names.")
            else:
                st.success("EF CSV loaded.")
                st.dataframe(df_ef.head(5))
    else:
        st.info("No custom EF uploaded — defaults used in model page")

    if pm_ef_file is not None:
        df_pm_ef, err = read_csv_file(pm_ef_file)
        if err:
            st.error(err)
        else:
            st.session_state.pm_ef_df = df_pm_ef
            st.success("PM EF loaded.")
            st.dataframe(df_pm_ef.head(5))

    if trip_file is not None:
        df_trip, err = read_csv_file(trip_file)
        if err:
            st.error(err)
        else:
            st.session_state.trips_df = df_trip
            st.success("Trips CSV loaded.")
            st.dataframe(df_trip.head(5))

with tab_ghg:
    st.header("GHG Emission Model")
    st.markdown("Choose EF source and run GHG calculation (CO2, CH4, N2O → CO2e).")
    ef_option = st.selectbox("EF source", ["Uploaded EF (if provided)", "Use scaffold IPCC sample"], index=0)
    if ef_option == "Use scaffold IPCC sample":
        ef_df = pd.read_csv(StringIO(DEFAULT_EF_IPCC))
    else:
        ef_df = st.session_state.ef_df if st.session_state.ef_df is not None else pd.read_csv(StringIO(DEFAULT_EF_IPCC))
        if st.session_state.ef_df is None:
            st.info("No uploaded EF found; using scaffold sample.")

    st.subheader("Vehicle distribution")
    vtypes = list(ef_df["vehicle_type"].unique()) if "vehicle_type" in ef_df.columns else ["BRT Bus","Standard Bus","Support Vehicle"]
    vehicle_dist = {}
    with st.expander("Set vehicle shares"):
        for vt in vtypes:
            vehicle_dist[vt] = st.number_input(f"{vt} share", min_value=0.0, max_value=1.0, value=0.5 if vt=="Standard Bus" else 0.25, step=0.05, key=f"ghg_share_{vt}")

    prefer_copert = st.checkbox("Prefer COPERT calculation path (only if copert.py available)", value=False)
    run = st.button("Run GHG calculation")
    if run:
        if st.session_state.links_df is None:
            st.error("Upload link CSV in Project Setup first.")
        else:
            try:
                if prefer_copert and _HAS_COPERT:
                    st.info("Running COPERT-based path (note: paste your original loops into the COPERT hook to preserve full authenticity).")
                    copert_obj = copert.Copert if _HAS_COPERT else None
                    ghg_res, ghg_summary = calculate_with_copert_stub(st.session_state.links_df, copert_obj, {})
                else:
                    ghg_res, ghg_summary = calculate_from_ef_table(st.session_state.links_df, ef_df, vehicle_dist, gwp)
                st.session_state.ghg_results = ghg_res
                st.success("GHG calculation complete")
                st.dataframe(ghg_res.head(20))
                st.metric("Total CO2e (g)", f"{ghg_summary['Total_CO2e_g']:.2f}")
            except Exception as e:
                st.error(f"GHG calculation error: {e}")

with tab_pm:
    st.header("PM Emission Model")
    st.markdown("Simple EF-based PM calculation (stub).")
    if st.session_state.pm_ef_df is None:
        st.info("No PM EF uploaded; using simple defaults.")
    pm_dist = {}
    pm_vtypes = ["BRT Bus","Standard Bus"]
    with st.expander("Vehicle shares for PM"):
        for vt in pm_vtypes:
            pm_dist[vt] = st.number_input(f"{vt} share (PM)", min_value=0.0, max_value=1.0, value=0.5 if vt=="Standard Bus" else 0.5, step=0.05, key=f"pm_share_{vt}")
    run_pm = st.button("Run PM calculation")
    if run_pm:
        try:
            efpm = st.session_state.pm_ef_df if st.session_state.pm_ef_df is not None else pd.DataFrame({
                "vehicle_type":["BRT Bus","Standard Bus"],
                "pollutant":["PM2.5","PM2.5"],
                "EF_g_km":[0.5,0.3]
            })
            pm_res = pm_calculate_from_ef(st.session_state.links_df, efpm, pm_dist)
            st.session_state.pm_results = pm_res
            st.success("PM calc done")
            st.dataframe(pm_res.head(20))
            st.metric("Total PM2.5 (g)", f"{pm_res['PM2.5_g'].sum():.2f}")
        except Exception as e:
            st.error(f"PM calc error: {e}")

with tab_lamata:
    st.header("LAMATA Fleet - Trip-based")
    mode = st.radio("Mode", ["Single trip","Bulk CSV"])
    if mode == "Single trip":
        start_lon = st.number_input("Start lon", value=x_center if 'x_center' in locals() else 3.4100, format="%.6f")
        start_lat = st.number_input("Start lat", value=y_center if 'y_center' in locals() else 6.4500, format="%.6f")
        end_lon = st.number_input("End lon", value=start_lon+0.01, format="%.6f")
        end_lat = st.number_input("End lat", value=start_lat+0.01, format="%.6f")
        vehicle_type = st.selectbox("Vehicle type", ["BRT Bus","Standard Bus","Support Vehicle"])
        distance_km = st.number_input("Distance km (optional)", value=0.0, format="%.3f")
        fuel_l = st.number_input("Fuel liters (optional)", value=0.0, format="%.3f")
        run_trip = st.button("Estimate trip emissions")
        if run_trip:
            trips_df = pd.DataFrame([{
                "trip_id":"SINGLE","start_lon":start_lon,"start_lat":start_lat,
                "end_lon":end_lon,"end_lat":end_lat,"vehicle_type":vehicle_type,
                "distance_km":distance_km if distance_km>0 else np.nan,
                "fuel_l": fuel_l if fuel_l>0 else np.nan
            }])
            st.session_state.trips_df = trips_df
            efdf = st.session_state.ef_df if st.session_state.ef_df is not None else pd.read_csv(StringIO(DEFAULT_EF_IPCC))
            lamata_res = lamata_trip_estimate(trips_df, efdf)
            st.session_state.lamata_results = lamata_res
            st.success("Trip estimate done")
            st.dataframe(lamata_res)
    else:
        trip_upload = st.file_uploader("Upload trip CSV", key="trip_csv")
        if trip_upload is not None:
            trips_df, err = read_csv_file(trip_upload)
            if err:
                st.error(err)
            else:
                st.session_state.trips_df = trips_df
                st.dataframe(trips_df.head(10))
                run_bulk = st.button("Run bulk trips")
                if run_bulk:
                    efdf = st.session_state.ef_df if st.session_state.ef_df is not None else pd.read_csv(StringIO(DEFAULT_EF_IPCC))
                    lamata_res = lamata_trip_estimate(trips_df, efdf)
                    st.session_state.lamata_results = lamata_res
                    st.success("Bulk trip calc done")
                    st.dataframe(lamata_res.head(30))

with tab_results:
    st.header("Results & Visualization")
    st.subheader("Map (Matplotlib stub)")
    metric_choice = st.selectbox("Color by", ["CO2e_g (GHG)","PM2.5_g (PM)","None"])
    fig = plt.figure(figsize=(8,6), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title("Emission Map (stub)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    if st.session_state.links_df is not None:
        df = st.session_state.links_df.reset_index().head(500)
        xs = np.linspace(x_center-0.02, x_center+0.02, len(df))
        ys = np.linspace(y_center-0.02, y_center+0.02, len(df))
        ax.scatter(xs, ys, s=10, alpha=0.8)
        if st.session_state.ghg_results is not None and metric_choice=="CO2e_g (GHG)":
            merged = pd.merge(df, st.session_state.ghg_results, on="OSM_ID", how="left")
            sizes = (merged["CO2e_g"].fillna(0) / (merged["CO2e_g"].max() + 1e-9)) * 100 + 5
            ax.scatter(xs, ys, s=sizes, alpha=0.6)
    else:
        ax.text(0.5,0.5,"Upload link CSV and run calculations to see map", ha="center", va="center", transform=ax.transAxes)
    st.pyplot(fig)
    st.session_state.map_fig = fig

    st.subheader("Downloads")
    if st.session_state.ghg_results is not None:
        st.download_button("⬇️ Download GHG CSV", st.session_state.ghg_results.to_csv(index=False).encode("utf-8"), file_name="ghg_results.csv", mime="text/csv")
    else:
        st.info("Run GHG model to download results")
    if st.session_state.pm_results is not None:
        st.download_button("⬇️ Download PM CSV", st.session_state.pm_results.to_csv(index=False).encode("utf-8"), file_name="pm_results.csv", mime="text/csv")
    else:
        st.info("Run PM model to download results")
    if st.session_state.lamata_results is not None:
        st.download_button("⬇️ Download Trip CSV", st.session_state.lamata_results.to_csv(index=False).encode("utf-8"), file_name="lamata_trips.csv", mime="text/csv")

    if st.session_state.map_fig is not None:
        buf = BytesIO()
        st.session_state.map_fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        st.download_button("⬇️ Download Map PNG", data=buf, file_name="emission_map.png", mime="image/png")

st.markdown("---")
st.caption("This merged app keeps your local copert/osm modules untouched and provides EF-based GHG calculations. To fully preserve EXACT original behavior, paste your original calculation loops into the COPERT hook (calculate_with_copert_stub) where noted.")
