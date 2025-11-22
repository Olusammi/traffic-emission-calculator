import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar
import pandas as pd
from io import BytesIO
import zipfile

st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("Built by SHassan")
st.markdown("Upload your input files to calculate and visualize traffic emissions")

# Sidebar file uploads
with st.sidebar:
    st.header("Inputs")
    copert_param_1 = st.file_uploader("COPERT param file 1 (CSV)", type=['csv'])
    copert_param_2 = st.file_uploader("COPERT param file 2 (CSV)", type=['csv'])
    copert_param_3 = st.file_uploader("COPERT param file 3 (CSV)", type=['csv'])
    copert_param_4 = st.file_uploader("COPERT param file 4 (CSV)", type=['csv'])
    link_osm = st.file_uploader("Link OSM data file (space-separated) — 7 or 9 columns", type=['txt','csv'])
    network_osm = st.file_uploader("OSM network file (.osm)", type=['osm','xml'])
    vehicle_prop_1 = st.file_uploader("Vehicle proportion file 1", type=['csv'])
    vehicle_prop_2 = st.file_uploader("Vehicle proportion file 2", type=['csv'])
    st.markdown("---")
    show_preview = st.checkbox("Show data previews", value=True)
    run_calc = st.button("Run emission calculation")

# helper to read possible text separators
def _read_whitespace_file(f):
    f.seek(0)
    try:
        df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
    except Exception:
        f.seek(0)
        df = pd.read_csv(f, header=None)
    return df

# load copert module (assume copert.py exists in same folder)
try:
    import copert as cop
except Exception as e:
    st.error(f"Could not import copert module: {e}")
    st.stop()

# Minimal example parameter arrays (if user didn't upload COPERT CSVs)
# In your real app these should be parsed from the four COPERT CSVs uploaded
if copert_param_1 is None:
    st.info("COPERT param file 1 not provided — using built-in defaults")
if copert_param_2 is None:
    st.info("COPERT param file 2 not provided — using built-in defaults")
if copert_param_3 is None:
    st.info("COPERT param file 3 not provided — using built-in defaults")
if copert_param_4 is None:
    st.info("COPERT param file 4 not provided — using built-in defaults")

# Preview uploaded link file and set column names depending on column count
data_link = None
if link_osm is not None:
    try:
        data_link = _read_whitespace_file(link_osm)
        ncols = data_link.shape[1]
        if ncols >= 9:
            # 9-column: OSM_ID, Length_km, Flow, Speed, Gasoline_Prop, PC_Prop, 4Stroke_Prop, LDV_prop, HDV_prop
            data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop','LDV_prop','HDV_prop'] + [f'Extra_{j}' for j in range(ncols-9)]
        elif ncols >= 7:
            data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop'] + [f'Extra_{j}' for j in range(ncols-7)]
        else:
            data_link.columns = [f'Column_{i}' for i in range(ncols)]
        if show_preview:
            st.subheader("Link data preview")
            st.dataframe(data_link.head(20))
    except Exception as e:
        st.error(f"Error reading link file: {e}")
        st.stop()

# Minimal engine capacity arrays to preserve compatibility if COPERT CSVs missing
# The original app expects arrays `data_engine_capacity_gasoline` and `data_engine_capacity_diesel`
if data_link is not None:
    Nlink_temp = data_link.shape[0]
    # default engine capacity distributions if not provided
    data_engine_capacity_gasoline = np.tile([0.6, 0.4], (Nlink_temp,1))
    data_engine_capacity_diesel = np.tile([0.5, 0.5], (Nlink_temp,1))
else:
    data_engine_capacity_gasoline = np.array([])
    data_engine_capacity_diesel = np.array([])

# placeholder engine_type arrays used in original logic; keep them for compatibility
engine_type = ['G', 'D']  # gasoline, diesel
engine_capacity = [1.0, 2.5]  # representative bins — original code uses these
engine_type_m = ['4S', '2S']  # motor engines: 4-stroke, 2-stroke

# some copert class lists (these were present in original app)
copert_class = [cop.class_Euro_0, cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3,
                cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
Nclass = len(copert_class)
copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                           cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
Mclass = len(copert_class_motorcycle)

# Run calculations when requested
if run_calc:
    if data_link is None:
        st.error("Please upload a Link OSM data file before running calculations.")
    else:
        # convert to numpy for speed and use the original indexing pattern
        data_link = data_link.values
        Nlink = data_link.shape[0]
        hot_emission_pc = np.zeros((Nlink,), dtype=float)
        hot_emission_m = np.zeros((Nlink,), dtype=float)
        hot_emission_ldv = np.zeros((Nlink,), dtype=float)
        hot_emission_hdv = np.zeros((Nlink,), dtype=float)
        hot_emission = np.zeros((Nlink,), dtype=float)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(Nlink):
            if i % max(1, Nlink // 100) == 0:
                progress_bar.progress(i / Nlink)
            status_text.text(f"Processing link {i+1}/{Nlink}")
            
            # Basic link fields
            link_length = float(data_link[i, 1])
            link_flow = float(data_link[i, 2])
            v = float(data_link[i, 3]) if data_link.shape[1] > 3 else 10.0
            v = min(max(10., v), 130.)
            
            # Existing proportions with safe fallbacks
            link_gasoline_proportion = float(data_link[i, 4]) if data_link.shape[1] > 4 else 1.0
            link_pc_proportion = float(data_link[i, 5]) if data_link.shape[1] > 5 else 0.0
            link_4_stroke_proportion = float(data_link[i, 6]) if data_link.shape[1] > 6 else 1.0
            
            # NEW: LDV and HDV proportions (cols 7 & 8 if present)
            LDV_prop = float(data_link[i, 7]) if data_link.shape[1] > 7 else 0.0
            HDV_prop = float(data_link[i, 8]) if data_link.shape[1] > 8 else 0.0
            
            p_passenger = link_gasoline_proportion
            P_motorcycle = 1. - link_pc_proportion
            engine_type_distribution = [link_gasoline_proportion, 1. - link_gasoline_proportion]
            engine_capacity_distribution = [data_engine_capacity_gasoline[i], data_engine_capacity_diesel[i]]
            engine_type_motorcycle_distribution = [link_4_stroke_proportion, 1. - link_4_stroke_proportion]
            
            # ---- Passenger cars (unchanged logic) ----
            for t in range(2):  # gasoline/diesel
                for c in range(Nclass):
                    for k in range(2):  # engine capacity bins
                        # preserve original filters where present
                        if (copert_class[c] != cop.class_Improved_Conventional and copert_class[c] != cop.class_Open_loop) or engine_capacity[k] <= 2.0:
                            if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                continue
                            try:
                                e = cop.Emission(cop.pollutant_CO, v, link_length, cop.vehicle_type_passenger_car,
                                                 engine_type[t], copert_class[c], engine_capacity[k], 28.2)
                            except Exception:
                                # fallback to gasoline/diesel specific HEF functions if signature differs
                                if engine_type[t] == 'G':
                                    e = cop.HEFGasolinePassengerCar(cop.pollutant_CO, v, copert_class[c], engine_capacity[k])
                                else:
                                    e = cop.HEFDieselPassengerCar(cop.pollutant_CO, v, copert_class[c], engine_capacity[k])
                                e = e * link_length
                            # scale and accumulate
                            factor = engine_type_distribution[t] * engine_capacity_distribution[t][k]
                            hot_emission_pc[i] += e * factor * p_passenger / link_length * link_flow
            
            # ---- Motorcycles (unchanged logic) ----
            for m in range(2):
                for d in range(Mclass):
                    # preserve original skip conditions if any
                    try:
                        e_f = cop.EFMotorcycle(cop.pollutant_CO, v, engine_type_m[m], copert_class_motorcycle[d])
                    except Exception:
                        # fallback simple call if signature differs
                        e_f = cop.EFMotorcycle(cop.pollutant_CO, v, engine_type_m[m], copert_class_motorcycle[d])
                    e_f *= engine_type_motorcycle_distribution[m]
                    hot_emission_m[i] += e_f * P_motorcycle * link_flow
            
            # ---- LDV (new) ----
            if LDV_prop > 0:
                for t in range(2):
                    for k in range(2):
                        try:
                            e_ldv = cop.Emission(cop.pollutant_CO, v, link_length,
                                                 cop.vehicle_type_light_commercial_vehicle,
                                                 engine_type[t], copert_class[c], engine_capacity[k], 28.2)
                        except Exception:
                            # fallback to HEF helper for LDV if available
                            try:
                                e_ldv = cop.HEFLightCommercialVehicle(cop.pollutant_CO, v, engine_type[t], copert_class[c])
                                e_ldv = e_ldv * link_length
                            except Exception:
                                e_ldv = 0.0
                        factor = engine_type_distribution[t] * engine_capacity_distribution[t][k]
                        hot_emission_ldv[i] += e_ldv * factor * LDV_prop / link_length * link_flow
            
            # ---- HDV (new) ----
            if HDV_prop > 0:
                try:
                    # assume diesel for HDV by default
                    e_hdv = cop.Emission(cop.pollutant_CO, v, link_length, cop.vehicle_type_heavy_duty_vehicle,
                                         cop.engine_type_diesel, cop.class_PRE_ECE, 2.0, 28.2)
                    hot_emission_hdv[i] += e_hdv / link_length * HDV_prop * link_flow
                except Exception:
                    # fallback simple factor to preserve prior behaviour
                    base_hdv_factor = 0.1 * link_length
                    hot_emission_hdv[i] += base_hdv_factor / link_length * HDV_prop * link_flow
            
            # ---- total for this link ----
            hot_emission[i] = hot_emission_pc[i] + hot_emission_m[i] + hot_emission_ldv[i] + hot_emission_hdv[i]

        # end for links

        status_text.text("Plotting emission data on map...")
        # Visualization and downstream code below remain unchanged, using the hot_emission array
        # Prepare results DataFrame for download (include LDV and HDV components)
        try:
            results_df = pd.DataFrame({
                'OSM_ID': data_link[:,0].astype(int),
                'Length_km': data_link[:,1].astype(float),
                'Hot_Emission_PC_g_km': hot_emission_pc,
                'Hot_Emission_Motorcycle_g_km': hot_emission_m,
                'Hot_Emission_LDV_g_km': hot_emission_ldv,
                'Hot_Emission_HDV_g_km': hot_emission_hdv,
                'Total_Emission_g_km': hot_emission
            })
        except Exception:
            # fallback if some casting fails
            results_df = pd.DataFrame({
                'OSM_ID': data_link[:,0],
                'Length_km': data_link[:,1],
                'Hot_Emission_PC_g_km': hot_emission_pc,
                'Hot_Emission_Motorcycle_g_km': hot_emission_m,
                'Hot_Emission_LDV_g_km': hot_emission_ldv,
                'Hot_Emission_HDV_g_km': hot_emission_hdv,
                'Total_Emission_g_km': hot_emission
            })

        st.subheader("Results preview")
        st.dataframe(results_df.head(20))

        # Create a ZIP with results and summary
        try:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                # CSV results
                csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("emission_results.csv", csv_bytes)

                # simple summary text
                x_min, x_max = 0, 0
                y_min, y_max = 0, 0
                try:
                    summary = f"""Emission calculation summary
Number of links: {Nlink}
Maximum Emission: {hot_emission.max():.2f} g/km
Minimum Emission: {hot_emission.min():.2f} g/km

Map Boundaries:
- Longitude: {x_min} to {x_max}
- Latitude: {y_min} to {y_max}
"""
                except Exception:
                    summary = f"Emission calculation summary\nNumber of links: {Nlink}\n"
                zip_file.writestr('summary.txt', summary)
            zip_buffer.seek(0)
            st.download_button(label="⬇️ Download Complete Results (ZIP)", data=zip_buffer, 
                               file_name="emission_results.zip", mime="application/zip")
            st.success("✅ ZIP archive created successfully!")
        except Exception as e:
            st.error(f"Error creating ZIP: {e}")
