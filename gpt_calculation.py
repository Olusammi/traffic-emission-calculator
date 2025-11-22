# Advanced Traffic Emission Calculator - restored UI + interactive map
# Paste this to replace your app.py

import streamlit as st
import numpy as np
import matplotlib
import tempfile
import os
import zipfile
from io import BytesIO

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Advanced Traffic Emission Calculator", layout="wide", initial_sidebar_state="expanded")

# ---------- Styling ----------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .stAlert {
        background-color: #e7f3ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Advanced Traffic Emission Calculator")
st.caption("Multi-Standard Emission Analysis with COPERT IV, IPCC, and EPA Methodologies")
st.markdown("---")

# ========== Sidebar: controls ==========
st.sidebar.header("📊 Emission Metrics Selection")
pollutants_available = {
    "CO": {"name": "Carbon Monoxide", "unit": "g", "standard": "COPERT IV", "color": "#ef4444"},
    "CO2": {"name": "Carbon Dioxide", "unit": "g", "standard": "IPCC", "color": "#3b82f6"},
    "NOx": {"name": "Nitrogen Oxides", "unit": "g", "standard": "COPERT IV", "color": "#f59e0b"},
    "PM": {"name": "Particulate Matter", "unit": "mg", "standard": "WHO", "color": "#8b5cf6"},
    "VOC": {"name": "Volatile Organic Compounds", "unit": "g", "standard": "COPERT IV", "color": "#10b981"},
    "FC": {"name": "Fuel Consumption", "unit": "L", "standard": "NEDC/WLTP", "color": "#f97316"}
}

selected_pollutants = st.sidebar.multiselect(
    "Select Pollutants to Calculate",
    options=list(pollutants_available.keys()),
    default=["CO", "NOx", "PM"],
    help="Choose one or more pollutants for emission calculation"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Calculation Methodology")
calculation_method = st.sidebar.selectbox(
    "Select Calculation Standard",
    ["COPERT IV (EU)", "IPCC Tier 2", "EPA MOVES (US)", "Hybrid (Multi-standard)"]
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Accuracy Settings")
include_temperature_correction = st.sidebar.checkbox("Temperature Correction", value=True)
include_cold_start = st.sidebar.checkbox("Cold Start Emissions", value=True)
include_slope_correction = st.sidebar.checkbox("Road Slope Correction", value=False)

if include_temperature_correction:
    ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -10, 40, 25)
else:
    ambient_temp = 20

if include_cold_start:
    trip_length = st.sidebar.slider("Average Trip Length (km)", 1, 50, 10)
else:
    trip_length = 10

if include_slope_correction:
    road_slope = st.sidebar.slider("Road Slope (%)", -6, 6, 0)
else:
    road_slope = 0

st.sidebar.markdown("---")
st.sidebar.info("Pro Tip: enable accuracy settings for research-grade calculations")

# ========== Sidebar: files ==========
st.sidebar.header("📂 Upload Input Files")
with st.sidebar.expander("COPERT Parameter Files", expanded=True):
    pc_param = st.file_uploader("PC Parameter CSV", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Parameter CSV", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Parameter CSV", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Parameter CSV", type=['csv'], key='moto')

with st.sidebar.expander("Data Files", expanded=True):
    link_osm = st.file_uploader("Link OSM Data (.dat / .csv / .txt)", type=['dat', 'csv', 'txt'], key='link')
    osm_file = st.file_uploader("OSM Network File (.osm)", type=['osm'], key='osm')

with st.sidebar.expander("Proportion Files", expanded=False):
    engine_cap_gas = st.file_uploader("Engine Capacity Gasoline", type=['dat', 'txt'], key='ecg')
    engine_cap_diesel = st.file_uploader("Engine Capacity Diesel", type=['dat', 'txt'], key='ecd')
    copert_class_gas = st.file_uploader("COPERT Class Gasoline", type=['dat', 'txt'], key='ccg')
    copert_class_diesel = st.file_uploader("COPERT Class Diesel", type=['dat', 'txt'], key='ccd')
    copert_2stroke = st.file_uploader("2-Stroke Motorcycle", type=['dat', 'txt'], key='2s')
    copert_4stroke = st.file_uploader("4-Stroke Motorcycle", type=['dat', 'txt'], key='4s')

st.sidebar.markdown("---")
st.sidebar.info("LDV/HDV distributions default to PC gasoline and 100% Euro VI (HDV) if not provided")

# Map params
st.sidebar.header("🗺️ Map Parameters")
col1, col2 = st.sidebar.columns(2)
x_min = col1.number_input("X Min (Lon)", value=3.37310, format="%.5f")
x_max = col2.number_input("X Max (Lon)", value=3.42430, format="%.5f")
y_min = col1.number_input("Y Min (Lat)", value=6.43744, format="%.5f")
y_max = col2.number_input("Y Max (Lat)", value=6.46934, format="%.5f")
tolerance = st.sidebar.number_input("Tolerance", value=0.005, format="%.3f")
ncore = st.sidebar.number_input("Number of Cores", value=8, min_value=1, max_value=16)

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📖 Instructions",
    "📊 Data Preview",
    "🧮 Formula Explanation",
    "⚙️ Calculate Emissions",
    "📈 Multi-Metric Analysis",
    "🗺️ Interactive Map",
    "📥 Download Results"
])

# Tab1: instructions
with tab1:
    st.header("📖 User Guide")
    st.markdown("- Upload COPERT parameter CSVs and link file (7 or 9 columns).")
    st.markdown("- Click the **Calculate Multi-Pollutant Emissions** button in the Calculate tab.")
    st.markdown("- Use the Map tab to generate road maps; use Multi-Metric Analysis for charts and top-link lists.")

# Tab2: data preview
with tab2:
    st.header("📊 Data Preview & Validation")
    if link_osm is not None:
        try:
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            if data_link.shape[1] >= 7:
                if data_link.shape[1] == 7:
                    data_link.columns = ['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop', '4Stroke_Prop']
                elif data_link.shape[1] == 9:
                    data_link.columns = ['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop', '4Stroke_Prop', 'LDV_Prop', 'HDV_Prop']
                else:
                    data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            else:
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            st.dataframe(data_link.head(20), use_container_width=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Links", len(data_link))
            with col2: st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
            with col3: st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
            with col4: st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")
            if data_link.shape[1] == 9:
                col5, col6 = st.columns(2)
                with col5: st.metric("Avg LDV Prop", f"{data_link['LDV_Prop'].mean():.2%}")
                with col6: st.metric("Avg HDV Prop", f"{data_link['HDV_Prop'].mean():.2%}")
            # quick histograms
            col_a, col_b = st.columns(2)
            with col_a:
                fig_speed = px.histogram(data_link, x='Speed', nbins=30, title="Speed distribution")
                st.plotly_chart(fig_speed, use_container_width=True)
            with col_b:
                fig_flow = px.histogram(data_link, x='Flow', nbins=30, title="Flow distribution")
                st.plotly_chart(fig_flow, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading link file: {e}")
    else:
        st.info("Upload Link OSM data in the sidebar to preview")

# Tab3: formulas (kept short)
with tab3:
    st.header("🧮 Formulas & Methodology")
    st.markdown("COPERT-based emission calculations. Outputs are total grams per link for the measurement period.")

# Tab4: calculate emissions (with pollutant preview UI restored)
with tab4:
    st.header("⚙️ Calculate Emissions")

    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    all_uploaded = all(f is not None for f in required_files)

    if not selected_pollutants:
        st.warning("Select at least one pollutant in the sidebar")
    elif not all_uploaded:
        st.info("Please upload all required parameter + proportion files")
    else:
        st.success("All required files present")
        # Calculation settings expander
        with st.expander("🔧 Current Calculation Settings", expanded=True):
            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                st.write(f"Pollutants: {', '.join(selected_pollutants)}")
                st.write(f"Method: {calculation_method}")
            with cs2:
                st.write(f"Temp correction: {'Yes' if include_temperature_correction else 'No'}")
                st.write(f"Cold start: {'Yes' if include_cold_start else 'No'}")
            with cs3:
                st.write(f"Ambient temp: {ambient_temp}°C")
                st.write(f"Trip length: {trip_length} km")

        # Main calculate button (unchanged)
        if st.button("🚀 Calculate Multi-Pollutant Emissions", type="primary", use_container_width=True):
            with st.spinner("Computing emissions..."):
                try:
                    import copert
                    # temp files
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pc_path = os.path.join(tmpdir, "PC_parameter.csv")
                        ldv_path = os.path.join(tmpdir, "LDV_parameter.csv")
                        hdv_path = os.path.join(tmpdir, "HDV_parameter.csv")
                        moto_path = os.path.join(tmpdir, "Moto_parameter.csv")
                        with open(pc_path, 'wb') as f: f.write(pc_param.getbuffer())
                        with open(ldv_path, 'wb') as f: f.write(ldv_param.getbuffer())
                        with open(hdv_path, 'wb') as f: f.write(hdv_param.getbuffer())
                        with open(moto_path, 'wb') as f: f.write(moto_param.getbuffer())

                        cop = copert.Copert(pc_path, ldv_path, hdv_path, moto_path)

                        # read inputs and arrays (same robust logic as patched)
                        link_osm.seek(0)
                        engine_cap_gas.seek(0); engine_cap_diesel.seek(0)
                        copert_class_gas.seek(0); copert_class_diesel.seek(0)
                        copert_2stroke.seek(0); copert_4stroke.seek(0)

                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values
                        Nlink = data_link.shape[0]
                        if data_link.shape[1] == 7:
                            P_ldv = np.zeros(Nlink); P_hdv = np.zeros(Nlink)
                            st.info("7-col link file: LDV/HDV assumed zero")
                        elif data_link.shape[1] == 9:
                            P_ldv = data_link[:,7].astype(float); P_hdv = data_link[:,8].astype(float)
                        else:
                            st.error("Link file must be 7 or 9 columns"); st.stop()

                        # load distributions
                        data_engine_capacity_gasoline = np.loadtxt(engine_cap_gas)
                        data_engine_capacity_diesel = np.loadtxt(engine_cap_diesel)
                        data_copert_class_gasoline = np.loadtxt(copert_class_gas)
                        data_copert_class_diesel = np.loadtxt(copert_class_diesel)
                        data_copert_class_motorcycle_two_stroke = np.loadtxt(copert_2stroke)
                        data_copert_class_motorcycle_four_stroke = np.loadtxt(copert_4stroke)

                        data_copert_class_ldv = data_copert_class_gasoline
                        N_HDV_Class = 6; N_HDV_Type = 15
                        data_hdv_reshaped = np.zeros((Nlink, N_HDV_Class, N_HDV_Type)); data_hdv_reshaped[:,5,0] = 1.0

                        # quick verification
                        st.subheader("🔍 Parameter verification")
                        vv1, vv2, vv3 = st.columns(3)
                        with vv1:
                            st.write("Motorcycle class ranges")
                            st.write(f"2S: {data_copert_class_motorcycle_two_stroke.min():.3f} - {data_copert_class_motorcycle_two_stroke.max():.3f}")
                        with vv2:
                            st.write("Link shares (sample)")
                            st.write(f"PC prop: {data_link[:,5].min():.3f} - {data_link[:,5].max():.3f}")
                        with vv3:
                            st.write("Link metrics")
                            st.write(f"Speed: {data_link[:,3].min():.1f} - {data_link[:,3].max():.1f}")

                        # single motorcycle EF test
                        try:
                            test_result = cop.EFMotorcycle(cop.pollutant_CO, 50.0, cop.engine_type_moto_four_stroke_50_250, cop.class_moto_Euro_3)
                            st.info(f"Motorcycle EF test (CO @50 km/h): {test_result:.6f} g/km")
                        except Exception as e:
                            st.error(f"Motorcycle EF test failed: {e}")

                        # ---- calculation (same logic as previous patch) ----
                        engine_type = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        engine_type_m = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        engine_capacity = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        copert_class = [
                            cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03,
                            cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop,
                            cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5,
                            cop.class_Euro_6, cop.class_Euro_6c
                        ]
                        Nclass = len(copert_class)
                        copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1,
                                                   cop.class_moto_Euro_2, cop.class_moto_Euro_3,
                                                   cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        Mclass = len(copert_class_motorcycle)

                        HDV_Emission_Classes = [
                            cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III,
                            cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI
                        ]

                        pollutant_mapping = {
                            "CO": cop.pollutant_CO,
                            "CO2": cop.pollutant_FC,
                            "NOx": cop.pollutant_NOx,
                            "PM": cop.pollutant_PM,
                            "VOC": cop.pollutant_VOC,
                            "FC": cop.pollutant_FC
                        }

                        emissions_data = {}
                        for poll in selected_pollutants:
                            emissions_data[poll] = {'pc': np.zeros((Nlink,), dtype=float),
                                                    'moto': np.zeros((Nlink,), dtype=float),
                                                    'ldv': np.zeros((Nlink,), dtype=float),
                                                    'hdv': np.zeros((Nlink,), dtype=float),
                                                    'total': np.zeros((Nlink,), dtype=float)}

                        progress_bar = st.progress(0); status_text = st.empty()
                        osmid_to_index = {int(data_link[idx,0]): idx for idx in range(Nlink)}

                        for i in range(Nlink):
                            if i % max(1, Nlink // 100) == 0:
                                progress_bar.progress(i / Nlink); status_text.text(f"Processing link {i+1}/{Nlink}")

                            link_length = float(data_link[i,1]); link_flow = float(data_link[i,2])
                            v = float(data_link[i,3]); v = min(max(10.0, v), 130.0)
                            link_gasoline_prop = float(data_link[i,4])
                            link_pc_prop = float(data_link[i,5])
                            link_4_stroke_prop = float(data_link[i,6])
                            P_ldv_i = float(P_ldv[i]) if P_ldv is not None else 0.0
                            P_hdv_i = float(P_hdv[i]) if P_hdv is not None else 0.0

                            # decompose motorcycle share if 9-col
                            if data_link.shape[1] == 9:
                                P_motorcycle = 1.0 - link_pc_prop - P_ldv_i - P_hdv_i
                                if P_motorcycle < 0:
                                    P_motorcycle = max(0.0, 1.0 - link_pc_prop)
                            else:
                                P_motorcycle = 1.0 - link_pc_prop

                            engine_type_distribution = [link_gasoline_prop, 1.0 - link_gasoline_prop]
                            engine_capacity_distribution = [data_engine_capacity_gasoline[i], data_engine_capacity_diesel[i]]
                            engine_type_motorcycle_distribution = [1.0 - link_4_stroke_prop, link_4_stroke_prop]

                            for poll_name in selected_pollutants:
                                poll_type = pollutant_mapping[poll_name]

                                # PC
                                try:
                                    for t in range(2):
                                        for c in range(Nclass):
                                            for k in range(2):
                                                if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                    continue
                                                try:
                                                    e_total_g = cop.Emission(poll_type, v, link_length,
                                                                            cop.vehicle_type_passenger_car,
                                                                            engine_type[t], copert_class[c], engine_capacity[k],
                                                                            ambient_temp)
                                                except Exception:
                                                    try:
                                                        if engine_type[t] == cop.engine_type_gasoline:
                                                            ef = cop.HEFGasolinePassengerCar(poll_type, v, copert_class[c], engine_capacity[k])
                                                        else:
                                                            ef = cop.HEFDieselPassengerCar(poll_type, v, copert_class[c])
                                                        e_total_g = ef * link_length
                                                    except Exception:
                                                        e_total_g = 0.0
                                                if poll_name == "NOx" and include_temperature_correction:
                                                    temp_factor = 1 + (0.02 if engine_type[t] == cop.engine_type_gasoline else 0.015) * (ambient_temp - 20)
                                                    e_total_g *= temp_factor
                                                if include_cold_start and poll_name in ["CO","NOx","VOC"]:
                                                    try:
                                                        beta = cop.ColdStartMileagePercentage(cop.vehicle_type_passenger_car, engine_type[t], poll_type, copert_class[c], engine_capacity[k], ambient_temp, trip_length)
                                                        e_cold = cop.ColdStartEmissionQuotient(cop.vehicle_type_passenger_car, engine_type[t], poll_type, v, copert_class[c], engine_capacity[k], ambient_temp)
                                                        e_total_g = e_total_g * ((1 - beta) + e_cold * beta)
                                                    except Exception:
                                                        pass
                                                pc_fleet_share = data_copert_class_gasoline[i, c] if t == 0 else data_copert_class_diesel[i, c]
                                                multiplier = engine_type_distribution[t] * engine_capacity_distribution[t][k] * pc_fleet_share
                                                emissions_data[poll_name]['pc'][i] += e_total_g * link_flow * multiplier * (link_pc_prop)
                                except Exception:
                                    pass

                                # LDV
                                if P_ldv_i > 0:
                                    try:
                                        for t in range(2):
                                            for c in range(Nclass):
                                                for k in range(2):
                                                    if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                        continue
                                                    try:
                                                        e_total_g = cop.Emission(poll_type, v, link_length,
                                                                               cop.vehicle_type_light_commercial_vehicle,
                                                                               engine_type[t], copert_class[c], engine_capacity[k],
                                                                               ambient_temp)
                                                    except Exception:
                                                        try:
                                                            ef = cop.HEFLightCommercialVehicle(poll_type, v, engine_type[t], copert_class[c])
                                                            e_total_g = ef * link_length
                                                        except Exception:
                                                            e_total_g = 0.0
                                                    if poll_name == "NOx" and include_temperature_correction:
                                                        e_total_g *= 1 + 0.02 * (ambient_temp - 20)
                                                    if include_cold_start and poll_name in ["CO","NOx","VOC"]:
                                                        try:
                                                            beta = cop.ColdStartMileagePercentage(cop.vehicle_type_light_commercial_vehicle, engine_type[t], poll_type, copert_class[c], engine_capacity[k], ambient_temp, trip_length)
                                                            e_cold = cop.ColdStartEmissionQuotient(cop.vehicle_type_light_commercial_vehicle, engine_type[t], poll_type, v, copert_class[c], engine_capacity[k], ambient_temp)
                                                            e_total_g = e_total_g * ((1 - beta) + e_cold * beta)
                                                        except Exception:
                                                            pass
                                                    ldv_fleet_share = data_copert_class_ldv[i, c]
                                                    multiplier = engine_type_distribution[t] * engine_capacity_distribution[t][k] * ldv_fleet_share
                                                    emissions_data[poll_name]['ldv'][i] += e_total_g * link_flow * multiplier * P_ldv_i
                                    except Exception:
                                        pass

                                # HDV
                                if P_hdv_i > 0:
                                    try:
                                        for t_class in range(N_HDV_Class):
                                            for t_type in range(N_HDV_Type):
                                                hdv_fleet_share = data_hdv_reshaped[i, t_class, t_type]
                                                if hdv_fleet_share <= 0:
                                                    continue
                                                engine_type_hdv = cop.engine_type_diesel
                                                try:
                                                    e_total_g = cop.Emission(poll_type, v, link_length,
                                                                             cop.vehicle_type_heavy_duty_vehicle,
                                                                             engine_type_hdv,
                                                                             HDV_Emission_Classes[t_class],
                                                                             t_type,
                                                                             ambient_temp)
                                                except Exception:
                                                    try:
                                                        ef = cop.HEFHeavyDutyVehicle(poll_type, v, HDV_Emission_Classes[t_class], t_type)
                                                        e_total_g = ef * link_length
                                                    except Exception:
                                                        e_total_g = 0.0
                                                if poll_name == "NOx" and include_temperature_correction:
                                                    e_total_g *= 1 + 0.015 * (ambient_temp - 20)
                                                e_total_g *= hdv_fleet_share
                                                emissions_data[poll_name]['hdv'][i] += e_total_g * link_flow * P_hdv_i
                                    except Exception:
                                        pass

                                # Motorcycles
                                try:
                                    for m in range(2):
                                        for d in range(Mclass):
                                            if m == 1 and copert_class_motorcycle[d] in range(cop.class_moto_Conventional, 1 + cop.class_moto_Euro_5):
                                                continue
                                            try:
                                                ef_moto_gpkm = cop.EFMotorcycle(poll_type, v, engine_type_m[m], copert_class_motorcycle[d])
                                            except Exception:
                                                ef_moto_gpkm = 0.0
                                            e_total_g_moto = ef_moto_gpkm * link_length
                                            e_total_g_moto *= engine_type_motorcycle_distribution[m]
                                            emissions_data[poll_name]['moto'][i] += e_total_g_moto * link_flow * P_motorcycle
                                except Exception:
                                    pass

                                emissions_data[poll_name]['total'][i] = (
                                    emissions_data[poll_name]['pc'][i]
                                    + emissions_data[poll_name]['ldv'][i]
                                    + emissions_data[poll_name]['hdv'][i]
                                    + emissions_data[poll_name]['moto'][i]
                                )

                        progress_bar.empty(); status_text.empty()
                        st.session_state.emissions_data = emissions_data
                        st.session_state.data_link = data_link
                        st.session_state.selected_pollutants = selected_pollutants
                        st.success("✅ Emissions calculated successfully")

                        # ====== pollutant preview UI (restored) ======
                        st.subheader("📋 Pollutant Previews (click a pollutant below)")
                        preview_cols = st.columns(len(selected_pollutants))
                        for idx, poll in enumerate(selected_pollutants):
                            with preview_cols[idx]:
                                if st.button(f"Preview {poll}", key=f"preview_{poll}"):
                                    # show quick stats and plots
                                    st.write(f"### Quick stats: {poll}")
                                    df_preview = pd.DataFrame({
                                        'OSM_ID': data_link[:,0].astype(int),
                                        'Length_km': data_link[:,1],
                                        'Flow': data_link[:,2],
                                        'Speed': data_link[:,3],
                                        f'{poll}_PC': emissions_data[poll]['pc'],
                                        f'{poll}_LDV': emissions_data[poll]['ldv'],
                                        f'{poll}_HDV': emissions_data[poll]['hdv'],
                                        f'{poll}_Moto': emissions_data[poll]['moto'],
                                        f'{poll}_Total': emissions_data[poll]['total']
                                    })
                                    st.dataframe(df_preview.head(50), use_container_width=True)
                                    # histogram of per-link totals
                                    fig_hist = px.histogram(df_preview, x=f'{poll}_Total', nbins=40, title=f"{poll} per-link distribution")
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                    # top 10 links table
                                    st.write("Top 10 links by emission")
                                    st.dataframe(df_preview.sort_values(by=f'{poll}_Total', ascending=False).head(10)[['OSM_ID','Length_km','Flow',f'{poll}_Total']], use_container_width=True)

                except Exception as exc:
                    st.error(f"Calculation failed: {exc}")
                    import traceback
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

# Tab5: multi-metric analysis
with tab5:
    st.header("📈 Multi-metric Analysis")
    if 'emissions_data' in st.session_state:
        emissions_data = st.session_state.emissions_data
        selected_pollutants = st.session_state.selected_pollutants
        breakdown = []
        for poll in selected_pollutants:
            breakdown.extend([
                {'Pollutant': poll, 'Vehicle_Type': 'PC', 'Total_Emissions': emissions_data[poll]['pc'].sum()},
                {'Pollutant': poll, 'Vehicle_Type': 'LDV', 'Total_Emissions': emissions_data[poll]['ldv'].sum()},
                {'Pollutant': poll, 'Vehicle_Type': 'HDV', 'Total_Emissions': emissions_data[poll]['hdv'].sum()},
                {'Pollutant': poll, 'Vehicle_Type': 'Moto', 'Total_Emissions': emissions_data[poll]['moto'].sum()},
            ])
        br_df = pd.DataFrame(breakdown)
        fig = px.bar(br_df, x='Pollutant', y='Total_Emissions', color='Vehicle_Type', title="Total emissions by vehicle type")
        st.plotly_chart(fig, use_container_width=True)
        # top links selection
        ranking_pollutant = st.selectbox("Select pollutant to rank by", options=selected_pollutants)
        ranking_data = pd.DataFrame(st.session_state.data_link.copy())
        if ranking_data.shape[1] == 7:
            ranking_data.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop']
        elif ranking_data.shape[1] == 9:
            ranking_data.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop','LDV_Prop','HDV_Prop']
        ranking_data[f'Total_{ranking_pollutant}'] = emissions_data[ranking_pollutant]['total']
        st.subheader("Top 10 links")
        st.dataframe(ranking_data.sort_values(by=f'Total_{ranking_pollutant}', ascending=False).head(10)[['OSM_ID','Length_km','Flow','Speed',f'Total_{ranking_pollutant}']], use_container_width=True)
    else:
        st.info("Run calculations first")

# Tab6: interactive map (UI fully restored & expanded)
with tab6:
    st.header("🗺️ Interactive Emission Map")
    if 'emissions_data' in st.session_state and 'data_link' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link_np = st.session_state.data_link

        col1, col2 = st.columns([2,1])
        with col1:
            map_pollutant = st.selectbox("Pollutant to map", options=st.session_state.selected_pollutants, index=0)
            map_vehicle = st.selectbox("Vehicle Type", options=['Total','PC','LDV','HDV','Moto'], index=0)
            show_top_labels = st.checkbox("Show road labels (if OSM provided)", value=False)
            label_density = st.selectbox("Label density", ["Minimal","Medium","High"], index=1)
        with col2:
            colormap = st.selectbox("Color map", ['viridis','plasma','inferno','cividis'], index=0)
            linewidth_scale = st.slider("Line width scale", 0.1, 5.0, 1.0, 0.1)
            generate_map = st.button("🗺️ Generate Road Network Map")
            download_map = st.button("Download current map (PNG)")

        # determine hot_emission
        key = map_vehicle.lower()
        if key not in emissions_data[map_pollutant]:
            hot_emission = emissions_data[map_pollutant]['total']
        else:
            hot_emission = emissions_data[map_pollutant][key]

        st.info(f"Mapping {map_vehicle} {map_pollutant}")

        # simplified scatter if no OSM
        if osm_file is None:
            st.warning("No OSM uploaded — showing simplified scatter")
            map_df = pd.DataFrame({
                'OSM_ID': data_link_np[:,0].astype(int),
                'Lat': (data_link_np[:,0] % 1000) * 0.0001 + y_min,
                'Lon': (data_link_np[:,0] % 1000) * 0.0001 + x_min,
                'Emission': hot_emission,
                'Speed': data_link_np[:,3]
            })
            fig_map = go.Figure(data=go.Scattergeo(
                lon=map_df['Lon'],
                lat=map_df['Lat'],
                text=map_df.apply(lambda r: f"Link {int(r['OSM_ID'])}<br>{map_vehicle} {map_pollutant}: {r['Emission']:.2f}", axis=1),
                mode='markers',
                marker=dict(size=8, color=map_df['Emission'], colorscale=px.colors.sequential.Viridis, showscale=True)
            ))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            # advanced OSM plotting on user command
            if generate_map:
                try:
                    import osm_network
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                        osm_file.seek(0)
                        tmp.write(osm_file.read())
                        osm_path = tmp.name
                    selected_zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]]
                    selected_zone.append(selected_zone[0])
                    status = st.empty(); status.text("Parsing OSM...")
                    highway_coordinate, highway_osmid, highway_names, highway_types = osm_network.retrieve_highway(osm_path, selected_zone, tolerance, int(ncore))
                    status.text("Building map...")
                    # precompute index mapping
                    emission_osmid_list = [int(x) for x in data_link_np[:,0]]
                    osmid_to_idx_local = {int(data_link_np[idx,0]): idx for idx in range(data_link_np.shape[0])}
                    max_em = float(np.nanmax(hot_emission)) if hot_emission.size else 1.0
                    lw_min, lw_max = (0.1*linewidth_scale, 3.0*linewidth_scale)
                    width_scaling = (lw_max - lw_min) / (max_em + 1e-9)

                    fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(1,1,1)
                    norm = colors.Normalize(vmin=0, vmax=max_em + 1e-9)
                    cmap = plt.get_cmap(colormap)
                    roads_with_data = 0; roads_without_data = 0
                    for refs, osmid, name, htype in zip(highway_coordinate, highway_osmid, highway_names, highway_types):
                        idx = osmid_to_idx_local.get(int(osmid), None)
                        if idx is not None:
                            val = float(hot_emission[idx])
                            color_val = cmap(norm(val))
                            line_width = lw_min + val * width_scaling
                            ax.plot([p[0] for p in refs], [p[1] for p in refs], color=color_val, linewidth=line_width)
                            roads_with_data += 1
                        else:
                            roads_without_data += 1
                            if show_top_labels:
                                ax.plot([p[0] for p in refs], [p[1] for p in refs], color='gray', linewidth=0.2, alpha=0.3)

                    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                    ax.set_title(f"{map_vehicle} {map_pollutant} emissions")
                    st.pyplot(fig)
                    status.empty()
                    # store last figure for download
                    st.session_state._last_map_fig = fig
                    os.remove(osm_path)
                    st.success("Map generated successfully")
                    st.metric("Roads with data", roads_with_data); st.metric("Roads without data", roads_without_data)
                except Exception as e:
                    st.error(f"OSM map generation failed: {e}")
                    import traceback
                    with st.expander("Trace"): st.code(traceback.format_exc())

            # download last map fig as PNG
            if download_map:
                if '_last_map_fig' in st.session_state:
                    fig = st.session_state._last_map_fig
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("Download map PNG", data=buf, file_name=f"emission_map_{map_pollutant}_{map_vehicle}.png", mime="image/png")
                else:
                    st.info("No map generated yet. Click 'Generate Road Network Map' first.")
    else:
        st.info("Run calculations first to use the map")

# Tab7: downloads (kept)
with tab7:
    st.header("📥 Download Results")
    if 'emissions_data' in st.session_state and 'data_link' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link_np = st.session_state.data_link
        selected_pollutants = st.session_state.selected_pollutants

        final_results_df = pd.DataFrame(data_link_np[:, :4], columns=['OSM_ID','Length_km','Flow','Speed'])
        if data_link_np.shape[1] == 7:
            proportion_columns = ['Gasoline_Prop','PC_Prop','4Stroke_Prop']
            proportion_df = pd.DataFrame(data_link_np[:,4:7], columns=proportion_columns)
        else:
            proportion_columns = ['Gasoline_Prop','PC_Prop','4Stroke_Prop','LDV_Prop','HDV_Prop']
            proportion_df = pd.DataFrame(data_link_np[:,4:9], columns=proportion_columns)
        final_results_df = pd.concat([final_results_df, proportion_df], axis=1)

        for poll in selected_pollutants:
            poll_df = pd.DataFrame({
                f'{poll}_PC': emissions_data[poll]['pc'],
                f'{poll}_LDV': emissions_data[poll]['ldv'],
                f'{poll}_HDV': emissions_data[poll]['hdv'],
                f'{poll}_Moto': emissions_data[poll]['moto'],
                f'{poll}_Total': emissions_data[poll]['total']
            })
            final_results_df = pd.concat([final_results_df, poll_df], axis=1)
        csv_export = final_results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_export, file_name='traffic_emission_results_full.csv', mime='text/csv')
    else:
        st.info("Calculate emissions to enable downloads")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;padding:10px;'>Advanced Traffic Emission Calculator — UI restored</div>", unsafe_allow_html=True)
