import streamlit as st
import numpy as np
import matplotlib
import tempfile
import os
import zipfile
import requests
from io import BytesIO

# Visualization Imports
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Advanced Traffic Emission Calculator", layout="wide", initial_sidebar_state="expanded")

# ==================== CONFIGURATION ====================
# Update this URL if your repository or folder name is different
# I used "default" (standard spelling). Change to "defualt" if that is exactly how you named it.
REPO_USER = "Olusammi"
REPO_NAME = "traffic-emission-calculator"
REPO_BRANCH = "main"
DEFAULT_FOLDER = "default"  # Change to "defualt" if necessary
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{REPO_BRANCH}/{DEFAULT_FOLDER}/"

# Default Filenames map
DEFAULT_FILES = {
    "pc_param": "PC_parameter.csv",
    "ldv_param": "LDV_parameter.csv",
    "hdv_param": "HDV_parameter.csv",
    "moto_param": "Moto_parameter.csv",
    "link_osm": "link_osm_with-ldv.dat",
    "osm_file": "selected_zone-lagos.osm", # Assumed .osm extension, check your repo
    "engine_cap_gas": "engine_capacity_gasoline.dat",
    "engine_cap_diesel": "engine_capacity_diesel.dat",
    "copert_class_gas": "copert_class_proportion_gasoline.dat",
    "copert_class_diesel": "copert_class_proportion_diesel.dat",
    "copert_2stroke": "copert_class_proportion_2_stroke_motorcycle_more_50.dat",
    "copert_4stroke": "copert_class_proportion_4_stroke_motorcycle_50_250.dat"
}

# ==================== CUSTOM CSS ====================
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

st.title("Vehicle Emission Calculator")
st.caption("Multi-Standard Emission Analysis with COPERT IV, IPCC, and EPA Methodologies")
st.markdown("---")

# ==================== HELPER FUNCTIONS ====================

@st.cache_data(show_spinner=False)
def load_default_file_from_github(filename):
    """
    Fetches a file from GitHub and returns it as a BytesIO object.
    Cached so it doesn't re-download on every interaction.
    """
    url = GITHUB_BASE_URL + filename
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        return None

def get_file_input(label, file_key, file_type):
    """
    Standardizes the logic: Check User Upload -> Else Use Default -> Else None
    """
    # 1. Try User Upload
    uploaded_file = st.file_uploader(label, type=file_type, key=file_key)
    
    if uploaded_file is not None:
        return uploaded_file
    
    # 2. Try Default File
    default_filename = DEFAULT_FILES.get(file_key)
    if default_filename:
        # Load from cache
        default_content = load_default_file_from_github(default_filename)
        if default_content:
            # We add a small caption to let the user know default is being used
            st.caption(f"🔹 *Using default file from GitHub:* `{default_filename}`")
            return default_content
        else:
            st.caption(f"⚠️ Default file `{default_filename}` not found on GitHub.")
    
    return None

# ==================== SIDEBAR: EMISSION METRICS SELECTION ====================
st.sidebar.header("📊 Emission Metrics Selection")

# Multi-pollutant selection
pollutants_available = {
    "CO": {"name": "Carbon Monoxide", "unit": "g/km", "standard": "COPERT IV", "color": "#ef4444"},
    "CO2": {"name": "Carbon Dioxide", "unit": "g/km", "standard": "IPCC", "color": "#3b82f6"},
    "NOx": {"name": "Nitrogen Oxides", "unit": "g/km", "standard": "COPERT IV", "color": "#f59e0b"},
    "PM": {"name": "Particulate Matter", "unit": "mg/km", "standard": "WHO", "color": "#8b5cf6"},
    "VOC": {"name": "Volatile Organic Compounds", "unit": "g/km", "standard": "COPERT IV", "color": "#10b981"},
    "FC": {"name": "Fuel Consumption", "unit": "L/100km", "standard": "NEDC/WLTP", "color": "#f97316"}
}

selected_pollutants = st.sidebar.multiselect(
    "Select Pollutants to Calculate",
    options=list(pollutants_available.keys()),
    default=["CO", "NOx", "PM"],
    help="Choose one or more pollutants for emission calculation"
)

# Display info about selected pollutants
if selected_pollutants:
    st.sidebar.markdown("### Selected Metrics Info")
    for pollutant in selected_pollutants:
        info = pollutants_available[pollutant]
        st.sidebar.markdown(f"""
        <div style='background-color: {info['color']}22; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {info["color"]}'>
            <strong>{pollutant}</strong>: {info['name']}<br>
            <small>Standard: {info['standard']}</small><br>
            <small>Unit: {info['unit']}</small>
        </div>
        """, unsafe_allow_html=True)

# Calculation methodology
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Calculation Methodology")
calculation_method = st.sidebar.selectbox(
    "Select Calculation Standard",
    ["COPERT IV (EU)", "IPCC Tier 2", "EPA MOVES (US)", "Hybrid (Multi-standard)"],
    help="Choose the international standard for emission calculations"
)

# Accuracy settings
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

# ==================== UNIT CONVERSION SETTINGS ====================
st.sidebar.markdown("---")
st.sidebar.header("📏 Unit Conversion")

unit_conversion_options = {
    "CO": {"g/km": {"name": "g/km", "factor": 1.0}, "kg/km": {"name": "kg/km", "factor": 0.001}, "tonnes/km": {"name": "t/km", "factor": 1e-6}},
    "CO2": {"g/km": {"name": "g/km", "factor": 1.0}, "kg/km": {"name": "kg/km", "factor": 0.001}, "tonnes/km": {"name": "t/km", "factor": 1e-6}},
    "NOx": {"g/km": {"name": "g/km", "factor": 1.0}, "kg/km": {"name": "kg/km", "factor": 0.001}, "mg/km": {"name": "mg/km", "factor": 1000.0}},
    "PM": {"mg/km": {"name": "mg/km", "factor": 1.0}, "g/km": {"name": "g/km", "factor": 0.001}, "kg/km": {"name": "kg/km", "factor": 1e-6}},
    "VOC": {"g/km": {"name": "g/km", "factor": 1.0}, "kg/km": {"name": "kg/km", "factor": 0.001}},
    "FC": {"L/100km": {"name": "L/100km", "factor": 1.0}, "L/km": {"name": "L/km", "factor": 0.01}}
}

selected_units = {}
for poll in pollutants_available.keys():
    if poll in unit_conversion_options:
        selected_unit = st.sidebar.selectbox(f"{poll} Unit", list(unit_conversion_options[poll].keys()), key=f"unit_{poll}")
        selected_units[poll] = selected_unit
    else:
        selected_units[poll] = pollutants_available[poll]['unit']

st.session_state.selected_units = selected_units
st.session_state.unit_conversion_options = unit_conversion_options

# ==================== FILE UPLOADS (WITH DEFAULT FALLBACK) ====================
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Input Files")

# COPERT Files
copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=True)
with copert_files:
    pc_param = get_file_input("PC Parameter CSV", 'pc_param', ['csv'])
    ldv_param = get_file_input("LDV Parameter CSV", 'ldv_param', ['csv'])
    hdv_param = get_file_input("HDV Parameter CSV", 'hdv_param', ['csv'])
    moto_param = get_file_input("Moto Parameter CSV", 'moto_param', ['csv'])

# Data Files
data_files = st.sidebar.expander("Data Files", expanded=True)
with data_files:
    link_osm = get_file_input("Link OSM Data (.dat/.csv)", 'link_osm', ['dat', 'csv', 'txt'])
    osm_file = get_file_input("OSM Network File (.osm)", 'osm_file', ['osm'])

# Proportion Files
proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
with proportion_files:
    engine_cap_gas = get_file_input("Engine Cap Gasoline", 'engine_cap_gas', ['dat', 'txt'])
    engine_cap_diesel = get_file_input("Engine Cap Diesel", 'engine_cap_diesel', ['dat', 'txt'])
    copert_class_gas = get_file_input("Class Gasoline", 'copert_class_gas', ['dat', 'txt'])
    copert_class_diesel = get_file_input("Class Diesel", 'copert_class_diesel', ['dat', 'txt'])
    copert_2stroke = get_file_input("2-Stroke Moto", 'copert_2stroke', ['dat', 'txt'])
    copert_4stroke = get_file_input("4-Stroke Moto", 'copert_4stroke', ['dat', 'txt'])

st.sidebar.markdown("---")

# Map parameters
st.sidebar.header("🗺️ Map Parameters")
col1, col2 = st.sidebar.columns(2)
x_min = col1.number_input("X Min (Lon)", value=3.37310, format="%.5f")
x_max = col2.number_input("X Max (Lon)", value=3.42430, format="%.5f")
y_min = col1.number_input("Y Min (Lat)", value=6.43744, format="%.5f")
y_max = col2.number_input("Y Max (Lat)", value=6.46934, format="%.5f")
tolerance = st.sidebar.number_input("Tolerance", value=0.005, format="%.3f")
ncore = st.sidebar.number_input("Number of Cores", value=8, min_value=1, max_value=16)

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📖 Instructions",
    "📊 Data Preview",
    "🧮 Formula Explanation",
    "⚙️ Calculate Emissions",
    "📈 Multi-Metric Analysis",
    "🗺️ Interactive Map",
    "📥 Download Results"
])

# Utility Functions
def convert_emission_value(value, pollutant, from_unit, to_unit, distance_km=None):
    if from_unit == to_unit: return value
    if pollutant not in unit_conversion_options: return value
    conversions = unit_conversion_options[pollutant]
    if to_unit not in conversions: return value
    if to_unit in ["mpg", "km/L"]:
        if value == 0: return 0
        if to_unit == "mpg": return 235.214 / value
        elif to_unit == "km/L": return 100 / value
    if "year" in to_unit:
        annual_distance = distance_km if distance_km else 15000
        base_value = value * conversions[to_unit]["factor"]
        return base_value * annual_distance
    return value * conversions[to_unit]["factor"]

def format_emission_value(value, unit):
    if value == 0: return "0.00"
    elif value < 0.001: return f"{value:.6f}"
    elif value < 0.1: return f"{value:.4f}"
    elif value < 10: return f"{value:.3f}"
    elif value < 1000: return f"{value:.2f}"
    else: return f"{value:.1f}"

# ==================== TAB 1-3 (Instructions, Preview, Formulas) ====================
with tab1:
    st.header("📖 User Guide")
    st.info("The app now automatically loads default files from GitHub if no files are uploaded.")
    st.markdown("""
    **Using Defaults:**
    If you see "Using default file from GitHub" in the sidebar, the app is using the standard dataset provided in your repository.
    To use your own data, simply upload a file in the corresponding slot, and it will override the default.
    """)

with tab2:
    st.header("📊 Data Preview")
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
            st.dataframe(data_link.head(20), use_container_width=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Links", len(data_link))
            with col2: st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
            with col3: st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
            with col4: st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")
        except Exception as e:
            st.error(f"❌ Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file (or ensure default is loaded)")

with tab3:
    st.header("🧮 Formula Explanation")
    st.info("Mathematics behind COPERT IV calculations.")

# ==================== TAB 4: CALCULATE EMISSIONS ====================
with tab4:
    st.header("⚙️ Calculate Emissions")
    
    # Check if all files are present (either uploaded or default)
    required_files_objs = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    all_present = all(f is not None for f in required_files_objs)

    if not selected_pollutants:
        st.warning("⚠️ Please select at least one pollutant from the sidebar")
    elif all_present:
        st.success("✅ All required files ready (Using Uploads or Defaults)")
        
        if st.button("🚀 Calculate Multi-Pollutant Emissions", type="primary", use_container_width=True):
            with st.spinner("Computing emissions..."):
                try:
                    import copert
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save temp files for COPERT class which requires paths
                        paths = {}
                        for name, obj in [('pc', pc_param), ('ldv', ldv_param), ('hdv', hdv_param), ('moto', moto_param)]:
                            p = os.path.join(tmpdir, f"{name}.csv")
                            with open(p, 'wb') as f: f.write(obj.getbuffer())
                            paths[name] = p

                        cop = copert.Copert(paths['pc'], paths['ldv'], paths['hdv'], paths['moto'])

                        # Reset file pointers
                        for f in [link_osm, engine_cap_gas, engine_cap_diesel, copert_class_gas, copert_class_diesel, copert_2stroke, copert_4stroke]:
                            f.seek(0)

                        # Load Data
                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values
                        Nlink = data_link.shape[0]

                        # Load Distributions
                        d_ec_gas = np.loadtxt(engine_cap_gas)
                        d_ec_diesel = np.loadtxt(engine_cap_diesel)
                        d_cc_gas = np.loadtxt(copert_class_gas)
                        d_cc_diesel = np.loadtxt(copert_class_diesel)
                        d_c_2s = np.loadtxt(copert_2stroke)
                        d_c_4s = np.loadtxt(copert_4stroke)

                        # LDV/HDV Defaults
                        if data_link.shape[1] == 7:
                            P_ldv, P_hdv = np.zeros(Nlink), np.zeros(Nlink)
                        else:
                            P_ldv, P_hdv = data_link[:, 7], data_link[:, 8]

                        d_cc_ldv = d_cc_gas
                        N_HDV_Class, N_HDV_Type = 6, 15
                        d_hdv = np.zeros((Nlink, N_HDV_Class, N_HDV_Type))
                        d_hdv[:, 5, 0] = 1.0

                        # Mappings
                        eng = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        eng_m = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        cap = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        cls_pc = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03,
                                  cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop, cop.class_Euro_1,
                                  cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                        cls_moto = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2, 
                                    cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        cls_hdv = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III,
                                   cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
                        
                        poll_map = {"CO": cop.pollutant_CO, "CO2": cop.pollutant_FC, "NOx": cop.pollutant_NOx,
                                    "PM": cop.pollutant_PM, "VOC": cop.pollutant_VOC, "FC": cop.pollutant_FC}

                        # Results Init
                        emissions_data = {}
                        fuel_emissions_data = {}
                        for p in selected_pollutants:
                            emissions_data[p] = {'pc': np.zeros(Nlink), 'ldv': np.zeros(Nlink), 'hdv': np.zeros(Nlink),
                                                 'moto': np.zeros(Nlink), 'total': np.zeros(Nlink)}
                            fuel_emissions_data[p] = {'gasoline': np.zeros(Nlink), 'diesel': np.zeros(Nlink)}

                        prog = st.progress(0)
                        
                        # Calculation Loop
                        for i in range(Nlink):
                            if i % max(1, Nlink//100) == 0: prog.progress(i/Nlink)
                            
                            L, flow, v = data_link[i, 1], data_link[i, 2], min(max(10., data_link[i, 3]), 130.)
                            prop_gas, prop_pc, prop_4s = data_link[i, 4], data_link[i, 5], data_link[i, 6]
                            
                            p_moto = 1. - prop_pc
                            p_ldv_i, p_hdv_i = float(P_ldv[i]), float(P_hdv[i])
                            
                            d_eng = [prop_gas, 1.-prop_gas]
                            d_cap = [d_ec_gas[i], d_ec_diesel[i]]
                            d_moto = [prop_4s, 1.-prop_4s]
                            
                            for p_name in selected_pollutants:
                                p_type = poll_map[p_name]
                                
                                # PC
                                for t in range(2):
                                    for c in range(len(cls_pc)):
                                        for k in range(2):
                                            if t==1 and k==0 and cls_pc[c] in range(cop.class_Euro_1, cop.class_Euro_3+1): continue
                                            try:
                                                e = cop.Emission(p_type, v, L, cop.vehicle_type_passenger_car, eng[t], cls_pc[c], cap[k], ambient_temp)
                                            except: e = 0.0
                                            
                                            if p_name=="NOx" and include_temperature_correction: e *= (1 + 0.02*(ambient_temp-20))
                                            
                                            share = d_cc_gas[i, c] if t==0 else d_cc_diesel[i, c]
                                            val = e * d_eng[t] * d_cap[t][k] * share * prop_pc / L * flow
                                            
                                            emissions_data[p_name]['pc'][i] += val
                                            fuel_emissions_data[p_name]['gasoline' if t==0 else 'diesel'][i] += val

                                # LDV
                                if p_ldv_i > 0:
                                    for t in range(2):
                                        for c in range(len(cls_pc)):
                                            for k in range(2):
                                                if t==1 and k==0 and cls_pc[c] in range(cop.class_Euro_1, cop.class_Euro_3+1): continue
                                                try:
                                                    e = cop.Emission(p_type, v, L, cop.vehicle_type_light_commercial_vehicle, eng[t], cls_pc[c], cap[k], ambient_temp)
                                                except: e = 0.0
                                                if p_name=="NOx" and include_temperature_correction: e *= (1 + 0.02*(ambient_temp-20))
                                                val = e * d_eng[t] * d_cap[t][k] * d_cc_ldv[i, c] * p_ldv_i / L * flow
                                                emissions_data[p_name]['ldv'][i] += val
                                                fuel_emissions_data[p_name]['gasoline' if t==0 else 'diesel'][i] += val

                                # HDV
                                if p_hdv_i > 0:
                                    for tc in range(N_HDV_Class):
                                        for tt in range(N_HDV_Type):
                                            share = d_hdv[i, tc, tt]
                                            if share <= 0: continue
                                            try:
                                                e = cop.Emission(p_type, v, L, cop.vehicle_type_heavy_duty_vehicle, cop.engine_type_diesel, cls_hdv[tc], tt, ambient_temp)
                                            except: e = 0.0
                                            if p_name=="NOx" and include_temperature_correction: e *= (1 + 0.015*(ambient_temp-20))
                                            val = e * share * p_hdv_i / L * flow
                                            emissions_data[p_name]['hdv'][i] += val
                                            fuel_emissions_data[p_name]['diesel'][i] += val

                                # Moto
                                for m in range(2):
                                    for mc in range(len(cls_moto)):
                                        if m==0 and cls_moto[mc] >= cop.class_moto_Euro_1: continue
                                        try: e = cop.EFMotorcycle(p_type, v, eng_m[m], cls_moto[mc])
                                        except: e = 0.0
                                        val = e * d_moto[m] * p_moto * flow
                                        emissions_data[p_name]['moto'][i] += val
                                        fuel_emissions_data[p_name]['gasoline'][i] += val

                                emissions_data[p_name]['total'][i] = (emissions_data[p_name]['pc'][i] + emissions_data[p_name]['ldv'][i] +
                                                                      emissions_data[p_name]['hdv'][i] + emissions_data[p_name]['moto'][i])

                        prog.empty()
                        st.session_state.emissions_data = emissions_data
                        st.session_state.fuel_emissions_data = fuel_emissions_data
                        st.session_state.data_link = data_link
                        st.success("✅ Emissions Calculated Successfully!")
                
                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("Missing required files (Upload or Defaults).")

# ==================== TAB 5: MULTI-METRIC ANALYSIS ====================
with tab5:
    st.header("📈 Multi-Metric Analysis")

    if 'emissions_data' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link = st.session_state.data_link
        
        col1, col2 = st.columns([3, 1])
        with col1: st.markdown("Select analysis views")
        with col2: use_converted = st.checkbox("Use Converted Units", value=True)
        
        if selected_pollutants:
            # FUEL ANALYSIS
            st.subheader("⛽ Emissions by Fuel Type")
            c1, c2 = st.columns([2, 1])
            with c1: 
                fuel_poll = st.selectbox("Pollutant (Fuel)", selected_pollutants, key='fp')
            with c2: 
                fuel_chart = st.selectbox("Chart Type", ["Pie Chart", "Bar Chart"], key='fc')
            
            # Helper to aggregate
            gas_prop = data_link[:, 4].mean()
            diesel_prop = 1 - gas_prop
            
            pc_t = emissions_data[fuel_poll]['pc'].sum()
            ldv_t = emissions_data[fuel_poll]['ldv'].sum()
            hdv_t = emissions_data[fuel_poll]['hdv'].sum()
            moto_t = emissions_data[fuel_poll]['moto'].sum()
            
            gas_sum = pc_t * gas_prop + ldv_t * gas_prop + moto_t
            dsl_sum = pc_t * diesel_prop + ldv_t * diesel_prop + hdv_t
            
            fuel_df = pd.DataFrame([
                {'Fuel': 'Gasoline', 'Value': gas_sum},
                {'Fuel': 'Diesel', 'Value': dsl_sum}
            ])
            
            if fuel_chart == "Pie Chart":
                fig = px.pie(fuel_df, values='Value', names='Fuel', title=f"{fuel_poll} by Fuel", 
                             color='Fuel', color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'})
            else:
                fig = px.bar(fuel_df, x='Fuel', y='Value', color='Fuel', title=f"{fuel_poll} by Fuel",
                             color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # VEHICLE ANALYSIS
            st.subheader("🚗 Emissions by Vehicle Type")
            c1, c2 = st.columns([2, 1])
            with c1: 
                veh_poll = st.selectbox("Pollutant (Vehicle)", selected_pollutants, key='vp')
            with c2: 
                veh_chart = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"], key='vc')
                
            veh_data = []
            for v in ['pc', 'ldv', 'hdv', 'moto']:
                veh_data.append({'Type': v.upper(), 'Value': emissions_data[veh_poll][v].sum()})
            veh_df = pd.DataFrame(veh_data)
            
            color_map = {'PC': '#667eea', 'LDV': '#f59e0b', 'HDV': '#ef4444', 'MOTO': '#10b981'}
            if veh_chart == "Bar Chart":
                fig = px.bar(veh_df, x='Type', y='Value', color='Type', title=f"{veh_poll} by Vehicle", color_discrete_map=color_map)
            else:
                fig = px.pie(veh_df, values='Value', names='Type', title=f"{veh_poll} by Vehicle", color='Type', color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Select pollutants to view analysis.")
    else:
        st.info("Calculate emissions first.")

# ==================== TAB 6: INTERACTIVE MAP (FIXED) ====================
with tab6:
    st.header("🗺️ Interactive Map")
    
    # Check if calculation results AND OSM file exist (either uploaded or default)
    if 'emissions_data' in st.session_state and osm_file is not None:
        
        # --- GEO DATA PREPARATION (Auto-parse logic) ---
        if 'geo_data' not in st.session_state:
            with st.spinner("Parsing OSM network for map visualization... (This runs once)"):
                try:
                    import osm_network
                    
                    # Prepare OSM file (Handles both UploadedFile and BytesIO)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                        osm_file.seek(0)
                        tmp.write(osm_file.read())
                        osm_path = tmp.name
                    
                    zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    coords, ids, names, types = osm_network.retrieve_highway(osm_path, zone, tolerance, int(ncore))
                    
                    st.session_state.geo_data = {'coords': coords, 'ids': ids, 'names': names}
                    os.unlink(osm_path)
                except Exception as e:
                    st.error(f"Error parsing OSM file: {e}")
                    st.stop()
        
        # --- MAP CONTROLS ---
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            map_poll = st.selectbox("Pollutant Layer", selected_pollutants)
            map_style = st.selectbox("Base Map", ["carto-positron", "carto-darkmatter", "open-street-map"])
        with c2:
            color_theme = st.selectbox("Color Theme", ["Reds", "Jet", "Viridis", "Plasma", "Inferno"])
            line_scale = st.slider("Line Width Scale", 1.0, 5.0, 2.5)
        with c3:
            st.markdown("**Filters**")
            f_speed = st.slider("Filter Speed (km/h)", 0, 130, (0, 130))
        
        # --- MAP RENDERING ---
        try:
            emis = st.session_state.emissions_data[map_poll]['total']
            link_data = st.session_state.data_link
            geo = st.session_state.geo_data
            
            id_to_idx = {int(row[0]): i for i, row in enumerate(link_data)}
            
            map_rows = []
            for coords, oid, name in zip(geo['coords'], geo['ids'], geo['names']):
                if oid in id_to_idx:
                    idx = id_to_idx[oid]
                    val = emis[idx]
                    sp = link_data[idx, 3]
                    if f_speed[0] <= sp <= f_speed[1]:
                        map_rows.append({'oid': oid, 'val': val, 'coords': coords, 'name': name, 'speed': sp})
            
            map_df = pd.DataFrame(map_rows)
            
            if not map_df.empty:
                # Quantile binning for coloring
                try:
                    map_df['quartile'] = pd.qcut(map_df['val'], 4, labels=["Low", "Medium", "High", "Critical"], duplicates='drop')
                except:
                    map_df['quartile'] = pd.cut(map_df['val'], 4, labels=["Low", "Medium", "High", "Critical"])
                
                # Theme Mapping
                themes = {"Jet": px.colors.sequential.Jet, "Viridis": px.colors.sequential.Viridis, 
                          "Reds": px.colors.sequential.Reds, "Inferno": px.colors.sequential.Inferno, 
                          "Plasma": px.colors.sequential.Plasma}
                colors_scale = themes.get(color_theme, px.colors.sequential.Reds)
                
                qs = map_df['quartile'].unique()
                if hasattr(qs, 'sort_values'): qs = qs.sort_values()
                
                fig = go.Figure()
                
                # Draw lines by group
                for i, q in enumerate(qs):
                    subset = map_df[map_df['quartile'] == q]
                    if subset.empty: continue
                    
                    c_lats, c_lons = [], []
                    for cs in subset['coords']:
                        unzipped = list(zip(*cs))
                        c_lons.extend(unzipped[0] + (None,))
                        c_lats.extend(unzipped[1] + (None,))
                    
                    c_idx = int(i / (len(qs)-1 or 1) * (len(colors_scale)-1))
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=c_lats, lon=c_lons,
                        mode='lines',
                        line=dict(width=line_scale, color=colors_scale[c_idx]),
                        name=f"{q} Emission",
                        hoverinfo='skip'
                    ))
                
                # Interactive Hover Points
                mid_lats = [c[len(c)//2][1] for c in map_df['coords']]
                mid_lons = [c[len(c)//2][0] for c in map_df['coords']]
                hover_txt = [f"<b>{r['name']}</b><br>ID: {r['oid']}<br>E: {r['val']:.2f}<br>V: {r['speed']}" for _, r in map_df.iterrows()]
                
                fig.add_trace(go.Scattermapbox(
                    lat=mid_lats, lon=mid_lons,
                    mode='markers',
                    marker=dict(size=5, opacity=0),
                    text=hover_txt,
                    hoverinfo='text',
                    name='Info'
                ))

                fig.update_layout(
                    mapbox_style=map_style,
                    mapbox_center={"lat": (y_min+y_max)/2, "lon": (x_min+x_max)/2},
                    mapbox_zoom=12,
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=600,
                    legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found for selected filters.")
        except Exception as e:
            st.error(f"Error rendering map: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("⚠️ Please Calculate Emissions first (Tab 4).")
        if osm_file is None:
            st.warning("OSM Network File is also missing (check Upload or Default).")

# ==================== TAB 7: DOWNLOAD ====================
with tab7:
    st.header("📥 Download Results")
    
    if 'emissions_data' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link = st.session_state.data_link
        
        st.subheader("Export Options")
        convert = st.checkbox("Export Converted Units", value=False)
        
        # Prepare DataFrame
        df_export = pd.DataFrame(data_link[:, :4], columns=['OSM_ID', 'Length', 'Flow', 'Speed'])
        
        for p in selected_pollutants:
            p_unit = unit_conversion_options[p][selected_units[p]]['name'] if convert else pollutants_available[p]['unit']
            # Get values (converted if requested)
            vals = emissions_data[p]['total']
            if convert:
                 orig = pollutants_available[p]['unit']
                 target = selected_units[p]
                 vals = [convert_emission_value(v, p, orig, target) for v in vals]
            
            df_export[f"{p}_Total ({p_unit})"] = vals
            
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV Results", csv, "emission_results.csv", "text/csv")
        
        if st.button("📦 Generate Full ZIP Package"):
             with BytesIO() as buf:
                 with zipfile.ZipFile(buf, 'w') as z:
                     z.writestr("emissions.csv", df_export.to_csv(index=False))
                     z.writestr("summary.txt", f"Report Generated: {pd.Timestamp.now()}\nTotal Links: {len(df_export)}")
                 st.download_button("⬇️ Download ZIP", buf.getvalue(), "emissions_pkg.zip", "application/zip")
    else:
        st.info("Calculate emissions first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Standards: EEA Guidebook 2019, IPCC 2019 Guidelines, WHO Air Quality Standards</p>
    <p>© 2025 - Developed by SHassan</p>
</div>
""", unsafe_allow_html=True)
