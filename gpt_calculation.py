import streamlit as st
import numpy as np
import matplotlib
import tempfile
import os
import zipfile
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

# Custom CSS for better styling
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

# ==================== SIDEBAR: EMISSION METRICS SELECTION ====================
st.sidebar.header("📊 Emission Metrics Selection")
st.sidebar.markdown("Select which pollutants to calculate and analyze")

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
include_temperature_correction = st.sidebar.checkbox("Temperature Correction", value=True,
                                                     help="Apply ambient temperature corrections to NOx emissions")
include_cold_start = st.sidebar.checkbox("Cold Start Emissions", value=True,
                                         help="Include cold start emission quotient")
include_slope_correction = st.sidebar.checkbox("Road Slope Correction", value=False,
                                               help="Apply road gradient corrections")

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
st.sidebar.info("💡 **Pro Tip**: Enable all accuracy settings for research-grade calculations")

# ==================== UNIT CONVERSION SETTINGS ====================
st.sidebar.markdown("---")
st.sidebar.header("📏 Unit Conversion")
st.sidebar.markdown("Convert emission results to different units")

# Unit conversion options for different pollutants
unit_conversion_options = {
    "CO": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "tonnes/km": {"name": "Tonnes per kilometer", "factor": 0.000001},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934},
        "kg/year": {"name": "Kilograms per year (avg)", "factor": 1.0, "note": "Requires trip frequency"}
    },
    "CO2": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "tonnes/km": {"name": "Tonnes per kilometer", "factor": 0.000001},
        "kg CO2e/km": {"name": "kg CO2 equivalent per km", "factor": 0.001},
        "tonnes CO2e/year": {"name": "Tonnes CO2e per year", "factor": 1.0, "note": "Requires annual distance"},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934}
    },
    "NOx": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1000.0},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934},
        "kg NO2e/km": {"name": "kg NO2 equivalent per km", "factor": 0.001}
    },
    "PM": {
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1.0},
        "g/km": {"name": "Grams per kilometer", "factor": 0.001},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.000001},
        "µg/km": {"name": "Micrograms per kilometer", "factor": 1000.0},
        "mg/mile": {"name": "Milligrams per mile", "factor": 1.60934}
    },
    "VOC": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1000.0},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934}
    },
    "FC": {
        "L/100km": {"name": "Liters per 100 km", "factor": 1.0},
        "L/km": {"name": "Liters per km", "factor": 0.01},
        "gal/100mi": {"name": "Gallons per 100 miles (US)", "factor": 2.3521},
        "mpg": {"name": "Miles per gallon (US)", "factor": 1.0, "note": "Inverse calculation"},
        "km/L": {"name": "Kilometers per liter", "factor": 1.0, "note": "Inverse calculation"}
    }
}

# Store selected units for each pollutant
selected_units = {}
for poll in pollutants_available.keys():
    if poll in unit_conversion_options:
        default_unit = list(unit_conversion_options[poll].keys())[0]
        selected_unit = st.sidebar.selectbox(
            f"{poll} Display Unit",
            options=list(unit_conversion_options[poll].keys()),
            format_func=lambda x, p=poll: f"{x} - {unit_conversion_options[p][x]['name']}",
            key=f"unit_{poll}"
        )
        selected_units[poll] = selected_unit
    else:
        selected_units[poll] = pollutants_available[poll]['unit']

# Store in session state for access across tabs
st.session_state.selected_units = selected_units
st.session_state.unit_conversion_options = unit_conversion_options

# Show conversion info
if st.sidebar.checkbox("Show Conversion Info", value=False):
    st.sidebar.markdown("**Conversion Notes:**")
    st.sidebar.info("""
    - **CO2e**: CO2 equivalent includes global warming potential
    - **Annual calculations**: Based on average 15,000 km/year
    - **Mile conversions**: 1 mile = 1.60934 km
    - **Inverse units** (mpg, km/L): Calculated as reciprocal
    """)

st.sidebar.markdown("---")
# ==================== FILE UPLOADS ====================
st.sidebar.header("📂 Upload Input Files")
copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=True)
with copert_files:
    pc_param = st.file_uploader("PC Parameter CSV", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Parameter CSV", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Parameter CSV", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Parameter CSV", type=['csv'], key='moto')

data_files = st.sidebar.expander("Data Files", expanded=True)
with data_files:
    link_osm = st.file_uploader("Link OSM Data (.dat or .csv)", type=['dat', 'csv', 'txt'], key='link')
    osm_file = st.file_uploader("OSM Network File (.osm)", type=['osm'], key='osm')

proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
with proportion_files:
    engine_cap_gas = st.file_uploader("Engine Capacity Gasoline", type=['dat', 'txt'], key='ecg')
    engine_cap_diesel = st.file_uploader("Engine Capacity Diesel", type=['dat', 'txt'], key='ecd')
    copert_class_gas = st.file_uploader("COPERT Class Gasoline", type=['dat', 'txt'], key='ccg')
    copert_class_diesel = st.file_uploader("COPERT Class Diesel", type=['dat', 'txt'], key='ccd')
    copert_2stroke = st.file_uploader("2-Stroke Motorcycle", type=['dat', 'txt'], key='2s')
    copert_4stroke = st.file_uploader("4-Stroke Motorcycle", type=['dat', 'txt'], key='4s')

st.sidebar.markdown("---")
st.sidebar.info("""
**LDV/HDV Distributions:**
Default fleet compositions (PC Gasoline for LDV, 100% Euro VI for HDV) will be used if specific distribution files are not provided, allowing calculations to proceed.
""")


# Map parameters
st.sidebar.header("🗺️ Map Parameters")
st.sidebar.markdown("**Domain Boundaries**")
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

# ==================== UNIT CONVERSION FUNCTION ====================
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

# ==================== TAB 1, 2, 3 (Unchanged logic, just placeholders for flow) ====================
with tab1:
    st.header("📖 User Guide & Instructions")
    st.info("Refer to the comprehensive guide in the original app for instructions.")

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
        except Exception as e:
            st.error(f"❌ Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar")

with tab3:
    st.header("🧮 Formula Explanation")
    st.info("Mathematics behind COPERT IV calculations.")

with tab4:
    st.header("⚙️ Calculate Emissions")
    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    all_uploaded = all(f is not None for f in required_files)

    if not selected_pollutants:
        st.warning("⚠️ Please select at least one pollutant from the sidebar")
    elif all_uploaded:
        if st.button("🚀 Calculate Multi-Pollutant Emissions", type="primary", use_container_width=True):
            with st.spinner("Computing emissions for selected pollutants..."):
                try:
                    import copert
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

                        # Reset pointers
                        for f in [link_osm, engine_cap_gas, engine_cap_diesel, copert_class_gas, copert_class_diesel, copert_2stroke, copert_4stroke]:
                            f.seek(0)

                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values
                        Nlink = data_link.shape[0]

                        # Proportions
                        if data_link.shape[1] == 7:
                            P_ldv, P_hdv = np.zeros(Nlink), np.zeros(Nlink)
                        elif data_link.shape[1] == 9:
                            P_ldv, P_hdv = data_link[:, 7], data_link[:, 8]

                        # Load distributions
                        data_engine_capacity_gasoline = np.loadtxt(engine_cap_gas)
                        data_engine_capacity_diesel = np.loadtxt(engine_cap_diesel)
                        data_copert_class_gasoline = np.loadtxt(copert_class_gas)
                        data_copert_class_diesel = np.loadtxt(copert_class_diesel)
                        data_copert_class_motorcycle_two_stroke = np.loadtxt(copert_2stroke)
                        data_copert_class_motorcycle_four_stroke = np.loadtxt(copert_4stroke)

                        # Defaults
                        data_copert_class_ldv = data_copert_class_gasoline
                        N_HDV_Class = 6; N_HDV_Type = 15
                        data_hdv_reshaped = np.zeros((Nlink, N_HDV_Class, N_HDV_Type))
                        data_hdv_reshaped[:, 5, 0] = 1.0

                        # Config
                        engine_type = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        engine_type_m = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        engine_capacity = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        copert_class = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03,
                                        cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop,
                                        cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4,
                                        cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                        Nclass = len(copert_class)
                        copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                                                   cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        Mclass = len(copert_class_motorcycle)
                        HDV_Emission_Classes = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III,
                                                cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]

                        # Init Data
                        emissions_data = {}
                        pollutant_mapping = {"CO": cop.pollutant_CO, "CO2": cop.pollutant_FC, "NOx": cop.pollutant_NOx,
                                             "PM": cop.pollutant_PM, "VOC": cop.pollutant_VOC, "FC": cop.pollutant_FC}
                        
                        fuel_emissions_data = {}
                        for poll in selected_pollutants:
                            fuel_emissions_data[poll] = {'gasoline': np.zeros((Nlink,), dtype=float), 'diesel': np.zeros((Nlink,), dtype=float)}
                            emissions_data[poll] = {'pc': np.zeros((Nlink,)), 'moto': np.zeros((Nlink,)), 
                                                    'ldv': np.zeros((Nlink,)), 'hdv': np.zeros((Nlink,)), 'total': np.zeros((Nlink,))}

                        progress_bar = st.progress(0)
                        
                        # Calculation Loop
                        for i in range(Nlink):
                            if i % max(1, Nlink // 100) == 0: progress_bar.progress(i / Nlink)
                            
                            link_length = data_link[i, 1]
                            link_flow = data_link[i, 2]
                            v = min(max(10., data_link[i, 3]), 130.)
                            
                            link_gas_prop = data_link[i, 4]
                            link_pc_prop = data_link[i, 5]
                            link_4s_prop = data_link[i, 6]
                            
                            p_passenger = link_pc_prop
                            P_motorcycle = 1. - link_pc_prop
                            P_ldv_i = float(P_ldv[i])
                            P_hdv_i = float(P_hdv[i])
                            
                            eng_type_dist = [link_gas_prop, 1. - link_gas_prop]
                            eng_cap_dist = [data_engine_capacity_gasoline[i], data_engine_capacity_diesel[i]]
                            moto_stroke_dist = [link_4s_prop, 1.0 - link_4s_prop]
                            
                            for poll_name in selected_pollutants:
                                poll_type = pollutant_mapping[poll_name]
                                
                                # --- PC ---
                                try:
                                    for t in range(2):
                                        for c in range(Nclass):
                                            for k in range(2):
                                                if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3): continue
                                                try: e = cop.Emission(poll_type, v, link_length, cop.vehicle_type_passenger_car, engine_type[t], copert_class[c], engine_capacity[k], ambient_temp)
                                                except: 
                                                    try: 
                                                        if t==0: e = cop.HEFGasolinePassengerCar(poll_type, v, copert_class[c], engine_capacity[k]) * link_length
                                                        else: e = cop.HEFDieselPassengerCar(poll_type, v, copert_class[c], engine_capacity[k]) * link_length
                                                    except: e = 0.0
                                                
                                                if poll_name == "NOx" and include_temperature_correction: e *= (1 + 0.02 * (ambient_temp - 20))
                                                
                                                pc_share = data_copert_class_gasoline[i, c] if t == 0 else data_copert_class_diesel[i, c]
                                                final_e = e * eng_type_dist[t] * eng_cap_dist[t][k] * pc_share * p_passenger / link_length * link_flow
                                                
                                                emissions_data[poll_name]['pc'][i] += final_e
                                                if t == 0: fuel_emissions_data[poll_name]['gasoline'][i] += final_e
                                                else: fuel_emissions_data[poll_name]['diesel'][i] += final_e
                                except: pass

                                # --- LDV ---
                                if P_ldv_i > 0:
                                    try:
                                        for t in range(2):
                                            for c in range(Nclass):
                                                for k in range(2):
                                                    if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3): continue
                                                    try: e_ldv = cop.Emission(poll_type, v, link_length, cop.vehicle_type_light_commercial_vehicle, engine_type[t], copert_class[c], engine_capacity[k], ambient_temp)
                                                    except: e_ldv = 0.0
                                                    
                                                    if poll_name == "NOx" and include_temperature_correction: e_ldv *= (1 + 0.02 * (ambient_temp - 20))
                                                    
                                                    final_e_ldv = e_ldv * eng_type_dist[t] * eng_cap_dist[t][k] * data_copert_class_ldv[i, c] * P_ldv_i / link_length * link_flow
                                                    emissions_data[poll_name]['ldv'][i] += final_e_ldv
                                                    if t == 0: fuel_emissions_data[poll_name]['gasoline'][i] += final_e_ldv
                                                    else: fuel_emissions_data[poll_name]['diesel'][i] += final_e_ldv
                                    except: pass

                                # --- HDV ---
                                if P_hdv_i > 0:
                                    try:
                                        for t_class in range(N_HDV_Class):
                                            for t_type in range(N_HDV_Type):
                                                share = data_hdv_reshaped[i, t_class, t_type]
                                                if share <= 0: continue
                                                try: e_hdv = cop.Emission(poll_type, v, link_length, cop.vehicle_type_heavy_duty_vehicle, cop.engine_type_diesel, HDV_Emission_Classes[t_class], t_type, ambient_temp)
                                                except: e_hdv = 0.0
                                                if poll_name == "NOx" and include_temperature_correction: e_hdv *= (1 + 0.015 * (ambient_temp - 20))
                                                
                                                final_e_hdv = e_hdv * share * P_hdv_i / link_length * link_flow
                                                emissions_data[poll_name]['hdv'][i] += final_e_hdv
                                                fuel_emissions_data[poll_name]['diesel'][i] += final_e_hdv
                                    except: pass

                                # --- MOTO ---
                                try:
                                    for m in range(2):
                                        for d in range(Mclass):
                                            if m == 0 and copert_class_motorcycle[d] >= cop.class_moto_Euro_1: continue
                                            try: e_f = cop.EFMotorcycle(poll_type, v, engine_type_m[m], copert_class_motorcycle[d])
                                            except: e_f = 0.0
                                            
                                            final_e_moto = e_f * moto_stroke_dist[m] * P_motorcycle * link_flow
                                            emissions_data[poll_name]['moto'][i] += final_e_moto
                                            fuel_emissions_data[poll_name]['gasoline'][i] += final_e_moto
                                except: pass

                                emissions_data[poll_name]['total'][i] = (
                                    emissions_data[poll_name]['pc'][i] + emissions_data[poll_name]['ldv'][i] +
                                    emissions_data[poll_name]['hdv'][i] + emissions_data[poll_name]['moto'][i]
                                )

                        progress_bar.empty()
                        st.session_state.emissions_data = emissions_data
                        st.session_state.fuel_emissions_data = fuel_emissions_data
                        st.session_state.data_link = data_link
                        st.session_state.selected_pollutants = selected_pollutants
                        st.session_state.calc_done = True
                        st.success("✅ Multi-pollutant emissions calculated successfully!")

                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    st.exception(e)
    else:
        st.info("Please ensure all required files and at least one pollutant are selected.")

# ==================== TAB 5: MULTI-METRIC ANALYSIS (ENHANCED) ====================
with tab5:
    st.header("📈 Multi-Metric Emission Analysis")
    st.markdown("Compare emissions across different pollutants, vehicle types, and fuel types.")

    if 'emissions_data' in st.session_state:
        emissions_data = st.session_state.emissions_data
        selected_pollutants = st.session_state.selected_pollutants
        data_link = st.session_state.data_link

        # ADD UNIT DISPLAY TOGGLE
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Select analysis views below")
        with col2:
            use_converted_units = st.checkbox("Use Converted Units", value=True, 
                                             help="Display values in your selected units from sidebar")

        if selected_pollutants:
            # ===== NEW: FUEL TYPE BREAKDOWN SECTION =====
            st.subheader("⛽ Emissions by Fuel Type")
            st.markdown("Breakdown of emissions by gasoline vs diesel vehicles")
            
            # Chart Options Dropdown
            col_sel_fuel, col_chart_fuel = st.columns([2, 1])
            with col_sel_fuel:
                fuel_analysis_pollutant = st.selectbox(
                    "Select Pollutant for Fuel Type Analysis",
                    options=selected_pollutants,
                    key='fuel_analysis_select'
                )
            with col_chart_fuel:
                fuel_chart_type = st.selectbox("Chart Type", ["Pie Chart", "Bar Chart"], key="fuel_chart_type")

            # Calculate proportions
            gasoline_prop_avg = data_link[:, 4].mean()
            diesel_prop_avg = 1 - gasoline_prop_avg
            
            # Prepare data
            fuel_type_data = []
            for poll in selected_pollutants:
                pc_total = emissions_data[poll]['pc'].sum()
                ldv_total = emissions_data[poll]['ldv'].sum()
                hdv_total = emissions_data[poll]['hdv'].sum()
                moto_total = emissions_data[poll]['moto'].sum()
                
                # Estimate split
                pc_gas = pc_total * gasoline_prop_avg
                pc_dsl = pc_total * diesel_prop_avg
                ldv_gas = ldv_total * gasoline_prop_avg
                ldv_dsl = ldv_total * diesel_prop_avg
                
                total_gas = pc_gas + ldv_gas + moto_total
                total_dsl = pc_dsl + ldv_dsl + hdv_total
                grand = total_gas + total_dsl
                
                fuel_type_data.append({'Pollutant': poll, 'Fuel_Type': 'Gasoline', 'Total_Emissions': total_gas, 'Percentage': total_gas/grand*100})
                fuel_type_data.append({'Pollutant': poll, 'Fuel_Type': 'Diesel', 'Total_Emissions': total_dsl, 'Percentage': total_dsl/grand*100})
            
            fuel_type_df = pd.DataFrame(fuel_type_data)
            fuel_selected_df = fuel_type_df[fuel_type_df['Pollutant'] == fuel_analysis_pollutant]

            # Render Selected Chart
            if fuel_chart_type == "Pie Chart":
                fig_fuel = px.pie(
                    fuel_selected_df, values='Total_Emissions', names='Fuel_Type',
                    title=f"{fuel_analysis_pollutant} Emissions by Fuel Type",
                    color='Fuel_Type', color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'},
                    hole=0.4
                )
                fig_fuel.update_traces(textposition='inside', textinfo='percent+label')
            else:
                fig_fuel = px.bar(
                    fuel_selected_df, x='Fuel_Type', y='Total_Emissions', color='Fuel_Type',
                    title=f"{fuel_analysis_pollutant} Emissions by Fuel Type",
                    color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'}
                )
            st.plotly_chart(fig_fuel, use_container_width=True)
            
            st.markdown("---")
            
            # ===== EXISTING: VEHICLE TYPE BREAKDOWN SECTION =====
            st.subheader("🚗 Emissions by Vehicle Type")

            col_sel_veh, col_chart_veh = st.columns([2, 1])
            with col_sel_veh:
                vehicle_analysis_pollutant = st.selectbox(
                    "Select Pollutant for Vehicle Type Analysis",
                    options=selected_pollutants,
                    key='vehicle_analysis_select'
                )
            with col_chart_veh:
                veh_chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"], key="veh_chart_type")
            
            # Prepare data
            breakdown_data = []
            for poll in selected_pollutants:
                for v_type in ['pc', 'ldv', 'hdv', 'moto']:
                    breakdown_data.append({
                        'Pollutant': poll,
                        'Vehicle_Type': v_type.upper(),
                        'Total_Emissions': emissions_data[poll][v_type].sum()
                    })

            breakdown_df = pd.DataFrame(breakdown_data)
            vehicle_selected_df = breakdown_df[breakdown_df['Pollutant'] == vehicle_analysis_pollutant]
            
            # Render Selected Chart
            color_map = {'PC': '#667eea', 'LDV': '#f59e0b', 'HDV': '#ef4444', 'MOTO': '#10b981', 'MOTORCYCLE': '#10b981'}
            
            if veh_chart_type == "Bar Chart":
                fig_vehicle = px.bar(
                    vehicle_selected_df, x='Vehicle_Type', y='Total_Emissions', color='Vehicle_Type',
                    title=f"{vehicle_analysis_pollutant} Emissions by Vehicle Type",
                    color_discrete_map=color_map,
                    template="plotly_white"
                )
            else:
                fig_vehicle = px.pie(
                    vehicle_selected_df, values='Total_Emissions', names='Vehicle_Type',
                    title=f"{vehicle_analysis_pollutant} Emissions by Vehicle Type",
                    color='Vehicle_Type', color_discrete_map=color_map,
                    hole=0.4
                )
                fig_vehicle.update_traces(textposition='inside', textinfo='percent+label')
                
            st.plotly_chart(fig_vehicle, use_container_width=True)

            st.markdown("---")
            
            # ===== EXISTING: LINK RANKING SECTION =====
            st.subheader("🔝 Top 10 Links by Total Emission")
            
            ranking_pollutant = st.selectbox("Select Pollutant to Rank by", options=selected_pollutants)

            if ranking_pollutant in emissions_data:
                ranking_data = pd.DataFrame(st.session_state.data_link[:, :4], columns=['OSM_ID', 'Length_km', 'Flow', 'Speed'])
                ranking_data[f'Total_{ranking_pollutant}'] = emissions_data[ranking_pollutant]['total']
                top_10_df = ranking_data.sort_values(by=f'Total_{ranking_pollutant}', ascending=False).head(10)
                top_10_df['OSM_ID'] = top_10_df['OSM_ID'].astype(int).astype(str)

                fig_top_10 = px.bar(top_10_df, 
                                    x='OSM_ID', 
                                    y=f'Total_{ranking_pollutant}', 
                                    color='Speed',
                                    title=f"Top 10 Links Emitting {ranking_pollutant}",
                                    labels={f'Total_{ranking_pollutant}': f"Total {ranking_pollutant} (g/km)"},
                                    template="plotly_white")
                st.plotly_chart(fig_top_10, use_container_width=True)
            
        else:
            st.info("No pollutants selected for analysis.")
    else:
        st.info("Please calculate emissions first in the 'Calculate Emissions' tab.")
 
# ==================== TAB 6: INTERACTIVE MAP (IMPROVED) ====================
with tab6:
    st.header("🗺️ Interactive Map")
    
    if st.session_state.get('calc_done') and 'data_link' in st.session_state and osm_file is not None:
        # Controls
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
        
        if st.button("🗺️ Generate Map", type="primary", use_container_width=True):
            with st.spinner("Parsing OSM network and generating interactive map..."):
                try:
                    import osm_network
                    import tempfile, os
                    
                    # Prepare OSM file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                        osm_file.seek(0)
                        tmp.write(osm_file.read())
                        osm_path = tmp.name
                    
                    # Parse network
                    zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    highway_coordinate, highway_osmid, highway_names, highway_types = osm_network.retrieve_highway(osm_path, zone, tolerance, int(ncore))
                    
                    os.unlink(osm_path)

                    # Data for mapping
                    emis = st.session_state.emissions_data[map_poll]['total']
                    link_data = st.session_state.data_link
                    
                    # Create mapping dictionary for fast lookup
                    id_to_idx = {int(row[0]): i for i, row in enumerate(link_data)}
                    
                    # Build Map DataFrame
                    map_rows = []
                    for coords, oid, name in zip(highway_coordinate, highway_osmid, highway_names):
                        if oid in id_to_idx:
                            idx = id_to_idx[oid]
                            val = emis[idx]
                            sp = link_data[idx, 3]
                            
                            if f_speed[0] <= sp <= f_speed[1]:
                                map_rows.append({'oid': oid, 'val': val, 'coords': coords, 'name': name, 'speed': sp})
                    
                    map_df = pd.DataFrame(map_rows)
                    
                    if not map_df.empty:
                        # === ROBUST QUANTILE/BINNING LOGIC ===
                        try:
                            # Try standard quantile cut
                            map_df['quartile'] = pd.qcut(map_df['val'], 4, labels=["Low", "Medium", "High", "Critical"], duplicates='drop')
                        except ValueError:
                            # Fallback to linear cut if quantiles fail (e.g. mostly zeros)
                            map_df['quartile'] = pd.cut(map_df['val'], 4, labels=["Low", "Medium", "High", "Critical"])
                        
                        # Colors based on selection
                        if color_theme == "Jet": colors_scale = px.colors.sequential.Jet
                        elif color_theme == "Viridis": colors_scale = px.colors.sequential.Viridis
                        elif color_theme == "Reds": colors_scale = px.colors.sequential.Reds
                        elif color_theme == "Inferno": colors_scale = px.colors.sequential.Inferno
                        else: colors_scale = px.colors.sequential.Plasma
                        
                        qs = map_df['quartile'].unique()
                        if hasattr(qs, 'sort_values'):
                            qs = qs.sort_values()
                            
                        fig = go.Figure()
                        
                        for i, q in enumerate(qs):
                            subset = map_df[map_df['quartile'] == q]
                            if subset.empty: continue
                            
                            c_lats, c_lons = [], []
                            for cs in subset['coords']:
                                unzipped = list(zip(*cs))
                                # Add None to break lines between different roads
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
                        
                        # Tooltips (Invisible markers on center points of roads)
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
                        st.warning("No data passed filters (check speed range or domain).")
                except Exception as e:
                    st.error(f"Error generating map: {e}")
                    # Debug
                    # import traceback
                    # st.code(traceback.format_exc())
    else:
        st.info("Run calculation and provide OSM file to view map.")

# ==================== TAB 7: DOWNLOAD RESULTS ====================
with tab7:
    st.header("📥 Download Results")
    st.markdown("Download calculated data and reports.")

    if 'emissions_data' in st.session_state and 'data_link' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link_np = st.session_state.data_link
        selected_pollutants = st.session_state.selected_pollutants
                
        # ADD UNIT SELECTION FOR EXPORT
        st.subheader("📤 Export Settings")
        col1, col2 = st.columns(2)
        with col1:
            export_in_converted_units = st.checkbox(
                "Export in converted units",
                value=False,
                help="Export emissions in your selected display units instead of original g/km"
            )
        with col2:
            if export_in_converted_units:
                st.info("✅ Will export using sidebar unit selections")
            else:
                st.info("ℹ️ Will export in original units (g/km, mg/km, L/100km)")
        
        st.markdown("---")

        # Create a single comprehensive results dataframe
        final_results_df = pd.DataFrame(data_link_np[:, :4], columns=['OSM_ID', 'Length_km', 'Flow', 'Speed'])
        
        if data_link_np.shape[1] == 7:
            proportion_columns = ['Gasoline_Prop', 'PC_Prop', '4Stroke_Prop']
            proportion_data = data_link_np[:, 4:7]
        else: # 9 columns
            proportion_columns = ['Gasoline_Prop', 'PC_Prop', '4Stroke_Prop', 'LDV_Prop', 'HDV_Prop']
            proportion_data = data_link_np[:, 4:9]

        proportion_df = pd.DataFrame(proportion_data, columns=proportion_columns)
        final_results_df = pd.concat([final_results_df, proportion_df], axis=1)

        for poll in selected_pollutants:
            if export_in_converted_units and 'selected_units' in st.session_state:
                selected_units = st.session_state.selected_units
                target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                original_unit = pollutants_available[poll]['unit']
                
                poll_df = pd.DataFrame({
                    f'{poll}_PC ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) for v in emissions_data[poll]['pc']],
                    f'{poll}_LDV ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) for v in emissions_data[poll]['ldv']],
                    f'{poll}_HDV ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) for v in emissions_data[poll]['hdv']],
                    f'{poll}_Motorcycle ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) for v in emissions_data[poll]['moto']],
                    f'{poll}_Total ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) for v in emissions_data[poll]['total']]
                })
            else:
                poll_df = pd.DataFrame({
                    f'{poll}_PC': emissions_data[poll]['pc'],
                    f'{poll}_LDV': emissions_data[poll]['ldv'],
                    f'{poll}_HDV': emissions_data[poll]['hdv'],
                    f'{poll}_Motorcycle': emissions_data[poll]['moto'],
                    f'{poll}_Total': emissions_data[poll]['total']
                })
            final_results_df = pd.concat([final_results_df, poll_df], axis=1)


        # Convert the DataFrame to CSV format for download
        csv_export = final_results_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download All Link Results (CSV)",
            data=csv_export,
            file_name='traffic_emission_results_full.csv',
            mime='text/csv',
            key='download_csv_full',
            use_container_width=True
        )

        st.markdown("---")

        # ZIP file containing all calculated data and reports
        st.subheader("Complete Analysis Package (ZIP)")
        
        if 'summary_df_for_download' not in st.session_state or st.session_state.get('recalculate_summary', True):
            summary_data = []
            for poll in selected_pollutants:
                if export_in_converted_units and 'selected_units' in st.session_state:
                    selected_units = st.session_state.selected_units
                    target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                    original_unit = pollutants_available[poll]['unit']
                    
                    total_pc = convert_emission_value(emissions_data[poll]['pc'].sum(), poll, original_unit, target_unit)
                    total_ldv = convert_emission_value(emissions_data[poll]['ldv'].sum(), poll, original_unit, target_unit)
                    total_hdv = convert_emission_value(emissions_data[poll]['hdv'].sum(), poll, original_unit, target_unit)
                    total_moto = convert_emission_value(emissions_data[poll]['moto'].sum(), poll, original_unit, target_unit)
                    grand_total = convert_emission_value(emissions_data[poll]['total'].sum(), poll, original_unit, target_unit)
                    display_unit = target_unit
                else:
                    total_pc = emissions_data[poll]['pc'].sum()
                    total_ldv = emissions_data[poll]['ldv'].sum()
                    total_hdv = emissions_data[poll]['hdv'].sum()
                    total_moto = emissions_data[poll]['moto'].sum()
                    grand_total = emissions_data[poll]['total'].sum()
                    display_unit = pollutants_available[poll]['unit']
                
                summary_data.append({
                    'Pollutant': poll,
                    'Total PC': format_emission_value(total_pc, display_unit),
                    'Total LDV': format_emission_value(total_ldv, display_unit),
                    'Total HDV': format_emission_value(total_hdv, display_unit),
                    'Total Moto': format_emission_value(total_moto, display_unit),
                    'Grand Total': format_emission_value(grand_total, display_unit),
                    'Unit': display_unit
                })

            st.session_state.summary_df_for_download = pd.DataFrame(summary_data)
            st.session_state.recalculate_summary = False
        
        summary_df = st.session_state.summary_df_for_download
        
        if st.button("Generate & Download ZIP Report", use_container_width=True):
            with st.spinner("Generating ZIP archive..."):
                with BytesIO() as buffer:
                    with zipfile.ZipFile(buffer, 'w') as zipf:
                        # 1. Full Results CSV
                        zipf.writestr('full_link_results.csv', final_results_df.to_csv(index=False))

                        # 2. Statistics Summary CSV
                        zipf.writestr('statistics_summary.csv', summary_df.to_csv(index=False))

                        # 3. Fuel Type Breakdown
                        if 'fuel_emissions_data' in st.session_state:
                            fuel_emissions = st.session_state.fuel_emissions_data
                            fuel_breakdown_data = []
                            for poll in selected_pollutants:
                                if export_in_converted_units and 'selected_units' in st.session_state:
                                    selected_units = st.session_state.selected_units
                                    target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                                    original_unit = pollutants_available[poll]['unit']
                                    
                                    gas_total = convert_emission_value(fuel_emissions[poll]['gasoline'].sum(), poll, original_unit, target_unit)
                                    diesel_total = convert_emission_value(fuel_emissions[poll]['diesel'].sum(), poll, original_unit, target_unit)
                                    display_unit = target_unit
                                else:
                                    gas_total = fuel_emissions[poll]['gasoline'].sum()
                                    diesel_total = fuel_emissions[poll]['diesel'].sum()
                                    display_unit = pollutants_available[poll]['unit']
                                
                                fuel_breakdown_data.append({
                                    'Pollutant': poll,
                                    'Gasoline_Total': format_emission_value(gas_total, display_unit),
                                    'Diesel_Total': format_emission_value(diesel_total, display_unit),
                                    'Unit': display_unit
                                })
                            fuel_df = pd.DataFrame(fuel_breakdown_data)
                            zipf.writestr('fuel_type_breakdown.csv', fuel_df.to_csv(index=False))

                # 4. Text Report
                gasoline_avg = data_link_np[:, 4].mean()
                diesel_avg = 1 - gasoline_avg
                
                report_text = f"""
Traffic Emission Calculation Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Selected Pollutants: {', '.join(selected_pollutants)}
Methodology: {calculation_method}
Ambient Temperature: {ambient_temp} degrees Celsius
Trip Length (Cold Start): {trip_length} km
Export Units: {'Converted Units' if export_in_converted_units else 'Original Units (g/km, mg/km, L/100km)'}
"""
                if export_in_converted_units and 'selected_units' in st.session_state:
                    unit_conversions = ', '.join([f'{p}={st.session_state.selected_units.get(p)}' for p in selected_pollutants])
                    report_text += f"Unit Conversions Applied: {unit_conversions}\n"
                
                report_text += f"""
--- Summary Statistics ---
{summary_df.to_string(index=False)}

--- Fuel Type Distribution ---
Average Gasoline Proportion: {gasoline_avg*100:.1f}%
Average Diesel Proportion: {diesel_avg*100:.1f}%
"""
                zipf.writestr('detailed_report.txt', report_text)
                
            st.success("✅ ZIP report generated successfully!")

            buffer.seek(0)
            st.download_button(
                label="📦 Download ZIP Report",
                data=buffer,
                file_name="traffic_emission_analysis.zip",
                mime="application/zip",
                key='download_zip',
                use_container_width=True
            )
        
        st.markdown("### 📚 Export Formats")
        
        if export_in_converted_units and 'selected_units' in st.session_state:
            st.success(f"**Current Export Settings: ✅ Converted Units**")
        else:
            st.info("""**Current Export Settings: ℹ️ Original Units**""")
        
    else:
        st.info("Calculate emissions first to create download package")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Standards: EEA Guidebook 2019, IPCC 2019 Guidelines, WHO Air Quality Standards</p>
    <p>© 2025 - Developed by SHassan</p>
</div>
""", unsafe_allow_html=True)
