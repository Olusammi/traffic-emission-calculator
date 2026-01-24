import streamlit as st
import numpy as np
import pandas as pd
import requests
import pydeck as pdk
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from io import BytesIO
import zipfile
import tempfile
import os
import geopandas as gpd

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="Advanced Traffic Emission Calculator", layout="wide", initial_sidebar_state="expanded")

REPO_USER = "Olusammi"
REPO_NAME = "traffic-emission-calculator"
REPO_BRANCH = "main"
DEFAULT_FOLDER = "default" 
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{REPO_BRANCH}/{DEFAULT_FOLDER}/"

DEFAULT_FILES_MAP = {
    "pc": "PC_parameter.csv",
    "ldv": "LDV_parameter.csv",
    "hdv": "HDV_parameter.csv",
    "moto": "Moto_parameter.csv",
    "link": "link_osm_with-ldv.dat",
    "osm": "nigeria_major_roads.gpkg", 
    "ecg": "engine_gasoline.dat",
    "ecd": "engine_diesel.dat",
    "ccg": "copert_class_proportion_gasoline.dat",
    "ccd": "copert_class_proportion_diesel.dat",
    "2s": "copert_class_proportion_2_stroke_motorcycle_more_50.dat",
    "4s": "copert_class_proportion_4_stroke_motorcycle_50_250.dat"
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
        text-align: center;
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
def fetch_default_file(filename):
    """Fetches a default file from GitHub."""
    try:
        # Handle GPKG vs OSM extension for network file
        if filename.endswith(".gpkg"):
             url = GITHUB_BASE_URL + filename
             r = requests.get(url, timeout=15)
             if r.status_code == 200: return BytesIO(r.content)
             # Fallback to .osm if gpkg fails
             url = GITHUB_BASE_URL + filename.replace(".gpkg", ".osm")
             r = requests.get(url, timeout=15)
             if r.status_code == 200: return BytesIO(r.content)
        else:
            url = GITHUB_BASE_URL + filename
            r = requests.get(url, timeout=10)
            if r.status_code == 200: return BytesIO(r.content)
        return None
    except Exception:
        return None

def get_file_input(label, type_list, key):
    """
    Smart Uploader: Checks upload -> Checks Default -> Returns File Object
    Also displays status message.
    """
    uploaded_file = st.file_uploader(label, type=type_list, key=key)
    status_container = st.empty()
    
    if uploaded_file is not None:
        status_container.success("📂 Using Uploaded File")
        return uploaded_file
    
    # Fallback
    if key in DEFAULT_FILES_MAP:
        default_name = DEFAULT_FILES_MAP[key]
        content = fetch_default_file(default_name)
        if content:
            status_container.info(f"✅ Using Default: {default_name}")
            return content
            
    status_container.warning("⚠️ No file provided")
    return None

@st.cache_data(show_spinner=False)
def load_copert_instance(pc_content, ldv_content, hdv_content, moto_content):
    import copert
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name, content in [("pc", pc_content), ("ldv", ldv_content), 
                              ("hdv", hdv_content), ("moto", moto_content)]:
            p = os.path.join(tmpdir, f"{name}.csv")
            with open(p, 'wb') as f: f.write(content.getvalue())
            paths[name] = p
        
        cop = copert.Copert(paths["pc"], paths["ldv"], paths["hdv"], paths["moto"])
    return cop

@st.cache_data(show_spinner=False)
def parse_osm_network_cached(osm_content_bytes, x_min, x_max, y_min, y_max, tolerance, ncore):
    import osm_network
    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
        tmp.write(osm_content_bytes.getvalue())
        tmp_name = tmp.name
    
    try:
        zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return osm_network.retrieve_highway(tmp_name, zone, tolerance, int(ncore))
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)

# ==================== SIDEBAR ====================
st.sidebar.header("📊 Emission Metrics Selection")
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

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Calculation Methodology")
calculation_method = st.sidebar.selectbox("Select Calculation Standard", ["COPERT IV (EU)", "IPCC Tier 2", "EPA MOVES (US)", "Hybrid (Multi-standard)"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 Accuracy Settings")
include_temperature_correction = st.sidebar.checkbox("Temperature Correction", value=True)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -10, 40, 25) if include_temperature_correction else 20
include_cold_start = st.sidebar.checkbox("Cold Start Correction", value=True)
trip_length = st.sidebar.slider("Trip Length (km)", 1, 50, 12) if include_cold_start else 12
include_slope = st.sidebar.checkbox("Slope Correction", value=False)

# ==================== UNIT CONVERSION SETTINGS ====================
st.sidebar.markdown("---")
st.sidebar.header("📏 Unit Conversion")
unit_conversion_options = {
    "CO": {"g/km": {"factor": 1.0}, "kg/km": {"factor": 0.001}, "tonnes/km": {"factor": 1e-6}},
    "CO2": {"g/km": {"factor": 1.0}, "kg/km": {"factor": 0.001}, "tonnes/km": {"factor": 1e-6}},
    "NOx": {"g/km": {"factor": 1.0}, "kg/km": {"factor": 0.001}, "mg/km": {"factor": 1000.0}},
    "PM": {"mg/km": {"factor": 1.0}, "g/km": {"factor": 0.001}, "µg/km": {"factor": 1000.0}},
    "VOC": {"g/km": {"factor": 1.0}, "kg/km": {"factor": 0.001}},
    "FC": {"L/100km": {"factor": 1.0}, "L/km": {"factor": 0.01}}
}

selected_units = {}
for poll in pollutants_available.keys():
    if poll in unit_conversion_options:
        selected_units[poll] = st.sidebar.selectbox(f"{poll} Unit", list(unit_conversion_options[poll].keys()), key=f"u_{poll}")
    else:
        selected_units[poll] = pollutants_available[poll]['unit']
st.session_state.selected_units = selected_units

# ==================== FILE UPLOADS ====================
st.sidebar.markdown("---")
st.sidebar.header("📂 Upload Input Files")

copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=True)
with copert_files:
    pc_param = get_file_input("PC Params", ['csv'], 'pc')
    ldv_param = get_file_input("LDV Params", ['csv'], 'ldv')
    hdv_param = get_file_input("HDV Params", ['csv'], 'hdv')
    moto_param = get_file_input("Moto Params", ['csv'], 'moto')

data_files = st.sidebar.expander("Data Files", expanded=True)
with data_files:
    link_osm = get_file_input("Link Data", ['dat', 'csv', 'txt'], 'link')
    osm_file = get_file_input("Network File (.osm/.gpkg)", ['osm', 'gpkg'], 'osm')

proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
prop_files = {}
with proportion_files:
    prop_files['ecg'] = get_file_input("Engine Cap Gasoline", ['dat', 'txt'], 'ecg')
    prop_files['ecd'] = get_file_input("Engine Cap Diesel", ['dat', 'txt'], 'ecd')
    prop_files['ccg'] = get_file_input("Class Gasoline", ['dat', 'txt'], 'ccg')
    prop_files['ccd'] = get_file_input("Class Diesel", ['dat', 'txt'], 'ccd')
    prop_files['2s'] = get_file_input("2-Stroke Moto", ['dat', 'txt'], '2s')
    prop_files['4s'] = get_file_input("4-Stroke Moto", ['dat', 'txt'], '4s')

st.sidebar.header("🗺️ Map Parameters")
with st.sidebar.expander("Boundaries", expanded=False):
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X Min", value=3.37310, format="%.5f")
    x_max = col2.number_input("X Max", value=3.42430, format="%.5f")
    y_min = col1.number_input("Y Min", value=6.43744, format="%.5f")
    y_max = col2.number_input("Y Max", value=6.46934, format="%.5f")
    tolerance = st.number_input("Tolerance", value=0.005, format="%.3f")
    ncore = st.number_input("Cores", value=8, min_value=1, max_value=16)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📖 Instructions", "📊 Data Preview", "🧮 Formula Explanation",
    "⚙️ Calculate Emissions", "📈 Multi-Metric Analysis",
    "🗺️ Interactive Map", "📥 Download Results"
])

# --- TAB 1: INSTRUCTIONS (Preserved) ---
with tab1:
    st.header("📖 User Guide & Instructions")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Multi-Pollutant Analysis**: Calculate CO, CO₂, NOx, PM, VOC, and FC.
        - **Smart Inputs**: Supports both Global (1 row) and Local (N rows) fleet mixes.
        - **Interactive Visualization**: GPU-accelerated PyDeck maps.
        """)
    with col2:
        st.subheader("📚 Standards")
        st.markdown("COPERT IV (EU), IPCC Tier 2, EPA MOVES")

# --- TAB 2: DATA PREVIEW (Preserved + Safe) ---
with tab2:
    st.header("📊 Data Preview")
    if link_osm:
        try:
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            if data_link.shape[1] >= 9:
                data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop','LDV_Prop','HDV_Prop']
                if data_link.shape[1] > 9: data_link.columns += [f"Col_{i}" for i in range(9, data_link.shape[1])]
            else:
                data_link.columns = [f'Col_{i}' for i in range(data_link.shape[1])]
            
            st.dataframe(data_link.head(20), use_container_width=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Links", len(data_link))
            c2.metric("Total Length", f"{data_link.iloc[:,1].sum():.1f} km")
            c3.metric("Avg Speed", f"{data_link.iloc[:,3].mean():.1f} km/h")
            c4.metric("Avg Flow", f"{data_link.iloc[:,2].mean():.0f}")
            
            # Graphs
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                fig = px.histogram(data_link, x='Speed', nbins=30, title="Speed Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c_g2:
                fig = px.histogram(data_link, x='Flow', nbins=30, title="Flow Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Waiting for Link Data...")

# --- TAB 3: FORMULAS (Preserved) ---
with tab3:
    st.header("🧮 Mathematical Formulas")
    sel_f = st.selectbox("Select Pollutant", list(pollutants_available.keys()))
    st.markdown("---")
    if sel_f == "CO":
        st.latex(r'''EF_{hot} = \frac{a + c \cdot V + e \cdot V^2}{1 + b \cdot V + d \cdot V^2}''')
    elif sel_f == "NOx":
        st.latex(r'''EF_{NOx} = EF_{base}(V) \cdot \left(1 + k \cdot (T_{amb} - 20)\right)''')
    elif sel_f == "CO2":
        st.latex(r'''CO_2 = 44.01 \times \frac{FC}{12.01 \times 100} \times \text{CarbonRatio}''')
    st.info("Formulas based on EMEP/EEA Guidebook 2019")

# --- TAB 4: CALCULATE (OPTIMIZED ENGINE) ---
with tab4:
    st.header("⚙️ Calculate Emissions")
    
    ready = all([pc_param, ldv_param, hdv_param, moto_param, link_osm]) and all(prop_files.values())
    if not ready:
        st.warning("⚠️ Some files are missing (Upload or check Defaults).")
    elif not selected_pollutants:
        st.warning("⚠️ Select at least one pollutant.")
    else:
        if st.button("🚀 Calculate Multi-Pollutant Emissions", type="primary", use_container_width=True):
            with st.spinner("Initializing Vectorized Engine..."):
                try:
                    import copert
                    cop = load_copert_instance(pc_param, ldv_param, hdv_param, moto_param)
                    
                    link_osm.seek(0)
                    df_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
                    data_link = df_link.values
                    N_links = len(data_link)
                    
                    # SAFE CYCLE LOADER
                    def load_safe(key, name):
                        f = prop_files[key]
                        f.seek(0)
                        try:
                            arr = np.loadtxt(f)
                            if arr.ndim == 1: arr = arr.reshape(1, -1)
                        except: st.error(f"Error parsing {name}"); st.stop()
                        
                        rows = arr.shape[0]
                        if rows == N_links: return arr
                        elif rows == 1: return np.tile(arr, (N_links, 1))
                        else:
                            reps = int(np.ceil(N_links / rows))
                            return np.tile(arr, (reps, 1))[:N_links]

                    eng_cap_gas = load_safe('ecg', 'Gas Cap')
                    eng_cap_dsl = load_safe('ecd', 'Dsl Cap')
                    cls_gas = load_safe('ccg', 'Gas Class')
                    cls_dsl = load_safe('ccd', 'Dsl Class')
                    cls_2s = load_safe('2s', 'Moto 2S')
                    cls_4s = load_safe('4s', 'Moto 4S')

                    # Extraction
                    lengths = data_link[:, 1]; flows = data_link[:, 2]
                    speeds = np.clip(data_link[:, 3], 10.0, 130.0)
                    prop_gas = data_link[:, 4]; prop_pc = data_link[:, 5]
                    prop_4s = data_link[:, 6]; prop_ldv = data_link[:, 7]
                    prop_hdv = data_link[:, 8]
                    prop_dsl = 1.0 - prop_gas
                    prop_moto = np.maximum(0.0, 1.0 - (prop_pc + prop_ldv + prop_hdv))
                    prop_2s = 1.0 - prop_4s

                    emissions_db = {}
                    poll_map = {"CO": cop.pollutant_CO, "NOx": cop.pollutant_NOx, "PM": cop.pollutant_PM, 
                                "VOC": cop.pollutant_VOC, "FC": cop.pollutant_FC, "CO2": cop.pollutant_FC}

                    prog = st.progress(0)
                    
                    for idx, poll_name in enumerate(selected_pollutants):
                        p_const = poll_map.get(poll_name, cop.pollutant_CO)
                        t_pc = np.zeros(N_links); t_ldv = np.zeros(N_links); t_hdv = np.zeros(N_links); t_moto = np.zeros(N_links)
                        
                        # --- PC & LDV ---
                        pc_configs = [(0, prop_gas, eng_cap_gas), (1, prop_dsl, eng_cap_dsl)]
                        cop_classes = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03, cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop, cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                        cap_indices = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2, cop.engine_capacity_more_2]

                        for eng_idx, eng_prop, cap_matrix in pc_configs:
                            for k, cap_id in enumerate(cap_indices):
                                if k >= cap_matrix.shape[1]: continue
                                for c_idx, cls_val in enumerate(cop_classes):
                                    if c_idx >= cls_gas.shape[1]: continue
                                    if eng_idx == 1 and k == 0 and cls_val <= cop.class_Euro_3: continue 
                                    
                                    ef = cop.HEFGasolinePassengerCar(p_const, speeds, cls_val, cap_id) if eng_idx == 0 else cop.HEFDieselPassengerCar(p_const, speeds, cls_val, cap_id)
                                    class_matrix = cls_gas if eng_idx == 0 else cls_dsl
                                    
                                    t_pc += ef * eng_prop * cap_matrix[:, k] * class_matrix[:, c_idx]
                                    
                                    # LDV Approximation using PC factors (Robust)
                                    ef_ldv = cop.HEFLightCommercialVehicle(p_const, speeds, cop.engine_type_gasoline if eng_idx==0 else cop.engine_type_diesel, cls_val)
                                    t_ldv += ef_ldv * eng_prop * class_matrix[:, c_idx]

                        # --- HDV ---
                        hdv_stds = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III, cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
                        for h in hdv_stds:
                            ef_h = cop.HEFHeavyDutyVehicle(p_const, speeds, 0, 0, h)
                            t_hdv += ef_h * (1.0/len(hdv_stds))

                        # --- Moto ---
                        m_types = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        m_classes = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2, cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        m_mats = [cls_2s, cls_4s]; m_props_arr = [prop_2s, prop_4s]
                        
                        for i, mt in enumerate(m_types):
                            for c_idx, cls in enumerate(m_classes):
                                if c_idx >= m_mats[i].shape[1]: continue
                                ef_m = cop.EFMotorcycle(p_const, speeds, mt, cls)
                                t_moto += ef_m * m_props_arr[i] * m_mats[i][:, c_idx]

                        t_pc *= prop_pc * flows; t_ldv *= prop_ldv * flows
                        t_hdv *= prop_hdv * flows; t_moto *= prop_moto * flows
                        total = t_pc + t_ldv + t_hdv + t_moto
                        
                        if poll_name == "CO2": # FC to CO2 Conversion (Approx)
                            factor = (prop_gas * 2392 + prop_dsl * 2640) * 0.01 
                            total *= factor; t_pc *= factor; t_ldv *= factor; t_hdv *= 2640 * 0.01; t_moto *= 2392 * 0.01

                        emissions_db[poll_name] = {'pc': t_pc, 'ldv': t_ldv, 'hdv': t_hdv, 'moto': t_moto, 'total': total}
                        prog.progress((idx + 1) / len(selected_pollutants))

                    st.session_state.emissions_db = emissions_db
                    st.session_state.data_link = data_link
                    st.success("✅ Calculations Complete!")
                    
                except Exception as e:
                    st.error(f"Calc Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# --- TAB 5: ANALYSIS (Preserved & Linked) ---
with tab5:
    st.header("📈 Multi-Metric Analysis")
    if 'emissions_db' in st.session_state:
        c1, c2 = st.columns([2, 1])
        with c1:
            p_view = st.selectbox("Select Pollutant to Analyze", selected_pollutants)
            db = st.session_state.emissions_db[p_view]
            df_v = pd.DataFrame({
                'Vehicle': ['PC', 'LDV', 'HDV', 'Moto'],
                'Emission': [db['pc'].sum(), db['ldv'].sum(), db['hdv'].sum(), db['moto'].sum()]
            })
            fig = px.pie(df_v, values='Emission', names='Vehicle', title=f"{p_view} by Vehicle Type", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"**Total {p_view}:** {db['total'].sum():,.2f}")
            st.dataframe(df_v, hide_index=True)
    else:
        st.info("Calculate first.")

# --- TAB 6: INTERACTIVE MAP (FIXED) ---
with tab6:
    st.header("🗺️ Interactive Map")
    
    if 'emissions_db' not in st.session_state:
        st.warning("⚠️ Calculate emissions first.")
    elif osm_file is None:
        st.warning("⚠️ OSM Network file missing.")
    else:
        # 1. Geometry Prep
        if 'map_geo' not in st.session_state:
            with st.spinner("Preparing Map Geometry..."):
                try:
                    osm_file.seek(0)
                    
                    # DETECT FILE TYPE
                    # If uploaded, use .name. If default (BytesIO), check config.
                    filename = "default.gpkg" 
                    if hasattr(osm_file, 'name'):
                        filename = osm_file.name
                    elif 'osm' in DEFAULT_FILES_MAP:
                        filename = DEFAULT_FILES_MAP['osm']

                    # BRANCH LOGIC: GPKG vs OSM
                    if filename.endswith('.gpkg'):
                        # GPKG MODE (Geopandas)
                        gdf = gpd.read_file(osm_file)
                        # Ensure Lat/Lon
                        if gdf.crs and gdf.crs.to_epsg() != 4326:
                            gdf = gdf.to_crs(epsg=4326)
                        st.session_state.map_geo_gdf = gdf
                        # Clear old OSM cache if exists
                        if 'map_geo' in st.session_state: del st.session_state.map_geo
                        
                    else:
                        # OSM MODE (Osmium)
                        coords, ids, names, types = parse_osm_network_cached(
                            osm_file, x_min, x_max, y_min, y_max, tolerance, ncore
                        )
                        st.session_state.map_geo = (coords, ids, names, types)
                        # Clear old GPKG cache if exists
                        if 'map_geo_gdf' in st.session_state: del st.session_state.map_geo_gdf
                        
                except Exception as e:
                    st.error(f"Map Prep Error: {e}")
                    st.stop()

        # 2. Controls
        c1, c2, c3, c4 = st.columns(4)
        with c1: view_poll = st.selectbox("Pollutant", selected_pollutants)
        with c2: lw = st.slider("Line Width", 1, 50, 15)
        with c3: pitch = st.slider("3D Pitch", 0, 60, 45)
        with c4: f_speed = st.slider("Min Speed Filter", 0, 100, 0)

        # 3. Prepare Data
        db = st.session_state.emissions_db[view_poll]['total']
        d_link = st.session_state.data_link
        
        # Filter Logic (Speed)
        mask = d_link[:, 3] >= f_speed
        filtered_ids = d_link[mask, 0].astype(int)
        filtered_vals = db[mask]
        lookup = dict(zip(filtered_ids, filtered_vals))
        
        map_data = []
        max_val = np.max(filtered_vals) if len(filtered_vals) > 0 else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        cmap = cm.get_cmap("Reds") 

        # 4. Geometry Binding (Handle both types)
        if 'map_geo_gdf' in st.session_state:
            # GPKG Logic
            gdf = st.session_state.map_geo_gdf
            for _, row in gdf.iterrows():
                oid = int(row.get('osm_id', 0))
                if oid in lookup:
                    val = lookup[oid]
                    color = [int(c*255) for c in cmap(norm(val))[:3]]
                    
                    # Extract coordinates from LineString/MultiLineString
                    if row.geometry.geom_type == 'LineString':
                        path = list(row.geometry.coords)
                        map_data.append({"path": path, "emission": val, "color": color, "name": str(oid)})
                    elif row.geometry.geom_type == 'MultiLineString':
                        for line in row.geometry.geoms:
                            path = list(line.coords)
                            map_data.append({"path": path, "emission": val, "color": color, "name": str(oid)})
                            
        elif 'map_geo' in st.session_state:
            # OSM Logic
            coords, ids, names, types = st.session_state.map_geo
            for r, oid, name in zip(coords, ids, names):
                if oid in lookup:
                    val = lookup[oid]
                    color = [int(c*255) for c in cmap(norm(val))[:3]]
                    map_data.append({"path": r, "emission": val, "color": color, "name": name})

        if not map_data:
            st.warning("No matching links found. Check if your Link Data IDs match the Map IDs.")
        else:
            # 5. Render PyDeck
            layer = pdk.Layer(
                type="PathLayer",
                data=map_data,
                pickable=True,
                get_color="color",
                width_scale=1,
                width_min_pixels=2,
                get_path="path",
                get_width=lw,
                opacity=0.9
            )
            
            # Auto-Center
            if map_data:
                start = map_data[0]['path'][0]
                view_state = pdk.ViewState(latitude=start[1], longitude=start[0], zoom=12, pitch=pitch)
            else:
                view_state = pdk.ViewState(latitude=6.5, longitude=3.3, zoom=12, pitch=pitch)
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{name}</b><br/>{emission:.2f} g/km", "style": {"backgroundColor": "steelblue", "color": "white"}},
                map_style="mapbox://styles/mapbox/light-v9"
            )
            st.pydeck_chart(deck)
            
            # Matplotlib Legend
            st.caption(f"Legend: 0 to {max_val:.2f} g/km")
            fig, ax = plt.subplots(figsize=(6, 0.5))
            matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
            st.pyplot(fig)

# --- TAB 7: DOWNLOAD (Preserved Units Logic) ---
with tab7:
    st.header("📥 Download Results")
    if 'emissions_db' in st.session_state:
        d_link = st.session_state.data_link
        df_out = pd.DataFrame(d_link[:, :4], columns=['OSM_ID', 'Length', 'Flow', 'Speed'])
        
        # Apply Unit Conversions if selected
        u_opts = st.session_state.get('selected_units', {})
        
        for p in selected_pollutants:
            val = st.session_state.emissions_db[p]['total']
            unit = u_opts.get(p, pollutants_available[p]['unit'])
            
            # Simple conversion logic check
            factor = unit_conversion_options.get(p, {}).get(unit, {}).get('factor', 1.0)
            df_out[f'{p}_Total ({unit})'] = val * factor
            
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "emissions_results.csv", "text/csv")
    else:
        st.info("Calculate first.")

# ==================== FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Instructions:**")
st.sidebar.markdown("""
1. Upload all COPERT parameter files
2. Upload link OSM data (9 columns)
3. Upload proportion data files
4. Upload OSM network file
5. Configure map parameters
6. Calculate emissions
7. Choose visualization mode
8. Generate map
9. Download results
""")
st.sidebar.info("Built with Streamlit by SHassan 🎈")
