import streamlit as st
import numpy as np
import pandas as pd
import requests
import pydeck as pdk
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import BytesIO
import zipfile
import tempfile
import os

st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with Interactive Map")
st.caption("Built by SHassan | Optimized for Global & Local Fleet Definitions")

# ==================== CONFIGURATION: GITHUB DEFAULTS ====================
REPO_USER = "Olusammi"
REPO_NAME = "traffic-emission-calculator"
REPO_BRANCH = "main"
DEFAULT_FOLDER = "default" 
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{REPO_BRANCH}/{DEFAULT_FOLDER}/"

# Mapping your variable keys to your specific filenames
DEFAULT_FILES_MAP = {
    "pc": "PC_parameter.csv",
    "ldv": "LDV_parameter.csv",
    "hdv": "HDV_parameter.csv",
    "moto": "Moto_parameter.csv",
    "link": "link_osm_with-ldv.dat",
    "osm": "selected_zone-lagos.osm", # Must be .osm for the parser
    "ecg": "engine_gasoline.dat",
    "ecd": "engine_diesel.dat",
    "ccg": "copert_class_proportion_gasoline.dat",
    "ccd": "copert_class_proportion_diesel.dat",
    "2s": "copert_class_proportion_2_stroke_motorcycle_more_50.dat",
    "4s": "copert_class_proportion_4_stroke_motorcycle_50_250.dat"
}

# --- HELPER FUNCTIONS ---

@st.cache_data(show_spinner=False)
def fetch_default_file(filename):
    """Fetches a default file from GitHub."""
    try:
        url = f"{GITHUB_BASE_URL}{filename}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        return BytesIO(response.content)
    except Exception as e:
        return None

def get_input_file(uploader_obj, file_key):
    """Returns the uploaded file OR the default file if available."""
    if uploader_obj is not None:
        return uploader_obj
    
    # If no upload, try default
    if file_key in DEFAULT_FILES_MAP:
        default_name = DEFAULT_FILES_MAP[file_key]
        content = fetch_default_file(default_name)
        return content
    return None

@st.cache_data(show_spinner=False)
def load_copert_instance(pc_content, ldv_content, hdv_content, moto_content):
    import copert
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        # Write contents to temp files because Copert class expects paths
        for name, content in [("pc", pc_content), ("ldv", ldv_content), 
                              ("hdv", hdv_content), ("moto", moto_content)]:
            p = os.path.join(tmpdir, f"{name}.csv")
            with open(p, 'wb') as f: f.write(content.getvalue())
            paths[name] = p
        
        cop = copert.Copert(paths["pc"], paths["ldv"], paths["hdv"], paths["moto"])
    return cop

@st.cache_data(show_spinner=False)
def parse_osm_network(osm_content_bytes, x_min, x_max, y_min, y_max, tolerance, ncore):
    import osm_network
    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
        tmp.write(osm_content_bytes.getvalue())
        tmp_name = tmp.name
    
    try:
        zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return osm_network.retrieve_highway(tmp_name, zone, tolerance, int(ncore))
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)

# --- SIDEBAR ---
st.sidebar.header("📂 1. Configuration Files")
copert_files = st.sidebar.expander("COPERT Parameters", expanded=False)
with copert_files:
    pc_param = st.file_uploader("PC Params", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Params", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Params", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Params", type=['csv'], key='moto')

st.sidebar.header("📂 2. Network Data")
data_files = st.sidebar.expander("Link & Network", expanded=True)
with data_files:
    link_osm = st.file_uploader("Link Data (9 Cols)", type=['dat','csv','txt'], key='link')
    osm_file = st.file_uploader("OSM Network (.osm)", type=['osm'], key='osm')

st.sidebar.header("📂 3. Fleet Mix (Smart)")
st.sidebar.info("💡 **Optimization:** You can upload files with 1 row (Global Mix) OR N rows (Local Mix). Defaults loaded if empty.")

prop_uploads = {
    'ecg': st.file_uploader("Eng Cap Gas", key='ecg'),
    'ecd': st.file_uploader("Eng Cap Diesel", key='ecd'),
    'ccg': st.file_uploader("Class Gas", key='ccg'),
    'ccd': st.file_uploader("Class Diesel", key='ccd'),
    '2s': st.file_uploader("Moto 2-Stroke", key='2s'),
    '4s': st.file_uploader("Moto 4-Stroke", key='4s')
}

# Map parameters
st.sidebar.header("🗺️ Map Parameters")
with st.sidebar.expander("Boundaries", expanded=False):
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X Min", value=3.37310, format="%.5f")
    x_max = col2.number_input("X Max", value=3.42430, format="%.5f")
    y_min = col1.number_input("Y Min", value=6.43744, format="%.5f")
    y_max = col2.number_input("Y Max", value=6.46934, format="%.5f")
    tolerance = st.number_input("Tolerance", value=0.005, format="%.3f")
    ncore = st.number_input("Cores", value=8, min_value=1, max_value=16)

# --- RESOLVE INPUTS ---
# Determine which file to use (Upload vs Default)
f_pc = get_input_file(pc_param, 'pc')
f_ldv = get_input_file(ldv_param, 'ldv')
f_hdv = get_input_file(hdv_param, 'hdv')
f_moto = get_input_file(moto_param, 'moto')
f_link = get_input_file(link_osm, 'link')
f_osm = get_input_file(osm_file, 'osm')

f_props = {}
for k, uploader in prop_uploads.items():
    f_props[k] = get_input_file(uploader, k)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Validation", "⚙️ Calculate", "🗺️ Interactive Map", "📥 Download", "ℹ️ Help"])

with tab1:
    st.header("Data Validation")
    if f_link is not None:
        try:
            f_link.seek(0)
            data_link = pd.read_csv(f_link, sep=r'\s+', header=None, engine='python')
            
            # Smart Column Naming
            if data_link.shape[1] >= 9:
                cols = ['OSM_ID','Length_km','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop','LDV_Prop','HDV_Prop']
                if data_link.shape[1] > 9:
                    cols += [f"Col_{i}" for i in range(9, data_link.shape[1])]
                data_link.columns = cols
            else:
                data_link.columns = [f'Col_{i}' for i in range(data_link.shape[1])]
            
            st.dataframe(data_link.head(5))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Links", len(data_link))
            c2.metric("Total Length", f"{data_link.iloc[:,1].sum():.1f} km")
            
            # Validation Logic
            if data_link.shape[1] < 9:
                st.error(f"❌ Link Data has {data_link.shape[1]} columns. Required: 9.")
            else:
                st.success("✅ Link Data Structure Valid (9+ Columns)")
                if link_osm is None:
                    st.info("Using Default Link Data from GitHub.")
                
        except Exception as e:
            st.error(f"Error reading Link Data: {e}")
    else:
        st.info("Waiting for Link Data (Upload or Default)...")

with tab2:
    st.header("Calculate Emissions")
    
    # Check readiness
    missing_files = []
    if not f_pc: missing_files.append("PC Params")
    if not f_ldv: missing_files.append("LDV Params")
    if not f_link: missing_files.append("Link Data")
    # Check if all prop files are present (defaults or uploads)
    if not all(f_props.values()):
        missing_files.append("Fleet Mix Files")
    
    if missing_files:
        st.warning(f"⚠️ Missing files (Check GitHub defaults or Upload): {', '.join(missing_files)}")
    else:
        if st.button("🚀 Calculate Emissions (Smart Mode)", type="primary"):
            with st.spinner("Initializing & Optimizing Inputs..."):
                try:
                    import copert
                    
                    # 1. Load Copert
                    cop = load_copert_instance(f_pc, f_ldv, f_hdv, f_moto)

                    # 2. Process Link Data
                    f_link.seek(0)
                    df_link = pd.read_csv(f_link, sep=r'\s+', header=None, engine='python')
                    data_link = df_link.values
                    N_links = len(data_link)
                    
                    # 3. SMART LOADER (Broadcasting)
                    def load_prop_smart(key, name):
                        f = f_props.get(key)
                        f.seek(0)
                        try:
                            # Try loading as 2D array
                            arr = np.loadtxt(f)
                            if len(arr.shape) == 1:
                                arr = arr.reshape(1, -1)
                        except:
                            st.error(f"Could not parse {name}")
                            st.stop()

                        rows, cols = arr.shape
                        
                        if rows == N_links:
                            return arr
                        elif rows == 1:
                            return np.tile(arr, (N_links, 1))
                        else:
                            st.error(f"❌ Shape Error in **{name}**.\nFound {rows} rows. Expected either 1 (Global) or {N_links} (Per-Link).")
                            st.stop()

                    # Load Proportions
                    eng_cap_gas = load_prop_smart('ecg', 'Eng Cap Gas')
                    eng_cap_dsl = load_prop_smart('ecd', 'Eng Cap Diesel')
                    cls_gas = load_prop_smart('ccg', 'Class Gas')
                    cls_dsl = load_prop_smart('ccd', 'Class Diesel')
                    cls_2s = load_prop_smart('2s', 'Moto 2-Stroke')
                    cls_4s = load_prop_smart('4s', 'Moto 4-Stroke')

                    # 4. Extract Variables
                    lengths = data_link[:, 1]
                    flows = data_link[:, 2]
                    speeds = np.clip(data_link[:, 3], 10.0, 130.0)
                    prop_gas = data_link[:, 4]
                    prop_pc = data_link[:, 5]
                    prop_4s = data_link[:, 6]
                    prop_ldv = data_link[:, 7]
                    prop_hdv = data_link[:, 8]

                    # Derived
                    prop_dsl = 1.0 - prop_gas
                    prop_moto = np.maximum(0.0, 1.0 - (prop_pc + prop_ldv + prop_hdv))
                    prop_2s = 1.0 - prop_4s

                    # 5. Calculation Loop
                    total_pc = np.zeros_like(lengths)
                    total_ldv = np.zeros_like(lengths)
                    total_hdv = np.zeros_like(lengths)
                    total_moto = np.zeros_like(lengths)

                    prog = st.progress(0)
                    
                    # --- PC ---
                    pc_configs = [(0, prop_gas, eng_cap_gas), (1, prop_dsl, eng_cap_dsl)]
                    cop_classes = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, 
                                   cop.class_ECE_15_03, cop.class_ECE_15_04, cop.class_Improved_Conventional, 
                                   cop.class_Open_loop, cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3, 
                                   cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                    cap_indices = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2, cop.engine_capacity_more_2]

                    for eng_idx, eng_prop, cap_matrix in pc_configs:
                        for k, cap_id in enumerate(cap_indices): 
                            if k >= cap_matrix.shape[1]: continue
                            for c_idx, cls_val in enumerate(cop_classes):
                                if c_idx >= cls_gas.shape[1]: continue
                                if eng_idx == 1 and k == 0 and cls_val <= cop.class_Euro_3: continue 

                                ef = cop.HEFGasolinePassengerCar(cop.pollutant_CO, speeds, cls_val, cap_id) \
                                     if eng_idx == 0 else \
                                     cop.HEFDieselPassengerCar(cop.pollutant_CO, speeds, cls_val, cap_id)
                                
                                class_matrix = cls_gas if eng_idx == 0 else cls_dsl
                                total_pc += ef * eng_prop * cap_matrix[:, k] * class_matrix[:, c_idx]

                    total_pc *= prop_pc * flows
                    prog.progress(0.4)

                    # --- LDV ---
                    ldv_configs = [(0, prop_gas), (1, prop_dsl)]
                    for eng_idx, eng_prop in ldv_configs:
                        class_matrix = cls_gas if eng_idx == 0 else cls_dsl
                        etype = cop.engine_type_gasoline if eng_idx == 0 else cop.engine_type_diesel
                        for c_idx, cls_val in enumerate(cop_classes):
                             if c_idx >= class_matrix.shape[1]: continue
                             ef = cop.HEFLightCommercialVehicle(cop.pollutant_CO, speeds, etype, cls_val)
                             total_ldv += ef * eng_prop * class_matrix[:, c_idx]
                    
                    total_ldv *= prop_ldv * flows
                    prog.progress(0.6)

                    # --- HDV ---
                    hdv_stds = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III, 
                                cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
                    split = 1.0 / len(hdv_stds)
                    for h_std in hdv_stds:
                        ef = cop.HEFHeavyDutyVehicle(cop.pollutant_CO, speeds, 0, 0, h_std)
                        total_hdv += ef * split
                    
                    total_hdv *= prop_hdv * flows
                    prog.progress(0.8)

                    # --- MOTO ---
                    m_types = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                    m_props = [prop_2s, prop_4s]
                    m_matrices = [cls_2s, cls_4s]
                    m_classes = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                                 cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                    
                    for i, mtype in enumerate(m_types):
                        for c_idx, cls_val in enumerate(m_classes):
                            if c_idx >= m_matrices[i].shape[1]: continue
                            ef = cop.EFMotorcycle(cop.pollutant_CO, speeds, mtype, cls_val)
                            total_moto += ef * m_props[i] * m_matrices[i][:, c_idx]

                    total_moto *= prop_moto * flows
                    
                    # --- TOTAL ---
                    total_emissions = total_pc + total_ldv + total_hdv + total_moto
                    prog.progress(1.0)
                    
                    st.session_state.res = {
                        'id': data_link[:, 0],
                        'pc': total_pc, 'ldv': total_ldv, 'hdv': total_hdv, 'moto': total_moto,
                        'total': total_emissions
                    }
                    st.session_state.data_link = data_link
                    
                    st.success(f"✅ Calculation Complete for {N_links} links!")
                    st.metric("Total Emissions", f"{total_emissions.sum():.2f} g/km")

                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab3:
    st.header("Interactive Emission Map")
    if 'res' not in st.session_state:
        st.info("Calculate emissions first.")
    elif f_osm is None:
        st.warning("OSM file missing (Check Defaults or Upload).")
    else:
        st.markdown("Zoom, Pan, and Hover over roads to see emission details.")
        
        # UI for Map
        col1, col2 = st.columns([1, 3])
        with col1:
             st.markdown("**Settings**")
             line_width = st.slider("Line Width", 1, 50, 20)
             cmap_name = st.selectbox("Color Palette", ["jet", "viridis", "inferno", "RdYlGn_r"])
        
        if st.button("Generate Interactive Map", type="primary"):
            with st.spinner("Processing Geometry & Colors..."):
                try:
                    # 1. Parse OSM
                    f_osm.seek(0)
                    coords, osmids, names, types = parse_osm_network(
                        f_osm, x_min, x_max, y_min, y_max, tolerance, ncore
                    )
                    
                    # 2. Match Data
                    res = st.session_state.res
                    lookup = dict(zip(res['id'].astype(int), res['total']))
                    
                    # Prepare Data for PyDeck
                    map_data = []
                    max_val = np.max(res['total']) if len(res['total']) > 0 else 1.0
                    
                    # Colormap setup
                    cmap = cm.get_cmap(cmap_name)
                    norm = mcolors.Normalize(vmin=0, vmax=max_val)

                    for r, oid, name in zip(coords, osmids, names):
                        if oid in lookup:
                            val = lookup[oid]
                            # Get color [R, G, B] (PyDeck needs 0-255)
                            rgba = cmap(norm(val))
                            color = [int(c * 255) for c in rgba[:3]]
                            
                            map_data.append({
                                "path": r,
                                "name": name if name else "Unknown Road",
                                "emission": round(val, 2),
                                "color": color
                            })

                    df_map = pd.DataFrame(map_data)

                    if df_map.empty:
                        st.error("No OSM IDs matched data.")
                    else:
                        # 3. Render PyDeck
                        all_points = np.concatenate([d['path'] for d in map_data])
                        mean_lat = np.mean(all_points[:, 1])
                        mean_lon = np.mean(all_points[:, 0])

                        view_state = pdk.ViewState(
                            latitude=mean_lat,
                            longitude=mean_lon,
                            zoom=12,
                            pitch=0
                        )

                        layer = pdk.Layer(
                            type="PathLayer",
                            data=df_map,
                            pickable=True,
                            get_color="color",
                            width_scale=1,
                            width_min_pixels=2,
                            get_path="path",
                            get_width=line_width,
                        )

                        # Tooltip
                        tooltip = {
                            "html": "<b>Road:</b> {name} <br/> <b>Emission:</b> {emission} g/km",
                            "style": {"backgroundColor": "steelblue", "color": "white"}
                        }

                        r = pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip=tooltip,
                            map_style="mapbox://styles/mapbox/light-v9"
                        )
                        
                        st.pydeck_chart(r)
                        
                        # Legend (Text based for PyDeck)
                        st.caption(f"Color Scale: Low (0) to High ({max_val:.1f} g/km) using {cmap_name}")
                        
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())

with tab4:
    st.header("Download")
    if 'res' in st.session_state:
        res = st.session_state.res
        df = pd.DataFrame({
            'OSM_ID': res['id'].astype(int),
            'Total': res['total'],
            'PC': res['pc'],
            'LDV': res['ldv'],
            'HDV': res['hdv'],
            'Moto': res['moto']
        })
        
        st.download_button("Download CSV", df.to_csv(index=False), "emissions.csv")
    else:
        st.info("Calculate first to enable download.")

with tab5:
    st.markdown("""
    ### How Smart Mode Works
    1. **Global Mix:** Upload files with **1 row**. Applied to all links.
    2. **Local Mix:** Upload files matching **Link Count**. Applied per link.
    3. **Defaults:** If you don't upload a file, the system fetches default Nigerian fleet parameters from GitHub.
    
    ### Required Columns (9)
    `OSM_ID` `Length` `Flow` `Speed` `Gas_Prop` `PC_Prop` `4S_Prop` `LDV_Prop` `HDV_Prop`
    """)

# Footer
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
