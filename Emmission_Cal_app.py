import streamlit as st
import numpy as np
import pandas as pd
import requests
import pydeck as pdk
import matplotlib
import matplotlib.pyplot as plt  # <--- FIXED: Added this missing import
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from io import BytesIO
import zipfile
import tempfile
import os

st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("Built by SHassan")
st.markdown("Upload your input files to calculate and visualize traffic emissions")

# ==================== CONFIGURATION: GITHUB DEFAULTS ====================
REPO_USER = "Olusammi"
REPO_NAME = "traffic-emission-calculator"
REPO_BRANCH = "main"
DEFAULT_FOLDER = "default" 
GITHUB_BASE_URL = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{REPO_BRANCH}/{DEFAULT_FOLDER}/"

# Mapping keys to filenames
DEFAULT_FILES_MAP = {
    "pc": "PC_parameter.csv",
    "ldv": "LDV_parameter.csv",
    "hdv": "HDV_parameter.csv",
    "moto": "Moto_parameter.csv",
    "link": "link_osm_with-ldv.dat",
    "osm": "selected_zone-lagos.osm", 
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
    except Exception:
        return None

def get_file_content(uploader_obj, file_key, label_placeholder):
    """
    Returns (file_content_bytes, status_message).
    Priority: Uploaded -> Default -> None
    """
    if uploader_obj is not None:
        label_placeholder.success("📂 Using Uploaded File")
        return uploader_obj
    
    if file_key in DEFAULT_FILES_MAP:
        default_name = DEFAULT_FILES_MAP[file_key]
        content = fetch_default_file(default_name)
        if content:
            label_placeholder.info(f"✅ Using Default: {default_name}")
            return content
            
    label_placeholder.warning("⚠️ No file provided")
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

# --- SIDEBAR UI ---
st.sidebar.header("📂 Upload Input Files")

# 1. COPERT FILES
copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=True)
with copert_files:
    # PC
    pc_up = st.file_uploader("PC Parameter CSV", type=['csv'], key='pc')
    stat_pc = st.empty()
    f_pc = get_file_content(pc_up, 'pc', stat_pc)
    
    # LDV
    ldv_up = st.file_uploader("LDV Parameter CSV", type=['csv'], key='ldv')
    stat_ldv = st.empty()
    f_ldv = get_file_content(ldv_up, 'ldv', stat_ldv)
    
    # HDV
    hdv_up = st.file_uploader("HDV Parameter CSV", type=['csv'], key='hdv')
    stat_hdv = st.empty()
    f_hdv = get_file_content(hdv_up, 'hdv', stat_hdv)
    
    # Moto
    moto_up = st.file_uploader("Moto Parameter CSV", type=['csv'], key='moto')
    stat_moto = st.empty()
    f_moto = get_file_content(moto_up, 'moto', stat_moto)

# 2. DATA FILES
data_files = st.sidebar.expander("Data Files", expanded=True)
with data_files:
    link_up = st.file_uploader("Link OSM Data (.dat or .csv)", type=['dat','csv','txt'], key='link')
    stat_link = st.empty()
    f_link = get_file_content(link_up, 'link', stat_link)
    
    osm_up = st.file_uploader("OSM Network File (.osm)", type=['osm'], key='osm')
    stat_osm = st.empty()
    f_osm = get_file_content(osm_up, 'osm', stat_osm)

# 3. PROPORTION FILES
proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
f_props = {}
with proportion_files:
    def prop_input(label, key):
        u = st.file_uploader(label, key=key)
        s = st.empty()
        return get_file_content(u, key, s)

    f_props['ecg'] = prop_input("Engine Capacity Gasoline", 'ecg')
    f_props['ecd'] = prop_input("Engine Capacity Diesel", 'ecd')
    f_props['ccg'] = prop_input("COPERT Class Gasoline", 'ccg')
    f_props['ccd'] = prop_input("COPERT Class Diesel", 'ccd')
    f_props['2s'] = prop_input("2-Stroke Motorcycle", '2s')
    f_props['4s'] = prop_input("4-Stroke Motorcycle", '4s')

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

# --- MAIN APP LOGIC ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Instructions", "📊 Data Preview", "⚙️ Calculate Emissions", "🗺️ Emission Map", "📥 Download Results"])

with tab1:
    st.header("📖 User Guide & Instructions")
    # (Keeping your original instructions logic)
    instructions_url = "https://raw.githubusercontent.com/Olusammi/traffic-emission-calculator/refs/heads/main/instruction.md"
    try:
        response = requests.get(instructions_url, timeout=5)
        if response.status_code == 200:
            st.markdown(response.text)
        else:
            raise Exception("GitHub fetch failed")
    except Exception:
        try:
            with open("instructions.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.markdown("""
            ## Quick Start
            1. **Upload Files:** Use the sidebar. If you leave a field empty, the system attempts to load a default file from the GitHub repository.
            2. **Calculate:** Go to the Calculate tab.
            3. **Map:** Use the new Interactive Map tab to visualize.
            """)

with tab2:
    st.header("Data Preview")
    if f_link is not None:
        st.subheader("Link OSM Data")
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
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            
            st.dataframe(data_link.head(20))
            st.info(f"📌 Total links: {len(data_link)} | Columns: {data_link.shape[1]}")
            
            if data_link.shape[1] >= 9:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Length", f"{data_link['Length_km'].sum():.2f}")
                c2.metric("Avg Speed", f"{data_link['Speed'].mean():.2f}")
                c3.metric("Avg Flow", f"{data_link['Flow'].mean():.0f}")
            else:
                st.warning(f"⚠️ Expected 9 columns but found {data_link.shape[1]}")
        except Exception as e:
            st.error(f"Error reading Link Data: {e}")
    else:
        st.info("Waiting for Link Data (Upload or Default)...")

with tab3:
    st.header("Calculate Emissions")
    
    # Check if we have the minimum requirements
    ready = all([f_pc, f_ldv, f_hdv, f_moto, f_link]) and all(f_props.values())
    
    if ready:
        if st.button("🚀 Calculate Emissions", type="primary"):
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
                    
                    # 3. SMART LOADER (Flexible Logic)
                    def load_prop_flexible(key, name):
                        f = f_props.get(key)
                        f.seek(0)
                        try:
                            arr = np.loadtxt(f)
                            if len(arr.shape) == 1:
                                arr = arr.reshape(1, -1) # Ensure 2D
                        except:
                            st.error(f"Could not parse {name}")
                            st.stop()

                        rows, cols = arr.shape
                        
                        # Case A: Exact Match
                        if rows == N_links:
                            return arr
                        
                        # Case B: Global (1 row) -> Broadcast
                        elif rows == 1:
                            return np.tile(arr, (N_links, 1))
                        
                        # Case C: Mismatch -> Cycle/Repeat (Safe Fallback)
                        else:
                            st.toast(f"⚠️ {name}: Row count {rows} != Link count {N_links}. Cycling data to fit.", icon="⚠️")
                            # Resize/Tile logic
                            # Calculate how many times to repeat
                            reps = int(np.ceil(N_links / rows))
                            tiled = np.tile(arr, (reps, 1))
                            # Trim to exact length
                            return tiled[:N_links]

                    # Load Proportions
                    eng_cap_gas = load_prop_flexible('ecg', 'Eng Cap Gas')
                    eng_cap_dsl = load_prop_flexible('ecd', 'Eng Cap Diesel')
                    cls_gas = load_prop_flexible('ccg', 'Class Gas')
                    cls_dsl = load_prop_flexible('ccd', 'Class Diesel')
                    cls_2s = load_prop_flexible('2s', 'Moto 2-Stroke')
                    cls_4s = load_prop_flexible('4s', 'Moto 4-Stroke')

                    # 4. Extract Variables
                    lengths = data_link[:, 1]
                    flows = data_link[:, 2]
                    speeds = np.clip(data_link[:, 3], 10.0, 130.0)
                    prop_gas = data_link[:, 4]
                    prop_pc = data_link[:, 5]
                    prop_4s = data_link[:, 6]
                    prop_ldv = data_link[:, 7]
                    prop_hdv = data_link[:, 8]

                    prop_dsl = 1.0 - prop_gas
                    prop_moto = np.maximum(0.0, 1.0 - (prop_pc + prop_ldv + prop_hdv))
                    prop_2s = 1.0 - prop_4s

                    # 5. Calculation
                    total_pc = np.zeros_like(lengths)
                    total_ldv = np.zeros_like(lengths)
                    total_hdv = np.zeros_like(lengths)
                    total_moto = np.zeros_like(lengths)

                    prog = st.progress(0)
                    status_text = st.empty()
                    
                    # --- PC ---
                    status_text.text("Calculating PC...")
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
                    status_text.text("Calculating LDV...")
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
                    status_text.text("Calculating HDV...")
                    hdv_stds = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III, 
                                cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
                    split = 1.0 / len(hdv_stds)
                    for h_std in hdv_stds:
                        ef = cop.HEFHeavyDutyVehicle(cop.pollutant_CO, speeds, 0, 0, h_std)
                        total_hdv += ef * split
                    
                    total_hdv *= prop_hdv * flows
                    prog.progress(0.8)

                    # --- MOTO ---
                    status_text.text("Calculating Moto...")
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
                    status_text.text("✅ Calculation complete!")
                    
                    st.session_state.res = {
                        'id': data_link[:, 0],
                        'pc': total_pc, 'ldv': total_ldv, 'hdv': total_hdv, 'moto': total_moto,
                        'total': total_emissions
                    }
                    st.session_state.data_link = data_link
                    
                    st.success(f"✅ Emissions calculated successfully!")
                    
                    # Results Table
                    results_df = pd.DataFrame({
                        'OSM_ID': data_link[:, 0].astype(int), 
                        'Total (g/km)': total_emissions,
                        'PC': total_pc, 'LDV': total_ldv, 'HDV': total_hdv, 'Moto': total_moto
                    })
                    st.dataframe(results_df.head(100))
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total PC", f"{total_pc.sum():.2f}")
                    c2.metric("Total LDV", f"{total_ldv.sum():.2f}")
                    c3.metric("Total HDV", f"{total_hdv.sum():.2f}")
                    c4.metric("Total All", f"{total_emissions.sum():.2f}")

                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Files missing. Please check sidebar uploads or default file availability.")

with tab4:
    st.header("Emission Factor Map")
    
    if 'res' not in st.session_state:
        st.warning("⚠️ Please calculate emissions first")
    elif f_osm is None:
        st.warning("⚠️ Please upload OSM network file")
    else:
        # 1. PARSE GEOMETRY (The heavy part - Cached)
        if 'map_geo' not in st.session_state:
            with st.spinner("Preparing Map Geometry..."):
                try:
                    f_osm.seek(0)
                    coords, osmids, names, types = parse_osm_network(
                        f_osm, x_min, x_max, y_min, y_max, tolerance, ncore
                    )
                    st.session_state.map_geo = (coords, osmids, names, types)
                except Exception as e:
                    st.error(f"Map Parsing Error: {e}")
                    st.stop()
        
        st.info("📍 Map Geometry Ready. Adjust settings to update visualization live.")
        
        # 2. LIVE MAP SETTINGS
        col1, col2, col3 = st.columns(3)
        with col1:
            line_width = st.slider("Line Width", 1, 30, 10, key='lw_slider')
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8, key='op_slider')
        with col2:
            cmap_name = st.selectbox("Color Palette", ["Reds", "YlOrRd", "plasma", "inferno", "jet"], index=0)
            st.caption("Default: White -> Red (Reds)")
        with col3:
            st.markdown("**View**")
            pitch = st.slider("Pitch (3D)", 0, 60, 0)
            
        # 3. RENDER PYDECK (Fast)
        try:
            coords, osmids, names, types = st.session_state.map_geo
            res = st.session_state.res
            lookup = dict(zip(res['id'].astype(int), res['total']))
            
            # Build DataFrame for Deck
            data_list = []
            max_val = np.max(res['total']) if len(res['total']) > 0 else 1.0
            
            # Map colors
            norm = mcolors.Normalize(vmin=0, vmax=max_val)
            cmap = cm.get_cmap(cmap_name)
            
            # Background Roads (Gray)
            bg_paths = []
            
            for r, oid, name in zip(coords, osmids, names):
                if oid in lookup:
                    val = lookup[oid]
                    rgba = cmap(norm(val))
                    color = [int(c * 255) for c in rgba[:3]] # RGB 0-255
                    
                    data_list.append({
                        "path": r,
                        "name": name if name else "Unknown",
                        "emission": float(f"{val:.2f}"),
                        "color": color
                    })
                else:
                    bg_paths.append({"path": r})
            
            if not data_list:
                st.error("No matching links found.")
            else:
                # Calculate View State
                all_pts = np.concatenate([d['path'] for d in data_list])
                mid_lat = np.mean(all_pts[:,1])
                mid_lon = np.mean(all_pts[:,0])
                
                initial_view = pdk.ViewState(
                    latitude=mid_lat,
                    longitude=mid_lon,
                    zoom=12,
                    pitch=pitch,
                )
                
                # Layers
                bg_layer = pdk.Layer(
                    type="PathLayer",
                    data=bg_paths,
                    get_path="path",
                    get_color=[200, 200, 200, 100], # Faint Gray
                    width_min_pixels=1,
                    get_width=2
                )
                
                data_layer = pdk.Layer(
                    type="PathLayer",
                    data=data_list,
                    pickable=True,
                    get_path="path",
                    get_color="color",
                    opacity=opacity,
                    width_scale=1,
                    width_min_pixels=2,
                    get_width=line_width,
                    cap_rounded=True,
                    joint_rounded=True
                )
                
                tooltip = {
                    "html": "<b>{name}</b><br/>Emission: <b>{emission}</b> g/km",
                    "style": {"backgroundColor": "steelblue", "color": "white", "zIndex": "999"}
                }
                
                deck = pdk.Deck(
                    layers=[bg_layer, data_layer],
                    initial_view_state=initial_view,
                    tooltip=tooltip,
                    map_style="mapbox://styles/mapbox/light-v9"
                )
                
                st.pydeck_chart(deck)
                
                # Legend Gradient (FIXED)
                st.write("---")
                st.markdown(f"**Legend ({cmap_name}):** Low (0) → High ({max_val:.2f})")
                
                fig, ax = plt.subplots(figsize=(6, 0.5))
                cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
                cb.set_label("g/km")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Rendering Error: {e}")
            import traceback
            st.code(traceback.format_exc())

with tab5:
    st.header("Download Results")
    st.markdown("### 📊 Available Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emission Data**")
        if 'res' in st.session_state:
            res = st.session_state.res
            # Prepare extended dataframe
            results_df = pd.DataFrame({
                'OSM_ID': res['id'].astype(int), 
                'PC_g_km': res['pc'],
                'LDV_g_km': res['ldv'],
                'HDV_g_km': res['hdv'],
                'Moto_g_km': res['moto'],
                'Total_g_km': res['total']
            })
            csv = results_df.to_csv(index=False)
            st.download_button(label="⬇️ Download Emission Data CSV", data=csv, file_name="link_hot_emission_factor.csv", mime="text/csv")
        else:
            st.info("Calculate emissions first")
    
    st.markdown("---")
    st.markdown("### 📦 Download All Results")
    if 'res' in st.session_state:
        if st.button("📦 Create ZIP Archive"):
            with st.spinner("Creating ZIP archive..."):
                try:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        csv_data = results_df.to_csv(index=False)
                        zip_file.writestr('link_hot_emission_factor.csv', csv_data)
                        
                        summary = f"""Emission Calculation Summary
==================================
Total All Emissions: {res['total'].sum():.2f} g/km
"""
                        zip_file.writestr('summary.txt', summary)
                    
                    zip_buffer.seek(0)
                    st.download_button(label="⬇️ Download Complete Results (ZIP)", data=zip_buffer, 
                                       file_name="emission_results.zip", mime="application/zip")
                    st.success("✅ ZIP archive created successfully!")
                except Exception as e:
                    st.error(f"Error creating ZIP: {e}")
    else:
        st.info("Calculate emissions first to create ZIP archive")

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
