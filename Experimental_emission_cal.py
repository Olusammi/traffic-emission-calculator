import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection
import matplotlib.colorbar
import pandas as pd
from io import BytesIO
import zipfile
import tempfile
import os

st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("Built by SHassan")
st.markdown("Upload your input files to calculate and visualize traffic emissions")

# --- PERFORMANCE UTILS (Caching) ---
@st.cache_data(show_spinner=False)
def load_copert_instance(pc_content, ldv_content, hdv_content, moto_content):
    import copert
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name, content in [("pc", pc_content), ("ldv", ldv_content), 
                              ("hdv", hdv_content), ("moto", moto_content)]:
            p = os.path.join(tmpdir, f"{name}.csv")
            with open(p, 'wb') as f: f.write(content)
            paths[name] = p
        
        cop = copert.Copert(paths["pc"], paths["ldv"], paths["hdv"], paths["moto"])
    return cop

@st.cache_data(show_spinner=False)
def parse_osm_network(osm_content, x_min, x_max, y_min, y_max, tolerance, ncore):
    import osm_network
    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
        tmp.write(osm_content)
        tmp_name = tmp.name
    
    try:
        # Define zone polygon
        zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        return osm_network.retrieve_highway(tmp_name, zone, tolerance, int(ncore))
    finally:
        if os.path.exists(tmp_name): os.unlink(tmp_name)

# --- SIDEBAR ---
# Sidebar for file uploads
st.sidebar.header("📂 Upload Input Files")
copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=True)
with copert_files:
    pc_param = st.file_uploader("PC Parameter CSV", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Parameter CSV", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Parameter CSV", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Parameter CSV", type=['csv'], key='moto')

data_files = st.sidebar.expander("Data Files", expanded=True)
with data_files:
    link_osm = st.file_uploader("Link OSM Data (.dat or .csv)", type=['dat','csv','txt'], key='link')
    osm_file = st.file_uploader("OSM Network File (.osm)", type=['osm'], key='osm')

proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
with proportion_files:
    # Using dictionary to ensure we can access these easily later
    prop_uploads = {
        'ecg': st.file_uploader("Engine Capacity Gasoline", type=['dat','txt'], key='ecg'),
        'ecd': st.file_uploader("Engine Capacity Diesel", type=['dat','txt'], key='ecd'),
        'ccg': st.file_uploader("COPERT Class Gasoline", type=['dat','txt'], key='ccg'),
        'ccd': st.file_uploader("COPERT Class Diesel", type=['dat','txt'], key='ccd'),
        '2s': st.file_uploader("2-Stroke Motorcycle", type=['dat','txt'], key='2s'),
        '4s': st.file_uploader("4-Stroke Motorcycle", type=['dat','txt'], key='4s')
    }

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

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Instructions", "📊 Data Preview", "⚙️ Calculate Emissions", "🗺️ Emission Map", "📥 Download Results"])

with tab1:
    st.header("📖 User Guide & Instructions")
    
    # Try to load from GitHub
    instructions_url = "https://raw.githubusercontent.com/Olusammi/traffic-emission-calculator/refs/heads/main/instruction.md"
    
    try:
        import requests
        response = requests.get(instructions_url, timeout=5)
        if response.status_code == 200:
            st.markdown(response.text)
            st.success("✅ Instructions loaded from GitHub")
        else:
            # Fallback to local file
            try:
                with open("instructions.md", "r", encoding="utf-8") as f:
                    st.markdown(f.read())
                st.info("📄 Instructions loaded from local file")
            except FileNotFoundError:
                # Show basic instructions if neither source is available
                st.warning("⚠️ Detailed instructions file not found. Showing basic guide...")
                st.markdown("""
                ## Quick Start Guide
                
                ### 1️⃣ Upload Required Files
                Use the sidebar to upload all necessary files:
                - 4 COPERT parameter CSV files
                - Link OSM data file (9 columns: ID, Len, Flow, Speed, Gas, PC, 4S, LDV, HDV)
                - OSM network file (.osm)
                - 6 vehicle proportion files
                
                ### 2️⃣ Preview Your Data
                Go to "Data Preview" tab to verify your uploaded data looks correct.
                
                ### 3️⃣ Calculate Emissions
                Click the "Calculate Emissions" button and wait for processing to complete.
                
                ### 4️⃣ Visualize Results
                Choose from 3 visualization modes:
                - **Classic**: Original simple view
                - **Enhanced**: Smart labels and better visibility
                - **Custom**: Full control over all settings
                
                ### 5️⃣ Download Results
                Download your emission data as CSV, map as PNG, or complete ZIP archive.
                
                ---
                
                **For detailed instructions**: Place `instructions.md` in the same folder as this app,
                or update the GitHub URL in the code to point to your repository.
                
                **File Format Requirements**:
                - Link OSM data: 9 space-separated columns (OSM_ID, Length_km, Flow, Speed, Gasoline_Prop, PC_Prop, 4Stroke_Prop, LDV_Prop, HDV_Prop)
                - Proportion files: Single column of decimal values (0-1)
                - OSM file: Standard OpenStreetMap XML format
                """)
    except Exception as e:
        # Fallback if requests fails
        try:
            with open("instructions.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
            st.info("📄 Instructions loaded from local file")
        except FileNotFoundError:
            st.warning("⚠️ Could not load instructions. Place instructions.md in the app directory.")
            st.markdown("## Basic Usage\n\n1. Upload all required files\n2. Calculate emissions\n3. Generate visualization\n4. Download results")

with tab2:
    st.header("Data Preview")
    if link_osm is not None:
        st.subheader("Link OSM Data")
        try:
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            
            # Updated to 9 columns structure
            if data_link.shape[1] >= 9:
                data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop','LDV_Prop','HDV_Prop']
            else:
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            
            st.dataframe(data_link.head(20))
            st.info(f"📌 Total links: {len(data_link)} | Columns: {data_link.shape[1]}")
            
            if data_link.shape[1] >= 9:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
                with col2:
                    st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
                with col3:
                    st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")
            else:
                st.warning(f"⚠️ Expected 9 columns but found {data_link.shape[1]}")
        except Exception as e:
            st.error(f"Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar")

with tab3:
    st.header("Calculate Emissions")
    # Collect all required files
    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm] + list(prop_uploads.values())
    all_uploaded = all(f is not None for f in required_files)
    
    if all_uploaded:
        st.success("✅ All required files uploaded!")
        if st.button("🚀 Calculate Emissions", type="primary"):
            with st.spinner("Computing emissions..."):
                try:
                    import copert
                    
                    # 1. Initialize COPERT (using Cached function)
                    cop = load_copert_instance(pc_param.getvalue(), ldv_param.getvalue(), 
                                             hdv_param.getvalue(), moto_param.getvalue())

                    # 2. Prepare Data Arrays (Vectorized)
                    link_osm.seek(0)
                    data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values
                    
                    if data_link.shape[1] < 9:
                        st.error("Error: Link data must have 9 columns [ID, Len, Flow, Speed, Gas, PC, 4S, LDV, HDV]")
                        st.stop()

                    # Helper to load prop files
                    def load_prop(key): 
                        f = prop_uploads[key]; f.seek(0)
                        return np.loadtxt(f)
                    
                    eng_cap_gas = load_prop('ecg')
                    eng_cap_dsl = load_prop('ecd')
                    cls_gas = load_prop('ccg')
                    cls_dsl = load_prop('ccd')
                    cls_2s = load_prop('2s')
                    cls_4s = load_prop('4s')

                    # Extract Columns
                    lengths = data_link[:, 1]
                    flows = data_link[:, 2]
                    speeds = np.clip(data_link[:, 3], 10.0, 130.0)
                    prop_gas = data_link[:, 4]
                    prop_pc = data_link[:, 5]
                    prop_4s = data_link[:, 6]
                    prop_ldv = data_link[:, 7]
                    prop_hdv = data_link[:, 8]

                    # Derived Props
                    prop_dsl = 1.0 - prop_gas
                    prop_moto = np.maximum(0.0, 1.0 - (prop_pc + prop_ldv + prop_hdv))
                    prop_2s = 1.0 - prop_4s

                    # 3. Vectorized Calculations
                    
                    # Initialize accumulators
                    total_pc_emission = np.zeros_like(lengths)
                    total_ldv_emission = np.zeros_like(lengths)
                    total_hdv_emission = np.zeros_like(lengths)
                    total_moto_emission = np.zeros_like(lengths)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # --- PASSENGER CARS ---
                    status_text.text("Calculating PC emissions...")
                    eng_configs = [(0, prop_gas, eng_cap_gas), (1, prop_dsl, eng_cap_dsl)]
                    caps = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                    cop_classes = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, 
                                   cop.class_ECE_15_03, cop.class_ECE_15_04, cop.class_Improved_Conventional, 
                                   cop.class_Open_loop, cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3, 
                                   cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]

                    for eng_idx, eng_prop, cap_dist_matrix in eng_configs:
                        for k, cap_val in enumerate(caps):
                            for c_idx, cls_val in enumerate(cop_classes):
                                if (cls_val != cop.class_Improved_Conventional and cls_val != cop.class_Open_loop) or cap_val <= 2.0:
                                    if eng_idx == 1 and k == 0 and cls_val in range(cop.class_Euro_1, cop.class_Euro_3 + 1):
                                        continue

                                    ef = cop.HEFGasolinePassengerCar(cop.pollutant_CO, speeds, cls_val, cap_val) \
                                         if eng_idx == 0 else \
                                         cop.HEFDieselPassengerCar(cop.pollutant_CO, speeds, cls_val, cap_val)
                                    
                                    class_prop = cls_gas[:, c_idx] if eng_idx == 0 else cls_dsl[:, c_idx]
                                    total_pc_emission += ef * eng_prop * cap_dist_matrix[:, k] * class_prop
                    
                    total_pc_emission = total_pc_emission * prop_pc * flows
                    progress_bar.progress(0.4)

                    # --- LDV ---
                    status_text.text("Calculating LDV emissions...")
                    ldv_eng_types = [(0, prop_gas, cop.engine_type_gasoline), (1, prop_dsl, cop.engine_type_diesel)]
                    for eng_idx, eng_prop, etype in ldv_eng_types:
                        for c_idx, cls_val in enumerate(cop_classes):
                            ef_ldv = cop.HEFLightCommercialVehicle(cop.pollutant_CO, speeds, etype, cls_val)
                            class_prop = cls_gas[:, c_idx] if eng_idx == 0 else cls_dsl[:, c_idx]
                            total_ldv_emission += ef_ldv * eng_prop * class_prop
                    
                    total_ldv_emission = total_ldv_emission * prop_ldv * flows
                    progress_bar.progress(0.6)

                    # --- HDV ---
                    status_text.text("Calculating HDV emissions...")
                    hdv_classes = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III, 
                                   cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
                    hdv_split = 1.0 / len(hdv_classes)
                    
                    for h_cls in hdv_classes:
                        ef_hdv = cop.HEFHeavyDutyVehicle(cop.pollutant_CO, speeds, 0, 0, h_cls)
                        total_hdv_emission += ef_hdv * hdv_split
                    
                    total_hdv_emission = total_hdv_emission * prop_hdv * flows
                    progress_bar.progress(0.8)

                    # --- MOTO ---
                    status_text.text("Calculating Motorcycle emissions...")
                    moto_types = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                    moto_props = [prop_2s, prop_4s]
                    moto_cls_matrices = [cls_2s, cls_4s]
                    moto_classes = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                                    cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                    
                    for m_idx, m_type in enumerate(moto_types):
                        m_prop_link = moto_props[m_idx]
                        for d_idx, d_cls in enumerate(moto_classes):
                            ef_m = cop.EFMotorcycle(cop.pollutant_CO, speeds, m_type, d_cls)
                            cls_prop = moto_cls_matrices[m_idx][:, d_idx] if d_idx < moto_cls_matrices[m_idx].shape[1] else 0
                            total_moto_emission += ef_m * m_prop_link * cls_prop
                    
                    total_moto_emission = total_moto_emission * prop_moto * flows
                    
                    # --- FINAL AGGREGATION ---
                    hot_emission = total_pc_emission + total_ldv_emission + total_hdv_emission + total_moto_emission
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Calculation complete!")
                    
                    # Update Session State
                    st.session_state.hot_emission = hot_emission
                    st.session_state.hot_emission_pc = total_pc_emission
                    st.session_state.hot_emission_ldv = total_ldv_emission
                    st.session_state.hot_emission_hdv = total_hdv_emission
                    st.session_state.hot_emission_m = total_moto_emission
                    st.session_state.data_link = data_link
                    
                    # Display Results
                    st.success("✅ Emissions calculated successfully!")
                    results_df = pd.DataFrame({
                        'OSM_ID': data_link[:, 0].astype(int), 
                        'PC (g/km)': total_pc_emission,
                        'LDV (g/km)': total_ldv_emission,
                        'HDV (g/km)': total_hdv_emission,
                        'Moto (g/km)': total_moto_emission,
                        'Total (g/km)': hot_emission
                    })
                    st.dataframe(results_df)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total PC", f"{total_pc_emission.sum():.2f}")
                    c2.metric("Total LDV", f"{total_ldv_emission.sum():.2f}")
                    c3.metric("Total HDV", f"{total_hdv_emission.sum():.2f}")
                    c4.metric("Total All", f"{hot_emission.sum():.2f}")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please upload all required files")
        missing = []
        file_names = ['PC Parameter', 'LDV Parameter', 'HDV Parameter', 'Moto Parameter', 'Link OSM',
                      'Engine Cap Gas', 'Engine Cap Diesel', 'COPERT Class Gas', 'COPERT Class Diesel', '2-Stroke', '4-Stroke']
        for fname, fdata in zip(file_names, required_files):
            if fdata is None: missing.append(fname)
        st.error(f"Missing: {', '.join(missing)}")

with tab4:
    st.header("Emission Factor Map")
    has_emissions = 'hot_emission' in st.session_state
    if not has_emissions:
        st.warning("⚠️ Please calculate emissions first")
    elif osm_file is None:
        st.warning("⚠️ Please upload OSM network file")
    else:
        st.info("📍 Ready to generate emission map")
        st.subheader("🎨 Visualization Mode")
        viz_mode = st.radio("Select visualization style:", ["Classic (Original)", "Enhanced with Labels", "Custom"], 
                            horizontal=True, help="Classic: Original | Enhanced: Smart labels | Custom: Full control")
        st.markdown("---")
        
        # UI Configuration Logic
        if viz_mode == "Classic (Original)":
            st.markdown("**Classic Mode Settings**")
            col1, col2 = st.columns(2)
            with col1:
                colormap = st.selectbox("Color Map", ['jet','viridis','plasma','RdYlGn_r','hot'], index=0)
                fig_size = st.slider("Figure Size", 8, 16, 10)
            with col2:
                show_roads_without_data = st.checkbox("Show roads without emission data", value=False)
                add_grid = st.checkbox("Add grid lines", value=False)
            line_width_multiplier = 1.0
            show_labels = False
            enhanced_styling = False
            road_transparency = 1.0
            grid_alpha = 0.3
            label_density = "Minimal"
        elif viz_mode == "Enhanced with Labels":
            st.markdown("**Enhanced Mode Settings**")
            col1, col2 = st.columns(2)
            with col1:
                colormap = st.selectbox("Color Map", ['jet','viridis','plasma','RdYlGn_r','hot','coolwarm'], index=0)
                fig_size = st.slider("Figure Size", 8, 16, 12)
                line_width_multiplier = st.slider("Line Width Scale", 0.5, 5.0, 2.0, 0.5)
            with col2:
                label_density = st.selectbox("Road Label Density", ["Minimal (Major roads only)", "Medium (Top 25% emissions)", "High (Top 50% emissions)"], index=1)
                show_roads_without_data = st.checkbox("Show roads without emission data", value=True)
            show_labels = True
            enhanced_styling = True
            add_grid = True
            road_transparency = 0.8
            grid_alpha = 0.2
        else:
            st.markdown("**Custom Mode Settings**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Appearance**")
                colormap = st.selectbox("Color Map", ['jet','viridis','plasma','RdYlGn_r','hot','coolwarm','inferno'], index=0)
                fig_size = st.slider("Figure Size", 8, 20, 12)
                line_width_multiplier = st.slider("Line Width Scale", 0.1, 10.0, 2.0, 0.5)
                enhanced_styling = st.checkbox("Enhanced styling", value=True)
            with col2:
                st.markdown("**Road Display**")
                show_roads_without_data = st.checkbox("Show roads without emission data", value=True)
                road_transparency = st.slider("Road transparency", 0.0, 1.0, 0.8, 0.1)
                add_grid = st.checkbox("Add grid lines", value=True)
                grid_alpha = st.slider("Grid transparency", 0.0, 1.0, 0.2, 0.1) if add_grid else 0.2
            with col3:
                st.markdown("**Labels**")
                show_labels = st.checkbox("Show road labels", value=True)
                if show_labels:
                    label_density = st.selectbox("Label Density", ["Minimal (Major roads only)", "Medium", "High", "Maximum"], index=1)
                else:
                    label_density = "Minimal"

        st.markdown("---")
        if st.button("🗺️ Generate Map", type="primary", use_container_width=True):
            with st.spinner("Generating emission map (Optimized)..."):
                try:
                    hot_emission = st.session_state.hot_emission
                    data_link = st.session_state.data_link
                    
                    # 1. Parse OSM (Cached)
                    osm_file.seek(0)
                    highway_coordinate, highway_osmid, highway_names, highway_types = parse_osm_network(
                        osm_file.read(), x_min, x_max, y_min, y_max, tolerance, ncore
                    )
                    
                    st.text("OSM network parsed successfully!")
                    
                    # 2. Prepare for LineCollection (High Performance Plotting)
                    emission_osm_id = [int(x) for x in data_link[:, 0]]
                    emission_lookup = dict(zip(emission_osm_id, hot_emission))
                    
                    segments_with_data = []
                    emissions_for_segments = []
                    segments_no_data = []
                    
                    # Labels collection
                    labels_to_plot = []
                    
                    # Iterating through network once
                    for refs, osmid, name, htype in zip(highway_coordinate, highway_osmid, highway_names, highway_types):
                        if osmid in emission_lookup:
                            segments_with_data.append(refs)
                            emissions_for_segments.append(emission_lookup[osmid])
                            
                            # Label logic (Simplified for speed)
                            if show_labels and name:
                                labels_to_plot.append((refs, name, htype, emission_lookup[osmid]))
                        else:
                            if show_roads_without_data:
                                segments_no_data.append(refs)
                    
                    # Setup Plot
                    fig = plt.figure(figsize=(fig_size, fig_size - 1), dpi=100)
                    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
                    ax.set_aspect("equal", adjustable="box")
                    
                    max_val = np.max(hot_emission) if len(hot_emission) > 0 else 1.0
                    epsilon = 1e-9
                    
                    # Plot Segments with Data using LineCollection
                    if segments_with_data:
                        # Width calc
                        if viz_mode == "Classic (Original)":
                             # Classic mode implies width proportional to emission
                             # We can't do variable width easily in one LineCollection unless using 'linewidths' array
                             # but matplotlib handles it.
                             lw_min, lw_max = 0.00002, 0.00004
                             widths = [lw_min + (e * (lw_max - lw_min)/(max_val+epsilon)) for e in emissions_for_segments]
                             # Scaling up because LineCollection units are points, classic was data coords. 
                             # We'll stick to a standard viewing width for clarity in new implementation
                             widths = [(w * 20000) for w in widths] # rough adj
                        else:
                            widths = [1.0 * line_width_multiplier + (e/max_val * 2.0 * line_width_multiplier) for e in emissions_for_segments]

                        norm = plt.Normalize(vmin=0, vmax=max_val)
                        lc = LineCollection(segments_with_data, cmap=colormap, norm=norm, alpha=road_transparency)
                        lc.set_array(np.array(emissions_for_segments))
                        lc.set_linewidths(widths)
                        if enhanced_styling: lc.set_capstyle('round')
                        ax.add_collection(lc)
                        
                        # Colorbar
                        ax_c = fig.add_axes([0.85, 0.21, 0.03, 0.53])
                        cb = matplotlib.colorbar.ColorbarBase(ax_c, cmap=plt.cm.get_cmap(colormap), norm=norm, orientation="vertical")
                        cb.set_label("g/km", fontsize=12)
                    
                    # Plot Background Roads
                    if segments_no_data:
                        lc_bg = LineCollection(segments_no_data, colors='gray', linewidths=0.5, alpha=0.3)
                        ax.add_collection(lc_bg)
                    
                    # Plot Labels (Naive implementation for speed)
                    if show_labels and labels_to_plot:
                        # Simple label density filter
                        threshold = np.percentile(hot_emission, 75) if "Medium" in str(label_density) else 0
                        if "Minimal" in str(label_density): threshold = np.percentile(hot_emission, 90)
                        
                        seen_names = set()
                        for refs, name, htype, em in labels_to_plot:
                            if name in seen_names: continue
                            if em < threshold and "Maximum" not in str(label_density): continue
                            
                            mid = len(refs)//2
                            pt = refs[mid]
                            ax.text(pt[0], pt[1], name, fontsize=7, ha='center', va='center',
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                            seen_names.add(name)

                    # Final Formatting
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    if enhanced_styling: ax.set_facecolor('#f0f0f0')
                    ax.set_title("Emission Factor Map", fontsize=14)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    if add_grid: ax.grid(True, alpha=grid_alpha, linestyle='--')
                    
                    st.pyplot(fig)
                    st.session_state.emission_map_fig = fig
                    
                    st.metric("Segments Plotted", len(segments_with_data))
                    st.success("✅ Map generated successfully!")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab5:
    st.header("Download Results")
    st.markdown("### 📊 Available Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emission Data**")
        if 'hot_emission' in st.session_state:
            data_link = st.session_state.data_link
            hot_emission = st.session_state.hot_emission
            # Prepare extended dataframe
            results_df = pd.DataFrame({
                'OSM_ID': data_link[:, 0].astype(int), 
                'Length_km': data_link[:, 1], 
                'PC_g_km': st.session_state.hot_emission_pc,
                'LDV_g_km': st.session_state.hot_emission_ldv,
                'HDV_g_km': st.session_state.hot_emission_hdv,
                'Moto_g_km': st.session_state.hot_emission_m,
                'Total_g_km': hot_emission
            })
            csv = results_df.to_csv(index=False)
            st.download_button(label="⬇️ Download Emission Data CSV", data=csv, file_name="link_hot_emission_factor.csv", mime="text/csv")
        else:
            st.info("Calculate emissions first")
    with col2:
        st.markdown("**Emission Map**")
        if 'emission_map_fig' in st.session_state:
            buf = BytesIO()
            st.session_state.emission_map_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(label="⬇️ Download Map PNG", data=buf, file_name="emission_factor_map.png", mime="image/png")
        else:
            st.info("Generate map first")
    
    st.markdown("---")
    st.markdown("### 📦 Download All Results")
    if 'hot_emission' in st.session_state:
        if st.button("📦 Create ZIP Archive"):
            with st.spinner("Creating ZIP archive..."):
                try:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        csv_data = results_df.to_csv(index=False)
                        zip_file.writestr('link_hot_emission_factor.csv', csv_data)
                        
                        if 'emission_map_fig' in st.session_state:
                            map_buf = BytesIO()
                            st.session_state.emission_map_fig.savefig(map_buf, format='png', dpi=150, bbox_inches='tight')
                            map_buf.seek(0)
                            zip_file.writestr('emission_factor_map.png', map_buf.read())
                        
                        summary = f"""Emission Calculation Summary
==================================

Total Links: {len(hot_emission)}
Total PC Emissions: {st.session_state.hot_emission_pc.sum():.2f} g/km
Total LDV Emissions: {st.session_state.hot_emission_ldv.sum():.2f} g/km
Total HDV Emissions: {st.session_state.hot_emission_hdv.sum():.2f} g/km
Total Moto Emissions: {st.session_state.hot_emission_m.sum():.2f} g/km
Total All Emissions: {hot_emission.sum():.2f} g/km

Map Boundaries:
- Longitude: {x_min} to {x_max}
- Latitude: {y_min} to {y_max}
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
