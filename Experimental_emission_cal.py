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
import sys
import os

# --- Mock COPERT Module (Essential for running the app) import streamlit as st
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
    engine_cap_gas = st.file_uploader("Engine Capacity Gasoline", type=['dat','txt'], key='ecg')
    engine_cap_diesel = st.file_uploader("Engine Capacity Diesel", type=['dat','txt'], key='ecd')
    copert_class_gas = st.file_uploader("COPERT Class Gasoline", type=['dat','txt'], key='ccg')
    copert_class_diesel = st.file_uploader("COPERT Class Diesel", type=['dat','txt'], key='ccd')
    copert_2stroke = st.file_uploader("2-Stroke Motorcycle", type=['dat','txt'], key='2s')
    copert_4stroke = st.file_uploader("4-Stroke Motorcycle", type=['dat','txt'], key='4s')

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
                - Link OSM data file (7 columns)
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
                - Link OSM data: 7 space-separated columns (OSM_ID, Length_km, Flow, Speed, Gasoline_Prop, PC_Prop, 4Stroke_Prop)
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
            if data_link.shape[1] >= 7:
                data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop']
            else:
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            st.dataframe(data_link.head(20))
            st.info(f"📌 Total links: {len(data_link)} | Columns: {data_link.shape[1]}")
            if data_link.shape[1] >= 7:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
                with col2:
                    st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
                with col3:
                    st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")
            else:
                st.warning(f"⚠️ Expected 7 columns but found {data_link.shape[1]}")
        except Exception as e:
            st.error(f"Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar")

with tab3:
    st.header("Calculate Emissions")
    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    all_uploaded = all(f is not None for f in required_files)
    
    if all_uploaded:
        st.success("✅ All required files uploaded!")
        if st.button("🚀 Calculate Emissions", type="primary"):
            with st.spinner("Computing emissions..."):
                try:
                    import copert
                    import tempfile, os
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
                        link_osm.seek(0)
                        engine_cap_gas.seek(0)
                        engine_cap_diesel.seek(0)
                        copert_class_gas.seek(0)
                        copert_class_diesel.seek(0)
                        copert_2stroke.seek(0)
                        copert_4stroke.seek(0)
                        
                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values
                        data_engine_capacity_gasoline = np.loadtxt(engine_cap_gas)
                        data_engine_capacity_diesel = np.loadtxt(engine_cap_diesel)
                        data_copert_class_gasoline = np.loadtxt(copert_class_gas)
                        data_copert_class_diesel = np.loadtxt(copert_class_diesel)
                        data_copert_class_motorcycle_two_stroke = np.loadtxt(copert_2stroke)
                        data_copert_class_motorcycle_four_stroke = np.loadtxt(copert_4stroke)
                        
                        engine_type = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        engine_type_m = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        engine_capacity = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        copert_class = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03,
                                        cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop, cop.class_Euro_1,
                                        cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                        Nclass = len(copert_class)
                        copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                                                   cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        Mclass = len(copert_class_motorcycle)
                        
                        Nlink = data_link.shape[0]
                        hot_emission_pc = np.zeros((Nlink,), dtype=float)
                        hot_emission_m = np.zeros((Nlink,), dtype=float)
                        hot_emission = np.zeros((Nlink,), dtype=float)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(Nlink):
                            if i % max(1, Nlink // 100) == 0:
                                progress_bar.progress(i / Nlink)
                            status_text.text(f"Processing link {i+1}/{Nlink}")
                            
                            link_length = data_link[i, 1]
                            link_flow = data_link[i, 2]
                            v = min(max(10., data_link[i, 3]), 130.)
                            link_gasoline_proportion = data_link[i, 4]
                            link_pc_proportion = data_link[i, 5]
                            link_4_stroke_proportion = data_link[i, 6]
                            p_passenger = link_gasoline_proportion
                            P_motorcycle = 1. - link_pc_proportion
                            engine_type_distribution = [link_gasoline_proportion, 1. - link_gasoline_proportion]
                            engine_capacity_distribution = [data_engine_capacity_gasoline[i], data_engine_capacity_diesel[i]]
                            engine_type_motorcycle_distribution = [link_4_stroke_proportion, 1. - link_4_stroke_proportion]
                            
                            for t in range(2):
                                for c in range(Nclass):
                                    for k in range(2):
                                        if (copert_class[c] != cop.class_Improved_Conventional and copert_class[c] != cop.class_Open_loop) or engine_capacity[k] <= 2.0:
                                            if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                continue
                                            e = cop.Emission(cop.pollutant_CO, v, link_length, cop.vehicle_type_passenger_car, engine_type[t], copert_class[c], engine_capacity[k], 28.2)
                                            e *= engine_type_distribution[t] * engine_capacity_distribution[t][k]
                                            hot_emission_pc[i] += e * p_passenger / link_length * link_flow
                            
                            for m in range(2):
                                for d in range(Mclass):
                                    if m == 1 and copert_class_motorcycle[d] in range(cop.class_moto_Conventional, 1 + cop.class_moto_Euro_5):
                                        continue
                                    e_f = cop.EFMotorcycle(cop.pollutant_CO, v, engine_type_m[m], copert_class_motorcycle[d])
                                    e_f *= engine_type_motorcycle_distribution[m]
                                    hot_emission_m[i] += e_f * P_motorcycle * link_flow
                            hot_emission[i] = hot_emission_m[i] + hot_emission_pc[i]
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Calculation complete!")
                        st.session_state.hot_emission = hot_emission
                        st.session_state.hot_emission_pc = hot_emission_pc
                        st.session_state.hot_emission_m = hot_emission_m
                        st.session_state.data_link = data_link
                    
                    st.success("✅ Emissions calculated successfully!")
                    results_df = pd.DataFrame({'OSM_ID': data_link[:, 0].astype(int), 'Hot_Emission_PC (g/km)': hot_emission_pc,
                                               'Hot_Emission_Motorcycle (g/km)': hot_emission_m, 'Total_Emission (g/km)': hot_emission})
                    st.dataframe(results_df)
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Total PC Emissions", f"{hot_emission_pc.sum():.2f} g/km")
                    with col2: st.metric("Total Motorcycle Emissions", f"{hot_emission_m.sum():.2f} g/km")
                    with col3: st.metric("Total Emissions", f"{hot_emission.sum():.2f} g/km")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please upload all required files **[Download Here](https://drive.google.com/drive/folders/1KCu8y-mZ0XtBc6icFlvPnJMxLFM7YCKY?usp=sharing)** ")
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
            label_density = "Minimal (Major roads only)"
            rotate_labels = False
            enhanced_styling = False
            road_transparency = 1.0
            grid_alpha = 0.3
            label_font_size = 7
            min_label_distance = 0.002
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
                rotate_labels = st.checkbox("Rotate labels along roads", value=True)
            show_labels = True
            enhanced_styling = True
            add_grid = True
            road_transparency = 0.8
            grid_alpha = 0.2
            label_font_size = 7
            min_label_distance = 0.002
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
                    label_density = st.selectbox("Label Density", ["Minimal (Major roads only)", "Medium (Top 25% emissions)", 
                                                  "High (Top 50% emissions)", "Maximum (All named roads)"], index=1)
                    rotate_labels = st.checkbox("Rotate labels along roads", value=True)
                    label_font_size = st.slider("Label font size", 4, 12, 7)
                    min_label_distance = st.slider("Min distance between labels", 0.001, 0.01, 0.002, 0.001)
                else:
                    label_density = "Minimal (Major roads only)"
                    rotate_labels = False
                    label_font_size = 7
                    min_label_distance = 0.002
        
        st.markdown("---")
        if st.button("🗺️ Generate Map", type="primary", use_container_width=True):
            with st.spinner("Generating emission map..."):
                try:
                    import osm_network
                    hot_emission = st.session_state.hot_emission
                    data_link = st.session_state.data_link
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                        osm_file.seek(0)
                        tmp.write(osm_file.read())
                        osm_path = tmp.name
                    try:
                        selected_zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]]
                        selected_zone.append(selected_zone[0])
                        status_text = st.empty()
                        status_text.text("Parsing OSM network...")
                        highway_coordinate, highway_osmid, highway_names, highway_types = osm_network.retrieve_highway(osm_path, selected_zone, tolerance, int(ncore))
                        status_text.text("OSM network parsed successfully!")
                        max_emission_value = np.max(hot_emission)
                        epsilon = 1e-9
                        if viz_mode == "Classic (Original)":
                            lw_max = 0.00004
                            lw_min = 0.00002
                            width_scaling = (lw_max - lw_min) / (max_emission_value + epsilon) + lw_min
                            lw_nodata = 0.003
                        else:
                            lw_max = 3.0 * line_width_multiplier
                            lw_min = 0.5 * line_width_multiplier
                            width_scaling = (lw_max - lw_min) / (max_emission_value + epsilon)
                            lw_nodata = 0.3
                        color_scale = colors.Normalize(vmin=0, vmax=max_emission_value + epsilon)
                        scale_map = cmx.ScalarMappable(norm=color_scale, cmap=colormap)
                        emission_osm_id = [int(x) for x in data_link[:, 0]]
                        fig = plt.figure(figsize=(fig_size, fig_size - 1), dpi=100)
                        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
                        ax.set_aspect("equal", adjustable="box")
                        ax_c = fig.add_axes([0.85, 0.21, 0.03, 0.53])
                        cb = matplotlib.colorbar.ColorbarBase(ax_c, cmap=plt.cm.get_cmap(colormap), norm=color_scale, orientation="vertical")
                        cb.set_label("g/km", fontsize=12)
                        if enhanced_styling:
                            ax.set_facecolor('#f0f0f0')
                        status_text.text("Plotting emission data on map...")
                        roads_with_data = 0
                        roads_without_data = 0
                        for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid, highway_names, highway_types):
                            try:
                                i = emission_osm_id.index(osmid)
                            except:
                                i = None
                            if i is not None:
                                current_emission = hot_emission[i]
                                color_value = scale_map.to_rgba(current_emission)
                                if viz_mode == "Classic (Original)":
                                    line_width = current_emission * width_scaling
                                else:
                                    line_width = lw_min + (current_emission * width_scaling)
                                plot_kwargs = {'color': color_value, 'lw': line_width, 'alpha': road_transparency}
                                if enhanced_styling:
                                    plot_kwargs['solid_capstyle'] = 'round'
                                ax.plot([x[0] for x in refs], [x[1] for x in refs], **plot_kwargs)
                                roads_with_data += 1
                            else:
                                if show_roads_without_data:
                                    if viz_mode == "Classic (Original)":
                                        ax.plot([x[0] for x in refs], [x[1] for x in refs], "k-", lw=lw_nodata)
                                    else:
                                        ax.plot([x[0] for x in refs], [x[1] for x in refs], "gray", lw=lw_nodata, alpha=0.3)
                                    roads_without_data += 1
                        if show_labels and viz_mode != "Classic (Original)":
                            labeled_roads = {}
                            major_road_types = ['motorway', 'trunk', 'primary', 'secondary']
                            if label_density == "Minimal (Major roads only)":
                                emission_percentile = 90
                                major_only = True
                            elif label_density == "Medium (Top 25% emissions)":
                                emission_percentile = 75
                                major_only = False
                            elif label_density == "High (Top 50% emissions)":
                                emission_percentile = 50
                                major_only = False
                            else:
                                emission_percentile = 0
                                major_only = False
                            emission_threshold = np.percentile(hot_emission, emission_percentile)
                            for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid, highway_names, highway_types):
                                try:
                                    i = emission_osm_id.index(osmid)
                                    current_emission = hot_emission[i]
                                except:
                                    continue
                                if major_only:
                                    should_label = name and highway_type in major_road_types
                                else:
                                    should_label = name and (highway_type in major_road_types or current_emission >= emission_threshold)
                                if should_label:
                                    center_index = len(refs) // 2
                                    x_center = refs[center_index][0]
                                    y_center = refs[center_index][1]
                                    too_close = False
                                    if name in labeled_roads:
                                        for prev_x, prev_y in labeled_roads[name]:
                                            distance = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                                            if distance < min_label_distance:
                                                too_close = True
                                                break
                                    if not too_close:
                                        angle = 0
                                        if rotate_labels and len(refs) > 1:
                                            dx = refs[min(center_index + 1, len(refs) - 1)][0] - refs[max(center_index - 1, 0)][0]
                                            dy = refs[min(center_index + 1, len(refs) - 1)][1] - refs[max(center_index - 1, 0)][1]
                                            angle = np.degrees(np.arctan2(dy, dx))
                                            if angle > 90:
                                                angle -= 180
                                            elif angle < -90:
                                                angle += 180
                                        ax.text(x_center, y_center, str(name), fontsize=label_font_size, color='black', ha='center', va='center',
                                                rotation=angle, rotation_mode='anchor', bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                                                linewidth=0.5, boxstyle='round,pad=0.3'), zorder=100)
                                        if name not in labeled_roads:
                                            labeled_roads[name] = []
                                        labeled_roads[name].append((x_center, y_center))
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        if viz_mode == "Classic (Original)":
                            ax.set_title("Emission Factor Map", fontsize=14)
                        else:
                            ax.set_title("Emission Factor Map with Road Names", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Longitude", fontsize=12)
                        ax.set_ylabel("Latitude", fontsize=12)
                        if add_grid:
                            ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)
                        st.pyplot(fig)
                        st.session_state.emission_map_fig = fig
                        if show_labels and viz_mode != "Classic (Original)":
                            col1, col2, col3 = st.columns(3)
                            with col1: st.metric("Roads with Emission Data", roads_with_data)
                            with col2: st.metric("Roads without Data", roads_without_data)
                            with col3: st.metric("Unique Road Names Labeled", len(labeled_roads))
                        else:
                            col1, col2 = st.columns(2)
                            with col1: st.metric("Roads with Emission Data", roads_with_data)
                            with col2: st.metric("Roads without Data", roads_without_data)
                        status_text.empty()
                        st.success("✅ Map generated successfully!")
                    finally:
                        if os.path.exists(osm_path):
                            os.unlink(osm_path)
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
            results_df = pd.DataFrame({'OSM_ID': data_link[:, 0].astype(int), 'Length_km': data_link[:, 1], 'Emission_g_km': hot_emission})
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
                        data_link = st.session_state.data_link
                        hot_emission = st.session_state.hot_emission
                        hot_emission_pc = st.session_state.hot_emission_pc
                        hot_emission_m = st.session_state.hot_emission_m
                        results_df = pd.DataFrame({
                            'OSM_ID': data_link[:, 0].astype(int),
                            'Length_km': data_link[:, 1],
                            'Hot_Emission_PC_g_km': hot_emission_pc,
                            'Hot_Emission_Motorcycle_g_km': hot_emission_m,
                            'Total_Emission_g_km': hot_emission
                        })
                        csv_data = results_df.to_csv(index=False)
                        zip_file.writestr('link_hot_emission_factor.csv', csv_data)
                        if 'emission_map_fig' in st.session_state:
                            map_buf = BytesIO()
                            st.session_state.emission_map_fig.savefig(map_buf, format='png', dpi=150, bbox_inches='tight')
                            map_buf.seek(0)
                            zip_file.writestr('emission_factor_map.png', map_buf.read())
                        summary = f"""Emission Calculation Summary
==================================

Total Links Analyzed: {len(hot_emission)}
Total PC Emissions: {hot_emission_pc.sum():.2f} g/km
Total Motorcycle Emissions: {hot_emission_m.sum():.2f} g/km
Total Emissions: {hot_emission.sum():.2f} g/km

Average Emission per Link: {hot_emission.mean():.2f} g/km
Maximum Emission: {hot_emission.max():.2f} g/km
Minimum Emission: {hot_emission.min():.2f} g/km

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

pollutant = st.sidebar.selectbox(
       "Select Pollutant",
       ["CO", "CO2", "NOx", "PM2.5"],
       help="Choose which pollutant to calculate"
   )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Instructions:**")
st.sidebar.markdown("""
1. Upload all COPERT parameter files
2. Upload link OSM data (7 columns)
3. Upload proportion data files
4. Upload OSM network file
5. Configure map parameters
6. Calculate emissions
7. Choose visualization mode
8. Adjust settings and generate map
9. Download results
""")
st.sidebar.info("Built with Streamlit by SHassan 🎈")








---
# Since the actual 'copert' module is not provided, this mock
# is necessary to allow the Streamlit app to run without errors.
# In a real environment, the user would replace this with the actual import.
class MockCopert:
    # Pollutant Constants (Crucial: These are placeholders based on typical COPERT structure)
    pollutant_CO = 1
    pollutant_CO2 = 2
    pollutant_NOx = 3
    pollutant_PM2p5 = 4

    # Vehicle Type Constants
    vehicle_type_passenger_car = 10
    vehicle_type_light_duty_vehicle = 20
    vehicle_type_heavy_duty_vehicle = 30
    vehicle_type_motorcycle = 40

    # Mock Data Holders for EF (Emission Factor) parameters
    PC_data = None
    LDV_data = None
    HDV_data = None
    MOTO_data = None

    def __init__(self, pc_param, ldv_param, hdv_param, moto_param):
        # In a real implementation, these CSVs would be loaded and parsed
        # into internal COPERT data structures (like tables or interpolation functions).
        try:
            self.PC_data = pd.read_csv(pc_param) if pc_param else None
            self.LDV_data = pd.read_csv(ldv_param) if ldv_param else None
            self.HDV_data = pd.read_csv(hdv_param) if hdv_param else None
            self.MOTO_data = pd.read_csv(moto_param) if moto_param else None
        except Exception as e:
            st.error(f"Error loading mock COPERT parameter file: {e}")
            raise

    # Mock Emission Function - Simulates COPERT's main EF calculation
    def Emission(self, pollutant_const, speed_kmh, link_length_km, vehicle_type_const, 
                 engine_type, copert_class, engine_capacity_cc, ambient_temp_c):
        # This is a critical mock. It simulates an EF calculation (g/km)
        # and then scales it by link flow and distance.
        # Hot Emission = EF_Hot * Distance * Flow
        
        # A simple, mock EF based on speed and speed and temp (for demonstration only)
        # Actual COPERT uses complex lookups and polynomials.
        base_ef = 0.5 + 0.01 * speed_kmh + 0.05 * (ambient_temp_c - 20)
        
        if pollutant_const == self.pollutant_CO2:
            base_ef *= 1.5 # CO2 is typically much higher
        elif pollutant_const == self.pollutant_NOx:
            base_ef *= 0.8
        elif pollutant_const == self.pollutant_PM2p5:
            base_ef *= 0.1 # PM is typically very low
        
        # Assume 1 unit of flow for mock output
        flow_units = 1.0 
        
        hot_emission_g = base_ef * link_length_km * flow_units
        return hot_emission_g

    # Mock Motorcycle EF function (Simplified)
    def EFMotorcycle(self, pollutant_const, speed_kmh, engine_type, copert_class):
        base_ef = 0.3 + 0.005 * speed_kmh
        if pollutant_const == self.pollutant_CO:
            base_ef *= 1.2
        return base_ef

# Mock Copert Class End
# ---

# --- Utility Functions ---

def load_data(file_uploader):
    """Loads CSV/DAT data from Streamlit uploader, supporting space delimiters."""
    if file_uploader:
        try:
            # Read a chunk to sniff delimiter
            content = file_uploader.read().decode('utf-8')
            file_uploader.seek(0)
            
            # Priority 1: Check for common single-character delimiters
            sep = None
            if '\t' in content:
                sep = '\t'
            elif ';' in content:
                sep = ';'
            elif ',' in content:
                sep = ','

            # Determine read strategy
            if sep is not None:
                # Use detected single-character separator
                df = pd.read_csv(file_uploader, sep=sep, header=None, on_bad_lines='skip', encoding='utf-8')
            else:
                # Fallback (and user-requested) to robust whitespace separation (\s+)
                # This handles files where columns are separated by one or more spaces
                try:
                    df = pd.read_csv(file_uploader, sep='\s+', header=None, on_bad_lines='skip', 
                                     engine='python', skipinitialspace=True, encoding='utf-8')
                except Exception as e:
                    st.warning(f"Failed to read file using robust whitespace separator: {e}. Attempting default comma read.")
                    file_uploader.seek(0)
                    df = pd.read_csv(file_uploader, header=None, on_bad_lines='skip', encoding='utf-8')


            # Handle single column files which are often read as a single column with no header
            if df.shape[1] > 0:
                return df
            else:
                st.error("Could not parse file content. Please check delimiter or ensure file is not empty.")
                return None
        except Exception as e:
            st.error(f"Error reading file {file_uploader.name}: {e}")
            return None
    return None

def load_osm_xml(osm_file):
    """Loads the OSM XML file content (without parsing complex network structure)."""
    if osm_file:
        try:
            return osm_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading OSM file: {e}")
            return None
    return None

def get_pollutant_constant(pollutant_name, cop):
    """Maps user-friendly pollutant name to the COPERT constant."""
    if pollutant_name == "CO": return cop.pollutant_CO
    elif pollutant_name == "CO2": return cop.pollutant_CO2
    elif pollutant_name == "NOx": return cop.pollutant_NOx
    elif pollutant_name == "PM2.5": return cop.pollutant_PM2p5
    return cop.pollutant_CO # Default fallback

# --- Core Calculation Function ---

def calculate_emissions(data_link, copert_instance, pollutant_str, ambient_temp_c, st_container):
    """Performs the full COPERT emission calculation."""
    
    # Check for core data readiness
    if data_link is None or copert_instance is None:
        st_container.error("Calculation setup incomplete. Check file uploads.")
        return None, None
        
    # CRITICAL: Ensure the expected number of columns are present (min 7)
    if data_link.shape[1] < 7:
        st_container.error(f"Link OSM Data must have at least 7 columns, but only {data_link.shape[1]} were found. Please check your file format and delimiter.")
        return None, None

    # Determine pollutant constant
    pollutant_const = get_pollutant_constant(pollutant_str, copert_instance)
    
    st_container.markdown(f"### Running Emission Model for: **{pollutant_str}**")
    st_container.caption(f"Ambient Temperature set to: **{ambient_temp_c}°C**")

    # Access the columns from the loaded link data using .iloc (0-indexed column access)
    try:
        # Assuming the link data (data_link) has the following structure (0-indexed):
        # 0: Link ID
        # 1: X start (Longitude)
        # 2: Y start (Latitude)
        # 3: X end (Longitude)
        # 4: Y end (Latitude)
        # 5: Flow or Flow related data (unused in mock EF but expected in structure)
        # 6: Speed (km/h) - CRITICAL INPUT FOR EF
        
        # Link Geometry & Traffic
        x_s, y_s = data_link.iloc[:, 1], data_link.iloc[:, 2] # Start Coords (X, Y)
        x_e, y_e = data_link.iloc[:, 3], data_link.iloc[:, 4] # End Coords (X, Y)
        speeds = data_link.iloc[:, 6] # Speed in km/h

        # Mocking the length calculation. In a real scenario, this would be calculated
        # from coordinates or provided as a dedicated column.
        link_lengths_km = np.ones(len(data_link)) * 0.5 # Mock length 0.5 km

        Nlink = len(data_link)
        hot_emission = np.zeros(Nlink)
        hot_emission_m = np.zeros(Nlink) # Motorcycle emissions

        progress_bar = st_container.progress(0, text="Initializing calculation...")
        # Define a stepping threshold to avoid excessive Streamlit updates
        step_size = max(1, Nlink // 100) 
        
        # Iteration over all links
        for i in range(Nlink):
            v = speeds.iloc[i] # Link speed
            link_length = link_lengths_km[i]
            
            # --- Mocking PC Calculation ---
            # In a real app, proportions (prop_g1, prop_d2, etc.) would be used here
            # to determine the weight of each vehicle category (e.g., Euro class, engine size).
            mock_engine_type = "Gasoline" 
            mock_copert_class = "EURO 4"
            mock_engine_capacity = "1400-2000" # cc

            # Calculate Passenger Car EF and total emission
            e_pc = copert_instance.Emission(
                pollutant_const, v, link_length, copert_instance.vehicle_type_passenger_car, 
                mock_engine_type, mock_copert_class, mock_engine_capacity, ambient_temp_c
            )
            hot_emission[i] += e_pc

            # --- Mocking Motorcycle Calculation ---
            mock_engine_type_m = "4-Stroke"
            mock_copert_class_m = "MC>50"
            
            e_moto_factor = copert_instance.EFMotorcycle(
                pollutant_const, v, mock_engine_type_m, mock_copert_class_m
            )
            # Emission = EF * Distance * Flow
            e_moto = e_moto_factor * link_length * 1.0 # Mock flow=1.0
            hot_emission_m[i] += e_moto
            hot_emission[i] += e_moto
            
            # Update progress bar only when stepping past the threshold
            if i % step_size == 0 or i == Nlink - 1:
                progress_bar.progress((i + 1) / Nlink, text=f"Processing link {i+1} of {Nlink}...")

        progress_bar.empty() # Clear the progress bar after completion
        st_container.success("Calculation complete! Results are ready.")

        # Create a DataFrame for visualization (including coordinates)
        results_df = pd.DataFrame({
            'x_s': x_s.astype(float), 'y_s': y_s.astype(float), 
            'x_e': x_e.astype(float), 'y_e': y_e.astype(float),
            'hot_emission': hot_emission,
            'hot_emission_m': hot_emission_m
        })
        
        # Determine map bounds
        x_min, x_max = min(x_s.min(), x_e.min()), max(x_s.max(), x_e.max())
        y_min, y_max = min(y_s.min(), y_e.min()), max(y_s.max(), y_e.max())
        
        return results_df, (x_min, x_max, y_min, y_max)

    except Exception as e:
        # Clear progress bar on error
        try:
            progress_bar.empty()
        except:
            pass
        st_container.error(f"A critical error occurred during calculation. Please ensure columns [1] to [4] and [6] contain valid numeric data. Error: {e}")
        return None, None
        
# --- Streamlit App Layout ---

st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("A COPERT-based emission model using Streamlit")

# --- Initialize Session State ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'map_bounds' not in st.session_state:
    st.session_state.map_bounds = None
if 'pollutant' not in st.session_state:
    st.session_state.pollutant = 'CO2' # Default to CO2 for global warming focus

# --- Sidebar (Improved UI/UX) ---
st.sidebar.header("⚙️ Configuration & Data Upload")

# 1. Critical Inputs
st.sidebar.markdown("### 🧪 Pollutant & Environment")
st.session_state.pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["CO2", "CO", "NOx", "PM2.5"],
    index=0,
    key='pollutant_select',
    help="Choose the pollutant (g/km) to calculate and visualize."
)

ambient_temp_c = st.sidebar.slider(
    "Ambient Temperature (°C)",
    min_value=-10.0, max_value=40.0, value=28.2, step=0.1,
    help="Ambient temperature affects 'Hot Emission' factors in COPERT."
)

# 2. COPERT Parameter Files
st.sidebar.markdown("### 📄 COPERT Emission Factor Files")
copert_files = st.sidebar.expander("COPERT Parameter Files (CSV/TXT)", expanded=False)
with copert_files:
    pc_param = st.file_uploader("Passenger Car (PC) Parameter", type=['csv','txt'], key='pc')
    ldv_param = st.file_uploader("Light Duty Vehicle (LDV) Parameter", type=['csv','txt'], key='ldv')
    hdv_param = st.file_uploader("Heavy Duty Vehicle (HDV) Parameter", type=['csv','txt'], key='hdv')
    moto_param = st.file_uploader("Motorcycle (Moto) Parameter", type=['csv','txt'], key='moto')

# 3. Traffic and Proportion Data
st.sidebar.markdown("### 🛣️ Traffic & Fleet Data")
data_files = st.sidebar.expander("Traffic & Proportion Data", expanded=True)
with data_files:
    # Traffic Link Data
    link_osm_desc = "Link OSM Data (Expected Columns: [0] ID, [1] X_s, [2] Y_s, [3] X_e, [4] Y_e, [5] Flow, [6] Speed)"
    link_osm = st.file_uploader(link_osm_desc, type=['dat','csv','txt'], key='link')
    
    # OSM Network File (For map visualization)
    osm_file = st.file_uploader("OSM Network File (.osm XML)", type=['osm'], key='osm')
    
    # Proportion Data (Highly Region-Specific)
    prop_files = st.expander("Vehicle Fleet Proportion Files (1-column array)", expanded=False)
    with prop_files:
        st.caption("These files define the distribution of the fleet (e.g., Euro Class, Fuel Type).")
        # --- Adjusted to include '.dat' ---
        prop_g1 = st.file_uploader("Engine Capacity Gasoline (e.g., <1.4L)", type=['dat','csv','txt'], key='g1')
        prop_d2 = st.file_uploader("COPERT Class Diesel (e.g., EURO 4)", type=['dat','csv','txt'], key='d2')
        prop_h1 = st.file_uploader("HDV Class Distribution", type=['dat','csv','txt'], key='h1')
        prop_h2 = st.file_uploader("HDV Load Distribution", type=['dat','csv','txt'], key='h2')
        prop_m1 = st.file_uploader("Motorcycle 2-Stroke Proportion", type=['dat','csv','txt'], key='m1')
        prop_m2 = st.file_uploader("Motorcycle 4-Stroke Proportion", type=['dat','csv','txt'], key='m2')
        # -----------------------------------

# --- Main Tabs ---

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧪 Model Setup", "⚙️ Calculate Emissions", "🗺️ Visualize Map"])

# Tab 1: Data Overview
with tab1:
    st.header("Uploaded Data Status")
    
    link_data = load_data(link_osm)
    
    if link_data is not None:
        st.success(f"Link OSM Data loaded: {len(link_data)} links found with {link_data.shape[1]} columns.")
        st.subheader("First 5 Rows of Link Data (0-Indexed)")
        st.dataframe(link_data.head(5))
        st.caption("Ensure your columns match: **[1] X_s, [2] Y_s, [3] X_e, [4] Y_e, [6] Speed**")
    else:
        st.info("Please upload your Link OSM Data in the sidebar to view a preview.")


# Tab 2: Model Setup
with tab2:
    st.header("COPERT Model Initialization")
    
    required_copert = [pc_param, ldv_param, hdv_param, moto_param]
    copert_ready = all(required_copert)
    
    if copert_ready:
        try:
            # Initialize the mock COPERT instance
            cop = MockCopert(pc_param, ldv_param, hdv_param, moto_param)
            st.success("✅ COPERT model base parameters loaded successfully.")
            st.session_state.copert_instance = cop
            st.caption("Note: This app uses a mock COPERT implementation. For accurate results, replace the 'MockCopert' class with your actual COPERT library integration.")
        except Exception as e:
             st.error(f"❌ Error during COPERT initialization: {e}")
             st.session_state.copert_instance = None
    else:
        st.warning("⚠️ Waiting for all four COPERT Parameter Files to be uploaded.")
        missing_files = []
        if not pc_param: missing_files.append("PC Parameter")
        if not ldv_param: missing_files.append("LDV Parameter")
        if not hdv_param: missing_files.append("HDV Parameter")
        if not moto_param: missing_files.append("Motorcycle Parameter")
        st.markdown(f"**Missing Files:** {', '.join(missing_files)}")

# Tab 3: Calculate Emissions
with tab3:
    st.header("Start Emission Calculation")
    calc_container = st.container()

    link_data = load_data(link_osm) # Reload link_data here for the button click context

    if link_data is None:
        calc_container.error("Link OSM Data is required to run the calculation.")
    elif st.session_state.get('copert_instance') is None:
        calc_container.error("COPERT Model is not initialized. Check Tab 2 for file status.")
    else:
        copert_instance = st.session_state.copert_instance

        if st.button("🚀 Run Emission Calculation", type="primary"):
            st.session_state.results_df, st.session_state.map_bounds = calculate_emissions(
                link_data, 
                copert_instance, 
                st.session_state.pollutant, 
                ambient_temp_c,
                calc_container
            )

        # --- Display Results and Download ---
        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df
            
            st.markdown("---")
            st.subheader("Summary of Results")

            hot_emission = results_df['hot_emission']
            hot_emission_m = results_df['hot_emission_m']
            
            st.markdown(f"**Calculated Pollutant:** `{st.session_state.pollutant}`")
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Total Emissions (g)", f"{hot_emission.sum():,.2f}")
            col_res2.metric("Average Emission per Link (g)", f"{hot_emission.mean():.4f}")
            col_res3.metric("Max Emission (g)", f"{hot_emission.max():,.2f}")
            
            # Download Section
            st.markdown("---")
            st.subheader("Download Results")
            
            # Create a buffer for the zip file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # 1. Add Full Results CSV
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                zip_file.writestr(f'emissions_{st.session_state.pollutant.lower()}_full_results.csv', csv_data)
                
                # 2. Add Summary Text
                x_min, x_max, y_min, y_max = st.session_state.map_bounds
                summary = f"""
Emission Calculation Summary Report (COPERT Model Mock)
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Pollutant Calculated: {st.session_state.pollutant}
Ambient Temperature: {ambient_temp_c}°C
Total Links Processed: {len(results_df)}

Total Emissions (PC + Others): {hot_emission.sum():,.2f} g
Total Emissions (Motorcycle Only): {hot_emission_m.sum():,.2f} g
Average Emission per Link: {hot_emission.mean():.4f} g
Maximum Emission (Single Link): {hot_emission.max():,.2f} g

Map Boundaries (If Link Data used for map):
- Longitude (X): {x_min:.5f} to {x_max:.5f}
- Latitude (Y): {y_min:.5f} to {y_max:.5f}
"""
                zip_file.writestr('summary_report.txt', summary)
            
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download Complete Results (ZIP)", 
                data=zip_buffer, 
                file_name=f"emission_results_{st.session_state.pollutant.lower()}.zip", 
                mime="application/zip",
                key="download_zip_button"
            )
            st.success("✅ Results available for download.")

        else:
            st.info("Click the button above to start the calculation.")

# Tab 4: Visualize Map
with tab4:
    st.header("Visualization")
    
    if st.session_state.results_df is None:
        st.warning("Please calculate emissions in the 'Calculate Emissions' tab first.")
    else:
        results_df = st.session_state.results_df
        x_min, x_max, y_min, y_max = st.session_state.map_bounds
        
        st.subheader(f"Emission Map Visualization ({st.session_state.pollutant} in g)")
        
        col_map_settings, col_map_style = st.columns([1, 1])
        
        with col_map_settings:
            map_style = st.radio(
                "Map Mode",
                ("Enhanced Visualization", "Classic Map"),
                index=0,
                key='map_style',
                help="Choose a visualization style for the emission data."
            )

        with col_map_style:
            # Color map selection
            cmap_name = st.selectbox(
                "Color Map Theme",
                ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdYlGn_r', 'jet'],
                index=0,
                help="Select the color gradient for the lines."
            )
            line_width_multiplier = st.slider("Line Width Multiplier", 0.5, 5.0, 2.0, 0.1)

        # Plotting logic
        if st.button("Generate Map Visualization", key='generate_map_btn'):
            try:
                # 1. Setup Plot
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.set_aspect('equal', adjustable='box')
                
                # Set axis limits based on data bounds
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                # 2. Prepare Data and Color Mapping
                emission_data = results_df['hot_emission'].values
                
                # Normalize emission data for color and width
                norm = colors.Normalize(vmin=emission_data.min(), vmax=emission_data.max())
                cmap = plt.get_cmap(cmap_name)
                scalarMap = cmx.ScalarMappable(norm=norm, cmap=cmap)
                
                # 3. Enhanced vs. Classic Styling
                if map_style == "Enhanced Visualization":
                    # Use a power scale for width to highlight high emissions dramatically
                    width_norm = colors.PowerNorm(gamma=0.5, vmin=emission_data.min(), vmax=emission_data.max())
                    widths = width_norm(emission_data) * line_width_multiplier
                else: # Classic Map
                    # Use linear scaling for simplicity
                    width_norm = colors.Normalize(vmin=emission_data.min(), vmax=emission_data.max())
                    widths = width_norm(emission_data) * line_width_multiplier * 0.5 + 0.5 # Base width 0.5

                # 4. Draw Links
                for index, row in results_df.iterrows():
                    colorVal = scalarMap.to_rgba(row['hot_emission'])
                    
                    ax.plot(
                        [row['x_s'], row['x_e']], 
                        [row['y_s'], row['y_e']], 
                        color=colorVal, 
                        linewidth=widths[index],
                        alpha=0.8
                    )
                
                # 5. Add Colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(emission_data) # Required for non-scatter plots
                cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
                cbar.set_label(f'{st.session_state.pollutant} Emission (g)', rotation=270, labelpad=15)

                ax.set_title(f'Traffic Emission Visualization ({st.session_state.pollutant})')
                ax.set_xlabel('Longitude (X)')
                ax.set_ylabel('Latitude (Y)')
                ax.grid(True, linestyle=':', alpha=0.6)
                
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Error generating map. Please ensure your X/Y coordinates are valid numbers and the data is correctly formatted. Error: {e}")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🌐 COPERT Methodology**")
st.sidebar.markdown(
    "The calculation is based on the COPERT (Computer Programme to calculate Emissions from Road Transport) standard, which provides an internationally accepted methodology for estimating emissions based on vehicle fleet composition, traffic activity, and driving behavior. "
)
st.sidebar.markdown("*Note: The `copert` module is mocked for deployment. A real implementation requires a proprietary COPERT library.*")

