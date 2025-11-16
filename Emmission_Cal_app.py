import streamlit as st
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set backend before importing pyplot
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

# Content of the instruction tab (loaded from instructions_tab.md)
INSTRUCTION_CONTENT = """
# 🏭 Emission Calculation and Visualization Workflow

Welcome to the Emission Calculation and Visualization App! This tab provides detailed instructions on the entire process, starting with the necessary geospatial data preparation in QGIS and ending with the interpretation of the visualized emission factors.

This workflow is divided into three main stages: **QGIS Geospatial Data Preparation**, **Data Conversion and Formatting**, and **App Usage & Visualization**.

## Stage 1: QGIS Geospatial Data Preparation (Street Network Modeling)

This stage involves preparing the street network data, merging layers, and calculating the required variables within a Geographic Information System (GIS) environment (QGIS).

### I. Install QGIS

Ensure QGIS (a free and open-source Geographic Information System) is inst...
"""

# Sidebar for file uploads
st.sidebar.header("📂 Upload Input Files")

# File uploaders
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

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "⚙️ Calculate Emissions", "🗺️ Emission Map", "📥 Download Results"])

with tab1:
    st.header("Data Preview")

    if link_osm is not None:
        st.subheader("Link OSM Data")
        try:
            # Read space-separated .dat file
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')

            # Set column names
            if data_link.shape[1] >= 7:
                data_link.columns = ['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop', '4Stroke_Prop']
            else:
                # Generic column names if less than 7 columns
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]

            st.dataframe(data_link.head(20))
            st.info(f"📌 Total links: {len(data_link)} | Columns: {data_link.shape[1]}")

            # Display statistics only if we have the right columns
            if data_link.shape[1] >= 7:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
                with col2:
                    st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
                with col3:
                    st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")
            else:
                st.warning(f"⚠️ Expected 7 columns but found {data_link.shape[1]}. Please check your data file.")

        except Exception as e:
            st.error(f"Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar")

with tab2:
    st.header("Calculate Emissions")

    # Check if all required files are uploaded
    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]

    all_uploaded = all(f is not None for f in required_files)

    if all_uploaded:
        st.success("✅ All required files uploaded!")

        if st.button("🚀 Calculate Emissions", type="primary"):
            with st.spinner("Computing emissions... This may take a moment"):
                try:
                    # Check if copert module is available
                    try:
                        import copert
                        copert_available = True
                    except ImportError:
                        copert_available = False
                        st.error("❌ 'copert' module not found. Please ensure it's in your Python path.")

                    if copert_available:
                        # Save uploaded files temporarily
                        import tempfile
                        import os

                        with tempfile.TemporaryDirectory() as tmpdir:
                            # Save COPERT parameter files
                            pc_path = os.path.join(tmpdir, "PC_parameter.csv")
                            ldv_path = os.path.join(tmpdir, "LDV_parameter.csv")
                            hdv_path = os.path.join(tmpdir, "HDV_parameter.csv")
                            moto_path = os.path.join(tmpdir, "Moto_parameter.csv")

                            with open(pc_path, 'wb') as f:
                                f.write(pc_param.getbuffer())
                            with open(ldv_path, 'wb') as f:
                                f.write(ldv_param.getbuffer())
                            with open(hdv_path, 'wb') as f:
                                f.write(hdv_param.getbuffer())
                            with open(moto_path, 'wb') as f:
                                f.write(moto_param.getbuffer())

                            # Initialize COPERT
                            cop = copert.Copert(pc_path, ldv_path, hdv_path, moto_path)

                            # Load data - reset file pointers first
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

                            # Define engine types and classes
                            engine_type = [cop.engine_type_gasoline, cop.engine_type_diesel]
                            engine_type_m = [cop.engine_type_moto_two_stroke_more_50,
                                             cop.engine_type_moto_four_stroke_50_250]
                            engine_capacity = [cop.engine_capacity_0p8_to_1p4,
                                               cop.engine_capacity_1p4_to_2]

                            copert_class = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01,
                                            cop.class_ECE_15_02, cop.class_ECE_15_03, cop.class_ECE_15_04,
                                            cop.class_Improved_Conventional, cop.class_Open_loop,
                                            cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3,
                                            cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6,
                                            cop.class_Euro_6c]
                            Nclass = len(copert_class)

                            copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1,
                                                       cop.class_moto_Euro_2, cop.class_moto_Euro_3,
                                                       cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                            Mclass = len(copert_class_motorcycle)

                            # Calculate emissions
                            Nlink = data_link.shape[0]
                            hot_emission_pc = np.zeros((Nlink,), dtype=float)
                            hot_emission_m = np.zeros((Nlink,), dtype=float)
                            hot_emission = np.zeros((Nlink,), dtype=float)

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i in range(Nlink):
                                # Update progress
                                if i % max(1, Nlink // 100) == 0:
                                    progress_bar.progress(i / Nlink)
                                status_text.text(f"Processing link {i + 1}/{Nlink}")

                                # Extract link data (columns: OSM_ID, Length, Flow, Speed, Gas_Prop, PC_Prop, 4Stroke_Prop)
                                link_length = data_link[i, 1]
                                link_flow = data_link[i, 2]
                                v = min(max(10., data_link[i, 3]), 130.)
                                link_gasoline_proportion = data_link[i, 4]
                                link_pc_proportion = data_link[i, 5]
                                link_4_stroke_proportion = data_link[i, 6]

                                p_passenger = link_gasoline_proportion
                                P_motorcycle = 1. - link_pc_proportion

                                engine_type_distribution = [link_gasoline_proportion, 1. - link_gasoline_proportion]
                                engine_capacity_distribution = [data_engine_capacity_gasoline[i],
                                                                data_engine_capacity_diesel[i]]
                                engine_type_motorcycle_distribution = [link_4_stroke_proportion,
                                                                       1. - link_4_stroke_proportion]

                                # Passenger car emission
                                for t in range(2):
                                    for c in range(Nclass):
                                        for k in range(2):
                                            if (copert_class[c] != cop.class_Improved_Conventional
                                                    and copert_class[c] != cop.class_Open_loop) \
                                                    or engine_capacity[k] <= 2.0:
                                                if t == 1 and k == 0 \
                                                        and copert_class[c] in range(cop.class_Euro_1,
                                                                                     1 + cop.class_Euro_3):
                                                    continue
                                                e = cop.Emission(cop.pollutant_CO, v, link_length,
                                                                 cop.vehicle_type_passenger_car, engine_type[t],
                                                                 copert_class[c], engine_capacity[k], 28.2)
                                                e *= engine_type_distribution[t] \
                                                      * engine_capacity_distribution[t][k]
                                                hot_emission_pc[i] += e * p_passenger / link_length * link_flow

                                # Motorcycle emission
                                for m in range(2):
                                    for d in range(Mclass):
                                        if m == 1 \
                                                and copert_class_motorcycle[d] in range(cop.class_moto_Conventional,
                                                                                        1 + cop.class_moto_Euro_5):
                                            continue
                                        e_f = cop.EFMotorcycle(cop.pollutant_CO, v, engine_type_m[m],
                                                               copert_class_motorcycle[d])
                                        e_f *= engine_type_motorcycle_distribution[m]
                                        hot_emission_m[i] += e_f * P_motorcycle * link_flow

                                hot_emission[i] = hot_emission_m[i] + hot_emission_pc[i]

                            progress_bar.progress(1.0)
                            status_text.text("✅ Calculation complete!")

                            # Store results in session state
                            st.session_state.hot_emission = hot_emission
                            st.session_state.hot_emission_pc = hot_emission_pc
                            st.session_state.hot_emission_m = hot_emission_m
                            st.session_state.data_link = data_link

                        st.success("✅ Emissions calculated successfully!")

                        # Display results
                        results_df = pd.DataFrame({
                            'OSM_ID': data_link[:, 0].astype(int),
                            'Hot_Emission_PC (g/km)': hot_emission_pc,
                            'Hot_Emission_Motorcycle (g/km)': hot_emission_m,
                            'Total_Emission (g/km)': hot_emission
                        })

                        st.dataframe(results_df)

                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total PC Emissions", f"{hot_emission_pc.sum():.2f} g/km")
                        with col2:
                            st.metric("Total Motorcycle Emissions", f"{hot_emission_m.sum():.2f} g/km")
                        with col3:
                            st.metric("Total Emissions", f"{hot_emission.sum():.2f} g/km")

                except Exception as e:
                    st.error(f"Error during calculation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please upload all required files to proceed")
        missing = []
        file_names = ['PC Parameter', 'LDV Parameter', 'HDV Parameter', 'Moto Parameter',
                      'Link OSM', 'Engine Cap Gas', 'Engine Cap Diesel', 'COPERT Class Gas',
                      'COPERT Class Diesel', '2-Stroke', '4-Stroke']
        for fname, fdata in zip(file_names, required_files):
            if fdata is None:
                missing.append(fname)
        st.error(f"Missing files: {', '.join(missing)}")

with tab3:
    st.header("Emission Factor Map")

    # Check if emissions have been calculated
    has_emissions = 'hot_emission' in st.session_state

    if not has_emissions:
        st.warning("⚠️ Please calculate emissions first in the 'Calculate Emissions' tab")
    elif osm_file is None:
        st.warning("⚠️ Please upload OSM network file in the sidebar")
    else:
        st.info("📍 Ready to generate emission map")

        # Map visualization settings
        st.subheader("Map Visualization Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            colormap = st.selectbox("Color Map", ['jet', 'viridis', 'plasma', 'RdYlGn_r', 'hot', 'coolwarm'], index=0)
            fig_size = st.slider("Figure Size", 8, 16, 12)
            line_width_multiplier = st.slider("Line Width Scale", 0.5, 5.0, 2.0, 0.5)

        with col2:
            label_density = st.selectbox(
                "Road Label Density", 
                ["Minimal (Major roads only)", "Medium (Top 25% emissions)", "High (Top 50% emissions)"],
                index=1
            )
            show_minor_roads = st.checkbox("Show roads without emission data", value=True)
            rotate_labels = st.checkbox("Rotate labels along roads", value=True)

        st.markdown("---")

        # Convert label density selection to threshold
        if label_density == "Minimal (Major roads only)":
            emission_percentile = 90
            major_only = True
        elif label_density == "Medium (Top 25% emissions)":
            emission_percentile = 75
            major_only = False
        else:  # High
            emission_percentile = 50
            major_only = False

        if st.button("🗺️ Generate Map", type="primary"):
            with st.spinner("Generating emission map..."):
                try:
                    # Check if osm_network module is available
                    try:
                        import osm_network
                        osm_available = True
                    except ImportError:
                        osm_available = False
                        st.error("❌ 'osm_network' module not found. Generating simplified map...")

                    hot_emission = st.session_state.hot_emission
                    data_link = st.session_state.data_link

                    if osm_available:
                        # Full OSM implementation
                        import tempfile
                        import os

                        # Save OSM file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                            osm_file.seek(0)
                            tmp.write(osm_file.read())
                            osm_path = tmp.name

                        try:
                            # Domain to be displayed
                            selected_zone = [[x_min, y_max],
                                             [x_min, y_min],
                                             [x_max, y_min],
                                             [x_max, y_max]]
                            selected_zone.append(selected_zone[0])

                            # Retrieve highway data
                            status_text = st.empty()
                            status_text.text("Parsing OSM network...")

                            # Now correctly expecting 4 return values from the fixed osm_network.py
                            highway_coordinate, highway_osmid, highway_names, highway_types = osm_network.retrieve_highway(
                                osm_path, selected_zone, tolerance, int(ncore)
                            )
                            status_text.text("OSM network parsed successfully!")

                            # Prepare visualization parameters
                            max_emission_value = np.max(hot_emission)
                            epsilon = 1e-9

                            # IMPROVED: Better line width scaling to make emissions more visible
                            lw_max = 3.0 * line_width_multiplier
                            lw_min = 0.5 * line_width_multiplier
                            width_scaling = (lw_max - lw_min) / (max_emission_value + epsilon)
                            lw_nodata = 0.3  # Roads without emission data

                            color_scale = colors.Normalize(vmin=0, vmax=max_emission_value + epsilon)
                            scale_map = cmx.ScalarMappable(norm=color_scale, cmap=colormap)

                            emission_osm_id = [int(x) for x in data_link[:, 0]]

                            # Create figure
                            fig = plt.figure(figsize=(fig_size, fig_size - 1), dpi=100)

                            # Main plot
                            ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
                            ax.set_aspect("equal", adjustable="box")

                            # Colorbar
                            ax_c = fig.add_axes([0.85, 0.21, 0.03, 0.53])
                            cb = matplotlib.colorbar.ColorbarBase(
                                ax_c, cmap=plt.cm.get_cmap(colormap),
                                norm=color_scale,
                                orientation="vertical"
                            )
                            cb.set_label("g/km", fontsize=12)

                            # Plot highways
                            status_text.text("Plotting emission data on map...")
                            roads_with_data = 0
                            roads_without_data = 0

                            # IMPROVED: Collect road labels to avoid overlaps
                            labeled_roads = {}  # Store positions of already labeled roads
                            min_label_distance = 0.002  # Minimum distance between labels (in lat/lon)

                            # First pass: Plot all roads
                            for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid, 
                                                                       highway_names, highway_types):
                                try:
                                    i = emission_osm_id.index(osmid)
                                except:
                                    i = None

                                if i is not None:
                                    current_emission = hot_emission[i]
                                    color_value = scale_map.to_rgba(current_emission)
                                    
                                    # IMPROVED: Better line width calculation
                                    line_width = lw_min + (current_emission * width_scaling)
                                    
                                    ax.plot([x[0] for x in refs], [x[1] for x in refs],
                                            color=color_value,
                                            lw=line_width,
                                            alpha=0.8,  # Added transparency
                                            solid_capstyle='round')  # Smoother line ends
                                    roads_with_data += 1
                                else:
                                    if show_minor_roads:
                                        ax.plot([x[0] for x in refs], [x[1] for x in refs],
                                                "gray", lw=lw_nodata, alpha=0.3)
                                        roads_without_data += 1

                            # IMPROVED: Second pass - Smart road labeling
                            # Only label major roads with high emissions or important road types
                            major_road_types = ['motorway', 'trunk', 'primary', 'secondary']
                            emission_threshold = np.percentile(hot_emission, emission_percentile)
                            
                            for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid, 
                                                                       highway_names, highway_types):
                                try:
                                    i = emission_osm_id.index(osmid)
                                    current_emission = hot_emission[i]
                                except:
                                    continue
                                
                                # Determine if we should label this road
                                if major_only:
                                    should_label = name and highway_type in major_road_types
                                else:
                                    should_label = (
                                        name and 
                                        (highway_type in major_road_types or current_emission >= emission_threshold)
                                    )
                                
                                if should_label:
                                    # Calculate center point
                                    center_index = len(refs) // 2
                                    x_center = refs[center_index][0]
                                    y_center = refs[center_index][1]
                                    
                                    # Check if we already labeled this road nearby
                                    too_close = False
                                    if name in labeled_roads:
                                        for prev_x, prev_y in labeled_roads[name]:
                                            distance = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                                            if distance < min_label_distance:
                                                too_close = True
                                                break
                                    
                                    if not too_close:
                                        # Calculate angle of road segment for rotation
                                        angle = 0
                                        if rotate_labels and len(refs) > 1:
                                            dx = refs[min(center_index + 1, len(refs) - 1)][0] - refs[max(center_index - 1, 0)][0]
                                            dy = refs[min(center_index + 1, len(refs) - 1)][1] - refs[max(center_index - 1, 0)][1]
                                            angle = np.degrees(np.arctan2(dy, dx))
                                            
                                            # Keep text readable (not upside down)
                                            if angle > 90:
                                                angle -= 180
                                            elif angle < -90:
                                                angle += 180
                                        
                                        # Add text label with better styling
                                        ax.text(x_center, y_center, str(name),
                                                fontsize=7,
                                                color='black',
                                                ha='center', 
                                                va='center',
                                                rotation=angle,
                                                rotation_mode='anchor',
                                                bbox=dict(
                                                    facecolor='white', 
                                                    alpha=0.8, 
                                                    edgecolor='lightgray',
                                                    linewidth=0.5,
                                                    boxstyle='round,pad=0.3'
                                                ),
                                                zorder=100)  # Ensure labels are on top
                                        
                                        # Record this label position
                                        if name not in labeled_roads:
                                            labeled_roads[name] = []
                                        labeled_roads[name].append((x_center, y_center))

                            # Finalize plot
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_title("Emission Factor Map with Road Names", fontsize=14, fontweight='bold')
                            ax.set_xlabel("Longitude", fontsize=12)
                            ax.set_ylabel("Latitude", fontsize=12)
                            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                            
                            # Add background color
                            ax.set_facecolor('#f0f0f0')

                            st.pyplot(fig)

                            # Store figure in session state for download
                            st.session_state.emission_map_fig = fig

                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Roads with Emission Data", roads_with_data)
                            with col2:
                                st.metric("Roads without Data", roads_without_data)
                            with col3:
                                st.metric("Unique Road Names Labeled", len(labeled_roads))

                            status_text.empty()
                            st.success("✅ Map generated successfully!")

                        finally:
                            # Clean up temp file
                            if os.path.exists(osm_path):
                                os.unlink(osm_path)

                    else:
                        # Simplified visualization without OSM parsing
                        st.warning("Generating simplified map without OSM network parsing...")

                        max_emission_value = np.max(hot_emission)
                        color_scale = colors.Normalize(vmin=0, vmax=max_emission_value)

                        fig = plt.figure(figsize=(fig_size, fig_size - 1), dpi=100)
                        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
                        ax.set_aspect("equal", adjustable="box")

                        # Colorbar
                        ax_c = fig.add_axes([0.85, 0.21, 0.03, 0.53])
                        cb = matplotlib.colorbar.ColorbarBase(
                            ax_c, cmap=plt.cm.get_cmap(colormap),
                            norm=color_scale,
                            orientation="vertical"
                        )
                        cb.set_label("g/km")

                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_title("Emission Factor Map (Simplified)")
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        ax.grid(True, alpha=0.3)

                        # Plot emission distribution as scatter
                        scatter = ax.scatter(
                            np.random.uniform(x_min, x_max, len(hot_emission)),
                            np.random.uniform(y_min, y_max, len(hot_emission)),
                            c=hot_emission,
                            cmap=colormap,
                            s=50,
                            alpha=0.6
                        )

                        ax.text(0.5, 0.95, 'Install osm_network module for full road network visualization',
                                transform=ax.transAxes, ha='center', va='top',
                                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                        st.pyplot(fig)
                        st.session_state.emission_map_fig = fig
                        st.info("💡 Install 'osm_network' module for complete road network visualization")

                except Exception as e:
                    st.error(f"Error generating map: {e}")
                    import traceback
                    st.code(traceback.format_exc())

with tab4:
    st.header("Download Results")

    st.markdown("### 📊 Available Outputs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Emission Data**")
        if 'hot_emission' in st.session_state:
            data_link = st.session_state.data_link
            hot_emission = st.session_state.hot_emission

            # Create CSV data
            results_df = pd.DataFrame({
                'OSM_ID': data_link[:, 0].astype(int),
                'Length_km': data_link[:, 1],
                'Emission_g_km': hot_emission
            })
            csv = results_df.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Emission Data CSV",
                data=csv,
                file_name="link_hot_emission_factor.csv",
                mime="text/csv"
            )
        else:
            st.info("Calculate emissions first to download data")

    with col2:
        st.markdown("**Emission Map**")
        if 'emission_map_fig' in st.session_state:
            # Save figure to bytes
            buf = BytesIO()
            st.session_state.emission_map_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Map PNG",
                data=buf,
                file_name="emission_factor_map.png",
                mime="image/png"
            )
        else:
            st.info("Generate map first to download image")

    st.markdown("---")
    st.markdown("### 📦 Download All Results")

    if 'hot_emission' in st.session_state:
        if st.button("📦 Create ZIP Archive"):
            with st.spinner("Creating ZIP archive..."):
                try:
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add emission CSV
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

                        # Add map if available
                        if 'emission_map_fig' in st.session_state:
                            map_buf = BytesIO()
                            st.session_state.emission_map_fig.savefig(map_buf, format='png', dpi=150,
                                                                      bbox_inches='tight')
                            map_buf.seek(0)
                            zip_file.writestr('emission_factor_map.png', map_buf.read())

                        # Add summary report
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

                    st.download_button(
                        label="⬇️ Download Complete Results (ZIP)",
                        data=zip_buffer,
                        file_name="emission_results.zip",
                        mime="application/zip"
                    )

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
2. Upload link OSM data (7 columns)
3. Upload proportion data files
4. Upload OSM network file
5. Configure map parameters
6. Calculate emissions
7. Adjust visualization settings
8. Generate and download results
""")

st.sidebar.info("Built with Streamlit by SHassan 🎈")
