import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import zipfile
from io import BytesIO

# Visualization Imports
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as colors
import matplotlib.cm as cmx

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Traffic Emission Calculator", 
    layout="wide", 
    page_icon="🚗"
)

# Custom CSS for Power BI feel
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("Built by SHassan (Upgraded v2.0)")

# ==================== SIDEBAR ====================
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

st.sidebar.header("🗺️ Map Parameters")
col1, col2 = st.sidebar.columns(2)
x_min = col1.number_input("X Min (Lon)", value=3.37310, format="%.5f")
x_max = col2.number_input("X Max (Lon)", value=3.42430, format="%.5f")
y_min = col1.number_input("Y Min (Lat)", value=6.43744, format="%.5f")
y_max = col2.number_input("Y Max (Lat)", value=6.46934, format="%.5f")
tolerance = st.sidebar.number_input("Tolerance", value=0.005, format="%.3f")
ncore = st.sidebar.number_input("Number of Cores", value=8, min_value=1, max_value=16)

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Instructions", 
    "📊 Data Preview", 
    "⚙️ Calculate Emissions", 
    "🗺️ Interactive Map", 
    "📈 Analysis Charts",
    "📥 Download Results"
])

# --- TAB 1: INSTRUCTIONS (Kept Original) ---
with tab1:
    st.header("📖 User Guide")
    st.markdown("""
    ### Quick Start Guide
    1. **Upload Files**: Upload all required COPERT, Link, and Proportion files in the sidebar.
    2. **Calculate**: Go to the **Calculate Emissions** tab and run the model.
    3. **Visualize**: Use the **Interactive Map** to explore emissions with 4-class coloring.
    4. **Analyze**: Use the **Analysis Charts** tab to choose specific charts.
    """)

# --- TAB 2: DATA PREVIEW (Kept Original Logic) ---
with tab2:
    st.header("Data Preview")
    if link_osm is not None:
        try:
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            if data_link.shape[1] >= 7:
                data_link.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop']
            else:
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            st.dataframe(data_link.head(20))
        except Exception as e:
            st.error(f"Error reading data: {e}")

# --- TAB 3: CALCULATE EMISSIONS (ORIGINAL LOGIC PRESERVED) ---
with tab3:
    st.header("Calculate Emissions")
    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    
    if all(f is not None for f in required_files):
        if st.button("🚀 Calculate Emissions", type="primary"):
            with st.spinner("Computing emissions (Original Algorithm)..."):
                try:
                    import copert
                    # Create temp files for the COPERT module
                    with tempfile.TemporaryDirectory() as tmpdir:
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
                        d_ec_gas = np.loadtxt(engine_cap_gas)
                        d_ec_diesel = np.loadtxt(engine_cap_diesel)
                        d_cc_gas = np.loadtxt(copert_class_gas)
                        d_cc_diesel = np.loadtxt(copert_class_diesel)
                        d_c_2s = np.loadtxt(copert_2stroke)
                        d_c_4s = np.loadtxt(copert_4stroke)

                        # Setup Classes (Original)
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

                        # Calculation Loop
                        Nlink = data_link.shape[0]
                        hot_emission_pc = np.zeros((Nlink,), dtype=float)
                        hot_emission_m = np.zeros((Nlink,), dtype=float)
                        hot_emission = np.zeros((Nlink,), dtype=float)
                        
                        prog_bar = st.progress(0)
                        
                        for i in range(Nlink):
                            if i % 100 == 0: prog_bar.progress(i / Nlink)
                            
                            L = data_link[i, 1]
                            Flow = data_link[i, 2]
                            v = min(max(10., data_link[i, 3]), 130.)
                            prop_gas = data_link[i, 4]
                            prop_pc = data_link[i, 5]
                            prop_4s = data_link[i, 6]
                            
                            p_passenger = prop_gas
                            P_motorcycle = 1. - prop_pc
                            
                            eng_dist = [prop_gas, 1. - prop_gas]
                            cap_dist = [d_ec_gas[i], d_ec_diesel[i]]
                            moto_dist = [prop_4s, 1. - prop_4s]

                            # PC Calc
                            for t in range(2):
                                for c in range(Nclass):
                                    for k in range(2):
                                        if (copert_class[c] != cop.class_Improved_Conventional and copert_class[c] != cop.class_Open_loop) or engine_capacity[k] <= 2.0:
                                            if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                continue
                                            e = cop.Emission(cop.pollutant_CO, v, L, cop.vehicle_type_passenger_car, engine_type[t], copert_class[c], engine_capacity[k], 28.2)
                                            e *= eng_dist[t] * cap_dist[t][k]
                                            hot_emission_pc[i] += e * p_passenger / L * Flow
                            
                            # Moto Calc
                            for m in range(2):
                                for d in range(Mclass):
                                    if m == 1 and copert_class_motorcycle[d] in range(cop.class_moto_Conventional, 1 + cop.class_moto_Euro_5):
                                        continue
                                    e_f = cop.EFMotorcycle(cop.pollutant_CO, v, engine_type_m[m], copert_class_motorcycle[d])
                                    e_f *= moto_dist[m]
                                    hot_emission_m[i] += e_f * P_motorcycle * Flow
                            
                            hot_emission[i] = hot_emission_pc[i] + hot_emission_m[i]
                        
                        prog_bar.empty()
                        
                        # Store Results
                        st.session_state.hot_emission = hot_emission
                        st.session_state.hot_emission_pc = hot_emission_pc
                        st.session_state.hot_emission_m = hot_emission_m
                        st.session_state.data_link = data_link
                        
                        st.success("✅ Calculation Complete!")
                        
                        # Display Summary
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Total PC Emission", f"{hot_emission_pc.sum():.2f}")
                        with c2: st.metric("Total Moto Emission", f"{hot_emission_m.sum():.2f}")
                        with c3: st.metric("Grand Total", f"{hot_emission.sum():.2f}")

                except Exception as e:
                    st.error(f"Calculation Error: {e}")
    else:
        st.warning("Please upload all files in the sidebar first.")

# --- TAB 4: INTERACTIVE MAP (UPDATED AS REQUESTED) ---
with tab4:
    st.header("🗺️ Interactive Emission Map")
    
    if 'hot_emission' in st.session_state and osm_file is not None:
        import osm_network
        
        # --- MAP CONTROLS ---
        c1, c2, c3 = st.columns(3)
        with c1:
            color_scheme = st.selectbox("Color Theme", ["Jet (Blue-Red)", "White-Red", "Viridis"])
        with c2:
            map_style = st.selectbox("Base Map", ["carto-positron", "open-street-map", "carto-darkmatter"])
        with c3:
            st.info("Colors classified into 4 groups (Quartiles)")

        if st.button("Generate Interactive Map", type="primary"):
            with st.spinner("Parsing Geometry & Building Map..."):
                try:
                    # 1. Parse OSM
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                        osm_file.seek(0)
                        tmp.write(osm_file.read())
                        osm_path = tmp.name
                    
                    zone = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    coords, osmids, names, types = osm_network.retrieve_highway(osm_path, zone, tolerance, int(ncore))
                    os.unlink(osm_path)
                    
                    # 2. Map Data Preparation
                    data_link = st.session_state.data_link
                    emissions = st.session_state.hot_emission
                    
                    # Create lookup dictionaries
                    emission_map = {int(row[0]): val for row, val in zip(data_link, emissions)}
                    speed_map = {int(row[0]): row[3] for row in data_link}
                    flow_map = {int(row[0]): row[2] for row in data_link}
                    
                    # Build DataFrame for Plotly
                    map_rows = []
                    for cid, oid, name in zip(coords, osmids, names):
                        if oid in emission_map:
                            val = emission_map[oid]
                            # Simple geometry center for hover (lines are drawn separately)
                            mid = len(cid) // 2
                            map_rows.append({
                                'lat': cid[mid][1], 'lon': cid[mid][0],
                                'OSM_ID': oid, 'Name': name,
                                'Emission': val,
                                'Speed': speed_map.get(oid, 0),
                                'Flow': flow_map.get(oid, 0),
                                'geometry': cid
                            })
                    
                    df_map = pd.DataFrame(map_rows)
                    
                    # 3. Classify into 4 Groups (Quartiles)
                    df_map['Group'] = pd.qcut(df_map['Emission'], 4, labels=["1: Low", "2: Medium", "3: High", "4: Critical"])
                    
                    # 4. Color Scale Logic
                    if color_scheme == "Jet (Blue-Red)":
                        colors_seq = px.colors.sequential.Jet
                    elif color_scheme == "White-Red":
                        colors_seq = px.colors.sequential.Reds
                    else:
                        colors_seq = px.colors.sequential.Viridis

                    # 5. Build Plotly Map (Optimized Layering)
                    fig = go.Figure()
                    
                    # Iterate groups to add traces
                    groups = sorted(df_map['Group'].unique())
                    for idx, group in enumerate(groups):
                        subset = df_map[df_map['Group'] == group]
                        
                        # Build Line Segments (Gap Method for Performance)
                        lats, lons = [], []
                        for g in subset['geometry']:
                            l = list(zip(*g))
                            lons.extend(l[0] + (None,))
                            lats.extend(l[1] + (None,))
                        
                        # Determine color for this group
                        c_idx = int(idx / 3 * (len(colors_seq)-1))
                        color = colors_seq[c_idx]
                        
                        fig.add_trace(go.Scattermapbox(
                            lat=lats, lon=lons, mode='lines',
                            line=dict(width=3, color=color),
                            name=f"{group} Emission",
                            hoverinfo='none' # Hover handled by markers
                        ))

                    # Add Invisible Markers for Tooltips
                    fig.add_trace(go.Scattermapbox(
                        lat=df_map['lat'], lon=df_map['lon'], mode='markers',
                        marker=dict(size=8, opacity=0),
                        text=df_map.apply(lambda row: f"<b>{row['Name']}</b><br>ID: {row['OSM_ID']}<br>Emission: {row['Emission']:.2f}<br>Speed: {row['Speed']}", axis=1),
                        hoverinfo='text',
                        name='Info Points'
                    ))

                    fig.update_layout(
                        mapbox_style=map_style,
                        mapbox_zoom=12,
                        mapbox_center={"lat": (ymin+ymax)/2, "lon": (xmin+xmax)/2},
                        height=600,
                        margin={"r":0,"t":0,"l":0,"b":0},
                        legend=dict(orientation="h", y=1, x=0, bgcolor="rgba(255,255,255,0.8)")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"Mapped {len(df_map)} road segments.")
                    
                except Exception as e:
                    st.error(f"Map Generation Error: {e}")
    else:
        st.info("Please calculate emissions and upload OSM file first.")

# --- TAB 5: ANALYSIS CHARTS (NEW FEATURE) ---
with tab5:
    st.header("📈 Deep Dive Analysis")
    
    if 'hot_emission' in st.session_state:
        # Prepare DataFrame
        data_link = st.session_state.data_link
        df = pd.DataFrame(data_link, columns=['OSM_ID','Length','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop'])
        df['PC_Emission'] = st.session_state.hot_emission_pc
        df['Moto_Emission'] = st.session_state.hot_emission_m
        df['Total_Emission'] = st.session_state.hot_emission
        
        # --- Chart Selector ---
        chart_type = st.selectbox(
            "Select Chart Representation", 
            ["Emission Sources (Bar)", "Fuel Split (Pie)", "Speed vs Emission (Scatter)", "Top Polluters (Bar)"]
        )
        
        st.markdown("---")
        
        if chart_type == "Emission Sources (Bar)":
            # Stacked Bar
            total_pc = df['PC_Emission'].sum()
            total_moto = df['Moto_Emission'].sum()
            plot_df = pd.DataFrame([
                {'Type': 'Passenger Cars', 'Emission': total_pc},
                {'Type': 'Motorcycles', 'Emission': total_moto}
            ])
            fig = px.bar(plot_df, x='Type', y='Emission', color='Type', title="Total Emissions by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Fuel Split (Pie)":
            # Estimate Fuel (PC only for simplicity as per original logic)
            # Gas = PC_Emission * Gas_Prop + Moto (assuming Moto is gas)
            # This is an approximation based on the aggregate data available
            avg_gas_prop = df['Gas_Prop'].mean()
            total = df['Total_Emission'].sum()
            gas_est = total * avg_gas_prop # Rough estimate for viz
            dsl_est = total * (1 - avg_gas_prop)
            
            fig = px.pie(
                names=['Gasoline', 'Diesel'], 
                values=[gas_est, dsl_est], 
                title="Estimated Fuel Contribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Speed vs Emission (Scatter)":
            fig = px.scatter(
                df, x='Speed', y='Total_Emission', 
                color='Flow', size='Flow',
                hover_data=['OSM_ID'],
                title="Speed vs Emission Analysis",
                labels={'Total_Emission': 'Emission (g/km)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Top Polluters (Bar)":
            top_10 = df.nlargest(10, 'Total_Emission')
            top_10['Road_ID'] = top_10['OSM_ID'].astype(str)
            fig = px.bar(
                top_10, x='Road_ID', y='Total_Emission', color='Speed',
                title="Top 10 Highest Emitting Links",
                labels={'Road_ID': 'OSM ID'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("Calculate emissions first to view analysis.")

# --- TAB 6: DOWNLOAD (Kept Original Logic) ---
with tab6:
    st.header("📥 Download Results")
    if 'hot_emission' in st.session_state:
        df_res = pd.DataFrame({
            'OSM_ID': st.session_state.data_link[:,0].astype(int),
            'PC_Emission': st.session_state.hot_emission_pc,
            'Moto_Emission': st.session_state.hot_emission_m,
            'Total_Emission': st.session_state.hot_emission
        })
        
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV Results", csv, "emission_results.csv", "text/csv")
        
        # ZIP
        if st.button("📦 Generate ZIP Package"):
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w') as z:
                z.writestr("emissions.csv", df_res.to_csv(index=False))
                z.writestr("summary.txt", f"Total Emission: {df_res['Total_Emission'].sum()}")
            buf.seek(0)
            st.download_button("⬇️ Download ZIP", buf, "emission_pkg.zip", "application/zip")
    else:
        st.info("No results to download yet.")
