import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import zipfile
from io import BytesIO
import time

# Import local modules (Must be in the same directory)
try:
    import copert
    import osm_network
except ImportError:
    st.error("Critical modules (copert.py, osm_network.py) not found. Please ensure they are in the application directory.")

# ==================== CONFIGURATION & STYLING ====================
st.set_page_config(
    page_title="Traffic Emission Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚗"
)

# Custom CSS for "Power BI" feel
st.markdown("""
<style>
    /* Main container padding */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* Card styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #1f77b4; }
    .metric-label { font-size: 14px; color: #666; }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #f8f9fa; }
    
    /* Plot containers */
    .stPlotlyChart { border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS & MAPPINGS ====================
POLLUTANTS_AVAILABLE = {
    "CO": {"name": "Carbon Monoxide", "unit": "g/km", "color": "#ef4444"},
    "CO2": {"name": "Carbon Dioxide", "unit": "g/km", "color": "#3b82f6"},
    "NOx": {"name": "Nitrogen Oxides", "unit": "g/km", "color": "#f59e0b"},
    "PM": {"name": "Particulate Matter", "unit": "mg/km", "color": "#8b5cf6"},
    "VOC": {"name": "Volatile Organic Compounds", "unit": "g/km", "color": "#10b981"},
    "FC": {"name": "Fuel Consumption", "unit": "L/100km", "color": "#f97316"}
}

UNIT_CONVERSIONS = {
    "CO": {"g/km": 1.0, "kg/km": 0.001, "tonnes/km": 1e-6},
    "CO2": {"g/km": 1.0, "kg/km": 0.001, "tonnes/km": 1e-6},
    "NOx": {"g/km": 1.0, "kg/km": 0.001, "mg/km": 1000.0},
    "PM": {"mg/km": 1.0, "g/km": 0.001, "kg/km": 1e-6},
    "VOC": {"g/km": 1.0, "kg/km": 0.001},
    "FC": {"L/100km": 1.0, "L/km": 0.01}
}

# ==================== HELPER FUNCTIONS ====================
def convert_value(val, pollutant, target_unit):
    """Handles unit conversion based on predefined factors."""
    if pollutant in UNIT_CONVERSIONS and target_unit in UNIT_CONVERSIONS[pollutant]:
        return val * UNIT_CONVERSIONS[pollutant][target_unit]
    return val

@st.cache_data(show_spinner=False)
def parse_osm_data(osm_content_bytes, zone, tolerance, ncore):
    """Parses OSM file and returns road geometry. Cached."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
        tmp.write(osm_content_bytes)
        tmp_path = tmp.name
    
    try:
        coords, osmids, names, types = osm_network.retrieve_highway(
            tmp_path, zone, tolerance, ncore
        )
        return coords, osmids, names, types
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@st.cache_data(show_spinner=False)
def perform_calculation(
    _pc_file, _ldv_file, _hdv_file, _moto_file, 
    _link_data_df, 
    _ec_gas, _ec_diesel, _cc_gas, _cc_diesel, _c_2s, _c_4s,
    selected_polls, ambient_temp, trip_len, use_temp_corr, use_cold_start
):
    """Core COPERT calculation logic."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name, content in [('pc', _pc_file), ('ldv', _ldv_file), ('hdv', _hdv_file), ('moto', _moto_file)]:
            p = os.path.join(tmpdir, f"{name}.csv")
            with open(p, 'wb') as f: f.write(content)
            paths[name] = p
            
        cop = copert.Copert(paths['pc'], paths['ldv'], paths['hdv'], paths['moto'])
        
        # Load Distributions
        d_ec_gas = np.loadtxt(BytesIO(_ec_gas))
        d_ec_diesel = np.loadtxt(BytesIO(_ec_diesel))
        d_cc_gas = np.loadtxt(BytesIO(_cc_gas))
        d_cc_diesel = np.loadtxt(BytesIO(_cc_diesel))
        d_c_2s = np.loadtxt(BytesIO(_c_2s))
        d_c_4s = np.loadtxt(BytesIO(_c_4s))

        data_link = _link_data_df.values
        Nlink = len(data_link)
        
        if data_link.shape[1] == 7:
            P_ldv = np.zeros(Nlink)
            P_hdv = np.zeros(Nlink)
        else:
            P_ldv = data_link[:, 7]
            P_hdv = data_link[:, 8]

        d_cc_ldv = d_cc_gas 
        
        HDV_CLASSES = [cop.class_hdv_Euro_I, cop.class_hdv_Euro_II, cop.class_hdv_Euro_III,
                       cop.class_hdv_Euro_IV, cop.class_hdv_Euro_V, cop.class_hdv_Euro_VI]
        PC_CLASSES = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02,
                      cop.class_ECE_15_03, cop.class_ECE_15_04, cop.class_Improved_Conventional,
                      cop.class_Open_loop, cop.class_Euro_1, cop.class_Euro_2, cop.class_Euro_3,
                      cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
        MOTO_CLASSES = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2,
                        cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
        POLL_MAP = {
            "CO": cop.pollutant_CO, "CO2": cop.pollutant_FC, "NOx": cop.pollutant_NOx,
            "PM": cop.pollutant_PM, "VOC": cop.pollutant_VOC, "FC": cop.pollutant_FC
        }

        results = {p: {'pc': np.zeros(Nlink), 'ldv': np.zeros(Nlink), 'hdv': np.zeros(Nlink), 
                       'moto': np.zeros(Nlink), 'total': np.zeros(Nlink)} for p in selected_polls}
        fuel_results = {p: {'gas': np.zeros(Nlink), 'diesel': np.zeros(Nlink)} for p in selected_polls}

        # Optimized calculation loop
        for i in range(Nlink):
            L = data_link[i, 1]
            Flow = data_link[i, 2]
            V = min(max(10., data_link[i, 3]), 130.)
            prop_gas = data_link[i, 4]
            prop_pc = data_link[i, 5]
            prop_4s = data_link[i, 6]
            
            dist_eng_type = [prop_gas, 1.0 - prop_gas]
            dist_cap = [d_ec_gas[i], d_ec_diesel[i]]
            dist_moto_stroke = [prop_4s, 1.0 - prop_4s] 

            for p_name in selected_polls:
                p_type = POLL_MAP[p_name]
                pc_sum = 0.0
                gas_sum = 0.0
                dsl_sum = 0.0
                
                # PC
                for t in range(2): 
                    for c_idx, cls in enumerate(PC_CLASSES):
                        for k in range(2):
                            if t == 1 and k == 0 and cls in range(cop.class_Euro_1, 1 + cop.class_Euro_3): continue
                            try:
                                e = cop.Emission(p_type, V, L, cop.vehicle_type_passenger_car, 
                                                 [cop.engine_type_gasoline, cop.engine_type_diesel][t],
                                                 cls, [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2][k],
                                                 ambient_temp)
                            except: e = 0.0
                            if p_name == "NOx" and use_temp_corr: e *= (1 + 0.02 * (ambient_temp - 20))
                            share = d_cc_gas[i, c_idx] if t==0 else d_cc_diesel[i, c_idx]
                            factor = e * dist_eng_type[t] * dist_cap[t][k] * share
                            val = factor * prop_pc / L * Flow
                            pc_sum += val
                            if t == 0: gas_sum += val
                            else: dsl_sum += val
                results[p_name]['pc'][i] = pc_sum
                
                # LDV
                ldv_sum = 0.0
                if P_ldv[i] > 0:
                    for t in range(2):
                        for c_idx, cls in enumerate(PC_CLASSES):
                            for k in range(2):
                                if t == 1 and k == 0 and cls in range(cop.class_Euro_1, 1 + cop.class_Euro_3): continue
                                try:
                                    e = cop.Emission(p_type, V, L, cop.vehicle_type_light_commercial_vehicle,
                                                     [cop.engine_type_gasoline, cop.engine_type_diesel][t],
                                                     cls, [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2][k],
                                                     ambient_temp)
                                except: e = 0.0
                                if p_name == "NOx" and use_temp_corr: e *= (1 + 0.02 * (ambient_temp - 20))
                                share = d_cc_ldv[i, c_idx]
                                factor = e * dist_eng_type[t] * dist_cap[t][k] * share
                                val = factor * P_ldv[i] / L * Flow
                                ldv_sum += val
                                if t == 0: gas_sum += val
                                else: dsl_sum += val
                results[p_name]['ldv'][i] = ldv_sum
                
                # HDV
                hdv_sum = 0.0
                if P_hdv[i] > 0:
                    try:
                        e = cop.Emission(p_type, V, L, cop.vehicle_type_heavy_duty_vehicle,
                                         cop.engine_type_diesel, cop.class_hdv_Euro_VI, 0, ambient_temp)
                    except: e = 0.0
                    if p_name == "NOx" and use_temp_corr: e *= (1 + 0.015 * (ambient_temp - 20))
                    val = e * 1.0 * P_hdv[i] / L * Flow
                    hdv_sum += val
                    dsl_sum += val
                results[p_name]['hdv'][i] = hdv_sum
                
                # MOTO
                moto_sum = 0.0
                m_engines = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                for m_idx in range(2): 
                    for d_idx, cls in enumerate(MOTO_CLASSES):
                        if m_idx == 0 and cls >= cop.class_moto_Euro_1: continue 
                        try:
                            ef = cop.EFMotorcycle(p_type, V, m_engines[m_idx], cls)
                        except: ef = 0.0
                        ef *= dist_moto_stroke[m_idx]
                        val = ef * (1.0 - prop_pc) * Flow
                        moto_sum += val
                        gas_sum += val 
                results[p_name]['moto'][i] = moto_sum
                
                results[p_name]['total'][i] = pc_sum + ldv_sum + hdv_sum + moto_sum
                fuel_results[p_name]['gas'][i] = gas_sum
                fuel_results[p_name]['diesel'][i] = dsl_sum

    return results, fuel_results

# ==================== MAIN UI LAYOUT ====================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=64)
    st.title("Traffic Emission Calculator")
    st.caption("v2.1 | Research Grade")
    
    st.markdown("### 1. Data Configuration")
    with st.expander("📂 Parameter Files", expanded=True):
        pc_param = st.file_uploader("PC Params", type=['csv'], key='pc')
        ldv_param = st.file_uploader("LDV Params", type=['csv'], key='ldv')
        hdv_param = st.file_uploader("HDV Params", type=['csv'], key='hdv')
        moto_param = st.file_uploader("Moto Params", type=['csv'], key='moto')

    with st.expander("🛣️ Network Data", expanded=True):
        link_osm = st.file_uploader("Link Data (.dat/.txt)", type=['dat', 'txt', 'csv'])
        osm_file = st.file_uploader("Network Geometry (.osm)", type=['osm'])
        
    with st.expander("📊 Fleet Distributions"):
        ec_gas = st.file_uploader("Eng Cap Gas", type=['dat','txt'])
        ec_dsl = st.file_uploader("Eng Cap Diesel", type=['dat','txt'])
        cc_gas = st.file_uploader("Class Gas", type=['dat','txt'])
        cc_dsl = st.file_uploader("Class Diesel", type=['dat','txt'])
        c_2s = st.file_uploader("Moto 2S", type=['dat','txt'])
        c_4s = st.file_uploader("Moto 4S", type=['dat','txt'])

    st.markdown("### 2. Analysis Settings")
    sel_pollutants = st.multiselect("Pollutants", list(POLLUTANTS_AVAILABLE.keys()), default=["CO", "NOx", "PM"])
    
    st.markdown("### 3. Accuracy Controls")
    use_temp = st.checkbox("Temp Correction", True)
    temp = st.slider("Ambient Temp (°C)", -10, 40, 25)
    use_cold = st.checkbox("Cold Start", True)
    trip = st.slider("Trip Length (km)", 1, 50, 10)
    
    st.markdown("### 4. Map Boundary")
    c1, c2 = st.columns(2)
    xmin = c1.number_input("Min Lon", value=3.37310, format="%.5f")
    xmax = c2.number_input("Max Lon", value=3.42430, format="%.5f")
    ymin = c1.number_input("Min Lat", value=6.43744, format="%.5f")
    ymax = c2.number_input("Max Lat", value=6.46934, format="%.5f")

# --- DASHBOARD VIEW ---
st.title("📊 Traffic Emission Intelligence Dashboard")

# Check inputs
req_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm, osm_file, ec_gas, ec_dsl, cc_gas, cc_dsl, c_2s, c_4s]
if not all(req_files):
    st.info("👋 Welcome! Please upload all required parameter and network files in the sidebar to begin.")
    st.stop()

# Load Data
try:
    link_df = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
    if link_df.shape[1] not in [7, 9]:
        st.error(f"Link data must have 7 or 9 columns. Found {link_df.shape[1]}.")
        st.stop()
except Exception as e:
    st.error(f"Error reading Link Data: {e}")
    st.stop()

# --- CALCULATION TRIGGER ---
if 'calc_results' not in st.session_state:
    st.session_state.calc_done = False

if st.sidebar.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Processing Network & Emissions..."):
        res, fuel_res = perform_calculation(
            pc_param.getvalue(), ldv_param.getvalue(), hdv_param.getvalue(), moto_param.getvalue(),
            link_df,
            ec_gas.getvalue(), ec_dsl.getvalue(), cc_gas.getvalue(), cc_dsl.getvalue(), c_2s.getvalue(), c_4s.getvalue(),
            sel_pollutants, temp, trip, use_temp, use_cold
        )
        st.session_state.calc_results = res
        st.session_state.fuel_results = fuel_res
        
        zone = [[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        coords, osmids, names, types = parse_osm_data(osm_file.getvalue(), zone, 0.005, 8)
        
        # Build Geometry Lookups
        geom_map = {oid: c for oid, c in zip(osmids, coords)}
        name_map = {oid: n for oid, n in zip(osmids, names)}
        
        # Prepare Analysis Dataframe
        analysis_data = link_df.copy()
        analysis_data.columns = ['OSM_ID', 'Len', 'Flow', 'Speed', 'Prop_Gas', 'Prop_PC', 'Prop_4S'] + (['Prop_LDV', 'Prop_HDV'] if link_df.shape[1]==9 else [])
        for p in sel_pollutants:
            analysis_data[f"{p}_Total"] = res[p]['total']
        
        st.session_state.analysis_df = analysis_data
        st.session_state.geom_map = geom_map
        st.session_state.name_map = name_map
        st.session_state.calc_done = True

# --- DASHBOARD VIEW ---
if st.session_state.get('calc_done', False):
    
    # Unit Selector
    st.sidebar.markdown("### 5. Display Units")
    units = {}
    for p in sel_pollutants:
        u_opts = list(UNIT_CONVERSIONS.get(p, {}).keys())
        if not u_opts: u_opts = [POLLUTANTS_AVAILABLE[p]['unit']]
        units[p] = st.sidebar.selectbox(f"{p}", u_opts, index=0)

    tab_map, tab_charts, tab_data = st.tabs(["🗺️ Interactive Map", "📈 Deep Dive Analysis", "📥 Data Export"])
    
    # 1. INTERACTIVE MAP
    with tab_map:
        c_ctrl, c_viz = st.columns([1, 4])
        
        with c_ctrl:
            st.markdown("#### Map Layers")
            map_pollutant = st.selectbox("Pollutant Layer", sel_pollutants)
            map_style = st.selectbox("Base Style", ["carto-positron", "open-street-map", "carto-darkmatter"])
            
            st.markdown("#### Color Scheme")
            color_option = st.selectbox("Color Scale", ["Jet (Blue-Red)", "White-Red", "Viridis", "Plasma", "Inferno"])
            
            # Map color scale selection logic
            if color_option == "Jet (Blue-Red)":
                color_scale = "Jet"
            elif color_option == "White-Red":
                color_scale = "Reds"
            else:
                color_scale = color_option
            
            st.markdown("#### Real-time Filters")
            df = st.session_state.analysis_df
            f_speed = st.slider("Speed (km/h)", int(df['Speed'].min()), int(df['Speed'].max()), (0, 130))
            f_flow = st.slider("Flow (veh/h)", int(df['Flow'].min()), int(df['Flow'].max()), (0, int(df['Flow'].max())))
            
            mask = (df['Speed'].between(f_speed[0], f_speed[1])) & (df['Flow'].between(f_flow[0], f_flow[1]))
            filtered_df = df[mask].copy()
            
            tot = filtered_df[f"{map_pollutant}_Total"].sum()
            u_fac = convert_value(1, map_pollutant, units[map_pollutant])
            st.metric(f"Total {map_pollutant} (Visible)", f"{tot * u_fac:,.2f}", units[map_pollutant])

        with c_viz:
            geom_map = st.session_state.geom_map
            name_map = st.session_state.name_map
            
            # Binning for Performance
            filtered_df['color_bin'] = pd.qcut(filtered_df[f"{map_pollutant}_Total"], 5, labels=False, duplicates='drop')
            
            fig = go.Figure()
            
            # Use Plotly's built-in colorscales based on selection
            if color_scale == "Jet":
                c_seq = px.colors.sequential.Jet
            elif color_scale == "Reds":
                c_seq = px.colors.sequential.Reds
            elif color_scale == "Viridis":
                c_seq = px.colors.sequential.Viridis
            elif color_scale == "Plasma":
                c_seq = px.colors.sequential.Plasma
            elif color_scale == "Inferno":
                c_seq = px.colors.sequential.Inferno
            else:
                c_seq = px.colors.sequential.Jet

            bins = sorted(filtered_df['color_bin'].unique())
            for b in bins:
                subset = filtered_df[filtered_df['color_bin'] == b]
                b_lats = []
                b_lons = []
                
                val_mean = subset[f"{map_pollutant}_Total"].mean() * u_fac
                # Safely map bin to color index
                c_idx = int(b * (len(c_seq)-1) / (max(bins) if max(bins)>0 else 1))
                c_code = c_seq[c_idx]
                
                for _, row in subset.iterrows():
                    oid = int(row['OSM_ID'])
                    if oid in geom_map:
                        g = geom_map[oid]
                        ls = list(zip(*g))
                        b_lons.extend(ls[0] + (None,))
                        b_lats.extend(ls[1] + (None,))
                
                fig.add_trace(go.Scattermapbox(
                    lat=b_lats, lon=b_lons,
                    mode='lines',
                    line=dict(width=3, color=c_code),
                    name=f"Level {int(b)+1} (~{val_mean:.2f})",
                    hoverinfo='name' 
                ))
            
            # Centroid scatter for tooltip
            center_lats, center_lons, center_text = [], [], []
            for _, row in filtered_df.iterrows():
                oid = int(row['OSM_ID'])
                if oid in geom_map:
                    g = geom_map[oid]
                    mid = len(g)//2
                    center_lons.append(g[mid][0])
                    center_lats.append(g[mid][1])
                    t_val = row[f"{map_pollutant}_Total"] * u_fac
                    nm = name_map.get(oid, 'Unknown Road')
                    txt = f"<b>{nm}</b><br>ID: {oid}<br>Speed: {row['Speed']} km/h<br>Flow: {row['Flow']}<br><b>{map_pollutant}: {t_val:.4f} {units[map_pollutant]}</b>"
                    center_text.append(txt)
            
            fig.add_trace(go.Scattermapbox(
                lat=center_lats, lon=center_lons,
                mode='markers',
                marker=dict(size=10, opacity=0),
                text=center_text,
                hoverinfo='text',
                name='Info'
            ))

            fig.update_layout(
                mapbox_style=map_style,
                mapbox_zoom=10,
                mapbox_center={"lat": (ymin+ymax)/2, "lon": (xmin+xmax)/2},
                margin={"r":0,"t":0,"l":0,"b":0},
                height=600,
                legend=dict(orientation="h", y=1, x=0, bgcolor="rgba(255,255,255,0.8)")
            )
            st.plotly_chart(fig, use_container_width=True)

    # 2. ANALYSIS CHARTS
    with tab_charts:
        df = st.session_state.analysis_df
        
        # -- Row 1: Key Drivers (Speed & Inequality) --
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📉 Speed Efficiency Curve")
            st.caption("Do higher speeds reduce or increase emissions? (Look for the 'U-shape')")
            # Scatter of Speed vs Total Emission
            chart_poll = st.selectbox("Analyze Pollutant", sel_pollutants, key="scatter_p")
            u = units[chart_poll]
            fac = convert_value(1, chart_poll, u)
            
            # Add converted column for plotting
            plot_df = df.copy()
            plot_df['Emission_Val'] = plot_df[f"{chart_poll}_Total"] * fac
            
            fig_scat = px.scatter(
                plot_df, x="Speed", y="Emission_Val", 
                color="Flow", size="Flow",
                hover_data=["OSM_ID"],
                color_continuous_scale="Viridis",
                labels={"Emission_Val": f"{chart_poll} ({u})", "Speed": "Speed (km/h)"},
                title=f"{chart_poll} vs Speed Analysis"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with c2:
            st.markdown("### 📊 Pareto Analysis (Lorenz Curve)")
            st.caption("Inequality Check: What % of roads cause the most emissions?")
            # Sort by emission descending
            sorted_df = plot_df.sort_values(by="Emission_Val", ascending=False)
            sorted_df['Cum_Emission'] = sorted_df['Emission_Val'].cumsum()
            sorted_df['Cum_Emission_Pct'] = sorted_df['Cum_Emission'] / sorted_df['Emission_Val'].sum()
            sorted_df['Road_Pct'] = np.arange(1, len(sorted_df)+1) / len(sorted_df)
            
            fig_lorenz = px.line(
                sorted_df, x="Road_Pct", y="Cum_Emission_Pct",
                title="Emission Inequality (Lorenz Curve)",
                labels={"Road_Pct": "Cumulative % of Roads", "Cum_Emission_Pct": "Cumulative % of Emissions"}
            )
            # Add diagonal perfect equality line
            fig_lorenz.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_lorenz, use_container_width=True)

        # -- Row 2: Composition & Variance --
        c3, c4 = st.columns(2)
        
        with c3:
            st.markdown("### 📦 Emission Variance by Vehicle Type")
            st.caption("Distribution of emissions per link for each vehicle class")
            # Prepare long format for box plot
            res = st.session_state.calc_results
            box_data = []
            p = chart_poll # Use same pollutant as above
            for v_type in ['pc', 'ldv', 'hdv', 'moto']:
                vals = res[p][v_type] * fac
                # Sample if too large to prevent lag, or just plot all
                # Let's filter out zero-emission links for log scale clarity if needed
                active_vals = vals[vals > 0]
                box_data.extend([{'Type': v_type.upper(), 'Emission': v} for v in active_vals])
            
            box_df = pd.DataFrame(box_data)
            fig_box = px.box(
                box_df, x="Type", y="Emission", color="Type",
                title=f"{p} Emission Spread by Vehicle",
                labels={"Emission": f"{p} ({u})"}
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c4:
            st.markdown("### ⛽ Fuel Contribution Breakdown")
            f_res = st.session_state.fuel_results
            f_data = []
            for p in sel_pollutants:
                u_p = units[p]
                f = convert_value(1, p, u_p)
                f_data.append({'Pollutant': p, 'Fuel': 'Gasoline', 'Value': f_res[p]['gas'].sum() * f})
                f_data.append({'Pollutant': p, 'Fuel': 'Diesel', 'Value': f_res[p]['diesel'].sum() * f})
            
            f_df = pd.DataFrame(f_data)
            fig_f = px.bar(f_df, x="Pollutant", y="Value", color="Fuel", title="Gasoline vs Diesel Split", barmode='group')
            st.plotly_chart(fig_f, use_container_width=True)
            
    # 3. EXPORT
    with tab_data:
        st.markdown("### Download Full Results")
        export_df = st.session_state.analysis_df.copy()
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "emissions_results.csv", "text/csv")
        st.info("Full calculation data including inputs and results available.")

else:
    st.markdown("""
    <div style='padding: 50px; text-align: center; color: #666;'>
        <h2>waiting for simulation run...</h2>
        <p>Upload files in sidebar and click 'Run Simulation'</p>
    </div>
    """, unsafe_allow_html=True)
