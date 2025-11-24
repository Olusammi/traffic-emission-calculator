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

# Import local modules
try:
    import copert
    import osm_network
except ImportError:
    st.error("Critical modules (copert.py, osm_network.py) not found. Please ensure they are in the application directory.")

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Traffic Emission Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚗"
)

# Custom CSS for PowerBI-like look
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .metric-card {
        background-color: white; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa; }
    .stPlotlyChart { border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
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

# ==================== LOGIC FUNCTIONS ====================
def convert_value(val, pollutant, target_unit):
    if pollutant in UNIT_CONVERSIONS and target_unit in UNIT_CONVERSIONS[pollutant]:
        return val * UNIT_CONVERSIONS[pollutant][target_unit]
    return val

@st.cache_data(show_spinner=False)
def parse_osm_data(osm_content_bytes, zone, tolerance, ncore):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
        tmp.write(osm_content_bytes)
        tmp_path = tmp.name
    try:
        return osm_network.retrieve_highway(tmp_path, zone, tolerance, ncore)
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

@st.cache_data(show_spinner=False)
def perform_calculation(
    _pc_file, _ldv_file, _hdv_file, _moto_file, 
    _link_data_df, 
    _ec_gas, _ec_diesel, _cc_gas, _cc_diesel, _c_2s, _c_4s,
    selected_polls, ambient_temp, trip_len, use_temp_corr, use_cold_start
):
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

# ==================== MAIN UI ====================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=64)
    st.title("Traffic Emission Calculator")
    st.caption("v2.2 | Research Grade")
    
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

# --- MAIN DASHBOARD ---
st.title("📊 Traffic Emission Intelligence Dashboard")

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

# --- CALCULATION ---
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
        
        # Maps
        geom_map = {oid: c for oid, c in zip(osmids, coords)}
        name_map = {oid: n for oid, n in zip(osmids, names)}
        
        # DataFrame
        analysis_data = link_df.copy()
        analysis_data.columns = ['OSM_ID', 'Len', 'Flow', 'Speed', 'Prop_Gas', 'Prop_PC', 'Prop_4S'] + (['Prop_LDV', 'Prop_HDV'] if link_df.shape[1]==9 else [])
        for p in sel_pollutants:
            analysis_data[f"{p}_Total"] = res[p]['total']
        
        st.session_state.analysis_df = analysis_data
        st.session_state.geom_map = geom_map
        st.session_state.name_map = name_map
        st.session_state.calc_done = True

if st.session_state.get('calc_done', False):
    
    st.sidebar.markdown("### 5. Display Units")
    units = {}
    for p in sel_pollutants:
        u_opts = list(UNIT_CONVERSIONS.get(p, {}).keys())
        if not u_opts: u_opts = [POLLUTANTS_AVAILABLE[p]['unit']]
        units[p] = st.sidebar.selectbox(f"{p}", u_opts, index=0)

    tab_map, tab_charts, tab_data = st.tabs(["🗺️ Interactive Map", "📈 Deep Dive Analysis", "📥 Data Export"])
    
    # 1. MAP
    with tab_map:
        c_ctrl, c_viz = st.columns([1, 4])
        
        with c_ctrl:
            st.markdown("#### Layers")
            map_pollutant = st.selectbox("Pollutant", sel_pollutants)
            map_style = st.selectbox("Base Map", ["carto-positron", "open-street-map", "carto-darkmatter"])
            # Updated Map Color Options
            color_option = st.selectbox("Color Scale", ["Jet (Blue-Red)", "White-Red", "Viridis", "Inferno"], index=0)
            
            # Map Logic
            if color_option == "Jet (Blue-Red)": c_seq = px.colors.sequential.Jet
            elif color_option == "White-Red": c_seq = px.colors.sequential.Reds
            elif color_option == "Viridis": c_seq = px.colors.sequential.Viridis
            else: c_seq = px.colors.sequential.Inferno

            st.markdown("#### Filters")
            df = st.session_state.analysis_df
            f_speed = st.slider("Speed", int(df['Speed'].min()), int(df['Speed'].max()), (0, 130))
            f_flow = st.slider("Flow", int(df['Flow'].min()), int(df['Flow'].max()), (0, int(df['Flow'].max())))
            
            mask = (df['Speed'].between(f_speed[0], f_speed[1])) & (df['Flow'].between(f_flow[0], f_flow[1]))
            filtered_df = df[mask].copy()
            
            tot = filtered_df[f"{map_pollutant}_Total"].sum()
            u_fac = convert_value(1, map_pollutant, units[map_pollutant])
            st.metric(f"Total {map_pollutant}", f"{tot * u_fac:,.2f}", units[map_pollutant])

        with c_viz:
            geom_map = st.session_state.geom_map
            name_map = st.session_state.name_map
            
            # 4 Groups (Quartiles)
            filtered_df['color_bin'] = pd.qcut(filtered_df[f"{map_pollutant}_Total"], 4, labels=False, duplicates='drop')
            
            fig = go.Figure()
            bins = sorted(filtered_df['color_bin'].unique())
            
            bin_labels = ["Low", "Medium", "High", "Critical"]
            
            for b in bins:
                subset = filtered_df[filtered_df['color_bin'] == b]
                b_lats, b_lons = [], []
                val_mean = subset[f"{map_pollutant}_Total"].mean() * u_fac
                
                # Color mapping
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
                    name=f"{bin_labels[int(b)]} (Avg {val_mean:.2f})",
                    hoverinfo='name'
                ))
            
            # Tooltip layer
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
                lat=center_lats, lon=center_lons, mode='markers', marker=dict(size=10, opacity=0),
                text=center_text, hoverinfo='text', name='Info'
            ))

            fig.update_layout(
                mapbox_style=map_style, mapbox_zoom=10,
                mapbox_center={"lat": (ymin+ymax)/2, "lon": (xmin+xmax)/2},
                margin={"r":0,"t":0,"l":0,"b":0}, height=600,
                legend=dict(orientation="h", y=1, x=0, bgcolor="rgba(255,255,255,0.8)")
            )
            st.plotly_chart(fig, use_container_width=True)

    # 2. ANALYSIS
    with tab_charts:
        res = st.session_state.calc_results
        f_res = st.session_state.fuel_results
        
        # Row 1: Vehicle Composition
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🚗 Vehicle Composition")
            chart_type_v = st.selectbox("Chart Type", ["Bar Chart", "Donut Chart", "Pie Chart"], key="v_chart")
            p_v = st.selectbox("Pollutant", sel_pollutants, key="v_poll")
            u = units[p_v]
            fac = convert_value(1, p_v, u)
            
            v_data = [
                {'Type': 'PC', 'Value': res[p_v]['pc'].sum() * fac},
                {'Type': 'Moto', 'Value': res[p_v]['moto'].sum() * fac},
                {'Type': 'LDV', 'Value': res[p_v]['ldv'].sum() * fac},
                {'Type': 'HDV', 'Value': res[p_v]['hdv'].sum() * fac}
            ]
            v_df = pd.DataFrame(v_data)
            
            if chart_type_v == "Bar Chart":
                fig_v = px.bar(v_df, x="Type", y="Value", color="Type", title=f"{p_v} by Vehicle Type")
            elif chart_type_v == "Donut Chart":
                fig_v = px.pie(v_df, values="Value", names="Type", hole=0.4, title=f"{p_v} Share")
            else:
                fig_v = px.pie(v_df, values="Value", names="Type", title=f"{p_v} Share")
            st.plotly_chart(fig_v, use_container_width=True)

        # Row 2: Fuel Split
        with c2:
            st.markdown("### ⛽ Fuel Analysis")
            chart_type_f = st.selectbox("Chart Type", ["Grouped Bar", "Stacked Bar"], key="f_chart")
            f_data = []
            for p in sel_pollutants:
                u_p = units[p]
                f = convert_value(1, p, u_p)
                f_data.append({'Pollutant': p, 'Fuel': 'Gasoline', 'Value': f_res[p]['gas'].sum() * f})
                f_data.append({'Pollutant': p, 'Fuel': 'Diesel', 'Value': f_res[p]['diesel'].sum() * f})
            f_df = pd.DataFrame(f_data)
            
            if chart_type_f == "Grouped Bar":
                fig_f = px.bar(f_df, x="Pollutant", y="Value", color="Fuel", barmode='group', title="Fuel Contribution")
            else:
                fig_f = px.bar(f_df, x="Pollutant", y="Value", color="Fuel", barmode='stack', title="Fuel Contribution")
            st.plotly_chart(fig_f, use_container_width=True)
            
    # 3. EXPORT
    with tab_data:
        st.markdown("### Download Data")
        export_df = st.session_state.analysis_df.copy()
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "emissions_results.csv", "text/csv")

else:
    st.markdown("""
    <div style='padding: 50px; text-align: center; color: #666;'>
        <h2>waiting for simulation run...</h2>
        <p>Upload files in sidebar and click 'Run Simulation'</p>
    </div>
    """, unsafe_allow_html=True)
