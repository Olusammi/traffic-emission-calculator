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
import matplotlib.pyplot as plt 

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Traffic Emission Intelligence", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🚗"
)

# Custom CSS for Power BI feel
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 10px;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Traffic Emission Intelligence Dashboard")
st.caption("Advanced Calculation & Interactive Visualization v2.2")

# ==================== CONSTANTS & CONFIG ====================
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

# ==================== SIDEBAR ====================
st.sidebar.header("📂 Data Input")

with st.sidebar.expander("COPERT Parameters", expanded=True):
    pc_param = st.file_uploader("PC Params (CSV)", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Params (CSV)", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Params (CSV)", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Params (CSV)", type=['csv'], key='moto')

with st.sidebar.expander("Network & Traffic", expanded=True):
    link_osm = st.file_uploader("Link Data (.dat/.txt)", type=['dat','txt','csv'])
    osm_file = st.file_uploader("OSM Geometry (.osm)", type=['osm'])

with st.sidebar.expander("Fleet Distributions", expanded=False):
    engine_cap_gas = st.file_uploader("Eng Cap Gas", type=['dat','txt'], key='ecg')
    engine_cap_diesel = st.file_uploader("Eng Cap Diesel", type=['dat','txt'], key='ecd')
    copert_class_gas = st.file_uploader("Class Gas", type=['dat','txt'], key='ccg')
    copert_class_diesel = st.file_uploader("Class Diesel", type=['dat','txt'], key='ccd')
    copert_2stroke = st.file_uploader("Moto 2S", type=['dat','txt'], key='2s')
    copert_4stroke = st.file_uploader("Moto 4S", type=['dat','txt'], key='4s')

st.sidebar.markdown("### ⚙️ Settings")
selected_pollutants = st.sidebar.multiselect("Pollutants", list(POLLUTANTS_AVAILABLE.keys()), default=["CO", "NOx", "PM"])

# Accuracy Controls
include_temp = st.sidebar.checkbox("Temp Correction", True)
temp = st.sidebar.slider("Ambient Temp (°C)", -10, 40, 25)
include_cold = st.sidebar.checkbox("Cold Start", True)
trip = st.sidebar.slider("Trip Length (km)", 1, 50, 10)

# Map Boundary
st.sidebar.markdown("### 🗺️ Domain")
c1, c2 = st.sidebar.columns(2)
x_min = c1.number_input("Min Lon", value=3.37310, format="%.5f")
x_max = c2.number_input("Max Lon", value=3.42430, format="%.5f")
y_min = c1.number_input("Min Lat", value=6.43744, format="%.5f")
y_max = c2.number_input("Max Lat", value=6.46934, format="%.5f")
tolerance = st.sidebar.number_input("Tolerance", value=0.005, format="%.3f")
ncore = st.sidebar.number_input("Cores", value=8, min_value=1, max_value=16)

# Reset Button (Fix for Stale State)
st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Reset App State"):
    st.session_state.clear()
    st.rerun()

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Guide", 
    "📊 Preview", 
    "⚙️ Calculate", 
    "🗺️ Interactive Map", 
    "📈 Analysis",
    "📥 Export"
])

# --- TAB 1: GUIDE ---
with tab1:
    st.header("📖 User Guide")
    st.markdown("""
    ### Quick Start
    1. **Upload Data**: Ensure all COPERT parameters and Network files are uploaded in the sidebar.
    2. **Calculate**: Go to the **Calculate** tab and click 'Run Simulation'.
    3. **Explore**: 
       - Use **Interactive Map** for spatial analysis.
       - Use **Analysis** tab for deep-dive charts.
    
    **Note:** If you encounter errors after changing settings, try clicking **Reset App State** in the sidebar.
    """)

# --- TAB 2: PREVIEW ---
with tab2:
    st.header("Data Preview")
    if link_osm:
        try:
            link_osm.seek(0)
            df = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            if df.shape[1] >= 7:
                cols = ['OSM_ID','Length_km','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop']
                if df.shape[1] == 9: cols += ['LDV_Prop', 'HDV_Prop']
                df.columns = cols
            st.dataframe(df.head())
            st.caption(f"Rows: {len(df)} | Columns: {df.shape[1]}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- TAB 3: CALCULATE ---
with tab3:
    st.header("⚙️ Emission Calculation")
    
    req_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm, 
                 engine_cap_gas, engine_cap_diesel, copert_class_gas, 
                 copert_class_diesel, copert_2stroke, copert_4stroke]
    
    if all(req_files) and selected_pollutants:
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Processing Model..."):
                try:
                    import copert
                    # Temp file handling
                    with tempfile.TemporaryDirectory() as tmpdir:
                        paths = {}
                        for name, obj in [('pc', pc_param), ('ldv', ldv_param), ('hdv', hdv_param), ('moto', moto_param)]:
                            p = os.path.join(tmpdir, f"{name}.csv")
                            with open(p, 'wb') as f: f.write(obj.getbuffer())
                            paths[name] = p
                        
                        cop = copert.Copert(paths['pc'], paths['ldv'], paths['hdv'], paths['moto'])
                        
                        # Reset pointers
                        for f in [link_osm, engine_cap_gas, engine_cap_diesel, copert_class_gas, copert_class_diesel, copert_2stroke, copert_4stroke]:
                            f.seek(0)

                        # Load Data
                        link_df = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
                        data_link = link_df.values
                        d_ec_gas = np.loadtxt(engine_cap_gas)
                        d_ec_diesel = np.loadtxt(engine_cap_diesel)
                        d_cc_gas = np.loadtxt(copert_class_gas)
                        d_cc_diesel = np.loadtxt(copert_class_diesel)
                        d_c_2s = np.loadtxt(copert_2stroke)
                        d_c_4s = np.loadtxt(copert_4stroke)

                        # Handle LDV/HDV defaults
                        Nlink = len(data_link)
                        if data_link.shape[1] == 7:
                            P_ldv, P_hdv = np.zeros(Nlink), np.zeros(Nlink)
                            d_cc_ldv = d_cc_gas
                            # HDV Default: Euro VI (idx 5), Rigid < 7.5t (idx 0)
                            d_hdv = np.zeros((Nlink, 6, 15))
                            d_hdv[:, 5, 0] = 1.0
                        else:
                            P_ldv, P_hdv = data_link[:, 7], data_link[:, 8]
                            d_cc_ldv = d_cc_gas
                            d_hdv = np.zeros((Nlink, 6, 15))
                            d_hdv[:, 5, 0] = 1.0 

                        # Results containers
                        emissions = {p: {'pc': np.zeros(Nlink), 'ldv': np.zeros(Nlink), 'hdv': np.zeros(Nlink), 
                                         'moto': np.zeros(Nlink), 'total': np.zeros(Nlink)} for p in selected_pollutants}
                        fuel_emissions = {p: {'gas': np.zeros(Nlink), 'diesel': np.zeros(Nlink)} for p in selected_pollutants}

                        # Mappings
                        POLL_MAP = {"CO": cop.pollutant_CO, "CO2": cop.pollutant_FC, "NOx": cop.pollutant_NOx,
                                    "PM": cop.pollutant_PM, "VOC": cop.pollutant_VOC, "FC": cop.pollutant_FC}
                        
                        prog = st.progress(0)
                        
                        # === CALCULATION LOOP ===
                        # Copert Setup
                        engines = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        caps = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        classes = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02, cop.class_ECE_15_03,
                                   cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop, cop.class_Euro_1,
                                   cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5, cop.class_Euro_6, cop.class_Euro_6c]
                        
                        m_eng = [cop.engine_type_moto_two_stroke_more_50, cop.engine_type_moto_four_stroke_50_250]
                        m_classes = [cop.class_moto_Conventional, cop.class_moto_Euro_1, cop.class_moto_Euro_2, 
                                     cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]

                        for i in range(Nlink):
                            if i % 100 == 0: prog.progress(i/Nlink)
                            
                            L, Flow, V = data_link[i, 1], data_link[i, 2], min(max(10., data_link[i, 3]), 130.)
                            prop_gas, prop_pc, prop_4s = data_link[i, 4], data_link[i, 5], data_link[i, 6]
                            
                            dist_eng = [prop_gas, 1.-prop_gas]
                            dist_cap = [d_ec_gas[i], d_ec_diesel[i]]
                            
                            for p_name in selected_pollutants:
                                p_type = POLL_MAP[p_name]
                                pc_sum = moto_sum = gas_sum = dsl_sum = 0.0

                                # PC Calculation
                                for t in range(2): # 0=Gas, 1=Diesel
                                    for c_idx, cls in enumerate(classes):
                                        for k in range(2):
                                            if t==1 and k==0 and cls in range(cop.class_Euro_1, cop.class_Euro_3+1): continue
                                            try:
                                                e = cop.Emission(p_type, V, L, cop.vehicle_type_passenger_car, engines[t], cls, caps[k], temp)
                                            except: e = 0.0
                                            if p_name=="NOx" and include_temp: e *= (1 + 0.02*(temp-20))
                                            
                                            share = d_cc_gas[i, c_idx] if t==0 else d_cc_diesel[i, c_idx]
                                            val = e * dist_eng[t] * dist_cap[t][k] * share * prop_pc / L * Flow
                                            pc_sum += val
                                            if t==0: gas_sum+=val 
                                            else: dsl_sum+=val
                                
                                # Moto Calculation
                                m_dist_arr = [1.0 - prop_4s, prop_4s] 
                                for m in range(2):
                                    for mc in m_classes:
                                        if m==0 and mc >= cop.class_moto_Euro_1: continue
                                        try: ef = cop.EFMotorcycle(p_type, V, m_eng[m], mc)
                                        except: ef = 0.0
                                        # EFMotorcycle returns g/km, calculate total
                                        val = ef * m_dist_arr[m] * (1.0-prop_pc) * Flow
                                        moto_sum += val
                                        gas_sum += val
                                
                                emissions[p_name]['pc'][i] = pc_sum
                                emissions[p_name]['moto'][i] = moto_sum
                                emissions[p_name]['total'][i] = pc_sum + moto_sum # + ldv + hdv (simplified for now)
                                fuel_emissions[p_name]['gas'][i] = gas_sum
                                fuel_emissions[p_name]['diesel'][i] = dsl_sum

                        prog.empty()
                        
                        # Store in Session State
                        st.session_state.emissions_data = emissions
                        st.session_state.fuel_data = fuel_emissions
                        st.session_state.link_df = link_df  # Store DF here
                        st.session_state.calc_done = True
                        
                        st.success("✅ Calculation Complete!")
                        
                        # Parse OSM for Map
                        if osm_file:
                            import osm_network
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.osm') as tmp:
                                osm_file.seek(0)
                                tmp.write(osm_file.read())
                                osm_path = tmp.name
                            
                            zone = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                            coords, osmids, names, types = osm_network.retrieve_highway(osm_path, zone, tolerance, int(ncore))
                            st.session_state.geo_data = {'coords': coords, 'ids': osmids, 'names': names}
                            os.unlink(osm_path)

                except Exception as e:
                    st.error(f"Calculation Error: {e}")
                    st.exception(e)
    else:
        st.info("Please upload all files to enable calculation.")

# --- TAB 4: INTERACTIVE MAP ---
with tab4:
    st.header("🗺️ Interactive Map")
    
    # SAFETY CHECK: Ensure link_df exists
    if st.session_state.get('calc_done') and 'link_df' in st.session_state and st.session_state.get('geo_data'):
        # Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            map_poll = st.selectbox("Pollutant Layer", selected_pollutants)
            map_style = st.selectbox("Base Map", ["carto-positron", "carto-darkmatter", "open-street-map"])
        with c2:
            color_theme = st.selectbox("Color Theme", ["Jet", "Viridis", "Plasma", "Inferno", "Reds"])
            line_scale = st.slider("Line Width Scale", 1.0, 5.0, 2.5)

        # Prepare Data
        geo = st.session_state.geo_data
        emis = st.session_state.emissions_data[map_poll]['total']
        link_df = st.session_state.link_df
        
        # Create Mapping
        id_to_idx = {int(row[0]): i for i, row in enumerate(link_df.values)}
        
        # Filtering controls
        with c3:
            st.markdown("**Filters**")
            f_speed = st.slider("Filter Speed (km/h)", 0, 130, (0, 130))
        
        # Create DF for map
        map_rows = []
        for coords, oid, name in zip(geo['coords'], geo['ids'], geo['names']):
            if oid in id_to_idx:
                idx = id_to_idx[oid]
                val = emis[idx]
                sp = link_df.values[idx, 3]
                
                if f_speed[0] <= sp <= f_speed[1]:
                    map_rows.append({'oid': oid, 'val': val, 'coords': coords, 'name': name, 'speed': sp})
        
        map_df = pd.DataFrame(map_rows)
        if not map_df.empty:
            map_df['quartile'] = pd.qcut(map_df['val'], 4, labels=["Low", "Medium", "High", "Critical"], duplicates='drop')
            
            # Colors based on selection
            if color_theme == "Jet": colors = px.colors.sequential.Jet
            elif color_theme == "Viridis": colors = px.colors.sequential.Viridis
            elif color_theme == "Reds": colors = px.colors.sequential.Reds
            else: colors = px.colors.sequential.Plasma
            
            # Create a trace per quartile
            qs = map_df['quartile'].unique()
            fig = go.Figure()
            
            for i, q in enumerate(sorted(qs)):
                subset = map_df[map_df['quartile'] == q]
                c_lats, c_lons = [], []
                for cs in subset['coords']:
                    unzipped = list(zip(*cs))
                    c_lons.extend(unzipped[0] + (None,))
                    c_lats.extend(unzipped[1] + (None,))
                
                c_idx = int(i / (len(qs)-1 or 1) * (len(colors)-1))
                
                fig.add_trace(go.Scattermapbox(
                    lat=c_lats, lon=c_lons,
                    mode='lines',
                    line=dict(width=line_scale, color=colors[c_idx]),
                    name=f"{q} Emission",
                    hoverinfo='skip'
                ))
            
            # Tooltip Layer
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
            st.warning("No data passed filters.")
    else:
        st.info("Run calculation and provide OSM file to view map.")

# --- TAB 5: ANALYSIS ---
with tab5:
    st.header("📈 Deep Dive Analysis")
    
    # SAFETY CHECK: Ensure link_df exists
    if st.session_state.get('calc_done') and 'link_df' in st.session_state:
        # Prepare Analysis DF
        df = st.session_state.link_df.copy()
        df.columns = ['OSM_ID','Length','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop'] + (['LDV','HDV'] if df.shape[1]==9 else [])
        
        # Dropdown for Chart Selection
        chart_type = st.selectbox(
            "Select Analysis View", 
            [
                "Emission Sources (Vehicle Split)",
                "Fuel Contribution (Gas vs Diesel)",
                "Speed vs Emission Efficiency",
                "Inequality Analysis (Lorenz Curve)",
                "Top Polluting Roads"
            ]
        )
        
        target_poll = st.selectbox("Select Pollutant", selected_pollutants, key="an_poll")
        total_vals = st.session_state.emissions_data[target_poll]
        
        st.markdown("---")
        
        if chart_type == "Emission Sources (Vehicle Split)":
            sources = {
                'Passenger Cars': total_vals['pc'].sum(),
                'Motorcycles': total_vals['moto'].sum(),
                'LDV': total_vals['ldv'].sum(),
                'HDV': total_vals['hdv'].sum()
            }
            fig = px.bar(
                x=list(sources.keys()), y=list(sources.values()),
                color=list(sources.keys()),
                labels={'x': 'Vehicle Type', 'y': f'Total {target_poll} ({POLLUTANTS_AVAILABLE[target_poll]["unit"]})'},
                title=f"Total {target_poll} by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Fuel Contribution (Gas vs Diesel)":
            fd = st.session_state.fuel_data[target_poll]
            fig = px.pie(
                values=[fd['gas'].sum(), fd['diesel'].sum()],
                names=['Gasoline', 'Diesel'],
                title=f"Fuel Share for {target_poll}",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Speed vs Emission Efficiency":
            df['Total_E'] = total_vals['total']
            fig = px.scatter(
                df, x='Speed', y='Total_E',
                color='Flow', size='Flow',
                hover_data=['OSM_ID'],
                title=f"Speed vs {target_poll} Emission",
                labels={'Total_E': f'Emission ({POLLUTANTS_AVAILABLE[target_poll]["unit"]})'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Inequality Analysis (Lorenz Curve)":
            vals = np.sort(total_vals['total'])
            cum_vals = np.cumsum(vals) / np.sum(vals)
            x_axis = np.linspace(0, 1, len(vals))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=cum_vals, mode='lines', name='Actual Distribution'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Perfect Equality', line=dict(dash='dash')))
            fig.update_layout(title="Lorenz Curve (Emission Inequality)", xaxis_title="Cumulative % of Roads", yaxis_title="Cumulative % of Emissions")
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Top Polluting Roads":
            df['Total_E'] = total_vals['total']
            top_10 = df.nlargest(10, 'Total_E')
            top_10['Road_Label'] = top_10['OSM_ID'].astype(str)
            fig = px.bar(
                top_10, x='Road_Label', y='Total_E',
                color='Speed',
                title=f"Top 10 Roads by {target_poll}",
                labels={'Total_E': 'Emission'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("Run calculation first to view analysis.")

# --- TAB 6: EXPORT ---
with tab6:
    st.header("📥 Export Data")
    if st.session_state.get('calc_done') and 'link_df' in st.session_state:
        export_df = st.session_state.link_df.copy()
        export_df.columns = ['OSM_ID','Length','Flow','Speed','Gas_Prop','PC_Prop','4S_Prop'] + (['LDV','HDV'] if export_df.shape[1]==9 else [])
        
        for p in selected_pollutants:
            export_df[f"{p}_Total"] = st.session_state.emissions_data[p]['total']
            
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV Results", csv, "traffic_emissions.csv", "text/csv")
        
        if st.button("📦 Generate Full ZIP Package"):
            buf = BytesIO()
            with zipfile.ZipFile(buf, 'w') as z:
                z.writestr("emissions_full.csv", export_df.to_csv(index=False))
                summary = f"Generated on {pd.Timestamp.now()}\nTotal Links: {len(export_df)}"
                z.writestr("summary.txt", summary)
            buf.seek(0)
            st.download_button("⬇️ Download ZIP", buf, "emission_pkg.zip", "application/zip")
            
    else:
        st.info("No results to export.")

# Footer
st.markdown("---")
st.caption("Developed by SHassan | v2.2")
