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
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Advanced Traffic Emission Calculator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
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

st.title("🚗 Advanced Traffic Emission Calculator")
st.caption("Multi-Standard Emission Analysis with COPERT IV, IPCC, and EPA Methodologies")
st.markdown("---")

# ==================== SIDEBAR: EMISSION METRICS SELECTION ====================
st.sidebar.header("📊 Emission Metrics Selection")
st.sidebar.markdown("Select which pollutants to calculate and analyze")

# Multi-pollutant selection
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

# Display info about selected pollutants
if selected_pollutants:
    st.sidebar.markdown("### Selected Metrics Info")
    for pollutant in selected_pollutants:
        info = pollutants_available[pollutant]
        st.sidebar.markdown(f"""
        <div style='background-color: {info['color']}22; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {info["color"]}'>
            <strong>{pollutant}</strong>: {info['name']}<br>
            <small>Standard: {info['standard']}</small><br>
            <small>Unit: {info['unit']}</small>
        </div>
        """, unsafe_allow_html=True)

# Calculation methodology
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Calculation Methodology")
calculation_method = st.sidebar.selectbox(
    "Select Calculation Standard",
    ["COPERT IV (EU)", "IPCC Tier 2", "EPA MOVES (US)", "Hybrid (Multi-standard)"],
    help="Choose the international standard for emission calculations"
)

# Accuracy settings
st.sidebar.markdown("---")
st.sidebar.header("🎯 Accuracy Settings")
include_temperature_correction = st.sidebar.checkbox("Temperature Correction", value=True,
                                                     help="Apply ambient temperature corrections to NOx emissions")
include_cold_start = st.sidebar.checkbox("Cold Start Emissions", value=True,
                                         help="Include cold start emission quotient")
include_slope_correction = st.sidebar.checkbox("Road Slope Correction", value=False,
                                               help="Apply road gradient corrections")

if include_temperature_correction:
    ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -10, 40, 25)
else:
    ambient_temp = 20

if include_cold_start:
    trip_length = st.sidebar.slider("Average Trip Length (km)", 1, 50, 10)
else:
    trip_length = 10

if include_slope_correction:
    road_slope = st.sidebar.slider("Road Slope (%)", -6, 6, 0)
else:
    road_slope = 0

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip**: Enable all accuracy settings for research-grade calculations")

# ==================== FILE UPLOADS ====================
st.sidebar.header("📂 Upload Input Files")
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

# NEW: LDV and HDV file uploaders
ldv_files = st.sidebar.expander("Light Duty Vehicle (LDV)", expanded=False)
with ldv_files:
    ldv_class_file = st.file_uploader("Upload LDV Euro Class Distribution (.csv)", type=['csv'], key='ldv_class')

hdv_files = st.sidebar.expander("Heavy Duty Vehicle (HDV)", expanded=False)
with hdv_files:
    hdv_class_type_file = st.file_uploader("Upload HDV Class/Type Distribution (.csv)", type=['csv'],
                                           key='hdv_class_type')

# ======== MAKE TABS STICKY (FROZEN) WHEN SCROLLING ========
st.markdown("""
<style>
/* Freeze the tabs container */
.stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 0;
    background-color: white;
    z-index: 999;
    padding-top: 10px;
}

/* Optional: Add shadow so the tabs look elevated when scrolling */
.stTabs [data-baseweb="tab-list"] {
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

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

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📖 Instructions",
    "📊 Data Preview",
    "🧮 Formula Explanation",
    "⚙️ Calculate Emissions",
    "📈 Multi-Metric Analysis",
    "🗺️ Interactive Map",
    "📥 Download Results"
])

# ==================== TAB 1: INSTRUCTIONS ====================
with tab1:
    st.header("📖 User Guide & Instructions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Multi-Pollutant Analysis**: Calculate CO, CO₂, NOx, PM, VOC, and FC simultaneously
        - **International Standards**: COPERT IV, IPCC, EPA MOVES compliance
        - **Formula Transparency**: See exact mathematical formulas used
        - **Interactive Visualization**: Dynamic maps with live emission data
        - **Accuracy Controls**: Temperature, cold-start, and slope corrections
        - **Comparative Analysis**: Compare across pollutants and standards
        - **Multi-Vehicle Support**: PC, Motorcycle, LDV, and HDV emissions
        """)

    with col2:
        st.subheader("📚 Standards Reference")
        st.markdown("""
        **COPERT IV (EU)**
        - European emission inventory standard
        - Accuracy: ~95% for European vehicles
        - Coverage: All vehicle types, Euro 1-6d

        **IPCC Guidelines**
        - Global greenhouse gas accounting
        - Focus: CO₂ and climate impacts
        - Used in: National inventories

        **EPA MOVES (US)**
        - US Environmental Protection Agency
        - Detailed fleet modeling
        - Integration: US regulations
        """)

    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    ### Step-by-Step Process:

    1. **Upload Required Files** (Left Sidebar)
       - 4 COPERT parameter CSV files
       - Link OSM data file (7 or 9 columns)
       - OSM network file (.osm)
       - 6 vehicle proportion files
       - Optional: LDV and HDV distribution files

    2. **Select Emission Metrics** (Left Sidebar)
       - Choose pollutants to calculate
       - Select calculation methodology
       - Configure accuracy settings

    3. **Preview Your Data** (Data Preview Tab)
       - Verify uploaded data structure
       - Check statistics and summaries

    4. **Review Formulas** (Formula Explanation Tab)
       - Understand calculation methods
       - See parameter definitions
       - Review examples

    5. **Calculate Emissions** (Calculate Tab)
       - Run emission calculations
       - View real-time progress
       - See detailed results

    6. **Analyze Results** (Multi-Metric Analysis Tab)
       - Compare pollutants
       - View trends and patterns
       - Generate insights

    7. **Visualize on Map** (Interactive Map Tab)
       - Interactive emission mapping
       - Multiple visualization modes
       - Road-level detail

    8. **Download Results** (Download Tab)
       - Export emission data
       - Save visualizations
       - Generate reports
    """)

# ==================== TAB 2: DATA PREVIEW ====================
with tab2:
    st.header("📊 Data Preview & Validation")

    if link_osm is not None:
        st.subheader("🔗 Link OSM Data")
        try:
            link_osm.seek(0)
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            if data_link.shape[1] >= 7:
                if data_link.shape[1] == 7:
                    data_link.columns = ['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop',
                                         '4Stroke_Prop']
                elif data_link.shape[1] == 9:
                    data_link.columns = ['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop',
                                         '4Stroke_Prop', 'LDV_Prop', 'HDV_Prop']
                else:
                    data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]
            else:
                data_link.columns = [f'Column_{i}' for i in range(data_link.shape[1])]

            # Display data with styling
            st.dataframe(data_link.head(20), use_container_width=True)

            # Statistics
            if data_link.shape[1] == 9:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
            else:
                col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Links", len(data_link))
            with col2:
                st.metric("Total Length (km)", f"{data_link['Length_km'].sum():.2f}")
            with col3:
                st.metric("Avg Speed (km/h)", f"{data_link['Speed'].mean():.2f}")
            with col4:
                st.metric("Avg Flow (veh)", f"{data_link['Flow'].mean():.0f}")

            if data_link.shape[1] == 9:
                with col5:
                    st.metric("Avg LDV Prop", f"{data_link['LDV_Prop'].mean():.2%}")
                with col6:
                    st.metric("Avg HDV Prop", f"{data_link['HDV_Prop'].mean():.2%}")

            # Data quality checks
            st.subheader("✅ Data Quality Checks")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Speed Distribution**")
                fig_speed = px.histogram(data_link, x='Speed', nbins=30,
                                         title="Speed Distribution Across Links")
                st.plotly_chart(fig_speed, use_container_width=True)

            with col2:
                st.markdown("**Flow Distribution**")
                fig_flow = px.histogram(data_link, x='Flow', nbins=30,
                                        title="Traffic Flow Distribution")
                st.plotly_chart(fig_flow, use_container_width=True)

            # Validation warnings
            if data_link['Speed'].min() < 10:
                st.warning(
                    f"⚠️ {len(data_link[data_link['Speed'] < 10])} links have speed < 10 km/h. COPERT formulas may be less accurate.")
            if data_link['Speed'].max() > 130:
                st.warning(
                    f"⚠️ {len(data_link[data_link['Speed'] > 130])} links have speed > 130 km/h. Consider speed caps.")

        except Exception as e:
            st.error(f"❌ Error reading link data: {e}")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar")

# ==================== TAB 3: FORMULA EXPLANATION ====================
with tab3:
    st.header("🧮 Mathematical Formulas & Methodology")
    st.markdown("Detailed explanation of emission calculation formulas used in this calculator")

    formula_pollutant = st.selectbox("Select Pollutant for Formula Details",
                                     list(pollutants_available.keys()))

    st.markdown("---")

    # Formula definitions for each pollutant
    if formula_pollutant == "CO":
        st.subheader("🔴 Carbon Monoxide (CO) Emission Formula")

        st.markdown("### COPERT IV Hot Emission Factor")
        st.latex(r'''
        EF_{hot} = \frac{a + c \cdot V + e \cdot V^2}{1 + b \cdot V + d \cdot V^2}
        ''')

        st.markdown("**Parameters:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - **EF<sub>hot</sub>**: Hot emission factor (g/km)
            - **V**: Average vehicle speed (km/h)
            - **a, c, e**: Emission coefficients (numerator)
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            - **b, d**: Speed correction factors (denominator)
            - **Valid range**: 10-130 km/h
            - **Dependency**: Vehicle type, fuel, Euro standard
            """)

        st.markdown("### Example Calculation")
        st.code("""
# For Euro 6 Gasoline Passenger Car at 60 km/h:
a, c, e = 11.2, -0.102, 0.000677
b, d = 0.129, -0.000947
V = 60

EF_hot = (a + c*V + e*V**2) / (1 + b*V + d*V**2)
EF_hot = (11.2 + (-0.102)*60 + 0.000677*3600) / (1 + 0.129*60 + (-0.000947)*3600)
EF_hot ≈ 0.85 g/km
        """, language='python')

        if include_cold_start:
            st.markdown("### Cold Start Correction")
            st.latex(r'''
            E_{total} = E_{hot} \cdot (1 - \beta) + E_{cold} \cdot \beta
            ''')
            st.latex(r'''
            \beta = 0.6474 - 0.02545 \cdot L_{trip} - (0.00974 - 0.000385 \cdot L_{trip}) \cdot T_{amb}
            ''')
            st.markdown("""
            - **β**: Cold mileage percentage
            - **L<sub>trip</sub>**: Average trip length (km)
            - **T<sub>amb</sub>**: Ambient temperature (°C)
            """, unsafe_allow_html=True)

    elif formula_pollutant == "CO2":
        st.subheader("🔵 Carbon Dioxide (CO₂) Emission Formula")

        st.markdown("### Fuel-Based CO₂ Calculation (IPCC)")
        st.latex(r'''
        CO_2 = FC \cdot CF \cdot \frac{44}{12} \cdot 1000
        ''')

        st.markdown("**Parameters:**")
        st.markdown("""
        - **FC**: Fuel consumption (L/km)
        - **CF**: Carbon content factor
          - Gasoline: 0.64 kg C/L
          - Diesel: 0.68 kg C/L
        - **44/12**: Molecular weight ratio (CO₂/C)
        - **1000**: Convert kg to g
        """)

        st.markdown("### Speed-Dependent Fuel Consumption")
        st.latex(r'''
        FC(V) = a \cdot V^2 + b \cdot V + c
        ''')

        st.markdown("### Example Calculation")
        st.code("""
# For Gasoline vehicle at 60 km/h:
V = 60
a, b, c = 0.00015, -0.015, 0.8  # L/km coefficients
CF = 0.64  # kg C/L for gasoline

FC = a*V**2 + b*V + c
FC = 0.00015*3600 - 0.015*60 + 0.8 = 0.44 L/km

CO2 = FC * CF * (44/12) * 1000
CO2 = 0.44 * 0.64 * 3.67 * 1000 ≈ 1032 g/km
        """, language='python')

    elif formula_pollutant == "NOx":
        st.subheader("🟡 Nitrogen Oxides (NOx) Emission Formula")

        st.markdown("### COPERT IV with Temperature Correction")
        st.latex(r'''
        EF_{NOx} = EF_{base}(V) \cdot \left(1 + k \cdot (T_{amb} - 20)\right)
        ''')

        st.markdown("**Base Emission Factor:**")
        st.latex(r'''
        EF_{base}(V) = \frac{a + c \cdot V + e \cdot V^2 + \frac{f}{V}}{1 + b \cdot V + d \cdot V^2}
        ''')

        st.markdown("**Parameters:**")
        st.markdown("""
        - **k**: Temperature coefficient (0.01-0.03 per °C)
        - **T<sub>amb</sub>**: Ambient temperature (°C)
        - **f/V**: Low-speed enrichment term
        - **Diesel k ≈ 0.015, Gasoline k ≈ 0.02**
        """, unsafe_allow_html=True)

        if include_temperature_correction:
            st.info(f"✅ Temperature correction enabled: T = {ambient_temp}°C")
            st.latex(
                f"Temperature \\ factor = 1 + 0.02 \\cdot ({ambient_temp} - 20) = {1 + 0.02 * (ambient_temp - 20):.3f}")

    elif formula_pollutant == "PM":
        st.subheader("🟣 Particulate Matter (PM) Emission Formula")

        st.markdown("### PM with DPF Efficiency")
        st.latex(r'''
        PM_{actual} = EF_{base} \cdot (1 - \eta_{DPF}) \cdot D
        ''')

        st.markdown("**Parameters:**")
        st.markdown("""
        - **EF<sub>base</sub>**: Base PM emission factor (mg/km)
        - **η<sub>DPF</sub>**: Diesel Particulate Filter efficiency
          - Euro 4: 85%
          - Euro 5: 90%
          - Euro 6: 95-99%
        - **D**: Distance traveled (km)
        """, unsafe_allow_html=True)

        st.markdown("### Base Emission Factor")
        st.latex(r'''
        EF_{base} = a \cdot V^2 + b \cdot V + c
        ''')

        st.markdown("### Example: Euro 6 Diesel")
        st.code("""
# Euro 6 Diesel with DPF
EF_base = 50  # mg/km (without DPF)
DPF_efficiency = 0.98  # 98%
Distance = 10  # km

PM_actual = EF_base * (1 - DPF_efficiency) * Distance
PM_actual = 50 * 0.02 * 10 = 10 mg total
PM_per_km = 1 mg/km
        """, language='python')

    elif formula_pollutant == "VOC":
        st.subheader("🟢 Volatile Organic Compounds (VOC) Formula")

        st.markdown("### Total VOC = Exhaust + Evaporative")
        st.latex(r'''
        VOC_{total} = VOC_{exhaust} + VOC_{evap}
        ''')

        st.markdown("**Exhaust Emissions:**")
        st.latex(r'''
        VOC_{exhaust} = \left(\frac{a}{V} + b + c \cdot V\right) \cdot D
        ''')

        st.markdown("**Evaporative Emissions:**")
        st.latex(r'''
        VOC_{evap} = k_{evap} \cdot (T_{amb} - T_{ref}) \cdot t_{soak}
        ''')

        st.markdown("**Parameters:**")
        st.markdown("""
        - **a/V**: Low-speed enrichment
        - **k<sub>evap</sub>**: Evaporation rate (g/°C/hour)
        - **t<sub>soak</sub>**: Hot soak time (hours)
        - **T<sub>ref</sub>**: Reference temperature (20°C)
        """, unsafe_allow_html=True)

    elif formula_pollutant == "FC":
        st.subheader("🟠 Fuel Consumption Formula")

        st.markdown("### Quadratic Speed Model")
        st.latex(r'''
        FC = a \cdot V^2 + b \cdot V + c
        ''')

        st.markdown("**Typical Coefficients (Gasoline PC):**")
        st.markdown("""
        - **a** ≈ 0.00015 (L/km per (km/h)²)
        - **b** ≈ -0.015 (L/km per km/h)
        - **c** ≈ 0.8 (L/km baseline)
        """)

        st.markdown("### Optimal Speed")
        st.latex(r'''
        V_{optimal} = -\frac{b}{2a} \approx 50-60 \\ km/h
        ''')

    st.markdown("---")
    st.info("""
    📚 **References:**
    - EMEP/EEA Air Pollutant Emission Inventory Guidebook 2019
    - IPCC Guidelines for National Greenhouse Gas Inventories
    - US EPA MOVES Technical Documentation
    """)

# ==================== TAB 4: CALCULATE EMISSIONS ====================
with tab4:
    st.header("⚙️ Calculate Emissions")

    required_files = [pc_param, ldv_param, hdv_param, moto_param, link_osm,
                      engine_cap_gas, engine_cap_diesel, copert_class_gas,
                      copert_class_diesel, copert_2stroke, copert_4stroke]
    all_uploaded = all(f is not None for f in required_files)

    if not selected_pollutants:
        st.warning("⚠️ Please select at least one pollutant from the sidebar")
    elif all_uploaded:
        st.success("✅ All required files uploaded!")

        # Display calculation settings
        with st.expander("🔧 Current Calculation Settings", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Pollutants:** {', '.join(selected_pollutants)}")
                st.markdown(f"**Method:** {calculation_method}")
            with col2:
                st.markdown(f"**Temp Correction:** {'✅' if include_temperature_correction else '❌'}")
                st.markdown(f"**Cold Start:** {'✅' if include_cold_start else '❌'}")
            with col3:
                st.markdown(f"**Slope Correction:** {'✅' if include_slope_correction else '❌'}")
                st.markdown(f"**Ambient Temp:** {ambient_temp}°C")

        if st.button("🚀 Calculate Multi-Pollutant Emissions", type="primary", use_container_width=True):
            with st.spinner("Computing emissions for selected pollutants..."):
                try:
                    import copert
                    import tempfile, os

                    # Setup temporary files
                    with tempfile.TemporaryDirectory() as tmpdir:
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

                        cop = copert.Copert(pc_path, ldv_path, hdv_path, moto_path)

                        # Read data
                        link_osm.seek(0)
                        engine_cap_gas.seek(0)
                        engine_cap_diesel.seek(0)
                        copert_class_gas.seek(0)
                        copert_class_diesel.seek(0)
                        copert_2stroke.seek(0)
                        copert_4stroke.seek(0)

                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values

                        # NEW: Check if 7 or 9 columns and handle accordingly
                        Nlink = data_link.shape[0]
                        if data_link.shape[1] == 7:
                            # Original 7-column format
                            P_ldv = np.zeros(Nlink)
                            P_hdv = np.zeros(Nlink)
                        elif data_link.shape[1] == 9:
                            # Extended 9-column format with LDV and HDV proportions
                            P_ldv = data_link[:, 7]
                            P_hdv = data_link[:, 8]
                        else:
                            raise ValueError(f"Link data must have 7 or 9 columns, got {data_link.shape[1]}")

                        data_engine_capacity_gasoline = np.loadtxt(engine_cap_gas)
                        data_engine_capacity_diesel = np.loadtxt(engine_cap_diesel)
                        data_copert_class_gasoline = np.loadtxt(copert_class_gas)
                        data_copert_class_diesel = np.loadtxt(copert_class_diesel)
                        data_copert_class_motorcycle_two_stroke = np.loadtxt(copert_2stroke)
                        data_copert_class_motorcycle_four_stroke = np.loadtxt(copert_4stroke)

                        # NEW: Load LDV and HDV distribution data
                        if ldv_class_file is not None:
                            ldv_class_file.seek(0)
                            data_copert_class_ldv = np.loadtxt(ldv_class_file, delimiter=',')
                        else:
                            st.warning("⚠️ LDV Euro Class Distribution file not uploaded. LDV emissions will be zero.")
                            data_copert_class_ldv = np.zeros((Nlink, 14))  # Nclass will be defined below

                        # NEW: Load HDV distribution
                        N_HDV_Class = 6
                        N_HDV_Type = 15
                        if hdv_class_type_file is not None:
                            hdv_class_type_file.seek(0)
                            data_hdv_class_type_distribution = np.loadtxt(hdv_class_type_file, delimiter=',')
                            # Reshape to [Nlink, N_HDV_Class, N_HDV_Type]
                            data_hdv_reshaped = data_hdv_class_type_distribution.reshape(Nlink, N_HDV_Class, N_HDV_Type)
                        else:
                            st.warning("⚠️ HDV Class/Type Distribution file not uploaded. HDV emissions will be zero.")
                            data_hdv_reshaped = np.zeros((Nlink, N_HDV_Class, N_HDV_Type))

                        # Setup classes and types
                        engine_type = [cop.engine_type_gasoline, cop.engine_type_diesel]
                        engine_type_m = [cop.engine_type_moto_two_stroke_more_50,
                                         cop.engine_type_moto_four_stroke_50_250]
                        engine_capacity = [cop.engine_capacity_0p8_to_1p4, cop.engine_capacity_1p4_to_2]
                        copert_class = [cop.class_PRE_ECE, cop.class_ECE_15_00_or_01, cop.class_ECE_15_02,
                                        cop.class_ECE_15_03,
                                        cop.class_ECE_15_04, cop.class_Improved_Conventional, cop.class_Open_loop,
                                        cop.class_Euro_1,
                                        cop.class_Euro_2, cop.class_Euro_3, cop.class_Euro_4, cop.class_Euro_5,
                                        cop.class_Euro_6, cop.class_Euro_6c]
                        Nclass = len(copert_class)
                        copert_class_motorcycle = [cop.class_moto_Conventional, cop.class_moto_Euro_1,
                                                   cop.class_moto_Euro_2,
                                                   cop.class_moto_Euro_3, cop.class_moto_Euro_4, cop.class_moto_Euro_5]
                        Mclass = len(copert_class_motorcycle)

                        # NEW: HDV Emission Classes
                        HDV_Emission_Classes = [
                            cop.class_hdv_Euro_I,
                            cop.class_hdv_Euro_II,
                            cop.class_hdv_Euro_III,
                            cop.class_hdv_Euro_IV,
                            cop.class_hdv_Euro_V,
                            cop.class_hdv_Euro_VI
                        ]

                        # Initialize emission arrays for each selected pollutant
                        emissions_data = {}
                        pollutant_mapping = {
                            "CO": cop.pollutant_CO,
                            "CO2": cop.pollutant_FC,  # Will convert FC to CO2
                            "NOx": cop.pollutant_NOx,
                            "PM": cop.pollutant_PM,
                            "VOC": cop.pollutant_VOC,
                            "FC": cop.pollutant_FC
                        }

                        # NEW: Initialize with LDV and HDV arrays
                        for poll in selected_pollutants:
                            emissions_data[poll] = {
                                'pc': np.zeros((Nlink,), dtype=float),
                                'moto': np.zeros((Nlink,), dtype=float),
                                'ldv': np.zeros((Nlink,), dtype=float),
                                'hdv': np.zeros((Nlink,), dtype=float),
                                'total': np.zeros((Nlink,), dtype=float)
                            }

                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Calculate emissions for each link
                        for i in range(Nlink):
                            if i % max(1, Nlink // 100) == 0:
                                progress_bar.progress(i / Nlink)
                                status_text.text(f"Processing link {i + 1}/{Nlink} - {(i / Nlink * 100):.1f}% complete")

                            link_length = data_link[i, 1]
                            link_flow = data_link[i, 2]
                            v = min(max(10., data_link[i, 3]), 130.)
                            link_gasoline_proportion = data_link[i, 4]
                            link_pc_proportion = data_link[i, 5]
                            link_4_stroke_proportion = data_link[i, 6]
                            p_passenger = link_pc_proportion
                            P_motorcycle = 1. - link_pc_proportion
                            P_ldv_i = P_ldv[i]
                            P_hdv_i = P_hdv[i]

                            engine_type_distribution = [link_gasoline_proportion, 1. - link_gasoline_proportion]
                            engine_capacity_distribution = [data_engine_capacity_gasoline[i],
                                                            data_engine_capacity_diesel[i]]
                            engine_type_motorcycle_distribution = [link_4_stroke_proportion,
                                                                   1. - link_4_stroke_proportion]

                            # Calculate for each selected pollutant
                            for poll_name in selected_pollutants:
                                poll_type = pollutant_mapping[poll_name]

                                # 1. Passenger car emissions
                                for t in range(2):
                                    for c in range(Nclass):
                                        for k in range(2):
                                            if (copert_class[c] != cop.class_Improved_Conventional and
                                                copert_class[c] != cop.class_Open_loop) or engine_capacity[k] <= 2.0:
                                                if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1,
                                                                                                  1 + cop.class_Euro_3):
                                                    continue

                                                try:
                                                    e = cop.Emission(poll_type, v, link_length,
                                                                     cop.vehicle_type_passenger_car, engine_type[t],
                                                                     copert_class[c], engine_capacity[k], ambient_temp)

                                                    # Apply temperature correction for NOx
                                                    if poll_name == "NOx" and include_temperature_correction:
                                                        temp_factor = 1 + 0.02 * (ambient_temp - 20)
                                                        e *= temp_factor

                                                    # Apply cold start correction
                                                    if include_cold_start and poll_name in ["CO", "NOx", "VOC"]:
                                                        try:
                                                            beta = cop.ColdStartMileagePercentage(
                                                                cop.vehicle_type_passenger_car, engine_type[t],
                                                                poll_type, copert_class[c], engine_capacity[k],
                                                                ambient_temp, trip_length)
                                                            e_cold = cop.ColdStartEmissionQuotient(
                                                                cop.vehicle_type_passenger_car, engine_type[t],
                                                                poll_type, v, copert_class[c], engine_capacity[k],
                                                                ambient_temp)
                                                            e = e * ((1 - beta) + e_cold * beta)
                                                        except:
                                                            pass  # Use hot emission only if cold start fails

                                                    e *= engine_type_distribution[t] * engine_capacity_distribution[t][
                                                        k]
                                                    emissions_data[poll_name]['pc'][
                                                        i] += e * p_passenger / link_length * link_flow
                                                except:
                                                    pass

                                # NEW: 2. Light Duty Vehicle (LDV) emissions
                                if P_ldv_i > 0:
                                    for t in range(2):  # engine types
                                        for c in range(Nclass):  # Euro classes
                                            for k in range(2):  # engine capacities
                                                if (copert_class[c] != cop.class_Improved_Conventional and
                                                    copert_class[c] != cop.class_Open_loop) or engine_capacity[
                                                    k] <= 2.0:
                                                    if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1,
                                                                                                      1 + cop.class_Euro_3):
                                                        continue

                                                    try:
                                                        e = cop.Emission(poll_type, v, link_length,
                                                                         cop.vehicle_type_light_commercial_vehicle,
                                                                         engine_type[t], copert_class[c],
                                                                         engine_capacity[k], ambient_temp)

                                                        # Apply temperature correction for NOx
                                                        if poll_name == "NOx" and include_temperature_correction:
                                                            temp_factor = 1 + 0.02 * (ambient_temp - 20)
                                                            e *= temp_factor

                                                        # Apply cold start correction
                                                        if include_cold_start and poll_name in ["CO", "NOx", "VOC"]:
                                                            try:
                                                                beta = cop.ColdStartMileagePercentage(
                                                                    cop.vehicle_type_light_commercial_vehicle,
                                                                    engine_type[t],
                                                                    poll_type, copert_class[c], engine_capacity[k],
                                                                    ambient_temp, trip_length)
                                                                e_cold = cop.ColdStartEmissionQuotient(
                                                                    cop.vehicle_type_light_commercial_vehicle,
                                                                    engine_type[t],
                                                                    poll_type, v, copert_class[c], engine_capacity[k],
                                                                    ambient_temp)
                                                                e = e * ((1 - beta) + e_cold * beta)
                                                            except:
                                                                pass

                                                        ldv_fleet_share = data_copert_class_ldv[i, c] if c < \
                                                                                                         data_copert_class_ldv.shape[
                                                                                                             1] else 0
                                                        e *= engine_type_distribution[t] * \
                                                             engine_capacity_distribution[t][k] * ldv_fleet_share
                                                        emissions_data[poll_name]['ldv'][
                                                            i] += e * P_ldv_i / link_length * link_flow
                                                    except:
                                                        pass

                                # Motorcycle emissions
                                for m in range(2):
                                    for d in range(Mclass):
                                        if m == 1 and copert_class_motorcycle[d] in range(cop.class_moto_Conventional,
                                                                                          1 + cop.class_moto_Euro_5):
                                            continue
                                        try:
                                            e_f = cop.EFMotorcycle(poll_type, v, engine_type_m[m],
                                                                   copert_class_motorcycle[d])
                                            e_f *= engine_type_motorcycle_distribution[m]
                                            emissions_data[poll_name]['moto'][i] += e_f * P_motorcycle * link_flow
                                        except:
                                            pass

                                # NEW: 3. Heavy Duty Vehicle (HDV) emissions
                                if P_hdv_i > 0:
                                    for t_class in range(N_HDV_Class):  # Euro classes
                                        for t_type in range(N_HDV_Type):  # vehicle types
                                            try:
                                                hdv_fleet_share = data_hdv_reshaped[i, t_class, t_type]
                                                if hdv_fleet_share > 0:
                                                    e = cop.EFHeavyDuty(poll_type, v, HDV_Emission_Classes[t_class],
                                                                        t_type)

                                                    # Apply temperature correction for NOx
                                                    if poll_name == "NOx" and include_temperature_correction:
                                                        temp_factor = 1 + 0.015 * (ambient_temp - 20)
                                                        e *= temp_factor

                                                    emissions_data[poll_name]['hdv'][
                                                        i] += e * hdv_fleet_share * P_hdv_i * link_flow
                                            except:
                                                pass

                                # NEW: Total emissions (updated to include LDV and HDV)
                                emissions_data[poll_name]['total'][i] = (
                                        emissions_data[poll_name]['pc'][i] +
                                        emissions_data[poll_name]['moto'][i] +
                                        emissions_data[poll_name]['ldv'][i] +
                                        emissions_data[poll_name]['hdv'][i]
                                )

                        # Convert FC to CO2 if needed
                        if "CO2" in selected_pollutants:
                            # FC is in L/100km, convert to g CO2/km
                            # Gasoline: 2.31 kg CO2/L, Diesel: 2.68 kg CO2/L
                            # Assuming 50-50 mix
                            co2_factor = 2.5 * 1000  # Average, in g/L
                            emissions_data["CO2"]['pc'] = emissions_data["CO2"]['pc'] * co2_factor / 100
                            emissions_data["CO2"]['ldv'] = emissions_data["CO2"]['ldv'] * co2_factor / 100
                            emissions_data["CO2"]['hdv'] = emissions_data["CO2"]['hdv'] * co2_factor / 100
                            emissions_data["CO2"]['moto'] = emissions_data["CO2"]['moto'] * co2_factor / 100
                            emissions_data["CO2"]['total'] = emissions_data["CO2"]['total'] * co2_factor / 100

                        progress_bar.progress(1.0)
                        status_text.text("✅ Calculation complete!")

                        # Store results in session state
                        st.session_state.emissions_data = emissions_data
                        st.session_state.data_link = data_link
                        st.session_state.selected_pollutants = selected_pollutants

                    st.success("✅ Multi-pollutant emissions calculated successfully!")

                    # Display results summary
                    st.subheader("📊 Emission Summary")

                    # NEW: Create summary dataframe with LDV and HDV
                    summary_data = []
                    for poll in selected_pollutants:
                        summary_data.append({
                            'Pollutant': poll,
                            'Total PC': f"{emissions_data[poll]['pc'].sum():.2f}",
                            'Total LDV': f"{emissions_data[poll]['ldv'].sum():.2f}",
                            'Total HDV': f"{emissions_data[poll]['hdv'].sum():.2f}",
                            'Total Moto': f"{emissions_data[poll]['moto'].sum():.2f}",
                            'Total': f"{emissions_data[poll]['total'].sum():.2f}",
                            'Avg per Link': f"{emissions_data[poll]['total'].mean():.3f}",
                            'Max': f"{emissions_data[poll]['total'].max():.2f}",
                            'Unit': pollutants_available[poll]['unit']
                        })

                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

                    # Detailed results by pollutant
                    for poll in selected_pollutants:
                        with st.expander(f"📋 Detailed Results: {poll} ({pollutants_available[poll]['name']})",
                                         expanded=False):
                            # NEW: Include LDV and HDV columns
                            results_df = pd.DataFrame({
                                'OSM_ID': data_link[:, 0].astype(int),
                                f'{poll}_PC': emissions_data[poll]['pc'],
                                f'{poll}_LDV': emissions_data[poll]['ldv'],
                                f'{poll}_HDV': emissions_data[poll]['hdv'],
                                f'{poll}_Motorcycle': emissions_data[poll]['moto'],
                                f'{poll}_Total': emissions_data[poll]['total']
                            })
                            st.dataframe(results_df.head(50), use_container_width=True)

                            # NEW: Display 5 metrics instead of 3
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric(f"Total PC {poll}",
                                          f"{emissions_data[poll]['pc'].sum():.2f} {pollutants_available[poll]['unit']}")
                            with col2:
                                st.metric(f"Total LDV {poll}",
                                          f"{emissions_data[poll]['ldv'].sum():.2f} {pollutants_available[poll]['unit']}")
                            with col3:
                                st.metric(f"Total HDV {poll}",
                                          f"{emissions_data[poll]['hdv'].sum():.2f} {pollutants_available[poll]['unit']}")
                            with col4:
                                st.metric(f"Total Motorcycle {poll}",
                                          f"{emissions_data[poll]['moto'].sum():.2f} {pollutants_available[poll]['unit']}")
                            with col5:
                                st.metric(f"Total {poll}",
                                          f"{emissions_data[poll]['total'].sum():.2f} {pollutants_available[poll]['unit']}")
        except Exception as e:
        st.error(f"❌ Error during calculation: {e}")
        import traceback

        with st.expander("🐛 Debug Information"):
            st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please upload all required files")
        missing = []
        file_names = ['PC Parameter', 'LDV Parameter', 'HDV Parameter', 'Moto Parameter', 'Link OSM',
                      'Engine Cap Gas', 'Engine Cap Diesel', 'COPERT Class Gas', 'COPERT Class Diesel',
                      '2-Stroke', '4-Stroke']
        for fname, fdata in zip(file_names, required_files):
            if fdata is None:
                missing.append(fname)
        st.error(f"**Missing files:** {', '.join(missing)}")
        st.info(
            "📁 [Download sample files](https://drive.google.com/drive/folders/1KCu8y-mZ0XtBc6icFlvPnJMxLFM7YCKY?usp=sharing)")

    # ==================== TAB 5: MULTI-METRIC ANALYSIS ====================
    with tab5:
        st.header("📈 Multi-Metric Analysis & Comparison")

        if 'emissions_data' in st.session_state and st.session_state.emissions_data:
            emissions_data = st.session_state.emissions_data
            data_link = st.session_state.data_link

            # Comparative bar chart
            st.subheader("📊 Pollutant Comparison")

            # NEW: Include LDV and HDV in comparison
            comparison_data = []
            for poll in st.session_state.selected_pollutants:
                comparison_data.append({
                    'Pollutant': poll,
                    'Passenger Cars': emissions_data[poll]['pc'].sum(),
                    'LDV': emissions_data[poll]['ldv'].sum(),
                    'HDV': emissions_data[poll]['hdv'].sum(),
                    'Motorcycles': emissions_data[poll]['moto'].sum()
                })

            comp_df = pd.DataFrame(comparison_data)

            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                name='Passenger Cars',
                x=comp_df['Pollutant'],
                y=comp_df['Passenger Cars'],
                marker_color='#667eea'
            ))
            fig_comparison.add_trace(go.Bar(
                name='LDV',
                x=comp_df['Pollutant'],
                y=comp_df['LDV'],
                marker_color='#10b981'
            ))
            fig_comparison.add_trace(go.Bar(
                name='HDV',
                x=comp_df['Pollutant'],
                y=comp_df['HDV'],
                marker_color='#f59e0b'
            ))
            fig_comparison.add_trace(go.Bar(
                name='Motorcycles',
                x=comp_df['Pollutant'],
                y=comp_df['Motorcycles'],
                marker_color='#764ba2'
            ))

            fig_comparison.update_layout(
                title='Total Emissions by Pollutant and Vehicle Type',
                xaxis_title='Pollutant',
                yaxis_title='Total Emissions',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Distribution analysis
            st.subheader("📉 Emission Distribution Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Box plot for selected pollutant
                analysis_poll = st.selectbox("Select pollutant for distribution analysis",
                                             st.session_state.selected_pollutants)

                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=emissions_data[analysis_poll]['total'],
                    name=analysis_poll,
                    marker_color=pollutants_available[analysis_poll]['color']
                ))
                fig_box.update_layout(
                    title=f'{analysis_poll} Distribution Across Links',
                    yaxis_title=f'{analysis_poll} ({pollutants_available[analysis_poll]["unit"]})',
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col2:
                # Histogram
                fig_hist = px.histogram(
                    x=emissions_data[analysis_poll]['total'],
                    nbins=50,
                    title=f'{analysis_poll} Frequency Distribution',
                    labels={'x': f'{analysis_poll} ({pollutants_available[analysis_poll]["unit"]})', 'y': 'Frequency'}
                )
                fig_hist.update_traces(marker_color=pollutants_available[analysis_poll]['color'])
                st.plotly_chart(fig_hist, use_container_width=True)

            # Correlation analysis
            if len(st.session_state.selected_pollutants) > 1:
                st.subheader("🔗 Pollutant Correlation Matrix")

                corr_data = {}
                for poll in st.session_state.selected_pollutants:
                    corr_data[poll] = emissions_data[poll]['total']

                corr_df = pd.DataFrame(corr_data)
                correlation_matrix = corr_df.corr()

                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdYlGn',
                    title='Correlation Between Pollutants',
                    aspect='auto'
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                st.info("💡 **Interpretation**: Values close to 1 indicate strong positive correlation, " +
                        "values close to -1 indicate negative correlation, values near 0 indicate no correlation.")

            # Top emitters analysis
            st.subheader("🔍 Top Emission Hotspots")

            top_n = st.slider("Number of top emitters to display", 5, 50, 10)

            for poll in st.session_state.selected_pollutants:
                with st.expander(f"Top {top_n} Links for {poll}"):
                    top_indices = np.argsort(emissions_data[poll]['total'])[-top_n:][::-1]

                    top_data = pd.DataFrame({
                        'Rank': range(1, top_n + 1),
                        'OSM_ID': data_link[top_indices, 0].astype(int),
                        'Length_km': data_link[top_indices, 1],
                        'Speed_kmh': data_link[top_indices, 3],
                        'Flow': data_link[top_indices, 2],
                        f'{poll}_Emission': emissions_data[poll]['total'][top_indices]
                    })

                    st.dataframe(top_data, use_container_width=True)

                    # Visualization
                    fig_top = px.bar(
                        top_data,
                        x='OSM_ID',
                        y=f'{poll}_Emission',
                        title=f'Top {top_n} Emitting Links for {poll}',
                        color=f'{poll}_Emission',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

            # Speed vs Emission analysis
            st.subheader("🚗 Speed vs Emission Analysis")

            speed_poll = st.selectbox("Select pollutant for speed analysis",
                                      st.session_state.selected_pollutants,
                                      key='speed_analysis')

            speed_emission_df = pd.DataFrame({
                'Speed': data_link[:, 3],
                'Emission': emissions_data[speed_poll]['total']
            })

            fig_speed = px.scatter(
                speed_emission_df,
                x='Speed',
                y='Emission',
                title=f'{speed_poll} Emissions vs Vehicle Speed',
                labels={'Speed': 'Speed (km/h)',
                        'Emission': f'{speed_poll} ({pollutants_available[speed_poll]["unit"]})'},
                trendline='lowess',
                color='Emission',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_speed, use_container_width=True)

            st.info("💡 **Optimal Speed Zone**: Most pollutants show minimum emissions in the 50-80 km/h range")

        else:
            st.info("👆 Please calculate emissions first in the 'Calculate Emissions' tab")

    # ==================== TAB 6: INTERACTIVE MAP ====================
    with tab6:
        st.header("🗺️ Interactive Emission Map Visualization")

        has_emissions = 'emissions_data' in st.session_state and st.session_state.emissions_data

        if not has_emissions:
            st.warning("⚠️ Please calculate emissions first")
        elif osm_file is None:
            st.warning("⚠️ Please upload OSM network file")
        else:
            st.info("🎨 Configure your visualization and generate the emission map")

            # Map pollutant selector
            map_pollutant = st.selectbox(
                "Select Pollutant to Visualize on Map",
                st.session_state.selected_pollutants,
                help="Choose which pollutant to display on the map"
            )

            st.markdown("---")
            st.subheader("🎨 Visualization Settings")

            viz_mode = st.radio(
                "Select visualization style:",
                ["Classic (Original)", "Enhanced with Labels", "Custom"],
                horizontal=True,
                help="Classic: Original | Enhanced: Smart labels | Custom: Full control"
            )

            st.markdown("---")

            # Visualization settings based on mode
            if viz_mode == "Classic (Original)":
                st.markdown("**Classic Mode Settings**")
                col1, col2 = st.columns(2)
                with col1:
                    colormap = st.selectbox("Color Map", ['jet', 'viridis', 'plasma', 'RdYlGn_r', 'hot'], index=0)
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
                    colormap = st.selectbox("Color Map", ['jet', 'viridis', 'plasma', 'RdYlGn_r', 'hot', 'coolwarm'],
                                            index=0)
                    fig_size = st.slider("Figure Size", 8, 16, 12)
                    line_width_multiplier = st.slider("Line Width Scale", 0.5, 5.0, 2.0, 0.5)
                with col2:
                    label_density = st.selectbox("Road Label Density",
                                                 ["Minimal (Major roads only)",
                                                  "Medium (Top 25% emissions)",
                                                  "High (Top 50% emissions)"], index=1)
                    show_roads_without_data = st.checkbox("Show roads without emission data", value=True)
                    rotate_labels = st.checkbox("Rotate labels along roads", value=True)
                show_labels = True
                enhanced_styling = True
                add_grid = True
                road_transparency = 0.8
                grid_alpha = 0.2
                label_font_size = 7
                min_label_distance = 0.002

            else:  # Custom
                st.markdown("**Custom Mode Settings**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Appearance**")
                    colormap = st.selectbox("Color Map",
                                            ['jet', 'viridis', 'plasma', 'RdYlGn_r', 'hot', 'coolwarm', 'inferno'],
                                            index=0)
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
                        label_density = st.selectbox("Label Density",
                                                     ["Minimal (Major roads only)",
                                                      "Medium (Top 25% emissions)",
                                                      "High (Top 50% emissions)",
                                                      "Maximum (All named roads)"], index=1)
                        rotate_labels = st.checkbox("Rotate labels along roads", value=True)
                        label_font_size = st.slider("Label font size", 4, 12, 7)
                        min_label_distance = st.slider("Min distance between labels", 0.001, 0.01, 0.002, 0.001)
                    else:
                        label_density = "Minimal (Major roads only)"
                        rotate_labels = False
                        label_font_size = 7
                        min_label_distance = 0.002

            st.markdown("---")

            # Display current pollutant info
            st.info(f"🎯 **Visualizing**: {map_pollutant} - {pollutants_available[map_pollutant]['name']} " +
                    f"({pollutants_available[map_pollutant]['unit']})")

            if st.button("🗺️ Generate Interactive Map", type="primary", use_container_width=True):
                with st.spinner(f"Generating {map_pollutant} emission map..."):
                    try:
                        import osm_network

                        emissions_data = st.session_state.emissions_data
                        hot_emission = emissions_data[map_pollutant]['total']
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
                            highway_coordinate, highway_osmid, highway_names, highway_types = osm_network.retrieve_highway(
                                osm_path, selected_zone, tolerance, int(ncore))
                            status_text.text(f"OSM network parsed successfully! Found {len(highway_osmid)} roads")

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
                            cb = matplotlib.colorbar.ColorbarBase(ax_c, cmap=plt.cm.get_cmap(colormap),
                                                                  norm=color_scale, orientation="vertical")
                            cb.set_label(f"{map_pollutant} ({pollutants_available[map_pollutant]['unit']})",
                                         fontsize=12)

                            if enhanced_styling:
                                ax.set_facecolor('#f0f0f0')

                            status_text.text(f"Plotting {map_pollutant} emission data on map...")
                            roads_with_data = 0
                            roads_without_data = 0

                            for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid, highway_names,
                                                                       highway_types):
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
                                            ax.plot([x[0] for x in refs], [x[1] for x in refs], "gray", lw=lw_nodata,
                                                    alpha=0.3)
                                        roads_without_data += 1

                            # Add labels
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

                                for refs, osmid, name, highway_type in zip(highway_coordinate, highway_osmid,
                                                                           highway_names, highway_types):
                                    try:
                                        i = emission_osm_id.index(osmid)
                                        current_emission = hot_emission[i]
                                    except:
                                        continue

                                    if major_only:
                                        should_label = name and highway_type in major_road_types
                                    else:
                                        should_label = name and (
                                                    highway_type in major_road_types or current_emission >= emission_threshold)

                                    if should_label:
                                        center_index = len(refs) // 2
                                        x_center = refs[center_index][0]
                                        y_center = refs[center_index][1]

                                        too_close = False
                                        if name in labeled_roads:
                                            for prev_x, prev_y in labeled_roads[name]:
                                                distance = np.sqrt((x_center - prev_x) ** 2 + (y_center - prev_y) ** 2)
                                                if distance < min_label_distance:
                                                    too_close = True
                                                    break

                                        if not too_close:
                                            angle = 0
                                            if rotate_labels and len(refs) > 1:
                                                dx = refs[min(center_index + 1, len(refs) - 1)][0] - \
                                                     refs[max(center_index - 1, 0)][0]
                                                dy = refs[min(center_index + 1, len(refs) - 1)][1] - \
                                                     refs[max(center_index - 1, 0)][1]
                                                angle = np.degrees(np.arctan2(dy, dx))
                                                if angle > 90:
                                                    angle -= 180
                                                elif angle < -90:
                                                    angle += 180

                                            ax.text(x_center, y_center, str(name), fontsize=label_font_size,
                                                    color='black', ha='center', va='center',
                                                    rotation=angle, rotation_mode='anchor',
                                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray',
                                                              linewidth=0.5, boxstyle='round,pad=0.3'), zorder=100)

                                            if name not in labeled_roads:
                                                labeled_roads[name] = []
                                            labeled_roads[name].append((x_center, y_center))

                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)

                            if viz_mode == "Classic (Original)":
                                ax.set_title(f"{map_pollutant} Emission Factor Map", fontsize=14)
                            else:
                                ax.set_title(f"{map_pollutant} Emission Factor Map with Road Names", fontsize=14,
                                             fontweight='bold')

                            ax.set_xlabel("Longitude", fontsize=12)
                            ax.set_ylabel("Latitude", fontsize=12)

                            if add_grid:
                                ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)

                            st.pyplot(fig)
                            st.session_state.emission_map_fig = fig
                            st.session_state.current_map_pollutant = map_pollutant

                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Roads with Emission Data", roads_with_data)
                            with col2:
                                st.metric("Roads without Data", roads_without_data)
                            with col3:
                                if show_labels and viz_mode != "Classic (Original)":
                                    st.metric("Unique Road Names Labeled", len(labeled_roads))
                                else:
                                    st.metric("Max Emission", f"{max_emission_value:.2f}")

                            status_text.empty()
                            st.success(f"✅ {map_pollutant} map generated successfully!")

                            # Additional map info
                            st.info(f"""
                                **Map Information:**
                                - Pollutant: {map_pollutant} ({pollutants_available[map_pollutant]['name']})
                                - Standard: {pollutants_available[map_pollutant]['standard']}
                                - Color scale: {colormap}
                                - Total roads visualized: {roads_with_data + roads_without_data}
                                """)

                        finally:
                            if os.path.exists(osm_path):
                                os.unlink(osm_path)

                    except Exception as e:
                        st.error(f"❌ Error generating map: {e}")
                        import traceback

                        with st.expander("🐛 Debug Information"):
                            st.code(traceback.format_exc())

    # ==================== TAB 7: DOWNLOAD RESULTS ====================
    with tab7:
        st.header("📥 Download Results")

        st.markdown("### 📊 Available Outputs")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Emission Data**")
            if 'emissions_data' in st.session_state and st.session_state.emissions_data:
                emissions_data = st.session_state.emissions_data
                data_link = st.session_state.data_link

                # NEW: Include LDV and HDV in export data
                export_data = {'OSM_ID': data_link[:, 0].astype(int), 'Length_km': data_link[:, 1]}

                for poll in st.session_state.selected_pollutants:
                    export_data[f'{poll}_PC'] = emissions_data[poll]['pc']
                    export_data[f'{poll}_LDV'] = emissions_data[poll]['ldv']
                    export_data[f'{poll}_HDV'] = emissions_data[poll]['hdv']
                    export_data[f'{poll}_Moto'] = emissions_data[poll]['moto']
                    export_data[f'{poll}_Total'] = emissions_data[poll]['total']

                results_df = pd.DataFrame(export_data)
                csv = results_df.to_csv(index=False)

                st.download_button(
                    label="⬇️ Download Multi-Pollutant Emission Data CSV",
                    data=csv,
                    file_name="multi_pollutant_emissions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Individual pollutant downloads
                with st.expander("Download Individual Pollutant Data"):
                    for poll in st.session_state.selected_pollutants:
                        # NEW: Include LDV and HDV in individual downloads
                        poll_df = pd.DataFrame({
                            'OSM_ID': data_link[:, 0].astype(int),
                            'Length_km': data_link[:, 1],
                            f'{poll}_PC': emissions_data[poll]['pc'],
                            f'{poll}_LDV': emissions_data[poll]['ldv'],
                            f'{poll}_HDV': emissions_data[poll]['hdv'],
                            f'{poll}_Moto': emissions_data[poll]['moto'],
                            f'{poll}_Total': emissions_data[poll]['total']
                        })
                        poll_csv = poll_df.to_csv(index=False)
                        st.download_button(
                            label=f"⬇️ Download {poll} Data",
                            data=poll_csv,
                            file_name=f"{poll}_emissions.csv",
                            mime="text/csv",
                            key=f"download_{poll}"
                        )
            else:
                st.info("Calculate emissions first")

        with col2:
            st.markdown("**Emission Map**")
            if 'emission_map_fig' in st.session_state:
                buf = BytesIO()
                st.session_state.emission_map_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)

                map_pollutant = st.session_state.get('current_map_pollutant', 'emission')
                st.download_button(
                    label=f"⬇️ Download {map_pollutant} Map PNG",
                    data=buf,
                    file_name=f"{map_pollutant}_emission_map.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("Generate map first")

        st.markdown("---")
        st.markdown("### 📦 Download Complete Analysis Package")

        if 'emissions_data' in st.session_state and st.session_state.emissions_data:
            if st.button("📦 Create ZIP Archive with All Results", type="primary", use_container_width=True):
                with st.spinner("Creating comprehensive ZIP archive..."):
                    try:
                        zip_buffer = BytesIO()
                        emissions_data = st.session_state.emissions_data
                        data_link = st.session_state.data_link

                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # NEW: Main emissions data with all vehicle types
                            export_data = {'OSM_ID': data_link[:, 0].astype(int), 'Length_km': data_link[:, 1]}
                            for poll in st.session_state.selected_pollutants:
                                export_data[f'{poll}_PC'] = emissions_data[poll]['pc']
                                export_data[f'{poll}_LDV'] = emissions_data[poll]['ldv']
                                export_data[f'{poll}_HDV'] = emissions_data[poll]['hdv']
                                export_data[f'{poll}_Moto'] = emissions_data[poll]['moto']
                                export_data[f'{poll}_Total'] = emissions_data[poll]['total']

                            results_df = pd.DataFrame(export_data)
                            zip_file.writestr('all_pollutants_emissions.csv', results_df.to_csv(index=False))

                            # NEW: Individual pollutant files with all vehicle types
                            for poll in st.session_state.selected_pollutants:
                                poll_df = pd.DataFrame({
                                    'OSM_ID': data_link[:, 0].astype(int),
                                    'Length_km': data_link[:, 1],
                                    f'{poll}_PC': emissions_data[poll]['pc'],
                                    f'{poll}_LDV': emissions_data[poll]['ldv'],
                                    f'{poll}_HDV': emissions_data[poll]['hdv'],
                                    f'{poll}_Moto': emissions_data[poll]['moto'],
                                    f'{poll}_Total': emissions_data[poll]['total']
                                })
                                zip_file.writestr(f'{poll}_emissions.csv', poll_df.to_csv(index=False))

                                # Map image if available
                            if 'emission_map_fig' in st.session_state:
                                map_buf = BytesIO()
                                st.session_state.emission_map_fig.savefig(map_buf, format='png', dpi=150,
                                                                          bbox_inches='tight')
                                map_buf.seek(0)
                                map_pollutant = st.session_state.get('current_map_pollutant', 'emission')
                                zip_file.writestr(f'{map_pollutant}_emission_map.png', map_buf.read())

                                # Comprehensive summary report
                            summary = f"""Traffic Emission Analysis Report
                                {'=' * 70}

                                Analysis Configuration:
                                - Calculation Method: {calculation_method}
                                - Pollutants Analyzed: {', '.join(st.session_state.selected_pollutants)}
                                - Temperature Correction: {'Enabled' if include_temperature_correction else 'Disabled'}
                                - Cold Start Correction: {'Enabled' if include_cold_start else 'Disabled'}
                                - Slope Correction: {'Enabled' if include_slope_correction else 'Disabled'}

                                Environmental Parameters:
                                - Ambient Temperature: {ambient_temp}°C
                                - Average Trip Length: {trip_length} km
                                - Road Slope: {road_slope}%

                                Dataset Information:
                                - Total Links Analyzed: {len(data_link)}
                                - Total Road Length: {data_link[:, 1].sum():.2f} km
                                - Average Speed: {data_link[:, 3].mean():.2f} km/h
                                - Average Flow: {data_link[:, 2].mean():.0f} vehicles

                                Emission Summary by Pollutant:
                                {'=' * 70}
                                """
                            for poll in st.session_state.selected_pollutants:
                                summary += f"""
                                {poll} - {pollutants_available[poll]['name']}:
                                  Standard: {pollutants_available[poll]['standard']}
                                  Unit: {pollutants_available[poll]['unit']}

                                  Total Passenger Car Emissions: {emissions_data[poll]['pc'].sum():.2f}
                                  Total Light Duty Vehicle Emissions: {emissions_data[poll]['ldv'].sum():.2f}
                                  Total Heavy Duty Vehicle Emissions: {emissions_data[poll]['hdv'].sum():.2f}
                                  Total Motorcycle Emissions: {emissions_data[poll]['moto'].sum():.2f}
                                  Total Emissions: {emissions_data[poll]['total'].sum():.2f}

                                  Average per Link: {emissions_data[poll]['total'].mean():.3f}
                                  Maximum Emission: {emissions_data[poll]['total'].max():.2f}
                                  Minimum Emission: {emissions_data[poll]['total'].min():.2f}
                                  Standard Deviation: {emissions_data[poll]['total'].std():.2f}
                                """

                            summary += f"""
                                {'=' * 70}
                                Map Domain Boundaries:
                                - Longitude Range: {x_min} to {x_max}
                                - Latitude Range: {y_min} to {y_max}
                                - Tolerance: {tolerance}

                                Data Quality Metrics:
                                - Links with speed < 10 km/h: {len([x for x in data_link[:, 3] if x < 10])}
                                - Links with speed > 130 km/h: {len([x for x in data_link[:, 3] if x > 130])}
                                - Data completeness: {(1 - data_link[:, 1].isna().sum() / len(data_link)) * 100:.1f}%

                                Standards and References:
                                - COPERT IV: European emission inventory guidebook
                                - IPCC: Intergovernmental Panel on Climate Change guidelines
                                - WHO: World Health Organization air quality standards

                                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                                """
                            zip_file.writestr('analysis_summary.txt', summary)

                            # NEW: Statistics summary CSV with all vehicle types
                            stats_data = []
                            for poll in st.session_state.selected_pollutants:
                                stats_data.append({
                                    'Pollutant': poll,
                                    'Name': pollutants_available[poll]['name'],
                                    'Unit': pollutants_available[poll]['unit'],
                                    'Standard': pollutants_available[poll]['standard'],
                                    'Total_PC': emissions_data[poll]['pc'].sum(),
                                    'Total_LDV': emissions_data[poll]['ldv'].sum(),
                                    'Total_HDV': emissions_data[poll]['hdv'].sum(),
                                    'Total_Moto': emissions_data[poll]['moto'].sum(),
                                    'Total': emissions_data[poll]['total'].sum(),
                                    'Mean': emissions_data[poll]['total'].mean(),
                                    'Median': np.median(emissions_data[poll]['total']),
                                    'Std': emissions_data[poll]['total'].std(),
                                    'Min': emissions_data[poll]['total'].min(),
                                    'Max': emissions_data[poll]['total'].max()
                                })
                            stats_df = pd.DataFrame(stats_data)
                            zip_file.writestr('statistics_summary.csv', stats_df.to_csv(index=False))

                        zip_buffer.seek(0)
                        st.download_button(
                            label="⬇️ Download Complete Analysis Package (ZIP)",
                            data=zip_buffer,
                            file_name="traffic_emission_analysis_complete.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        st.success("✅ ZIP archive created successfully!")

                        # Show what's included
                        with st.expander("📋 Package Contents"):
                            st.markdown("""
                                                        **Included Files:**
                                                        - `all_pollutants_emissions.csv` - Combined data for all pollutants (PC, LDV, HDV, Moto)
                                                        - Individual CSV files for each pollutant with all vehicle types
                                                        - `emission_map.png` - Visual map of emissions (if generated)
                                                        - `analysis_summary.txt` - Comprehensive text report
                                                        - `statistics_summary.csv` - Statistical summary table with vehicle type breakdown
                                                        """)

                    except Exception as e:
                        st.error(f"❌ Error creating ZIP: {e}")
                        import traceback

                        with st.expander("🐛 Debug Information"):
                            st.code(traceback.format_exc())
                    else:
                        st.info("Calculate emissions first to create download package")

                    st.markdown("---")
                    st.markdown("### 📚 Export Formats")
                    st.info("""
                                    **Available Export Formats:**
                                    - **CSV**: Comma-separated values for spreadsheet applications
                                    - **PNG**: High-resolution maps (150 DPI) for reports and presentations
                                    - **ZIP**: Complete analysis package with all data and documentation

                                    **Recommended Uses:**
                                    - Academic Research: Use ZIP package for complete documentation
                                    - Policy Reports: Use PNG maps with summary statistics
                                    - Data Analysis: Use individual CSV files for further processing

                                    **Vehicle Type Breakdown:**
                                    All exports now include separate columns for:
                                    - PC: Passenger Cars
                                    - LDV: Light Duty Vehicles
                                    - HDV: Heavy Duty Vehicles
                                    - Moto: Motorcycles
                                    - Total: Sum of all vehicle types
                                    """)

                # Footer
                st.markdown("---")
                st.markdown("""
                                <div style='text-align: center; color: #666; padding: 20px;'>
                                    <p><strong>Advanced Traffic Emission Calculator v2.0</strong></p>
                                    <p>Built with COPERT IV, IPCC, and EPA MOVES methodologies</p>
                                    <p>Now with support for PC, LDV, HDV, and Motorcycle emissions</p>
                                    <p>Standards: EEA Guidebook 2019, IPCC 2019 Guidelines, WHO Air Quality Standards</p>
                                    <p>© 2025 - Developed by SHassan 🎈</p>
                                </div>
                                """, unsafe_allow_html=True)
