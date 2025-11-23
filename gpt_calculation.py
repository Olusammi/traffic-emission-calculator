import streamlit as st
import numpy as np
import matplotlib
import tempfile
import os
import zipfile
from io import BytesIO

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.colorbar
import pandas as pd
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

# ==================== UNIT CONVERSION SETTINGS ====================
st.sidebar.markdown("---")
st.sidebar.header("📏 Unit Conversion")
st.sidebar.markdown("Convert emission results to different units")

# Unit conversion options for different pollutants
unit_conversion_options = {
    "CO": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "tonnes/km": {"name": "Tonnes per kilometer", "factor": 0.000001},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934},
        "kg/year": {"name": "Kilograms per year (avg)", "factor": 1.0, "note": "Requires trip frequency"}
    },
    "CO2": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "tonnes/km": {"name": "Tonnes per kilometer", "factor": 0.000001},
        "kg CO2e/km": {"name": "kg CO2 equivalent per km", "factor": 0.001},
        "tonnes CO2e/year": {"name": "Tonnes CO2e per year", "factor": 1.0, "note": "Requires annual distance"},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934}
    },
    "NOx": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1000.0},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934},
        "kg NO2e/km": {"name": "kg NO2 equivalent per km", "factor": 0.001}
    },
    "PM": {
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1.0},
        "g/km": {"name": "Grams per kilometer", "factor": 0.001},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.000001},
        "µg/km": {"name": "Micrograms per kilometer", "factor": 1000.0},
        "mg/mile": {"name": "Milligrams per mile", "factor": 1.60934}
    },
    "VOC": {
        "g/km": {"name": "Grams per kilometer", "factor": 1.0},
        "kg/km": {"name": "Kilograms per kilometer", "factor": 0.001},
        "mg/km": {"name": "Milligrams per kilometer", "factor": 1000.0},
        "g/mile": {"name": "Grams per mile", "factor": 1.60934}
    },
    "FC": {
        "L/100km": {"name": "Liters per 100 km", "factor": 1.0},
        "L/km": {"name": "Liters per km", "factor": 0.01},
        "gal/100mi": {"name": "Gallons per 100 miles (US)", "factor": 2.3521},
        "mpg": {"name": "Miles per gallon (US)", "factor": 1.0, "note": "Inverse calculation"},
        "km/L": {"name": "Kilometers per liter", "factor": 1.0, "note": "Inverse calculation"}
    }
}

# Store selected units for each pollutant
selected_units = {}
for poll in pollutants_available.keys():
    if poll in unit_conversion_options:
        default_unit = list(unit_conversion_options[poll].keys())[0]
        selected_unit = st.sidebar.selectbox(
            f"{poll} Display Unit",
            options=list(unit_conversion_options[poll].keys()),
            format_func=lambda x, p=poll: f"{x} - {unit_conversion_options[p][x]['name']}",
            key=f"unit_{poll}"
        )
        selected_units[poll] = selected_unit
    else:
        selected_units[poll] = pollutants_available[poll]['unit']

# Store in session state for access across tabs
st.session_state.selected_units = selected_units
st.session_state.unit_conversion_options = unit_conversion_options

# Show conversion info
if st.sidebar.checkbox("Show Conversion Info", value=False):
    st.sidebar.markdown("**Conversion Notes:**")
    st.sidebar.info("""
    - **CO2e**: CO2 equivalent includes global warming potential
    - **Annual calculations**: Based on average 15,000 km/year
    - **Mile conversions**: 1 mile = 1.60934 km
    - **Inverse units** (mpg, km/L): Calculated as reciprocal
    """)

st.sidebar.markdown("---")
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

# NOTE: LDV and HDV distribution file uploaders are removed as requested.
st.sidebar.markdown("---")
st.sidebar.info("""
**LDV/HDV Distributions:**
Default fleet compositions (PC Gasoline for LDV, 100% Euro VI for HDV) will be used if specific distribution files are not provided, allowing calculations to proceed.
""")


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

# ==================== UNIT CONVERSION FUNCTION ====================
def convert_emission_value(value, pollutant, from_unit, to_unit, distance_km=None):
    """
    Convert emission values between different units
    
    Args:
        value: Original emission value
        pollutant: Pollutant type (CO, CO2, NOx, etc.)
        from_unit: Original unit
        to_unit: Target unit
        distance_km: Optional distance for annual calculations
    
    Returns:
        Converted value
    """
    if from_unit == to_unit:
        return value
    
    if pollutant not in unit_conversion_options:
        return value
    
    conversions = unit_conversion_options[pollutant]
    
    if to_unit not in conversions:
        return value
    
    # Handle inverse calculations (mpg, km/L)
    if to_unit in ["mpg", "km/L"]:
        if value == 0:
            return 0
        if to_unit == "mpg":
            # Convert L/100km to mpg
            return 235.214 / value if value != 0 else 0
        elif to_unit == "km/L":
            # Convert L/100km to km/L
            return 100 / value if value != 0 else 0
    
    # Handle annual conversions
    if "year" in to_unit:
        annual_distance = distance_km if distance_km else 15000  # Default 15,000 km/year
        base_value = value * conversions[to_unit]["factor"]
        return base_value * annual_distance
    
    # Standard conversion
    return value * conversions[to_unit]["factor"]

def format_emission_value(value, unit):
    """Format emission value based on magnitude"""
    if value == 0:
        return "0.00"
    elif value < 0.001:
        return f"{value:.6f}"
    elif value < 0.1:
        return f"{value:.4f}"
    elif value < 10:
        return f"{value:.3f}"
    elif value < 1000:
        return f"{value:.2f}"
    else:
        return f"{value:.1f}"

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
        - **Multi-Vehicle Support**: PC, Motorcycle, **LDV**, and **HDV** emissions
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
       - Link OSM data file (**7 or 9 columns**)
       - OSM network file (.osm)
       - 6 vehicle proportion files
       - **LDV and HDV distributions are now defaulted/inferred**

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
            # Read data using '\s+' as separator for space/tab/etc.
            data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python')
            
            # Flexible column naming based on file size
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

            # Statistics - adjust columns for 7 or 9 columns
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
                    import copert # Assumes copert.py is available or uploaded

                    # Setup temporary files
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pc_path = os.path.join(tmpdir, "PC_parameter.csv")
                        ldv_path = os.path.join(tmpdir, "LDV_parameter.csv")
                        hdv_path = os.path.join(tmpdir, "HDV_parameter.csv")
                        moto_path = os.path.join(tmpdir, "Moto_parameter.csv")

                        # Write uploaded content to temporary files for the COPERT class to load
                        with open(pc_path, 'wb') as f:
                            f.write(pc_param.getbuffer())
                        with open(ldv_path, 'wb') as f:
                            f.write(ldv_param.getbuffer())
                        with open(hdv_path, 'wb') as f:
                            f.write(hdv_param.getbuffer())
                        with open(moto_path, 'wb') as f:
                            f.write(moto_param.getbuffer())

                        # Initialize COPERT class
                        cop = copert.Copert(pc_path, ldv_path, hdv_path, moto_path)

                        # Read data
                        link_osm.seek(0)
                        engine_cap_gas.seek(0)
                        engine_cap_diesel.seek(0)
                        copert_class_gas.seek(0)
                        copert_class_diesel.seek(0)
                        copert_2stroke.seek(0)
                        copert_4stroke.seek(0)

                        # Load link data, supporting 7 or 9 columns
                        data_link = pd.read_csv(link_osm, sep=r'\s+', header=None, engine='python').values

                        Nlink = data_link.shape[0]

                        # --- Link Data Proportion Handling (LDV/HDV proportions P_ldv, P_hdv) ---
                        if data_link.shape[1] == 7:
                            # Original 7-column format: PC/Moto only. LDV/HDV proportions are assumed to be 0
                            P_ldv = np.zeros(Nlink)
                            P_hdv = np.zeros(Nlink)
                            st.warning("⚠️ Link data has only 7 columns. LDV and HDV flow proportions are assumed to be **0** for all links.")
                        elif data_link.shape[1] == 9:
                            # Extended 9-column format: PC, Moto, LDV, HDV
                            P_ldv = data_link[:, 7]
                            P_hdv = data_link[:, 8]
                        else:
                            st.error(f"❌ Link data must have 7 or 9 columns, got {data_link.shape[1]}")
                            st.stop()

                        # --- End Link Data Handling ---

                        # Load PC and Moto distribution files (these MUST be uploaded)
                        data_engine_capacity_gasoline = np.loadtxt(engine_cap_gas)
                        data_engine_capacity_diesel = np.loadtxt(engine_cap_diesel)
                        data_copert_class_gasoline = np.loadtxt(copert_class_gas)
                        data_copert_class_diesel = np.loadtxt(copert_class_diesel)
                        data_copert_class_motorcycle_two_stroke = np.loadtxt(copert_2stroke)
                        data_copert_class_motorcycle_four_stroke = np.loadtxt(copert_4stroke)

                        # --- NEW: LDV and HDV Distribution Defaults (No external file needed) ---

                        # 1. Default LDV Distribution: Use the Gasoline PC Euro Class distribution as a proxy for the entire LDV fleet.
                        st.info("ℹ️ Using **Passenger Car (Gasoline)** Euro Class distribution as the default for **LDV**.")
                        data_copert_class_ldv = data_copert_class_gasoline

                        # 2. Default HDV Distribution: Assumption of a modern, standard fleet.
                        N_HDV_Class = 6 # Euro I to VI
                        N_HDV_Type = 15 # Standard COPERT types
                        st.info("ℹ️ Using a **Default HDV Fleet** composition: **100% Euro VI** (cleanest standard) and **Rigid Truck < 7.5t** (Type 0).")
                        
                        data_hdv_reshaped = np.zeros((Nlink, N_HDV_Class, N_HDV_Type))
                        
                        # Set 100% of the fleet for each link to Euro VI (index 5) and HDV Type 0 (Rigid Truck < 7.5t, index 0)
                        data_hdv_reshaped[:, 5, 0] = 1.0 
                        
                        # --- End New Defaults ---

                        # ====================================================================
                        # 🔍 PARAMETER VERIFICATION SECTION - PUT THIS RIGHT HERE
                        # ====================================================================
                        st.subheader("🔍 Parameter Verification")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write("**Motorcycle Distribution Ranges:**")
                            st.write(f"2-stroke: {data_copert_class_motorcycle_two_stroke.min():.3f} to {data_copert_class_motorcycle_two_stroke.max():.3f}")
                            st.write(f"4-stroke: {data_copert_class_motorcycle_four_stroke.min():.3f} to {data_copert_class_motorcycle_four_stroke.max():.3f}")

                        with col2:
                            st.write("**Link Data Ranges:**")
                            st.write(f"PC Proportion: {data_link[:, 5].min():.3f} to {data_link[:, 5].max():.3f}")
                            st.write(f"Motorcycle Flow: {(1 - data_link[:, 5]).min():.3f} to {(1 - data_link[:, 5]).max():.3f}")
                            st.write(f"LDV Proportion: {P_ldv.min():.3f} to {P_ldv.max():.3f}")
                            st.write(f"HDV Proportion: {P_hdv.min():.3f} to {P_hdv.max():.3f}")

                        with col3:
                            st.write("**Calculation Parameters:**")
                            st.write(f"Speed Range: {data_link[:, 3].min():.1f} to {data_link[:, 3].max():.1f} km/h")
                            st.write(f"Flow Range: {data_link[:, 2].min():.1f} to {data_link[:, 2].max():.1f} vehicles")
                            st.write(f"Link Length Range: {data_link[:, 1].min():.3f} to {data_link[:, 1].max():.3f} km")

                        # MOTORCYCLE CALCULATION TEST
                        st.subheader("🧪 Motorcycle Calculation Test")
                        test_speed = 50.0
                        test_pollutant = cop.pollutant_CO
                        test_engine_type = cop.engine_type_moto_four_stroke_50_250
                        test_copert_class = cop.class_moto_Euro_3

                        try:
                            test_result = cop.EFMotorcycle(test_pollutant, test_speed, test_engine_type, test_copert_class)
                            st.write(f"Test Calculation - CO at 50 km/h, 4-stroke Euro 3: {test_result:.6f} g/km")
                            
                            if test_result == 0:
                                st.error("❌ Motorcycle test calculation returned 0 - there may be parameter issues")
                            else:
                                st.success("✅ Motorcycle test calculation successful!")
                                
                        except Exception as test_error:
                            st.error(f"❌ Motorcycle test calculation failed: {test_error}")

                        # ====================================================================
                        # END PARAMETER VERIFICATION SECTION
                        # ====================================================================

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

                        # HDV Emission Classes (for use in calculation loop)
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
                        # Add after line: emissions_data[poll] = {'pc': ..., 'total': ...}

                        # Also track fuel-specific emissions
                        fuel_emissions_data = {}
                        for poll in selected_pollutants:
                            fuel_emissions_data[poll] = {
                                'gasoline': np.zeros((Nlink,), dtype=float),
                                'diesel': np.zeros((Nlink,), dtype=float)
                            }
                        # Initialize with LDV and HDV arrays
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
                            # ... rest of your existing calculation code ... 
                        
                        # --- End New Defaults ---


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

                        # HDV Emission Classes (for use in calculation loop)
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

                        # Initialize with LDV and HDV arrays
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
                        # Calculate emissions for each link
                        for i in range(Nlink):
                            if i % max(1, Nlink // 100) == 0:
                                progress_bar.progress(i / Nlink)
                                status_text.text(f"Processing link {i + 1}/{Nlink} - {(i / Nlink * 100):.1f}% complete")
                        
                            link_length = data_link[i, 1]
                            link_flow = data_link[i, 2]
                            v = min(max(10., data_link[i, 3]), 130.)
                        
                            # Proportions read from link data
                            link_gasoline_proportion = data_link[i, 4]
                            link_pc_proportion = data_link[i, 5]
                            link_4_stroke_proportion = data_link[i, 6]
                        
                            # Flows on this link
                            p_passenger = link_pc_proportion
                            P_motorcycle = 1. - link_pc_proportion  # Assuming PC + Moto is base prior to LDV/HDV split
                            P_ldv_i = float(P_ldv[i]) if P_ldv is not None else 0.0
                            P_hdv_i = float(P_hdv[i]) if P_hdv is not None else 0.0
                        
                            engine_type_distribution = [link_gasoline_proportion, 1. - link_gasoline_proportion]
                            engine_capacity_distribution = [data_engine_capacity_gasoline[i], data_engine_capacity_diesel[i]]
                            # Replace whatever you currently set here with this correct stroke-based distribution:
                            engine_type_motorcycle_distribution = [link_4_stroke_proportion, 1.0 - link_4_stroke_proportion]

                        
                            # For convenience: ensure ldv copert class array exists
                            # data_copert_class_ldv is expected to be (Nlink, Nclass)
                            # data_hdv_reshaped is expected to be (Nlink, N_HDV_Class, N_HDV_Type)
                            for poll_name in selected_pollutants:
                                poll_type = pollutant_mapping[poll_name]
                        
                                # -------- Passenger Car (PC) ------------
                                try:
                                    for t in range(2):  # engine types: gasoline/diesel
                                        for c in range(Nclass):  # euro classes
                                            for k in range(2):  # engine capacity bins
                                                # Diesel/capacity filter preserved
                                                if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                    continue
                                                try:
                                                    e = cop.Emission(poll_type, v, link_length,
                                                                     cop.vehicle_type_passenger_car,
                                                                     engine_type[t],
                                                                     copert_class[c],
                                                                     engine_capacity[k],
                                                                     ambient_temp)
                                                except Exception:
                                                    # fallback to HEF if signature differs
                                                    try:
                                                        if engine_type[t] in (cop.engine_type_gasoline, getattr(cop, 'engine_type_gasoline', None)):
                                                            e = cop.HEFGasolinePassengerCar(poll_type, v, copert_class[c], engine_capacity[k])
                                                        else:
                                                            e = cop.HEFDieselPassengerCar(poll_type, v, copert_class[c], engine_capacity[k])
                                                        e = e * link_length
                                                    except Exception:
                                                        e = 0.0
                                                # temperature correction (NOx)
                                                if poll_name == "NOx" and include_temperature_correction:
                                                    temp_factor = 1 + 0.02 * (ambient_temp - 20)
                                                    e *= temp_factor
                                                # cold start
                                                if include_cold_start and poll_name in ["CO", "NOx", "VOC"]:
                                                    try:
                                                        beta = cop.ColdStartMileagePercentage(
                                                            cop.vehicle_type_passenger_car, engine_type[t], poll_type,
                                                            copert_class[c], engine_capacity[k], ambient_temp, trip_length)
                                                        e_cold = cop.ColdStartEmissionQuotient(
                                                            cop.vehicle_type_passenger_car, engine_type[t], poll_type,
                                                            v, copert_class[c], engine_capacity[k], ambient_temp)
                                                        e = e * ((1 - beta) + e_cold * beta)
                                                    except Exception:
                                                        pass
                        
                                                pc_fleet_share = data_copert_class_gasoline[i, c] if t == 0 else data_copert_class_diesel[i, c]
                                                e *= engine_type_distribution[t] * engine_capacity_distribution[t][k] * pc_fleet_share
                        
                                                #emissions_data[poll_name]['pc'][i] += e * p_passenger / link_length * link_flow
                                # Calculate once and store in variable
                                                final_e = e * p_passenger / link_length * link_flow
                                                
                                                # Add to PC total
                                                emissions_data[poll_name]['pc'][i] += final_e
                                                
                                                # Track by fuel type
                                                if 'fuel_emissions_data' in locals():
                                                    if t == 0:  # Gasoline
                                                        fuel_emissions_data[poll_name]['gasoline'][i] += final_e
                                                    else:  # Diesel
                                                        fuel_emissions_data[poll_name]['diesel'][i] += final_e
                                except Exception:
                                    # if PC block wholly fails, skip but continue
                                    pass
                        
                                # -------- LDV ------------
                                if P_ldv_i > 0:
                                    try:
                                        for t in range(2):
                                            for c in range(Nclass):
                                                for k in range(2):
                                                    # preserve same diesel/capacity skip rule
                                                    if t == 1 and k == 0 and copert_class[c] in range(cop.class_Euro_1, 1 + cop.class_Euro_3):
                                                        continue
                                                    try:
                                                        e_ldv = cop.Emission(poll_type, v, link_length,
                                                                             cop.vehicle_type_light_commercial_vehicle,
                                                                             engine_type[t], copert_class[c], engine_capacity[k], ambient_temp)
                                                    except Exception:
                                                        try:
                                                            e_ldv = cop.HEFLightCommercialVehicle(poll_type, v, engine_type[t], copert_class[c])
                                                            e_ldv = e_ldv * link_length
                                                        except Exception:
                                                            e_ldv = 0.0
                        
                                                    # temperature correction
                                                    if poll_name == "NOx" and include_temperature_correction:
                                                        temp_factor = 1 + 0.02 * (ambient_temp - 20)
                                                        e_ldv *= temp_factor
                                                    # cold start
                                                    if include_cold_start and poll_name in ["CO", "NOx", "VOC"]:
                                                        try:
                                                            beta = cop.ColdStartMileagePercentage(
                                                                cop.vehicle_type_light_commercial_vehicle, engine_type[t], poll_type,
                                                                copert_class[c], engine_capacity[k], ambient_temp, trip_length)
                                                            e_cold = cop.ColdStartEmissionQuotient(
                                                                cop.vehicle_type_light_commercial_vehicle, engine_type[t], poll_type,
                                                                v, copert_class[c], engine_capacity[k], ambient_temp)
                                                            e_ldv = e_ldv * ((1 - beta) + e_cold * beta)
                                                        except Exception:
                                                            pass
                        
                                                    # ldv fleet share (uses data_copert_class_ldv; defaulted to PC gasoline distribution)
                                                    ldv_fleet_share = data_copert_class_ldv[i, c]
                                                    e_ldv *= engine_type_distribution[t] * engine_capacity_distribution[t][k] * ldv_fleet_share
                        
                                                    #emissions_data[poll_name]['ldv'][i] += e_ldv * P_ldv_i / link_length * link_flow
                                                    final_e_ldv = e_ldv * P_ldv_i / link_length * link_flow
                                                    emissions_data[poll_name]['ldv'][i] += final_e_ldv
                                                    
                                                    # Track fuel-specific emissions
                                                    if 'fuel_emissions_data' in locals():
                                                        if t == 0:  # Gasoline
                                                            fuel_emissions_data[poll_name]['gasoline'][i] += final_e_ldv
                                                        else:  # Diesel
                                                            fuel_emissions_data[poll_name]['diesel'][i] += final_e_ldv
                                                    # === END NEW SECTION ===
                                    except Exception:
                                        pass
                        
                                # -------- HDV ------------
                                if P_hdv_i > 0:
                                    try:
                                        for t_class in range(N_HDV_Class):
                                            for t_type in range(N_HDV_Type):
                                                hdv_fleet_share = data_hdv_reshaped[i, t_class, t_type]
                                                if hdv_fleet_share <= 0:
                                                    continue
                                                engine_type_hdv = cop.engine_type_diesel  # HDVs: diesel by default
                                                try:
                                                    e_hdv = cop.Emission(poll_type, v, link_length,
                                                                         cop.vehicle_type_heavy_duty_vehicle,
                                                                         engine_type_hdv,
                                                                         HDV_Emission_Classes[t_class],
                                                                         t_type,
                                                                         ambient_temp)
                                                except Exception:
                                                    # fallback: if signature different, try a simple HEF or 0
                                                    try:
                                                        e_hdv = cop.HEFHeavyDutyVehicle(poll_type, v, HDV_Emission_Classes[t_class], t_type)
                                                        e_hdv = e_hdv * link_length
                                                    except Exception:
                                                        # very small base factor fallback to avoid zeros
                                                        e_hdv = 0.0
                        
                                                # temperature correction for HDV NOx (diesel)
                                                if poll_name == "NOx" and include_temperature_correction:
                                                    temp_factor = 1 + 0.015 * (ambient_temp - 20)
                                                    e_hdv *= temp_factor
                        
                                                e_hdv *= hdv_fleet_share
                                                emissions_data[poll_name]['hdv'][i] += e_hdv * P_hdv_i / link_length * link_flow
                                                # ADD:
                                                if 'fuel_emissions_data' in locals():
                                                    fuel_emissions_data[poll_name]['diesel'][i] += e_hdv * P_hdv_i / link_length * link_flow
                                    except Exception:
                                        pass
                        
                               # -------- Motorcycle (inside pollutant loop) ------------
                                try:
                                    for m in range(2):
                                        for d in range(Mclass):
                                            # preserve your skip condition for certain euro classes
                                            if m == 0 and copert_class_motorcycle[d] >= cop.class_moto_Euro_1:
                                                continue
                                            try:
                                                # Use poll_type (current pollutant) here, not a fixed pollutant
                                                e_f = cop.EFMotorcycle(poll_type, v, engine_type_m[m], copert_class_motorcycle[d])
                                            except Exception:
                                                # fallback: set to 0 and continue
                                                e_f = 0.0
                                
                                            # apply stroke distribution (2S/4S)
                                            e_f *= engine_type_motorcycle_distribution[m]
                                
                                            # ORIGINAL SCALING used in your working script — do NOT multiply by link_length
                                            emissions_data[poll_name]['moto'][i] += e_f * P_motorcycle * link_flow
                                            # ADD:
                                            if 'fuel_emissions_data' in locals():
                                                fuel_emissions_data[poll_name]['gasoline'][i] += e_f * P_motorcycle * link_flow
                                except Exception:
                                    # keep the app running even if motorcycle EF call fails for one pollutant/link
                                    pass


                        
                            # After computing all vehicle-type contributions for this link, compute totals per pollutant
                            for poll_name in selected_pollutants:
                                emissions_data[poll_name]['total'][i] = (
                                    emissions_data[poll_name]['pc'][i] +
                                    emissions_data[poll_name]['ldv'][i] +
                                    emissions_data[poll_name]['hdv'][i] +
                                    emissions_data[poll_name]['moto'][i]
                                )
                        
                        # end link loop
                        
                        progress_bar.empty()
                        status_text.empty()


                        # FINAL RESULTS HANDLING
                        # Store data in session state for cross-tab use
                        st.session_state.emissions_data = emissions_data
                        # After: st.session_state.emissions_data = emissions_data
                        st.session_state.fuel_emissions_data = fuel_emissions_data
                        st.session_state.data_link = data_link
                        st.session_state.selected_pollutants = selected_pollutants
                        st.success("✅ Multi-pollutant emissions calculated successfully!")

                        # Display results summary
                        st.subheader("📊 Emission Summary")
                        
                        # Get selected units
                        selected_units = st.session_state.get('selected_units', {})
                        unit_conversion_options = st.session_state.get('unit_conversion_options', {})
                        
                        # Create summary dataframe with converted units
                        summary_data = []
                        for poll in selected_pollutants:
                            target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                            original_unit = pollutants_available[poll]['unit']
                            
                            # Convert values
                            total_pc_converted = convert_emission_value(
                                emissions_data[poll]['pc'].sum(), poll, original_unit, target_unit
                            )
                            total_ldv_converted = convert_emission_value(
                                emissions_data[poll]['ldv'].sum(), poll, original_unit, target_unit
                            )
                            total_hdv_converted = convert_emission_value(
                                emissions_data[poll]['hdv'].sum(), poll, original_unit, target_unit
                            )
                            total_moto_converted = convert_emission_value(
                                emissions_data[poll]['moto'].sum(), poll, original_unit, target_unit
                            )
                            total_converted = convert_emission_value(
                                emissions_data[poll]['total'].sum(), poll, original_unit, target_unit
                            )
                            avg_converted = convert_emission_value(
                                emissions_data[poll]['total'].mean(), poll, original_unit, target_unit
                            )
                            max_converted = convert_emission_value(
                                emissions_data[poll]['total'].max(), poll, original_unit, target_unit
                            )
                            
                            summary_data.append({
                                'Pollutant': poll,
                                'Total PC': format_emission_value(total_pc_converted, target_unit),
                                'Total LDV': format_emission_value(total_ldv_converted, target_unit),
                                'Total HDV': format_emission_value(total_hdv_converted, target_unit),
                                'Total Moto': format_emission_value(total_moto_converted, target_unit),
                                'Total': format_emission_value(total_converted, target_unit),
                                'Avg per Link': format_emission_value(avg_converted, target_unit),
                                'Max': format_emission_value(max_converted, target_unit),
                                'Unit': target_unit
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Show conversion notice if units were changed
                        if any(selected_units.get(p) != pollutants_available[p]['unit'] for p in selected_pollutants):
                            st.info("ℹ️ Values have been converted to your selected display units")

                        # Detailed results by pollutant
                        for poll in selected_pollutants:
                            with st.expander(f"📋 Detailed Results: {poll} ({pollutants_available[poll]['name']})", expanded=False):
                                # Include LDV and HDV columns
                                results_df = pd.DataFrame({
                                    'OSM_ID': data_link[:, 0].astype(int),
                                    f'{poll}_PC': emissions_data[poll]['pc'],
                                    f'{poll}_LDV': emissions_data[poll]['ldv'],
                                    f'{poll}_HDV': emissions_data[poll]['hdv'],
                                    f'{poll}_Motorcycle': emissions_data[poll]['moto'],
                                    f'{poll}_Total': emissions_data[poll]['total']
                                })
                                st.dataframe(results_df.head(50), use_container_width=True)

                                # Display 5 metrics 
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric(f"Total PC {poll}", f"{emissions_data[poll]['pc'].sum():.2f} {pollutants_available[poll]['unit']}")
                                with col2:
                                    st.metric(f"Total LDV {poll}", f"{emissions_data[poll]['ldv'].sum():.2f} {pollutants_available[poll]['unit']}")
                                with col3:
                                    st.metric(f"Total HDV {poll}", f"{emissions_data[poll]['hdv'].sum():.2f} {pollutants_available[poll]['unit']}")
                                with col4:
                                    st.metric(f"Total Moto {poll}", f"{emissions_data[poll]['moto'].sum():.2f} {pollutants_available[poll]['unit']}")
                                with col5:
                                    st.metric(f"Grand Total {poll}", f"{emissions_data[poll]['total'].sum():.2f} {pollutants_available[poll]['unit']}")

                except Exception as e:
                    st.error(f"❌ An error occurred during calculation: {e}")
                    import traceback
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc())
    else:
        st.info("Please ensure all required files and at least one pollutant are selected to run the calculation.")

# ==================== TAB 5: MULTI-METRIC ANALYSIS (ENHANCED) ====================
with tab5:
    st.header("📈 Multi-Metric Emission Analysis")
    st.markdown("Compare emissions across different pollutants, vehicle types, and fuel types.")

    if 'emissions_data' in st.session_state:
        emissions_data = st.session_state.emissions_data
        selected_pollutants = st.session_state.selected_pollutants
        data_link = st.session_state.data_link

                    # ADD UNIT DISPLAY TOGGLE
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Select analysis views below")
        with col2:
            use_converted_units = st.checkbox("Use Converted Units", value=True, 
                                             help="Display values in your selected units from sidebar")

        if selected_pollutants:
            # ===== NEW: FUEL TYPE BREAKDOWN SECTION =====
            st.subheader("⛽ Emissions by Fuel Type")
            st.markdown("Breakdown of emissions by gasoline vs diesel vehicles")
            
            # Calculate fuel type proportions from link data
            gasoline_prop_avg = data_link[:, 4].mean()
            diesel_prop_avg = 1 - gasoline_prop_avg
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Gasoline Proportion", f"{gasoline_prop_avg*100:.1f}%")
            with col2:
                st.metric("Average Diesel Proportion", f"{diesel_prop_avg*100:.1f}%")

            for poll in selected_pollutants:
                pc_total = emissions_data[poll]['pc'].sum()
                
                if use_converted_units:
                    selected_units = st.session_state.get('selected_units', {})
                    target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                    original_unit = pollutants_available[poll]['unit']
                    pc_total = convert_emission_value(pc_total, poll, original_unit, target_unit)
             
            # Select pollutant for fuel type analysis
            fuel_analysis_pollutant = st.selectbox(
                "Select Pollutant for Fuel Type Analysis",
                options=selected_pollutants,
                key='fuel_analysis_select'
            )
            
            # Calculate emissions by fuel type for PC and LDV
            # Note: We'll estimate based on proportions since we don't store fuel-specific emissions
            fuel_type_data = []
            
            for poll in selected_pollutants:
                # For PC: split by gasoline/diesel proportion
                pc_total = emissions_data[poll]['pc'].sum()
                pc_gasoline_estimate = pc_total * gasoline_prop_avg
                pc_diesel_estimate = pc_total * diesel_prop_avg
                
                # For LDV: split by gasoline/diesel proportion
                ldv_total = emissions_data[poll]['ldv'].sum()
                ldv_gasoline_estimate = ldv_total * gasoline_prop_avg
                ldv_diesel_estimate = ldv_total * diesel_prop_avg
                
                # HDV: Typically 100% diesel
                hdv_total = emissions_data[poll]['hdv'].sum()
                
                # Motorcycles: Typically 100% gasoline
                moto_total = emissions_data[poll]['moto'].sum()
                
                # Total by fuel type
                total_gasoline = pc_gasoline_estimate + ldv_gasoline_estimate + moto_total
                total_diesel = pc_diesel_estimate + ldv_diesel_estimate + hdv_total
                
                fuel_type_data.append({
                    'Pollutant': poll,
                    'Fuel_Type': 'Gasoline',
                    'Total_Emissions': total_gasoline,
                    'Percentage': total_gasoline / (total_gasoline + total_diesel) * 100
                })
                fuel_type_data.append({
                    'Pollutant': poll,
                    'Fuel_Type': 'Diesel',
                    'Total_Emissions': total_diesel,
                    'Percentage': total_diesel / (total_gasoline + total_diesel) * 100
                })
            
            fuel_type_df = pd.DataFrame(fuel_type_data)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for selected pollutant
                fuel_selected_df = fuel_type_df[fuel_type_df['Pollutant'] == fuel_analysis_pollutant]
                fig_pie = px.pie(
                    fuel_selected_df,
                    values='Total_Emissions',
                    names='Fuel_Type',
                    title=f"{fuel_analysis_pollutant} Emissions by Fuel Type",
                    color='Fuel_Type',
                    color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'},
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart comparing all pollutants
                fig_fuel_bar = px.bar(
                    fuel_type_df,
                    x='Pollutant',
                    y='Total_Emissions',
                    color='Fuel_Type',
                    title="All Pollutants: Gasoline vs Diesel",
                    labels={'Total_Emissions': 'Total Emissions'},
                    color_discrete_map={'Gasoline': '#ff6b6b', 'Diesel': '#4dabf7'},
                    barmode='group',
                    template="plotly_white"
                )
                st.plotly_chart(fig_fuel_bar, use_container_width=True)
            
            # Detailed table
            st.markdown("**Detailed Fuel Type Breakdown**")
            fuel_summary = []
            for poll in selected_pollutants:
                poll_fuel_data = fuel_type_df[fuel_type_df['Pollutant'] == poll]
                gasoline_data = poll_fuel_data[poll_fuel_data['Fuel_Type'] == 'Gasoline'].iloc[0]
                diesel_data = poll_fuel_data[poll_fuel_data['Fuel_Type'] == 'Diesel'].iloc[0]
                
                fuel_summary.append({
                    'Pollutant': poll,
                    'Gasoline Emissions': f"{gasoline_data['Total_Emissions']:.2f}",
                    'Gasoline %': f"{gasoline_data['Percentage']:.1f}%",
                    'Diesel Emissions': f"{diesel_data['Total_Emissions']:.2f}",
                    'Diesel %': f"{diesel_data['Percentage']:.1f}%",
                    'Unit': pollutants_available[poll]['unit']
                })
            
            fuel_summary_df = pd.DataFrame(fuel_summary)
            st.dataframe(fuel_summary_df, use_container_width=True)
            
            st.markdown("---")
            
            # ===== EXISTING: VEHICLE TYPE BREAKDOWN SECTION =====
            st.subheader("🚗 Emissions by Vehicle Type")
            
            # Prepare data for breakdown chart
            breakdown_data = []
            for poll in selected_pollutants:
                breakdown_data.append({
                    'Pollutant': poll,
                    'Vehicle_Type': 'PC',
                    'Total_Emissions': emissions_data[poll]['pc'].sum()
                })
                breakdown_data.append({
                    'Pollutant': poll,
                    'Vehicle_Type': 'LDV',
                    'Total_Emissions': emissions_data[poll]['ldv'].sum()
                })
                breakdown_data.append({
                    'Pollutant': poll,
                    'Vehicle_Type': 'HDV',
                    'Total_Emissions': emissions_data[poll]['hdv'].sum()
                })
                breakdown_data.append({
                    'Pollutant': poll,
                    'Vehicle_Type': 'Motorcycle',
                    'Total_Emissions': emissions_data[poll]['moto'].sum()
                })

            breakdown_df = pd.DataFrame(breakdown_data)
            
            # --- Vehicle Type Breakdown Chart ---
            fig_breakdown = px.bar(breakdown_df, 
                                   x='Pollutant', 
                                   y='Total_Emissions', 
                                   color='Vehicle_Type',
                                   title="Total Emissions by Vehicle Type",
                                   labels={'Total_Emissions': 'Total Emissions (Sum of g/km * Flow)', 'Pollutant': 'Pollutant'},
                                   hover_data=['Total_Emissions', 'Vehicle_Type'],
                                   template="plotly_white")
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Pie chart for vehicle type distribution of selected pollutant
            vehicle_analysis_pollutant = st.selectbox(
                "Select Pollutant for Vehicle Type Pie Chart",
                options=selected_pollutants,
                key='vehicle_pie_select'
            )
            
            vehicle_selected_df = breakdown_df[breakdown_df['Pollutant'] == vehicle_analysis_pollutant]
            fig_vehicle_pie = px.pie(
                vehicle_selected_df,
                values='Total_Emissions',
                names='Vehicle_Type',
                title=f"{vehicle_analysis_pollutant} Emissions by Vehicle Type",
                color='Vehicle_Type',
                color_discrete_map={
                    'PC': '#667eea',
                    'LDV': '#f59e0b',
                    'HDV': '#ef4444',
                    'Motorcycle': '#10b981'
                },
                hole=0.4
            )
            fig_vehicle_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_vehicle_pie, use_container_width=True)

            st.markdown("---")
            
            # ===== EXISTING: LINK RANKING SECTION =====
            st.subheader("🔝 Top 10 Links by Total Emission")
            
            ranking_pollutant = st.selectbox("Select Pollutant to Rank by", options=selected_pollutants)

            if ranking_pollutant in emissions_data:
                ranking_data = st.session_state.data_link.copy()
                
                if ranking_data.shape[1] == 7:
                    ranking_data = pd.DataFrame(ranking_data, columns=['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop', '4Stroke_Prop'])
                elif ranking_data.shape[1] == 9:
                    ranking_data = pd.DataFrame(ranking_data, columns=['OSM_ID', 'Length_km', 'Flow', 'Speed', 'Gasoline_Prop', 'PC_Prop', '4Stroke_Prop', 'LDV_Prop', 'HDV_Prop'])
                
                ranking_data[f'Total_{ranking_pollutant}'] = emissions_data[ranking_pollutant]['total']

                top_10_df = ranking_data.sort_values(by=f'Total_{ranking_pollutant}', ascending=False).head(10)
                top_10_df['OSM_ID'] = top_10_df['OSM_ID'].astype(int)

                st.dataframe(top_10_df[['OSM_ID', 'Length_km', 'Flow', 'Speed', f'Total_{ranking_pollutant}']], use_container_width=True)

                fig_top_10 = px.bar(top_10_df, 
                                    x='OSM_ID', 
                                    y=f'Total_{ranking_pollutant}', 
                                    color='Speed',
                                    title=f"Top 10 Links Emitting {ranking_pollutant}",
                                    labels={f'Total_{ranking_pollutant}': f"Total {ranking_pollutant} (g/s or mg/s)"},
                                    template="plotly_white")
                st.plotly_chart(fig_top_10, use_container_width=True)
            else:
                st.warning("Please select a pollutant that has been calculated.")
            
            # ===== NEW: COMPARATIVE INSIGHTS SECTION =====
            st.markdown("---")
            st.subheader("💡 Comparative Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 Key Findings:**")
                
                # Find dominant vehicle type
                for poll in selected_pollutants:
                    pc = emissions_data[poll]['pc'].sum()
                    ldv = emissions_data[poll]['ldv'].sum()
                    hdv = emissions_data[poll]['hdv'].sum()
                    moto = emissions_data[poll]['moto'].sum()
                    
                    vehicle_totals = {'PC': pc, 'LDV': ldv, 'HDV': hdv, 'Motorcycle': moto}
                    dominant_vehicle = max(vehicle_totals, key=vehicle_totals.get)
                    dominant_percentage = (vehicle_totals[dominant_vehicle] / sum(vehicle_totals.values())) * 100
                    
                    st.markdown(f"""
                    **{poll}**: {dominant_vehicle} contributes **{dominant_percentage:.1f}%** of total emissions
                    """)
            
            with col2:
                st.markdown("**⛽ Fuel Type Insights:**")
                
                for poll in selected_pollutants:
                    poll_fuel_data = fuel_type_df[fuel_type_df['Pollutant'] == poll]
                    gasoline_pct = poll_fuel_data[poll_fuel_data['Fuel_Type'] == 'Gasoline']['Percentage'].iloc[0]
                    diesel_pct = poll_fuel_data[poll_fuel_data['Fuel_Type'] == 'Diesel']['Percentage'].iloc[0]
                    
                    dominant_fuel = 'Gasoline' if gasoline_pct > diesel_pct else 'Diesel'
                    dominant_fuel_pct = max(gasoline_pct, diesel_pct)
                    
                    st.markdown(f"""
                    **{poll}**: {dominant_fuel} vehicles contribute **{dominant_fuel_pct:.1f}%**
                    """)

        else:
            st.info("No pollutants selected for analysis.")
    else:
        st.info("Please calculate emissions first in the 'Calculate Emissions' tab.")
 
# ==================== TAB 6: INTERACTIVE MAP ====================
# ==================== TAB 6: INTERACTIVE MAP ====================
with tab6:
    st.header("🗺️ Interactive Emission Map")
    st.markdown("Visualize the calculated emissions on the road network.")
    
    if 'emissions_data' in st.session_state and 'data_link' in st.session_state:
        
        # Load data
        emissions_data = st.session_state.emissions_data
        data_link_np = st.session_state.data_link
        
        # NEW: Vehicle Type Selection for Mapping
        col1, col2 = st.columns(2)
        with col1:
            vehicle_types_to_map = ['Total', 'PC', 'LDV', 'HDV', 'Moto']
            map_type = st.selectbox(
                "Select Vehicle Type to Map", 
                options=vehicle_types_to_map,
                key='map_type_select',
                index=0,  # Default to 'Total'
                help="Choose which vehicle type's emissions to visualize on the map"
            )
        
        with col2:
            # Existing Pollutant Selection
            map_pollutant = st.selectbox(
                "Select Pollutant to Map", 
                options=st.session_state.selected_pollutants,
                key='map_pollutant_select',
                help="Choose which pollutant to visualize on the map"
            )
        
        # Load the emission data for the selected vehicle type and pollutant
        hot_emission = emissions_data[map_pollutant][map_type.lower()]
        
        st.info(f"📊 Mapping {map_type} {map_pollutant} emissions across the road network")
        
        # Check if OSM file is available for proper road network visualization
        if osm_file is None:
            st.warning("⚠️ Please upload OSM network file for proper road network visualization")
            
            # Fallback to simplified scatter plot if no OSM file
            st.subheader(f"Simplified {map_type} {map_pollutant} Emission Map")
            st.caption("Note: Upload OSM file for proper road network visualization")
            
            # Prepare data for simplified map visualization
            map_df = pd.DataFrame({
                'OSM_ID': data_link_np[:, 0],
                'Latitude': (data_link_np[:, 0] % 1000) * 0.0001 + y_min, # Placeholder Lat
                'Longitude': (data_link_np[:, 0] % 1000) * 0.0001 + x_min, # Placeholder Lon
                'Emission_Value': hot_emission,
                'Speed': data_link_np[:, 3]
            })
            
            # Create a scatter map for visualization
            fig_map = go.Figure(data=go.Scattergeo(
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda row: f"Link ID: {int(row['OSM_ID'])}<br>{map_type} {map_pollutant}: {row['Emission_Value']:.2f} {pollutants_available[map_pollutant]['unit']}", axis=1),
                mode='markers',
                marker=dict(
                    size=10,
                    opacity=0.8,
                    reversescale=True,
                    autocolorscale=False,
                    symbol='circle',
                    line=dict(width=1, color='rgba(102, 102, 102)'),
                    cmax=hot_emission.max(),
                    cmin=hot_emission.min(),
                    colorbar_title=f"{map_type} {map_pollutant}",
                    color=map_df['Emission_Value'],
                    colorscale=px.colors.sequential.Viridis
                )))

            fig_map.update_layout(
                geo=dict(
                    scope='world',
                    showland=True,
                    landcolor='rgb(217, 217, 217)',
                    subunitcolor='rgb(255, 255, 255)',
                    countrycolor='rgb(255, 255, 255)',
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)',
                    showsubunits=True,
                    showcountries=True,
                    lonaxis=dict(range=[x_min - tolerance, x_max + tolerance]),
                    lataxis=dict(range=[y_min - tolerance, y_max + tolerance]),
                ),
                title_text=f"Geographical Distribution of {map_type} {map_pollutant} Emissions",
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        else:
            # Use the sophisticated road network visualization when OSM file is available
            st.subheader("🎨 Visualization Settings")
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
            if st.button("🗺️ Generate Road Network Map", type="primary", use_container_width=True):
                with st.spinner("Generating emission map..."):
                    try:
                        import osm_network
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
                            emission_osm_id = [int(x) for x in data_link_np[:, 0]]
                            
                            fig = plt.figure(figsize=(fig_size, fig_size - 1), dpi=100)
                            ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
                            ax.set_aspect("equal", adjustable="box")
                            ax_c = fig.add_axes([0.85, 0.21, 0.03, 0.53])
                            cb = matplotlib.colorbar.ColorbarBase(ax_c, cmap=plt.cm.get_cmap(colormap), norm=color_scale, orientation="vertical")
                            cb.set_label(f"{map_pollutant} ({pollutants_available[map_pollutant]['unit']})", fontsize=12)
                            
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
                                ax.set_title(f"{map_type} {map_pollutant} Emission Map", fontsize=14)
                            else:
                                ax.set_title(f"{map_type} {map_pollutant} Emission Map with Road Names", fontsize=14, fontweight='bold')
                            
                            ax.set_xlabel("Longitude", fontsize=12)
                            ax.set_ylabel("Latitude", fontsize=12)
                            
                            if add_grid:
                                ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)
                            
                            st.pyplot(fig)
                            st.session_state.emission_map_fig = fig
                            
                            # Display statistics
                            if show_labels and viz_mode != "Classic (Original)":
                                col1, col2, col3 = st.columns(3)
                                with col1: 
                                    st.metric("Roads with Emission Data", roads_with_data)
                                with col2: 
                                    st.metric("Roads without Data", roads_without_data)
                                with col3: 
                                    st.metric("Unique Road Names Labeled", len(labeled_roads))
                            else:
                                col1, col2 = st.columns(2)
                                with col1: 
                                    st.metric("Roads with Emission Data", roads_with_data)
                                with col2: 
                                    st.metric("Roads without Data", roads_without_data)
                            
                            status_text.empty()
                            st.success("✅ Road network map generated successfully!")
                            
                        finally:
                            if os.path.exists(osm_path):
                                os.unlink(osm_path)
                                
                    except Exception as e:
                        st.error(f"❌ Error generating road network map: {e}")
                        import traceback
                        with st.expander("🐛 Debug Information"):
                            st.code(traceback.format_exc())
        
        # NEW: Add summary statistics for the selected vehicle type and pollutant
        # NEW: Add summary statistics for the selected vehicle type and pollutant
        st.markdown("---")
        st.subheader(f"📊 {map_type} {map_pollutant} Summary")
        
        # Get conversion settings
        selected_units = st.session_state.get('selected_units', {})
        target_unit = selected_units.get(map_pollutant, pollutants_available[map_pollutant]['unit'])
        original_unit = pollutants_available[map_pollutant]['unit']
        
        # Convert values
        total_converted = convert_emission_value(
            hot_emission.sum(), map_pollutant, original_unit, target_unit
        )
        avg_converted = convert_emission_value(
            hot_emission.mean(), map_pollutant, original_unit, target_unit
        )
        max_converted = convert_emission_value(
            hot_emission.max(), map_pollutant, original_unit, target_unit
        )
        min_converted = convert_emission_value(
            hot_emission.min(), map_pollutant, original_unit, target_unit
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Total {map_type} {map_pollutant}", 
                     f"{format_emission_value(total_converted, target_unit)} {target_unit}")
        with col2:
            st.metric(f"Average per Link", 
                     f"{format_emission_value(avg_converted, target_unit)} {target_unit}")
        with col3:
            st.metric(f"Maximum", 
                     f"{format_emission_value(max_converted, target_unit)} {target_unit}")
        with col4:
            st.metric(f"Minimum", 
                     f"{format_emission_value(min_converted, target_unit)} {target_unit}")
        
        # Update colorbar label on map
        if 'cb' in locals():
            cb.set_label(f"{map_pollutant} ({target_unit})", fontsize=12)
        
    else:
        st.info("Please calculate emissions first in the 'Calculate Emissions' tab.")

# ==================== TAB 7: DOWNLOAD RESULTS ====================
# ==================== TAB 7: DOWNLOAD RESULTS ====================
with tab7:
    st.header("📥 Download Results")
    st.markdown("Download calculated data and reports.")

    if 'emissions_data' in st.session_state and 'data_link' in st.session_state:
        emissions_data = st.session_state.emissions_data
        data_link_np = st.session_state.data_link
        selected_pollutants = st.session_state.selected_pollutants
                
        # ADD UNIT SELECTION FOR EXPORT
        st.subheader("📤 Export Settings")
        col1, col2 = st.columns(2)
        with col1:
            export_in_converted_units = st.checkbox(
                "Export in converted units",
                value=False,
                help="Export emissions in your selected display units instead of original g/km"
            )
        with col2:
            if export_in_converted_units:
                st.info("✅ Will export using sidebar unit selections")
            else:
                st.info("ℹ️ Will export in original units (g/km, mg/km, L/100km)")
        
        st.markdown("---")
        # ============ END NEW SECTION ============

        # Create a single comprehensive results dataframe
        final_results_df = pd.DataFrame(data_link_np[:, :4], columns=['OSM_ID', 'Length_km', 'Flow', 'Speed'])
        
        # Assign flexible column names for the proportions based on 7 or 9 columns
        if data_link_np.shape[1] == 7:
            proportion_columns = ['Gasoline_Prop', 'PC_Prop', '4Stroke_Prop']
            proportion_data = data_link_np[:, 4:7]
        else: # 9 columns
            proportion_columns = ['Gasoline_Prop', 'PC_Prop', '4Stroke_Prop', 'LDV_Prop', 'HDV_Prop']
            proportion_data = data_link_np[:, 4:9]

        proportion_df = pd.DataFrame(proportion_data, columns=proportion_columns)
        final_results_df = pd.concat([final_results_df, proportion_df], axis=1)

        # ============ REPLACE WITH THIS ============
        for poll in selected_pollutants:
            if export_in_converted_units and 'selected_units' in st.session_state:
                selected_units = st.session_state.selected_units
                target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                original_unit = pollutants_available[poll]['unit']
                
                poll_df = pd.DataFrame({
                    f'{poll}_PC ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) 
                                                    for v in emissions_data[poll]['pc']],
                    f'{poll}_LDV ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) 
                                                     for v in emissions_data[poll]['ldv']],
                    f'{poll}_HDV ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) 
                                                     for v in emissions_data[poll]['hdv']],
                    f'{poll}_Motorcycle ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) 
                                                            for v in emissions_data[poll]['moto']],
                    f'{poll}_Total ({target_unit})': [convert_emission_value(v, poll, original_unit, target_unit) 
                                                       for v in emissions_data[poll]['total']]
                })
            else:
                poll_df = pd.DataFrame({
                    f'{poll}_PC': emissions_data[poll]['pc'],
                    f'{poll}_LDV': emissions_data[poll]['ldv'],
                    f'{poll}_HDV': emissions_data[poll]['hdv'],
                    f'{poll}_Motorcycle': emissions_data[poll]['moto'],
                    f'{poll}_Total': emissions_data[poll]['total']
                })
            final_results_df = pd.concat([final_results_df, poll_df], axis=1)
        # ============ END REPLACEMENT ============


        # Convert the DataFrame to CSV format for download
        csv_export = final_results_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download All Link Results (CSV)",
            data=csv_export,
            file_name='traffic_emission_results_full.csv',
            mime='text/csv',
            key='download_csv_full',
            use_container_width=True
        )

        st.markdown("---")

        # ZIP file containing all calculated data and reports
        st.subheader("Complete Analysis Package (ZIP)")
        
        # IMPORTANT: Create summary_df BEFORE the button, but store it in session state
        # This ensures it's available when the button is clicked
        # ============ REPLACE WITH THIS ============
        if 'summary_df_for_download' not in st.session_state or st.session_state.get('recalculate_summary', True):
            summary_data = []
            for poll in selected_pollutants:
                # Determine which unit to use
                if export_in_converted_units and 'selected_units' in st.session_state:
                    selected_units = st.session_state.selected_units
                    target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                    original_unit = pollutants_available[poll]['unit']
                    
                    total_pc = convert_emission_value(emissions_data[poll]['pc'].sum(), poll, original_unit, target_unit)
                    total_ldv = convert_emission_value(emissions_data[poll]['ldv'].sum(), poll, original_unit, target_unit)
                    total_hdv = convert_emission_value(emissions_data[poll]['hdv'].sum(), poll, original_unit, target_unit)
                    total_moto = convert_emission_value(emissions_data[poll]['moto'].sum(), poll, original_unit, target_unit)
                    grand_total = convert_emission_value(emissions_data[poll]['total'].sum(), poll, original_unit, target_unit)
                    display_unit = target_unit
                else:
                    total_pc = emissions_data[poll]['pc'].sum()
                    total_ldv = emissions_data[poll]['ldv'].sum()
                    total_hdv = emissions_data[poll]['hdv'].sum()
                    total_moto = emissions_data[poll]['moto'].sum()
                    grand_total = emissions_data[poll]['total'].sum()
                    display_unit = pollutants_available[poll]['unit']
                
                summary_data.append({
                    'Pollutant': poll,
                    'Total PC': format_emission_value(total_pc, display_unit),
                    'Total LDV': format_emission_value(total_ldv, display_unit),
                    'Total HDV': format_emission_value(total_hdv, display_unit),
                    'Total Moto': format_emission_value(total_moto, display_unit),
                    'Grand Total': format_emission_value(grand_total, display_unit),
                    'Unit': display_unit
                })
        # ============ END REPLACEMENT ============
            st.session_state.summary_df_for_download = pd.DataFrame(summary_data)
            st.session_state.recalculate_summary = False
        
        # Now use the stored summary_df
        summary_df = st.session_state.summary_df_for_download
        
        if st.button("Generate & Download ZIP Report", use_container_width=True):
            with st.spinner("Generating ZIP archive..."):
                with BytesIO() as buffer:
                    with zipfile.ZipFile(buffer, 'w') as zipf:
                        # 1. Full Results CSV
                        zipf.writestr('full_link_results.csv', final_results_df.to_csv(index=False))

                        # 2. Statistics Summary CSV
                        zipf.writestr('statistics_summary.csv', summary_df.to_csv(index=False))

                        # 3. Fuel Type Breakdown (if available)
                        # ============ REPLACE WITH THIS ============
                        # 3. Fuel Type Breakdown (if available)
                        if 'fuel_emissions_data' in st.session_state:
                            fuel_emissions = st.session_state.fuel_emissions_data
                            fuel_breakdown_data = []
                            for poll in selected_pollutants:
                                if export_in_converted_units and 'selected_units' in st.session_state:
                                    selected_units = st.session_state.selected_units
                                    target_unit = selected_units.get(poll, pollutants_available[poll]['unit'])
                                    original_unit = pollutants_available[poll]['unit']
                                    
                                    gas_total = convert_emission_value(fuel_emissions[poll]['gasoline'].sum(), 
                                                                       poll, original_unit, target_unit)
                                    diesel_total = convert_emission_value(fuel_emissions[poll]['diesel'].sum(), 
                                                                          poll, original_unit, target_unit)
                                    display_unit = target_unit
                                else:
                                    gas_total = fuel_emissions[poll]['gasoline'].sum()
                                    diesel_total = fuel_emissions[poll]['diesel'].sum()
                                    display_unit = pollutants_available[poll]['unit']
                                
                                fuel_breakdown_data.append({
                                    'Pollutant': poll,
                                    'Gasoline_Total': format_emission_value(gas_total, display_unit),
                                    'Diesel_Total': format_emission_value(diesel_total, display_unit),
                                    'Unit': display_unit
                                })
                        # ============ END REPLACEMENT ============
                            fuel_df = pd.DataFrame(fuel_breakdown_data)
                            zipf.writestr('fuel_type_breakdown.csv', fuel_df.to_csv(index=False))

                        # 4. Text Report
                        gasoline_avg = data_link_np[:, 4].mean()
                        diesel_avg = 1 - gasoline_avg
                        
Methodology: {calculation_method}
Ambient Temperature: {ambient_temp}°C
Trip Length (Cold Start): {trip_length} km
Export Units: {'Converted Units' if export_in_converted_units else 'Original Units (g/km, mg/km, L/100km)'}
{f"Unit Conversions Applied: {', '.join([f'{p}={st.session_state.selected_units.get(p)}' for p in selected_pollutants])}" if export_in_converted_units and 'selected_units' in st.session_state else ''}

--- Summary Statistics ---
{summary_df.to_string(index=False)}

--- Fuel Type Distribution ---
Average Gasoline Proportion: {gasoline_avg*100:.1f}%
Average Diesel Proportion: {diesel_avg*100:.1f}%

--- Note on LDV/HDV Distribution ---
This calculation used a simplified fleet distribution for Light Duty Vehicles (LDV) and Heavy Duty Vehicles (HDV) as no specific files were uploaded.
LDV Fleet Composition: Defaulted to the uploaded Passenger Car (Gasoline) Euro Class Distribution.
HDV Fleet Composition: Defaulted to 100% Euro VI standard and Rigid Truck < 7.5t type.

--- Link Data Column Count ---
Link Data File Columns: {data_link_np.shape[1]}
If 7 columns, LDV/HDV flow proportions were assumed zero.
If 9 columns, LDV/HDV flow proportions were read from columns 8 and 9.

--- Data Preview (First 10 Rows of Full Results) ---
{final_results_df.head(10).to_string()}

--- Top 5 Emitting Links ---
"""
                        # Add top 5 links for each pollutant
                        for poll in selected_pollutants:
                            temp_df = final_results_df.copy()
                            temp_df['Total_Emission'] = emissions_data[poll]['total']
                            top_5 = temp_df.nlargest(5, 'Total_Emission')[['OSM_ID', 'Total_Emission']]
                            report_text += f"\n{poll} Top 5 Links:\n{top_5.to_string(index=False)}\n"

                        zipf.writestr('detailed_report.txt', report_text)
                        
                    st.success("✅ ZIP report generated successfully!")

                    buffer.seek(0)
                    st.download_button(
                        label="📦 Download ZIP Report",
                        data=buffer,
                        file_name="traffic_emission_analysis.zip",
                        mime="application/zip",
                        key='download_zip',
                        use_container_width=True
                    )
        
        # ============ REPLACE WITH THIS ============
        st.markdown("### 📚 Export Formats")
        
        # Show current export settings
        if export_in_converted_units and 'selected_units' in st.session_state:
            st.success(f"""
            **Current Export Settings: ✅ Converted Units**
            
            Your exports will use these units:
            {chr(10).join([f"- {poll}: {st.session_state.selected_units.get(poll, pollutants_available[poll]['unit'])}" for poll in selected_pollutants])}
            """)
        else:
            st.info("""
            **Current Export Settings: ℹ️ Original Units**
            
            Exports will use standard COPERT units:
            - CO, NOx, VOC: g/km
            - CO2: g/km  
            - PM: mg/km
            - FC: L/100km
            """)
        
        st.markdown("---")
        st.info("""
        **Available Export Formats:**
        - **CSV**: Comma-separated values for spreadsheet applications
        - **ZIP**: Complete analysis package with all data and documentation
        
        **Vehicle Type Breakdown:**
        All exports include separate columns for:
        - PC: Passenger Cars
        - LDV: Light Duty Vehicles
        - HDV: Heavy Duty Vehicles
        - Moto: Motorcycles
        - Total: Sum of all vehicle types
        
        **Fuel Type Data:**
        If fuel tracking is enabled, ZIP includes gasoline/diesel breakdown
        
        **Unit Conversion:**
        Enable "Export in converted units" to use your sidebar unit selections in all exports
        """)
        # ============ END REPLACEMENT ============
    else:
        st.info("Calculate emissions first to create download package")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with COPERT IV, IPCC, and EPA MOVES methodologies</p>
    <p>Now with support for PC, LDV, HDV, and Motorcycle emissions</p>
    <p>Standards: EEA Guidebook 2019, IPCC 2019 Guidelines, WHO Air Quality Standards</p>
    <p>© 2025 - Developed by Shassan</p>
</div>
""", unsafe_allow_html=True)







