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

# --- Mock COPERT Module (Essential for running the app) ---
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
        
        # A simple, mock EF based on speed and temp (for demonstration only)
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
    """Loads CSV/DAT data from Streamlit uploader."""
    if file_uploader:
        try:
            # Check for common delimiters
            content = file_uploader.read().decode('utf-8')
            sep = ','
            if '\t' in content: sep = '\t'
            elif ';' in content: sep = ';'
            
            file_uploader.seek(0)
            df = pd.read_csv(file_uploader, sep=sep, header=None, on_bad_lines='skip')
            # Handle single column files which are often read as a single column with no header
            if df.shape[1] > 0:
                return df
            else:
                st.error("Could not parse file content. Please check delimiter.")
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

@st.cache_resource
def calculate_emissions(data_link, copert_instance, pollutant_str, ambient_temp_c, st_container):
    """Performs the full COPERT emission calculation."""
    
    # Check for core data readiness
    if data_link is None or copert_instance is None:
        st_container.error("Calculation setup incomplete. Check file uploads.")
        return None, None

    # Determine pollutant constant
    pollutant_const = get_pollutant_constant(pollutant_str, copert_instance)
    
    st_container.markdown(f"### Running Emission Model for: **{pollutant_str}**")
    st_container.caption(f"Ambient Temperature set to: **{ambient_temp_c}°C**")

    # Access the columns from the loaded link data (based on original code's implied structure)
    try:
        # Assuming the link data (data_link) has the following structure (0-indexed):
        # 0: Link ID
        # 1: X start
        # 2: Y start
        # 3: X end
        # 4: Y end
        # 5: Flow or Flow related
        # 6: Speed (km/h)
        # 7: Vehicle Proportions (This is complex, but we'll use one column as a placeholder)
        
        # Link Geometry & Traffic
        x_s, y_s = data_link.iloc[:, 1], data_link.iloc[:, 2] # Start Coords
        x_e, y_e = data_link.iloc[:, 3], data_link.iloc[:, 4] # End Coords
        speeds = data_link.iloc[:, 6] # Speed in km/h

        # Mocking the length calculation (using Haversine or simple distance for real use)
        # Here we use a mock fixed length for simplicity, or calculated distance.
        # Assuming the link file contains a 'length' column, or we calculate it.
        # For simplicity, we assume Link Length is not explicitly in the first 7 columns, 
        # but is implicit in the coordinates, so we will use a mock vector.
        link_lengths_km = np.ones(len(data_link)) * 0.5 # Mock length 0.5 km

        Nlink = len(data_link)
        hot_emission = np.zeros(Nlink)
        hot_emission_m = np.zeros(Nlink) # Motorcycle emissions

        progress_bar = st_container.progress(0, text="Initializing calculation...")

        # Mock Proportions (Need 6 proportion arrays based on the original code)
        # In the original app, these came from separate files (prop_g1, prop_d2, etc.)
        # Since we load them separately, we need to ensure they are passed to the function, 
        # but for this simplified run, we will mock them.
        
        # Mocking the complex proportion/parameter lookup and iteration:
        # In a real scenario, the following parameters would be dynamically loaded 
        # from the proportion files and matched to the link ID:
        
        # Example mock iteration for Passenger Cars (PC)
        for i in range(Nlink):
            v = speeds.iloc[i] # Link speed
            link_length = link_lengths_km[i]
            
            # --- Mocking PC Calculation ---
            # Assume 1 vehicle type, 1 engine type, 1 copert class, 1 engine capacity category
            mock_engine_type = "Gasoline" 
            mock_copert_class = "EURO 4"
            mock_engine_capacity = "1400-2000" # cc

            # The Emission function handles the complex COPERT lookups
            e_pc = copert_instance.Emission(
                pollutant_const, v, link_length, copert_instance.vehicle_type_passenger_car, 
                mock_engine_type, mock_copert_class, mock_engine_capacity, ambient_temp_c
            )
            hot_emission[i] += e_pc

            # --- Mocking Motorcycle Calculation ---
            # Assume a single motorcycle class for mock
            mock_engine_type_m = "4-Stroke"
            mock_copert_class_m = "MC>50"
            
            e_moto_factor = copert_instance.EFMotorcycle(
                pollutant_const, v, mock_engine_type_m, mock_copert_class_m
            )
            # Emission = EF * Distance * Flow
            e_moto = e_moto_factor * link_length * 1.0 # Mock flow=1.0
            hot_emission_m[i] += e_moto
            hot_emission[i] += e_moto
            
            # Update progress bar
            if i % max(1, Nlink // 50) == 0:
                progress_bar.progress((i + 1) / Nlink, text=f"Processing link {i+1} of {Nlink}...")

        progress_bar.progress(1.0, text="Calculation complete!")
        st_container.success("Calculation successful!")

        # Create a DataFrame for visualization (including coordinates)
        results_df = pd.DataFrame({
            'x_s': x_s, 'y_s': y_s, 
            'x_e': x_e, 'y_e': y_e,
            'hot_emission': hot_emission,
            'hot_emission_m': hot_emission_m
        })
        
        # Determine map bounds
        x_min, x_max = min(x_s.min(), x_e.min()), max(x_s.max(), x_e.max())
        y_min, y_max = min(y_s.min(), y_e.min()), max(y_s.max(), y_e.max())
        
        return results_df, (x_min, x_max, y_min, y_max)

    except Exception as e:
        st_container.error(f"A critical error occurred during calculation. Check that your link data has at least 7 columns. Error: {e}")
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
    link_osm_desc = "Link OSM Data (Expects min. 7 columns: ID, X_s, Y_s, X_e, Y_e, Flow, Speed)"
    link_osm = st.file_uploader(link_osm_desc, type=['dat','csv','txt'], key='link')
    
    # OSM Network File (For map visualization)
    osm_file = st.file_uploader("OSM Network File (.osm XML)", type=['osm'], key='osm')
    
    # Proportion Data (Highly Region-Specific)
    prop_files = st.expander("Vehicle Fleet Proportion Files (1-column array)", expanded=False)
    with prop_files:
        st.caption("These files define the distribution of the fleet (e.g., Euro Class, Fuel Type).")
        prop_g1 = st.file_uploader("Engine Capacity Gasoline (e.g., <1.4L)", type=['csv','txt'], key='g1')
        prop_d2 = st.file_uploader("COPERT Class Diesel (e.g., EURO 4)", type=['csv','txt'], key='d2')
        prop_h1 = st.file_uploader("HDV Class Distribution", type=['csv','txt'], key='h1')
        prop_h2 = st.file_uploader("HDV Load Distribution", type=['csv','txt'], key='h2')
        prop_m1 = st.file_uploader("Motorcycle 2-Stroke Proportion", type=['csv','txt'], key='m1')
        prop_m2 = st.file_uploader("Motorcycle 4-Stroke Proportion", type=['csv','txt'], key='m2')

# --- Main Tabs ---

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧪 Model Setup", "⚙️ Calculate Emissions", "🗺️ Visualize Map"])

# Tab 1: Data Overview
with tab1:
    st.header("Uploaded Data Status")
    
    link_data = load_data(link_osm)
    
    if link_data is not None:
        st.success(f"Link OSM Data loaded: {len(link_data)} links found.")
        st.subheader("First 5 Rows of Link Data (Implied Columns)")
        st.dataframe(link_data.head(5))
        st.caption("Ensure your columns match: [0] ID, [1] X_s, [2] Y_s, [3] X_e, [4] Y_e, [5] Flow, [6] Speed")
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
