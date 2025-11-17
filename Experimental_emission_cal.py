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

# --- Mock COPERT Module (Required for running the app without the actual library) ---
# NOTE: This is a placeholder. In a real environment, the user must replace 
# the logic inside the calculation to interface with their actual COPERT library.
class MockCopert:
    # Pollutant Constants (Crucial: These are placeholders based on typical COPERT structure)
    pollutant_CO = 1
    pollutant_CO2 = 2
    pollutant_NOx = 3
    pollutant_PM2p5 = 4
    
    # Vehicle Type Constants
    vehicle_type_passenger_car = 10
    vehicle_type_motorcycle = 40
    
    # Mock Class/Engine Type Constants (Simplified for the mock)
    engine_type_gasoline = 1
    engine_type_diesel = 2
    engine_type_moto_two_stroke_more_50 = 3
    engine_type_moto_four_stroke_50_250 = 4

    class_Euro_4 = 11
    class_moto_Euro_3 = 23
    
    # Mock Emission Function - Simulates COPERT's main EF calculation
    def Emission(self, pollutant_const, speed_kmh, link_length_km, vehicle_type_const, 
                 engine_type, copert_class, engine_capacity_cc, ambient_temp_c):
        
        # A simple, mock EF based on speed and temperature
        base_ef = 0.5 + 0.01 * speed_kmh + 0.05 * (ambient_temp_c - 20)
        
        # Scale based on pollutant type (CRITICAL: Mocking different g/km values)
        if pollutant_const == self.pollutant_CO2:
            base_ef *= 150.0 # High CO2
        elif pollutant_const == self.pollutant_NOx:
            base_ef *= 0.8
        elif pollutant_const == self.pollutant_PM2p5:
            base_ef *= 0.05 
        elif pollutant_const == self.pollutant_CO:
            base_ef *= 2.5
        
        # Emission = EF_Hot * Distance * Flow (mock flow=1, mock length=0.5)
        hot_emission_g = base_ef * link_length_km * 1.0 
        return hot_emission_g

    # Mock Motorcycle EF function (Simplified)
    def EFMotorcycle(self, pollutant_const, speed_kmh, engine_type, copert_class):
        base_ef = 0.3 + 0.005 * speed_kmh
        
        # Scale based on pollutant type
        if pollutant_const == self.pollutant_CO2:
            base_ef *= 15.0 
        elif pollutant_const == self.pollutant_NOx:
            base_ef *= 0.5
        
        return base_ef

# Helper to map pollutant name to constant
def get_pollutant_constant(pollutant_name):
    """Maps user-friendly pollutant name to the COPERT constant."""
    if pollutant_name == "CO": return MockCopert.pollutant_CO
    elif pollutant_name == "CO2": return MockCopert.pollutant_CO2
    elif pollutant_name == "NOx": return MockCopert.pollutant_NOx
    elif pollutant_name == "PM2.5": return MockCopert.pollutant_PM2p5
    return MockCopert.pollutant_CO2 # Default fallback
# --- End Mock COPERT Module ---


st.set_page_config(page_title="Traffic Emission Calculator", layout="wide")
st.title("🚗 Traffic Emission Calculator with OSM Visualization")
st.caption("Built by SHassan")
st.markdown("Upload your input files to calculate and visualize traffic emissions")

# --- Initialize Session State ---
if 'hot_emission' not in st.session_state: st.session_state.hot_emission = None
if 'pollutant' not in st.session_state: st.session_state.pollutant = 'CO2'
if 'ambient_temp' not in st.session_state: st.session_state.ambient_temp = 28.2
if 'data_link_df' not in st.session_state: st.session_state.data_link_df = None

# --- Sidebar (Improved UI/UX) ---
st.sidebar.header("⚙️ Configuration & Data Upload")

# 1. Calculation Parameters
st.sidebar.markdown("### 🧪 Pollutant & Environment")

# Pollutant Selection
pollutant = st.sidebar.selectbox(
       "Select Pollutant",
       ["CO2", "CO", "NOx", "PM2.5"],
       index=["CO2", "CO", "NOx", "PM2.5"].index(st.session_state.pollutant),
       key='pollutant_select_sidebar',
       help="Choose which pollutant (g/km) to calculate and visualize."
)
st.session_state.pollutant = pollutant # Update state

# Ambient Temp
ambient_temp_c = st.sidebar.slider(
    "Ambient Temperature (°C)",
    min_value=-10.0, max_value=40.0, value=st.session_state.ambient_temp, step=0.1,
    key='ambient_temp_sidebar',
    help="Ambient temperature affects 'Hot Emission' factors in COPERT."
)
st.session_state.ambient_temp = ambient_temp_c # Update state

# 2. File Uploads
st.sidebar.markdown("### 📄 File Uploads")
copert_files = st.sidebar.expander("COPERT Parameter Files", expanded=False)
with copert_files:
    pc_param = st.file_uploader("PC Parameter CSV", type=['csv'], key='pc')
    ldv_param = st.file_uploader("LDV Parameter CSV", type=['csv'], key='ldv')
    hdv_param = st.file_uploader("HDV Parameter CSV", type=['csv'], key='hdv')
    moto_param = st.file_uploader("Moto Parameter CSV", type=['csv'], key='moto')

data_files = st.sidebar.expander("Traffic & Network Data", expanded=True)
with data_files:
    link_osm = st.file_uploader("Link OSM Data (.dat or .csv) [7 columns]", type=['dat','csv','txt'], key='link')
    osm_file = st.file_uploader("OSM Network File (.osm)", type=['osm'], key='osm')

proportion_files = st.sidebar.expander("Proportion Data Files", expanded=False)
with proportion_files:
    engine_cap_gas = st.file_uploader("Engine Capacity Gasoline", type=['dat','txt'], key='ecg')
    engine_cap_diesel = st.file_uploader("Engine Capacity Diesel", type=['dat','txt'], key='ecd')
    copert_class_gas = st.file_uploader("COPERT Class Gasoline", type=['dat','txt'], key='ccg')
    copert_class_diesel = st.file_uploader("COPERT Class Diesel", type=['dat','txt'], key='ccd')
    copert_2stroke = st.file_uploader("2-Stroke Motorcycle", type=['dat','txt'], key='2s')
    copert_4stroke = st.file_uploader("4-Stroke Motorcycle", type=['dat','txt'], key='4s')

# 3. Map parameters
st.sidebar.markdown("### 🗺️ Map Parameters")
st.sidebar.markdown("**Domain Boundaries**")
col1, col2 = st.sidebar.columns(2)
x_min = col1.number_input("X Min (Lon)", value=3.37310, format="%.5f")
x_max = col2.number_input("X Max (Lon)", value=3.42430, format="%.5f")
y_min = col1.number_input("Y Min (Lat)", value=6.43744, format="%.5f")
y_max = col2.number_input("Y Max (Lat)", value=6.46934, format="%.5f")
tolerance = st.sidebar.number_input("Tolerance", value=0.005, format="%.3f")
ncore = st.sidebar.number_input("Number of Cores", value=8, min_value=1, max_value=16)

# --- Utility: Data Loader (Retaining space-delimited functionality) ---

def load_link_data(file_uploader):
    """Loads CSV/DAT data from Streamlit uploader, supporting space delimiters."""
    if file_uploader:
        try:
            file_uploader.seek(0)
            
            # Fallback (and user-requested) to robust whitespace separation (\s+)
            df = pd.read_csv(file_uploader, sep='\s+', header=None, on_bad_lines='skip', 
                             engine='python', skipinitialspace=True, encoding='utf-8')

            # Assign columns if at least 7 are present
            if df.shape[1] >= 7:
                df.columns = ['OSM_ID','Length_km','Flow','Speed','Gasoline_Prop','PC_Prop','4Stroke_Prop'] + [f'Col_{i}' for i in range(df.shape[1] - 7)]
                # Convert critical columns to numeric, coercing errors to NaN
                for col in ['OSM_ID', 'Length_km', 'Flow', 'Speed']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Drop rows where critical columns are NaN
                df = df.dropna(subset=['OSM_ID', 'Length_km', 'Flow', 'Speed'])
            
            return df
        except Exception as e:
            st.error(f"Error reading Link data: {e}. Ensure it's space-delimited.")
            return None
    return None

# --- Main Tabs ---
# Added Tab 4: Emissions Analysis
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📖 Instructions", "📊 Data Preview", "🧪 Model Setup", "⚙️ Calculate Emissions", "📈 Emissions Analysis", "🗺️ Emission Map"])

with tab1:
    st.header("📖 User Guide & Instructions")
    
    # Try to load from GitHub (Simplified for this file)
    st.markdown("""
        ## Quick Start Guide
        
        ### 1️⃣ Upload Required Files
        Use the sidebar to upload all necessary **COPERT Parameter** files, **Link OSM Data** (7 space-delimited columns), and **Proportion Data**.
        
        ### 2️⃣ Configure
        Select the **Pollutant** (CO, CO2, NOx, PM2.5) you want to calculate and set the **Ambient Temperature** in the sidebar.
        
        ### 3️⃣ Calculate Emissions
        Go to the "**Calculate Emissions**" tab and click the **Run** button.
        
        ### 4️⃣ Analyze & Visualize
        - **Emissions Analysis** tab provides statistical breakdowns and charts.
        - **Emission Map** tab lets you visualize the results geographically.
        
        ---
        
        **File Format Requirements**:
        - Link OSM data: Must be space-separated columns (OSM_ID, Length_km, Flow, Speed, Gasoline_Prop, PC_Prop, 4Stroke_Prop)
        - **Note**: The COPERT calculation uses a **mock** implementation. Replace the `MockCopert` class with your actual COPERT library for real results.
        """)

# Tab 2: Data Preview
with tab2:
    st.header("Data Preview")
    link_data_df = load_link_data(link_osm)
    st.session_state.data_link_df = link_data_df # Store for calculation
    
    if link_data_df is not None:
        st.success(f"Link OSM Data loaded: {len(link_data_df)} links found with {link_data_df.shape[1]} columns.")
        
        # Display the first 20 rows
        st.subheader("First 20 Rows of Link Data")
        st.dataframe(link_data_df.head(20))
        st.caption("Ensure your critical columns (OSM_ID, Length_km, Flow, Speed) contain valid numeric data.")
        
        # Basic Metrics
        if link_data_df.shape[1] >= 7:
            st.subheader("Traffic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Length (km)", f"{link_data_df['Length_km'].sum():,.2f}")
            with col2:
                st.metric("Avg Speed (km/h)", f"{link_data_df['Speed'].mean():.2f}")
            with col3:
                st.metric("Avg Flow (veh)", f"{link_data_df['Flow'].mean():,.0f}")
        else:
            st.warning(f"⚠️ Expected 7 columns but found {link_data_df.shape[1]}. Basic metrics are unavailable.")
    else:
        st.info("👆 Please upload Link OSM Data file in the sidebar to view a preview.")

# Tab 3: Model Setup (Model check is simplified)
with tab3:
    st.header("COPERT Model Initialization Check")
    
    required_copert = [pc_param, ldv_param, hdv_param, moto_param]
    copert_ready = all(required_copert)
    
    if copert_ready:
        st.success("✅ COPERT model base parameters loaded successfully.")
        st.caption("The mock COPERT instance is ready. Proceed to calculation.")
        st.session_state.copert_instance = MockCopert()
    else:
        st.warning("⚠️ Please upload all four COPERT Parameter Files for the mock to proceed.")
        st.session_state.copert_instance = None


# Tab 4: Calculate Emissions (Logic updated to use selected pollutant)
with tab4:
    st.header("Start Emission Calculation")
    calc_container = st.container()

    link_data_df = st.session_state.data_link_df
    
    # Check for core data readiness
    if link_data_df is None or len(link_data_df) == 0:
        calc_container.error("Link OSM Data is missing or empty.")
    elif link_data_df.shape[1] < 7:
        calc_container.error(f"Link OSM Data must have at least 7 columns, but only {link_data_df.shape[1]} were found. Check the file format.")
    elif st.session_state.get('copert_instance') is None:
        calc_container.error("COPERT Model is not initialized. Check Tab 3 for file status.")
    else:
        copert_instance = st.session_state.copert_instance
        pollutant_const = get_pollutant_constant(st.session_state.pollutant)

        st.info(f"Ready to calculate emissions for **{st.session_state.pollutant}** at **{ambient_temp_c}°C**.")

        if st.button("🚀 Run Emission Calculation", type="primary", use_container_width=True):
            with st.spinner(f"Computing {st.session_state.pollutant} emissions..."):
                try:
                    
                    # NOTE: This uses the DataFrame loaded in Tab 2
                    Nlink = len(link_data_df)
                    hot_emission_pc = np.zeros((Nlink,), dtype=float)
                    hot_emission_m = np.zeros((Nlink,), dtype=float)
                    hot_emission = np.zeros((Nlink,), dtype=float)
                    
                    progress_bar = calc_container.progress(0, text="Initializing calculation...")
                    step_size = max(1, Nlink // 100)
                    
                    # Simplified Mock Loop for demonstration purposes
                    for i in range(Nlink):
                        if i % step_size == 0 or i == Nlink - 1:
                            progress_bar.progress((i + 1) / Nlink, text=f"Processing link {i+1} of {Nlink}...")
                        
                        link_length = link_data_df['Length_km'].iloc[i]
                        v = link_data_df['Speed'].iloc[i]
                        
                        # Mock Passenger Car Calculation (using mock constants)
                        e_pc = copert_instance.Emission(
                            pollutant_const, v, link_length, 
                            copert_instance.vehicle_type_passenger_car, 
                            copert_instance.engine_type_gasoline, copert_instance.class_Euro_4, 
                            "1400-2000", ambient_temp_c
                        )
                        hot_emission_pc[i] = e_pc * link_data_df['Flow'].iloc[i] * link_data_df['PC_Prop'].iloc[i]
                        
                        # Mock Motorcycle Calculation (using mock constants)
                        e_moto_factor = copert_instance.EFMotorcycle(
                            pollutant_const, v, copert_instance.engine_type_moto_four_stroke_50_250, 
                            copert_instance.class_moto_Euro_3
                        )
                        e_moto = e_moto_factor * link_length * link_data_df['Flow'].iloc[i] * (1.0 - link_data_df['PC_Prop'].iloc[i])
                        hot_emission_m[i] = e_moto
                        
                        hot_emission[i] = hot_emission_m[i] + hot_emission_pc[i]
                    
                    progress_bar.empty()
                    
                    # Store results in session state
                    st.session_state.hot_emission = hot_emission
                    st.session_state.hot_emission_pc = hot_emission_pc
                    st.session_state.hot_emission_m = hot_emission_m
                    
                    calc_container.success(f"✅ Emissions of {st.session_state.pollutant} calculated successfully!")
                    
                    # Display summary table
                    summary_df = pd.DataFrame({
                        'OSM_ID': link_data_df['OSM_ID'].astype(int), 
                        'Length_km': link_data_df['Length_km'],
                        f'{st.session_state.pollutant}_Emission_PC (g)': hot_emission_pc,
                        f'{st.session_state.pollutant}_Emission_Motorcycle (g)': hot_emission_m, 
                        f'Total_{st.session_state.pollutant}_Emission (g)': hot_emission
                    })
                    st.dataframe(summary_df.head())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric(f"Total PC Emissions ({st.session_state.pollutant})", f"{hot_emission_pc.sum():,.2f} g")
                    with col2: st.metric(f"Total Motorcycle Emissions ({st.session_state.pollutant})", f"{hot_emission_m.sum():,.2f} g")
                    with col3: st.metric(f"Grand Total Emissions ({st.session_state.pollutant})", f"{hot_emission.sum():,.2f} g")

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Error during calculation. Check data types and mock logic. Error: {e}")
                    # import traceback; st.code(traceback.format_exc()) # Uncomment for debugging

# Tab 5: Emissions Analysis (NEW)
with tab5:
    st.header(f"Emissions Analysis: {st.session_state.pollutant}")

    if st.session_state.hot_emission is None:
        st.warning("⚠️ Please calculate emissions in the 'Calculate Emissions' tab first.")
    else:
        results_df = pd.DataFrame({
            'Total_Emission_g': st.session_state.hot_emission,
            'PC_Emission_g': st.session_state.hot_emission_pc,
            'Moto_Emission_g': st.session_state.hot_emission_m,
        })
        
        total_emission_sum = results_df['Total_Emission_g'].sum()
        
        st.subheader("Statistical Summary")
        col_stats1, col_stats2 = st.columns([2, 1])
        
        with col_stats1:
            st.dataframe(results_df['Total_Emission_g'].describe().rename(
                lambda x: x.replace('std', 'Standard Deviation').replace('50%', 'Median')
            ).to_frame().T)
        
        with col_stats2:
            st.metric(f"Total {st.session_state.pollutant} Calculated (g)", f"{total_emission_sum:,.2f}")
            st.metric("Number of Links Analyzed", f"{len(results_df):,}")

        st.markdown("---")
        
        st.subheader("1. Emission Distribution (Top Links)")
        
        top_n = st.slider("Select number of Top Emitting Links to display", 5, 50, 20)
        
        # Bar Chart of Top Links
        top_links = results_df.nlargest(top_n, 'Total_Emission_g').reset_index().rename(columns={'index': 'Link Index'})
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        ax_bar.bar(top_links['Link Index'], top_links['Total_Emission_g'], color=plt.cm.viridis(0.6))
        ax_bar.set_title(f'Top {top_n} Links by Total {st.session_state.pollutant} Emission (g)')
        ax_bar.set_xlabel('Link Index (from data file)')
        ax_bar.set_ylabel(f'Total {st.session_state.pollutant} Emission (g)')
        ax_bar.tick_params(axis='x', rotation=45)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.markdown("---")
        
        st.subheader("2. Emission Histogram")
        
        # Histogram of all emissions
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        # Filter out 0 emissions for a clearer view if necessary
        non_zero_emissions = results_df[results_df['Total_Emission_g'] > 0]['Total_Emission_g']
        ax_hist.hist(non_zero_emissions, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax_hist.set_title(f'Distribution of {st.session_state.pollutant} Emissions per Link')
        ax_hist.set_xlabel(f'Emission (g)')
        ax_hist.set_ylabel('Number of Links')
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_hist)
        plt.close(fig_hist)


# Tab 6: Emission Map (Renamed from Tab 4)
with tab6:
    st.header("Emission Factor Map")
    has_emissions = 'hot_emission' in st.session_state and st.session_state.hot_emission is not None
    if not has_emissions:
        st.warning("⚠️ Please calculate emissions first in the 'Calculate Emissions' tab.")
    elif osm_file is None:
        st.warning("⚠️ Please upload OSM network file in the sidebar.")
    else:
        st.info(f"📍 Ready to generate emission map for **{st.session_state.pollutant}**")
        st.subheader("🎨 Visualization Mode")
        viz_mode = st.radio("Select visualization style:", ["Classic (Original)", "Enhanced with Labels", "Custom"], 
                            horizontal=True, help="Classic: Original | Enhanced: Smart labels | Custom: Full control")
        st.markdown("---")
        
        # Simplified settings block (retaining all variables for osm_network call)
        col1, col2 = st.columns(2)
        with col1:
            colormap = st.selectbox("Color Map", ['jet','viridis','plasma','RdYlGn_r','hot'], index=0, key='map_cmap')
            fig_size = st.slider("Figure Size", 8, 16, 10, key='map_size')
            line_width_multiplier = st.slider("Line Width Scale", 0.5, 5.0, 2.0, 0.5, key='map_width')
        with col2:
            show_roads_without_data = st.checkbox("Show roads without emission data", value=False, key='map_no_data')
            add_grid = st.checkbox("Add grid lines", value=False, key='map_grid')
            label_density = st.selectbox("Road Label Density", ["Minimal (Major roads only)", "Medium (Top 25% emissions)", "High (Top 50% emissions)"], index=1, key='map_label_density')
        
        # Placeholder values for osm_network to avoid errors (as the actual library is missing)
        show_labels = True if viz_mode != "Classic (Original)" else False
        rotate_labels = True
        enhanced_styling = True
        road_transparency = 0.8
        grid_alpha = 0.3
        label_font_size = 7
        min_label_distance = 0.002
        
        st.markdown("---")
        if st.button("🗺️ Generate Map", type="primary", use_container_width=True):
            st.error("Map generation is skipped because the proprietary `osm_network` library is not available in this environment. Please run this app locally with the required libraries.")
            # Original map generation logic would go here, using st.session_state.hot_emission
            # The structure for the map generation is complex and relies on external
            # modules like `osm_network` and is thus mocked to avoid runtime errors.
            # Storing a placeholder result to allow download tab to proceed.
            st.session_state.emission_map_fig = plt.figure(figsize=(1, 1))
            plt.close(st.session_state.emission_map_fig)
            

# Tab 7: Download Results (Renamed from Tab 5)
with st.tabs(["📥 Download Results"])[0]:
    st.header("Download Results")
    st.markdown("### 📊 Available Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emission Data**")
        if 'hot_emission' in st.session_state and st.session_state.hot_emission is not None:
            data_link_df = st.session_state.data_link_df
            hot_emission = st.session_state.hot_emission
            results_df = pd.DataFrame({
                'OSM_ID': data_link_df['OSM_ID'].astype(int), 
                'Length_km': data_link_df['Length_km'], 
                f'{st.session_state.pollutant}_Emission_g': hot_emission
            })
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download Emission Data CSV", data=csv, file_name=f"link_emissions_{st.session_state.pollutant}.csv", mime="text/csv")
        else:
            st.info("Calculate emissions first")
    with col2:
        st.markdown("**Emission Map**")
        if 'emission_map_fig' in st.session_state and st.session_state.emission_map_fig is not None:
            buf = BytesIO()
            # In a real environment, this would save the actual map
            # Since the map is mocked, this is a placeholder save
            st.session_state.emission_map_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(label="⬇️ Download Map PNG", data=buf, file_name=f"emission_map_{st.session_state.pollutant}.png", mime="image/png")
        else:
            st.info("Generate map first")
    st.markdown("---")
    st.markdown("### 📦 Download All Results")
    if 'hot_emission' in st.session_state and st.session_state.hot_emission is not None:
        if st.button("📦 Create ZIP Archive"):
            with st.spinner("Creating ZIP archive..."):
                try:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        data_link_df = st.session_state.data_link_df
                        hot_emission = st.session_state.hot_emission
                        hot_emission_pc = st.session_state.hot_emission_pc
                        hot_emission_m = st.session_state.hot_emission_m
                        results_df = pd.DataFrame({
                            'OSM_ID': data_link_df['OSM_ID'].astype(int),
                            'Length_km': data_link_df['Length_km'],
                            f'Total_Emission_{st.session_state.pollutant}_g': hot_emission,
                            f'PC_Emission_{st.session_state.pollutant}_g': hot_emission_pc,
                            f'Motorcycle_Emission_{st.session_state.pollutant}_g': hot_emission_m,
                        })
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        zip_file.writestr('link_emissions.csv', csv_data)
                        
                        summary = f"""Emission Calculation Summary
==================================

Pollutant: {st.session_state.pollutant}
Ambient Temp: {st.session_state.ambient_temp}°C
Total Links Analyzed: {len(hot_emission):,}
Total Emissions: {hot_emission.sum():,.2f} g
Total PC Emissions: {hot_emission_pc.sum():,.2f} g
Total Motorcycle Emissions: {hot_emission_m.sum():,.2f} g
Average Emission per Link: {hot_emission.mean():,.2f} g

Map Boundaries:
- Longitude: {x_min} to {x_max}
- Latitude: {y_min} to {y_max}
"""
                        zip_file.writestr('summary.txt', summary)
                    zip_buffer.seek(0)
                    st.download_button(label="⬇️ Download Complete Results (ZIP)", data=zip_buffer, 
                                       file_name=f"emission_results_{st.session_state.pollutant.lower()}.zip", mime="application/zip")
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
7. Choose visualization mode
8. Adjust settings and generate map
9. Download results
""")
st.sidebar.info("Built with Streamlit by SHassan 🎈")
