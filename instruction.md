# 🏭 Traffic Emission Calculation and Visualization App Guide

Welcome to the **Traffic Emission Calculator with OSM Visualization**! This application uses the **COPERT** (Computer Programme to Calculate Emissions from Road Transport) methodology to calculate and visualize vehicle emissions based on OpenStreetMap (OSM) road networks and user-provided traffic data.

## Key Outputs:

* Emission factors per road segment (g/km) for various pollutants (e.g., CO, NOx, PM).
* **Interactive maps** with color-coded and thickness-scaled emission intensity.
* Optional road name labels for easy identification.
* Downloadable results (CSV, PNG, ZIP).

---

## 2. Prerequisites

### 💻 Software Requirements

| Software | Purpose | Download Link | Notes |
|:---------|:--------|:--------------|:------|
| **QGIS** | Geospatial data preparation and network modeling. | [https://qgis.org/](https://qgis.org/) | Version 3.x or higher is recommended. |
| **Python** | Running the Streamlit application. | [https://python.org/](https://python.org/) | Python 3.8 or higher required. |
| **Python Packages** | Application dependencies. | `pip install streamlit numpy pandas matplotlib osmium` | The custom `copert` module must be in your Python path. |

### 📂 Data Requirements

You must provide your own **road link data** and utilize the provided **COPERT/Proportion files** for the calculation.

| File Name | Purpose | Format | Notes |
|:----------|:--------|:-------|:------|
| **link_osm.dat** | **Primary Input:** Road segment geometry and traffic data. | Space-separated DAT | Must contain **7 mandatory columns** (see Stage 2). |
| **network.osm** | OpenStreetMap network file for visualization. | OSM XML | Can be downloaded from QuickOSM or Overpass-Turbo. |
| **4 COPERT Parameter Files** | Emission factors for PC, LDV, HDV, Moto. | CSV | Provided auxiliary files. |
| **6 Vehicle Proportion Files** | Vehicle class/engine type distributions. | Single-column DAT | Provided auxiliary files. |

> **📥 Download Auxiliary Files:**
>
> All necessary COPERT parameter and Vehicle Proportion files must be downloaded before running the app. You can find them here:
>
> **Download All Required Auxiliary Input Files **[Download Here](https://drive.google.com/drive/folders/1KCu8y-mZ0XtBc6icFlvPnJMxLFM7YCKY?usp=sharing)**

---

## 3. Workflow Overview

The overall process is divided into three distinct and sequential stages:

**Stage 1: QGIS Geospatial Data Preparation** → **Stage 2: Data Conversion and Formatting** → **Stage 3: Application Usage & Visualization**

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: QGIS Geospatial Data Preparation                 │
│  • Download OSM data                                        │
│  • Process road networks                                    │
│  • Extract attributes (length, type, etc.)                  │
│  • Add traffic data                                         │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Data Conversion and Formatting                   │
│  • Export from QGIS to CSV                                 │
│  • Convert CSV to space-separated DAT                       │
│  • Validate data structure                                  │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Application Usage & Visualization                │
│  • Upload all required files                                │
│  • Calculate emissions                                      │
│  • Generate visualizations                                  │
│  • Download results                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1: QGIS Geospatial Data Preparation

This stage focuses on acquiring and preparing the street network data to create the `link_osm.dat` input file.

### 4.1. 🗺️ Street Network Data Acquisition

1. **Open QGIS** and start a **New Project**. Zoom to your study area.
2. **Install Plugin:** Go to **Plugins → Manage and Install Plugins...** and install the **"QuickOSM"** or **"OSM Downloader"** plugin.
3. **Download Data:** Use the plugin to download road network data for your specific geographic area.
   * *Recommended:* Use QuickOSM with **Key: highway** and **Value: (leave empty)** to capture all road types.
4. **Export Data:** Export the downloaded data as an OSM XML file (`network.osm`) and ensure the vector layers (lines/multilines) are loaded in QGIS.

### 4.2. 🧩 Data Processing and Layer Merging

1. **Merge Layers:** OSM data often separates road features into distinct layers. Consolidate them:
   * Open the **Processing Toolbox** (**View → Panels → Processing Toolbox**).
   * Search for and run the **"Merge Vector Layers"** tool.
   * Select **ONLY** the `lines` and `multilines` layers as input. The output is your single, complete street network layer.

### 4.3. 📊 Attribute Table Calculation

You must add and calculate the seven variables required by the emission model in the merged layer's attribute table.

1. **Access Table:** Right-click the merged layer and select **Open Attribute Table**.
2. **Calculate Attributes:** Use the **Field Calculator** to create the following seven fields (columns):

| Field Name | Description | Calculation/Data Source |
|:-----------|:------------|:------------------------|
| **OSM_ID** | Unique road segment ID. | Extract from raw OSM data field (e.g., `osm_id` or `id`). |
| **Length_km** | Physical length of the segment. | **Expression:** `$length / 1000` |
| **Flow** | Estimated traffic volume. | External data or **default values** (vehicles/hour). |
| **Speed** | Average vehicle speed. | External data or **default values** (km/h). |
| **Gasoline_Prop** | Proportion of gasoline-powered vehicles. | External data or **model assumptions** (0.0 to 1.0). |
| **PC_Prop** | Proportion of passenger cars (vs. motorcycles). | External data or **model assumptions** (0.0 to 1.0). |
| **4Stroke_Prop** | Proportion of 4-stroke motorcycles. | External data or **model assumptions** (0.0 to 1.0). |

> **💡 Traffic Data Tip:** If external traffic data is unavailable, use **default assumptions** based on road type:
> - **Motorway:** High Flow (2000+ veh/h), High Speed (100-120 km/h), Gasoline_Prop: 0.75, PC_Prop: 0.95, 4Stroke_Prop: 0.80
> - **Primary/Secondary:** Medium Flow (1000-2000 veh/h), Medium Speed (60-80 km/h), Gasoline_Prop: 0.70, PC_Prop: 0.85, 4Stroke_Prop: 0.75
> - **Residential:** Low Flow (200-500 veh/h), Low Speed (30-50 km/h), Gasoline_Prop: 0.65, PC_Prop: 0.80, 4Stroke_Prop: 0.70

---

## 5. Stage 2: Data Conversion and Formatting

The final input file for the app (`link_osm.dat`) must be a space-separated file with a specific column order.

### 5.1. Final Attribute Selection and CSV Export

1. **Clean Table:** Delete all irrelevant columns from the attribute table, ensuring only the 7 mandatory columns remain.
2. **Order Columns:** The columns must be in this **exact order**:
   ```
   OSM_ID, Length_km, Flow, Speed, Gasoline_Prop, PC_Prop, 4Stroke_Prop
   ```
3. **Export to CSV:**
   * Right-click the final merged layer → **Export → Save Feature As...**
   * Format: Choose **Comma Separated Values (CSV)**.

### 5.2. CSV to DAT Conversion (Crucial Step)

The Streamlit app requires a **space-separated .dat file**.

1. **Open in Editor:** Open the exported CSV file using a plain text editor (e.g., Notepad, VS Code, Sublime Text).
2. **Find and Replace (Separators):**
   * **Find:** The CSV field separator (typically `,` or `;`)
   * **Replace with:** A single space (` `)
3. **Find and Replace (Quotes):**
   * **Find:** Double Quotes (`"`)
   * **Replace with:** Nothing (delete them)
4. **Save as DAT:** Save the modified file with the **exact name** `link_osm.dat`.

**Example of correct format:**
```
123456 0.25 1200 45 0.75 0.85 0.70
234567 0.50 800 35 0.70 0.80 0.65
345678 0.15 500 30 0.65 0.75 0.60
```

---

## 6. Stage 3: Application Usage & Visualization

### 6.1. File Acquisition and Upload

1. **Upload:** Navigate to the sidebar of the web application. Use the dedicated sections to upload **all 11 required files**:
   - 4 COPERT Parameter Files (PC, LDV, HDV, Moto)
   - 1 Link OSM Data file (`link_osm.dat`)
   - 1 OSM Network file (`network.osm`)
   - 6 Vehicle Proportion Files (Engine Capacity Gasoline/Diesel, COPERT Class Gasoline/Diesel, 2-Stroke/4-Stroke Motorcycle)

### 6.2. Data Preview

1. Navigate to the **"Data Preview"** tab to verify your uploaded data:
   - Check that all 7 columns are correctly identified
   - Review the first 20 rows for any anomalies
   - Verify statistics (total length, average speed, average flow)

### 6.3. Calculate Emissions

1. **Validate Data:** The app will check for the presence of all 11 files and validate the structure of `link_osm.dat`. Look for a **green checkmark** indicating readiness.
2. **Run Calculation:** Navigate to the **"Calculate Emissions"** tab and click the **"Calculate Emissions"** button.
3. **Wait for Processing:** The app will display a progress bar. Processing time depends on network size (typically 1-5 minutes per 1000 road segments).
4. **Review Results:** Once complete, you'll see:
   - A table with emission factors per road segment
   - Summary statistics (Total PC Emissions, Total Motorcycle Emissions, Total Emissions)

### 6.4. Visualization Modes

The app offers three modes to present the emission data on the map:

| Mode | Best For | Key Features |
|:-----|:---------|:-------------|
| **Classic (Original)** | Academic papers, simple overview. | Minimal line widths, no road labels, traditional look. Exact replica of original visualization. |
| **Enhanced with Labels** | Presentations, reports, analysis. | Thicker lines (proportional to emissions), smart road name labels (no overlap), easy-to-read, modern styling. |
| **Custom** | Specific, fine-tuned outputs. | Complete control over line width scale (0.1x-10x), label density (4 levels), road transparency, grid transparency, label font size (4-12), minimum label distance. |

#### Classic Mode Settings:
- Color map selection
- Figure size
- Show/hide roads without emission data
- Optional grid lines

#### Enhanced Mode Settings:
- All Classic settings +
- Line width scale (0.5x - 5x)
- Label density (Minimal/Medium/High)
- Rotate labels along roads toggle
- Enhanced styling (background color, transparency)

#### Custom Mode Settings:
- All Enhanced settings +
- Road transparency slider (0.0 - 1.0)
- Grid transparency slider (0.0 - 1.0)
- Label font size slider (4 - 12)
- Minimum label distance slider (0.001 - 0.01)
- Maximum label density option

### 6.5. Generate and View Map

1. **Select Visualization Mode:** Choose Classic, Enhanced, or Custom based on your needs.
2. **Adjust Settings:** Configure the visualization parameters according to your preferences.
3. **Click "Generate Map":** Wait for the map to be generated (10-60 seconds depending on network size and label settings).
4. **Analyze Results:** View the color-coded emission map with optional road labels.

### 6.6. Download Results

The final **"Download Results"** tab allows you to download the results for further analysis or publication:

* **Individual Downloads:**
  * **Emission Data CSV:** Full dataset with OSM_ID, Length_km, and Emission_g_km
  * **Emission Map PNG:** High-resolution map image (150 DPI)

* **ZIP Archive:** Includes:
  * `link_hot_emission_factor.csv` - Full emission dataset (includes PC emissions, Motorcycle emissions, Total emissions)
  * `emission_factor_map.png` - Map visualization
  * `summary.txt` - Statistical summary with key metrics

---

## 7. Understanding Your Results

### Emission Values
- **g/km (grams per kilometer):** Total emissions per kilometer of road length
- **Hot_Emission_PC:** Emissions from passenger cars only
- **Hot_Emission_Motorcycle:** Emissions from motorcycles only
- **Total_Emission:** Combined emissions from all vehicles

### Map Interpretation
- **Color Scale:** Warmer colors (red/yellow) indicate higher emissions
- **Line Thickness:** In Enhanced/Custom modes, thicker lines represent higher emissions
- **Gray Roads:** Roads without emission data (exist in OSM but not in your link_osm.dat file)
- **Road Labels:** Major roads and high-emission routes are labeled for easy identification

### Color Map Guide
- **jet:** Traditional rainbow (blue → green → yellow → red)
- **viridis:** Perceptually uniform, colorblind-friendly
- **plasma:** High contrast, good for presentations
- **RdYlGn_r:** Red-Yellow-Green reversed (red = high emissions)
- **hot:** Black → red → yellow → white
- **coolwarm:** Blue (low) → white → red (high)
- **inferno:** Dark → vibrant (Custom mode only)

---

## 8. Troubleshooting

### 🛑 Common Issues and Solutions

| Problem | Solution |
|:--------|:---------|
| `"Expected 7 columns but found X"` | Verify your `link_osm.dat` is **space-separated** and contains **exactly 7 columns** in the correct order. Check for double quotes (`"`) and remove them. Make sure there are no extra spaces at the end of lines. |
| `"copert module not found"` | Ensure the custom COPERT Python module is installed and accessible in your Python environment path. Contact the developer for the module. |
| `"osm_network module not found"` | Install the dependency: `pip install osmium`. Make sure `osm_network.py` is in the same directory as the app. |
| **Map shows, but emission lines are invisible.** | Switch from **Classic** to **Enhanced** mode, or increase the line width scale in the settings. Classic mode uses microscopic line widths by design. |
| **Too many overlapping labels.** | Decrease the label density setting (e.g., from High to Minimal) or increase the minimum label distance in the **Custom** mode. Try increasing figure size to spread labels out. |
| **Calculation is very slow.** | Be patient - complex networks take time (1-5 min per 1000 segments). Close other applications to free up memory. Consider filtering less important roads in QGIS to reduce network size. |
| **OSM file not parsing correctly.** | Ensure OSM file is in XML format (.osm). Check file isn't corrupted. Try re-downloading from OSM. |
| **Roads without emission data.** | These roads exist in your OSM file but don't have matching OSM_IDs in your link_osm.dat file. This is normal if you filtered roads during QGIS processing. |

---

## 9. Tips for Best Results

### Data Quality
- **Use real traffic data** when available for more accurate results
- **Validate speed values:** Should be between 10-130 km/h
- **Check proportions:** All proportion values should be between 0.0 and 1.0
- **Verify OSM_IDs:** Must match between link_osm.dat and network.osm

### Visualization
- **For presentations:** Use Enhanced mode with Medium label density
- **For publications:** Use Classic mode or Enhanced with Minimal labels
- **For analysis:** Use Custom mode to highlight specific features
- **Large networks:** Reduce label density to avoid clutter
- **Small networks:** Can use Maximum label density for complete road identification

### Performance
- **Filter roads in QGIS** before export to focus on important routes
- **Use appropriate figure size:** Larger sizes take longer to generate
- **Disable labels** for quick previews, enable for final maps
- **Save intermediate results:** Download CSV after calculation in case of crashes

---

## 10. Additional Resources

- **COPERT Methodology:** [https://www.emisia.com/utilities/copert/](https://www.emisia.com/utilities/copert/)
- **OpenStreetMap Data:** [https://www.openstreetmap.org/](https://www.openstreetmap.org/)
- **QGIS Tutorials:** [https://docs.qgis.org/](https://docs.qgis.org/)
- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## 11. Citation and License

If you use this tool in your research or publications, please cite:

**Traffic Emission Calculator with OSM Visualization**  
Developed by SHassan  
Version 2.0 (November 2025)
