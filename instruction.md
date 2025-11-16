## 🏭 Traffic Emission Calculation and Visualization App Guide

Welcome to the **Traffic Emission Calculator with OSM Visualization**! This application uses the **COPERT** (Computer Programme to Calculate Emissions from Road Transport) methodology to calculate and visualize vehicle emissions based on OpenStreetMap (OSM) road networks and user-provided traffic data.

### Key Outputs:

* Emission factors per road segment ($\text{g/km}$) for various pollutants (e.g., CO, NOx, PM).
* **Interactive maps** with color-coded and thickness-scaled emission intensity.
* Optional road name labels for easy identification.
* Downloadable results (CSV, PNG, ZIP).

---

## 2. Prerequisites

### 💻 Software Requirements

| Software | Purpose | Download Link | Notes |
| :--- | :--- | :--- | :--- |
| **QGIS** | Geospatial data preparation and network modeling. | $\text{[https://qgis.org/](https://qgis.org/)}$ | Version 3.x or higher is recommended. |
| **Python** | Running the Streamlit application. | $\text{[https://python.org/](https://python.org/)}$ | Python 3.8 or higher required. |
| **Python Packages** | Application dependencies. | `pip install streamlit numpy pandas matplotlib osmium` | The custom `copert` module must be in your Python path. |

### 📂 Data Requirements

You must provide your own **road link data** and utilize the provided **COPERT/Proportion files** for the calculation.

| File Name | Purpose | Format | Notes |
| :--- | :--- | :--- | :--- |
| **link\_osm.dat** | **Primary Input:** Road segment geometry and traffic data. | Space-separated DAT | Must contain **7 mandatory columns** (see Stage 2). |
| **network.osm** | OpenStreetMap network file for visualization. | OSM XML | Can be downloaded from QuickOSM or Overpass-Turbo. |
| **4 COPERT Parameter Files** | Emission factors for PC, LDV, HDV, Moto. | CSV | Provided auxiliary files. |
| **6 Vehicle Proportion Files** | Vehicle class/engine type distributions. | Single-column DAT | Provided auxiliary files. |

> **📥 Download Auxiliary Files:**
>
> All necessary COPERT parameter and Vehicle Proportion files must be downloaded before running the app. You can find them here:
>
> **[Download All Required Auxiliary Input Files (Google Drive Link)]** *(Replace this text with your actual download link)*

---

## 3. Workflow Overview

The overall process is divided into three distinct and sequential stages:

$$\text{Stage 1: QGIS Geospatial Data Preparation} \rightarrow \text{Stage 2: Data Conversion and Formatting} \rightarrow \text{Stage 3: Application Usage & Visualization}$$

---

## 4. Stage 1: QGIS Geospatial Data Preparation

This stage focuses on acquiring and preparing the street network data to create the `link_osm.dat` input file.

### 4.1. 🗺️ Street Network Data Acquisition

1.  **Open QGIS** and start a **New Project**. Zoom to your study area.
2.  **Install Plugin:** Go to $\text{Plugins} \rightarrow \text{Manage and Install Plugins...}$ and install the **"QuickOSM"** or **"OSM Downloader"** plugin.
3.  **Download Data:** Use the plugin to download road network data for your specific geographic area.
    * *Recommended:* Use QuickOSM with $\text{Key: highway}$ and $\text{Value: (leave empty)}$ to capture all road types.
4.  **Export Data:** Export the downloaded data as an OSM XML file (`network.osm`) and ensure the vector layers (lines/multilines) are loaded in QGIS.

### 4.2. 🧩 Data Processing and Layer Merging

1.  **Merge Layers:** OSM data often separates road features into distinct layers. Consolidate them:
    * Open the **Processing Toolbox** ($\text{View} \rightarrow \text{Panels} \rightarrow \text{Processing Toolbox}$).
    * Search for and run the **"Merge Vector Layers"** tool.
    * Select **ONLY** the `lines` and `multilines` layers as input. The output is your single, complete street network layer.

### 4.3. 📊 Attribute Table Calculation

You must add and calculate the seven variables required by the emission model in the merged layer's attribute table.

1.  **Access Table:** Right-click the merged layer and select $\text{Open Attribute Table}$.
2.  **Calculate Attributes:** Use the **Field Calculator** to create the following seven fields (columns):

| Field Name | Description | Calculation/Data Source |
| :--- | :--- | :--- |
| **OSM\_ID** | Unique road segment ID. | Extract from raw OSM data field (e.g., `osm_id` or `id`). |
| **Length\_km** | Physical length of the segment. | $\text{Expression: (\$length / 1000)}$ |
| **Flow** | Estimated traffic volume. | External data or **default values** (vehicles/hour). |
| **Speed** | Average vehicle speed. | External data or **default values** ($\text{km/h}$). |
| **Gasoline\_Prop** | Proportion of gasoline-powered vehicles. | External data or **model assumptions** (0.0 to 1.0). |
| **PC\_Prop** | Proportion of passenger cars (vs. motorcycles). | External data or **model assumptions** (0.0 to 1.0). |
| **4Stroke\_Prop** | Proportion of 4-stroke motorcycles. | External data or **model assumptions** (0.0 to 1.0). |

> **💡 Traffic Data Tip:** If external traffic data is unavailable, use **default assumptions** based on road type (e.g., Motorway: High Flow, High Speed; Residential: Low Flow, Low Speed).

---

## 5. Stage 2: Data Conversion and Formatting

The final input file for the app (`link_osm.dat`) must be a space-separated file with a specific column order.

### 5.1. Final Attribute Selection and CSV Export

1.  **Clean Table:** Delete all irrelevant columns from the attribute table, ensuring only the 7 mandatory columns remain.
2.  **Order Columns:** The columns must be in this **exact order**:
    $$\text{OSM\_ID, Length\_km, Flow, Speed, Gasoline\_Prop, PC\_Prop, 4Stroke\_Prop}$$
3.  **Export to CSV:**
    * Right-click the final merged layer $\rightarrow \text{Export} \rightarrow \text{Save Feature As...}$
    * Format: Choose **Comma Separated Values (CSV)**.

### 5.2. CSV to DAT Conversion (Crucial Step)

The Streamlit app requires a **space-separated .dat file**.

1.  **Open in Editor:** Open the exported CSV file using a plain text editor (e.g., Notepad, VS Code).
2.  **Find and Replace (Separators):**
    * **Find:** The CSV field separator (typically `,` or `;`) and **Replace with:** A single space (` `).
3.  **Find and Replace (Quotes):**
    * **Find:** Double Quotes (`"`) and **Replace with:** Nothing (delete them).
4.  **Save as DAT:** Save the modified file with the **exact name** $\text{link\_osm.dat}$.

---

## 6. Stage 3: Application Usage & Visualization

### 6.1. File Acquisition and Upload

1.  **Upload:** Navigate to the sidebar of the web application. Use the dedicated sections to upload **all 11 required files**, including your `link_osm.dat` and the 10 auxiliary files you downloaded.

### 6.2. Calculation and Visualization

1.  **Validate Data:** The app will check for the presence of all 11 files and validate the structure of `link_osm.dat`. Look for a **green checkmark** indicating readiness.
2.  **Run Calculation:** Navigate to the **"Calculate Emissions"** tab (or similar section) and click the calculation button.
3.  **Plotting:** Proceed to the **"Emission Map"** tab to visualize your results.

### 6.3. Visualization Modes

The app offers several modes to present the emission data on the map:

| Mode | Best For | Key Features |
| :--- | :--- | :--- |
| **Classic (Original)** | Academic papers, simple overview. | Minimal line widths, no road labels, traditional look. |
| **Enhanced with Labels** | Presentations, reports, analysis. | Thicker lines (proportional to emissions), smart road name labels (no overlap), easy-to-read. |
| **Custom** | Specific, fine-tuned outputs. | Complete control over line width scale, label density, road transparency, and color maps. |

### 6.4. Download Results

The final tab allows you to download the results for further analysis or publication:

* **Individual Downloads:**
    * Emission Data CSV (Full dataset).
    * Emission Map PNG (High-resolution image).
* **ZIP Archive:** Includes the full emission dataset (`link_hot_emission_factor.csv`), the map visualization (`emission_factor_map.png`), and a statistical summary (`summary.txt`).

---

## 7. Troubleshooting

### 🛑 Common Issues and Solutions

| Problem | Solution |
| :--- | :--- |
| `"Expected 7 columns but found X"` | Verify your `link_osm.dat` is **space-separated** and contains **exactly 7 columns** in the correct order. Check for double quotes (`"`) and remove them. |
| `"copert module not found"` | Ensure the custom COPERT Python module is installed and accessible in your Python environment path. |
| `"osm_network module not found"` | Install the dependency: `pip install osmium`. |
| **Map shows, but emission lines are invisible.** | Switch from **Classic** to **Enhanced** mode, or increase the line width scale in the settings. |
| **Too many overlapping labels.** | Decrease the label density setting (e.g., from High to Minimal) or increase the minimum label distance in the **Custom** mode. |

---

Would you like me to refine the language or adjust the formatting of any specific section?
