# Traffic Emission Calculator - Setup Guide

## 📦 Quick Setup

### Option 1: GitHub Hosted Instructions (Recommended)

1. **Upload `instructions.md` to your GitHub repository**
   ```bash
   git add instructions.md
   git commit -m "Add detailed instructions"
   git push
   ```

2. **Update the app code** (line ~30 in `Emmission_Cal_app.py`):
   ```python
   instructions_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/instructions.md"
   ```
   Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repository name.

3. **Run the app**:
   ```bash
   streamlit run Emmission_Cal_app.py
   ```

### Option 2: Local Instructions File

1. **Place `instructions.md` in the same directory as your app**:
   ```
   your-project/
   ├── Emmission_Cal_app.py
   ├── instructions.md
   ├── osm_network.py
   └── copert.py
   ```

2. **Run the app**:
   ```bash
   streamlit run Emmission_Cal_app.py
   ```

The app will automatically fallback to the local file if GitHub is unavailable.

---

## 📁 Complete File Structure

```
traffic-emission-calculator/
├── Emmission_Cal_app.py          # Main Streamlit application
├── osm_network.py                 # OSM network parser (FIXED version)
├── copert.py                      # COPERT emission calculation module
├── instructions.md                # Detailed user guide
├── SETUP.md                       # This file
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
└── data/                          # (Optional) Sample data folder
    ├── sample_link_osm.dat
    ├── sample_network.osm
    └── copert_parameters/
        ├── PC_parameter.csv
        ├── LDV_parameter.csv
        ├── HDV_parameter.csv
        └── Moto_parameter.csv
```

---

## 🔧 Installation

### 1. Install Python Dependencies

Create a `requirements.txt` file:
```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
osmium>=3.6.0
requests>=2.31.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Install COPERT Module

If you have a custom COPERT module:
```bash
pip install -e /path/to/copert
```

Or place `copert.py` in the same directory as the app.

### 3. Verify osmium Installation

Test the osm_network module:
```bash
python -c "import osmium; print('osmium installed successfully')"
```

---

## 🚀 Running the App

### Local Development
```bash
streamlit run Emmission_Cal_app.py
```

The app will open in your browser at `http://localhost:8501`

### Production Deployment

####
