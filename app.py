import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
import numpy as np

# --- 1. CONFIG & DATABASE ---
st.set_page_config(page_title="Bendac Visualiser", layout="wide")

def load_data():
    """
    Tries to load data from CSV. 
    Falls back to hardcoded dictionary if file is missing/broken.
    """
    csv_file = "bendac_database.csv"
    
    # BACKUP DATA
    backup_data = {
        'Product Name': [
            'Bendac AccuVision', 'Bendac AccuVision', 
            'Bendac Krystl Max', 'Bendac Krystl Max', 'Bendac Krystl Max', 
            'Bendac Dura', 'Bendac Dura', 'Bendac Dura', 
            'Bendac COBi', 'Bendac COBi', 'Bendac COBi'
        ],
        'Width(mm)': [337.5, 337.5, 1000.0, 1000.0, 1000.0, 960.0, 960.0, 960.0, 600.0, 600.0, 600.0],
        'Height(mm)': [600.0, 600.0, 500.0, 500.0, 500.0, 960.0, 960.0, 960.0, 337.5, 337.5, 337.5],
        'Pitch(mm)': [0.9, 1.2, 1.9, 2.6, 3.9, 6.0, 8.0, 10.0, 0.9, 1.2, 1.2],
        'ResW(px)': [360, 270, 512, 384, 256, 144, 120, 96, 640, 480, 384],
        'ResH(px)': [640, 480, 256, 192, 128, 144, 120, 96, 360, 270, 216],
        'Power(W)': [100.0, 100.0, 300.0, 300.0, 300.0, 450.0, 520.0, 520.0, 121.5, 121.5, 121.5],
        'Color': [
            '#4a90e2', '#4a90e2', 
            '#50e3c2', '#50e3c2', '#50e3c2', 
            '#9013fe', '#9013fe', '#9013fe', 
            '#f5a623', '#f5a623', '#f5a623'
        ]
    }

    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            required_cols = list(backup_data.keys())
            if all(col in df.columns for col in required_cols):
                return df
            else:
                st.warning("CSV missing required columns. Using backup data.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}. Using backup data.")
    
    return pd.DataFrame(backup_data)

# Initialize Session State
if 'db' not in st.session_state:
    st.session_state.db = load_data()

df = st.session_state.db

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    else:
        st.title("BENDAC")
    
    st.caption("VISUALISER TOOL")
    st.header("Configuration")
    
    # Product Selection
    unique_products = df["Product Name"].unique()
    selected_prod = st.selectbox("Product Series", unique_products)
    
    # Filter for Pitches
    prod_rows = df[df["Product Name"] == selected_prod]
    available_pitches = sorted(prod_rows["Pitch(mm)"].unique())
    selected_pitch = st.selectbox("Pixel Pitch (mm)", available_pitches)
    
    # Get Data
    spec = prod_rows[prod_rows["Pitch(mm)"] == selected_pitch].iloc[0]
    
    st.divider()
    
    # Dimensions
    col1, col2 = st.columns(2)
    with col1:
        panels_w = st.number_input("Panels Wide", min_value=1, value=8)
    with col2:
        panels_h = st.number_input("Panels High", min_value=1, value=5)
        
    st.divider()
    
    # Curve Logic
    st.subheader("Curve Geometry")
    curve_radius = st.number_input("Radius (mm)", min_value=0.0, value=0.0, step=100.0)
    curve_angle = st.number_input("Total Angle (deg)", min_value=0.0, value=0.0, step=1.0)
    
    # Auto-Calc Logic
    is_curved = False
    calc_note = ""
    
    if curve_radius > 0 and curve_angle > 0:
        is_curved = True
        req_arc = 2 * math.pi * curve_radius * (curve_angle / 360)
        panels_needed = round(req_arc / spec["Width(mm)"])
        if panels_needed < 1: panels_needed = 1
        panels_w = panels_needed 
        calc_note = f"Auto-Adjusted Width to **{panels_needed} panels** based on Radius/Angle."

    elif curve_radius > 0:
        is_curved = True
        arc_len = panels_w * spec["Width(mm)"]
        rad_angle = arc_len / curve_radius
        curve_angle = math.degrees(rad_angle)
        
    elif curve_angle > 0:
        is_curved = True
        arc_len = panels_w * spec["Width(mm)"]
        rad_angle = math.radians(curve_angle)
        curve_radius = arc_len / rad_angle

    if calc_note:
        st.info(calc_note)

# --- 3. CALCULATIONS ---
total_w_mm = panels_w * spec["Width(mm)"]
total_h_mm = panels_h * spec["Height(mm)"]
total_res_w = panels_w * spec["ResW(px)"]
total_res_h = panels_h * spec["ResH(px)"]
total_power = (panels_w * panels_h * spec["Power(W)"]) / 1000

if is_curved and curve_radius > 0:
    rad_angle = math.radians(curve_angle)
    phys_w = 2 * curve_radius * math.sin(rad_angle/2)
    phys_d = curve_radius * (1 - math.cos(rad_angle/2))
else:
    phys_w = total_w_mm
    phys_d = 0

# --- 4. MAIN DISPLAY ---
st.subheader(f"{selected_prod} ({selected_pitch}mm)")

# Stats Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dimensions", f"{total_w_mm:,.0f} x {total_h_mm:,.0f} mm")
m2.metric("Resolution", f"{total_res_w} x {total_res_h} px")
m3.metric("Max Power", f"{total_power:.2f} kW")
m4.metric("Panel Count", f"{panels_w * panels_h}")

if is_curved:
    st.caption(f"**Curve Stats:** Radius: {curve_radius:,.0f}mm | Angle: {curve_angle:.1f}° | Footprint: {phys_w:,.0f}mm (W) x {phys_d:,.0f}mm (D)")

# --- 5. PLOTTING ---
plot_col1, plot_col2 = st.columns(2)

# Common Settings
person_h = 1750
panel_color = spec["Color"]

# --- FRONT VIEW (AX1) ---
with plot_col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5)) 
    ax1.set_title("FRONT VIEW (Unfolded)")
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Draw Screen
    rect = patches.Rectangle((0, 0), total_w_mm, total_h_mm, linewidth=1, edgecolor='white', facecolor=panel_color)
    ax1.add_patch(rect)

    # Draw Grid Lines
    # Logic: Draw all lines, rely on Matplotlib to handle rendering. 
    # Performance is usually fine for <5000 lines.
    for c in range(int(panels_w) + 1):
        x = c * spec["Width(mm)"]
        ax1.plot([x, x], [0, total_h_mm], color='white', linewidth=0.5)
    for r in range(int(panels_h) + 1):
        y = r * spec["Height(mm)"]
        ax1.plot([0, total_w_mm], [y, y], color='white', linewidth=0.5)

    # Draw Person (IMAGE)
    p_x = total_w_mm + 500 # Position 500mm to the right
    
    if os.path.exists("person.png"):
        try:
            img = mpimg.imread("person.png")
            img_h_px, img_w_px = img.shape[:2]
            aspect_ratio = img_w_px / img_h_px
            target_h = person_h
            target_w = target_h * aspect_ratio
            ax1.imshow(img, extent=[p_x, p_x + target_w, 0, target_h])
        except Exception as e:
            st.error(f"Error loading person.png: {e}")
            ax1.add_patch(patches.Rectangle((p_x, 0), 600, 1750, color="#ccc"))
    else:
        # Fallback if file missing
        ax1.add_patch(patches.Rectangle((p_x, 0), 600, 1750, color="#888"))

    # Auto-Scale
    ax1.autoscale_view()
    ax1.margins(0.1)
    st.pyplot(fig1, use_container_width=True)

# --- TOP VIEW (AX2) ---
with plot_col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.set_title("TOP VIEW (Plan)")
    ax2.set_aspect('equal')
    ax2.axis('off')

    panel_thick = 100

    if not is_curved:
        # Flat View
        start_x = -(total_w_mm / 2)
        # Screen
        rect_top = patches.Rectangle((start_x, 0), total_w_mm, panel_thick, linewidth=1, edgecolor='black', facecolor=panel_color)
        ax2.add_patch(rect_top)
        
        # Viewer Position
        person_y = 1000
        
    else:
        # Curved View
        center_x = 0
        center_y = curve_radius 
        
        start_angle = 270 - (curve_angle / 2)
        angle_step = curve_angle / panels_w
        
        current_a = math.radians(start_angle)
        step_a = math.radians(angle_step)
        
        for i in range(int(panels_w)):
            # Inner Points
            x1 = center_x + curve_radius * math.cos(current_a)
            y1 = center_y + curve_radius * math.sin(current_a)
            x2 = center_x + curve_radius * math.cos(current_a + step_a)
            y2 = center_y + curve_radius * math.sin(current_a + step_a)
            
            # Outer Points
            r_out = curve_radius + panel_thick
            x3 = center_x + r_out * math.cos(current_a + step_a)
            y3 = center_y + r_out * math.sin(current_a + step_a)
            x4 = center_x + r_out * math.cos(current_a)
            y4 = center_y + r_out * math.sin(current_a)
            
            poly = patches.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, edgecolor='white', facecolor=panel_color)
            ax2.add_patch(poly)
            
            current_a += step_a

        # Viewer Position
        person_y = min(curve_radius, phys_d + 3000) 

    # --- DRAW TOP PERSON IMAGE ---
    # Center the person at (0, person_y)
    if os.path.exists("top_person.png"):
        try:
            img_top = mpimg.imread("top_person.png")
            t_h_px, t_w_px = img_top.shape[:2]
            t_aspect = t_w_px / t_h_px
            
            # Define approximate real-world size for top-down person (Shoulder width ~600mm)
            target_top_w = 600
            target_top_h = target_top_w / t_aspect
            
            # Extent: [left, right, bottom, top]
            # Image centered at x=0, y=person_y
            extent = [
                -target_top_w / 2, 
                target_top_w / 2, 
                person_y - target_top_h / 2, 
                person_y + target_top_h / 2
            ]
            ax2.imshow(img_top, extent=extent)
            
        except Exception as e:
            st.error(f"Error loading top_person.png: {e}")
            # Fallback
            ax2.add_patch(patches.Circle((0, person_y), 200, color='#333'))
    else:
        # Fallback if file missing (Standard Circle)
        ax2.add_patch(patches.Circle((0, person_y), 200, color='#333'))
        # Note: Text label "Viewer" removed as requested

    ax2.autoscale_view()
    ax2.margins(0.1)

    st.pyplot(fig2, use_container_width=True)
