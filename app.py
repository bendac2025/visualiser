import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
import io
import numpy as np
from PIL import Image

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
        'Weight(kg)': [7.5, 7.5, 12.5, 12.5, 12.0, 28.0, 28.0, 28.0, 6.5, 6.5, 6.5],
        'MaxFPS(Hz)': [240, 240, 60, 60, 60, 60, 60, 60, 60, 60, 60],
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
            if 'Weight(kg)' not in df.columns: df['Weight(kg)'] = 10.0
            if 'MaxFPS(Hz)' not in df.columns: df['MaxFPS(Hz)'] = 60
            
            required_cols = ['Product Name', 'Width(mm)', 'Height(mm)', 'Pitch(mm)', 'ResW(px)', 'ResH(px)', 'Power(W)', 'Color']
            if all(col in df.columns for col in required_cols):
                return df
            else:
                st.warning("CSV missing required columns. Using backup data.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}. Using backup data.")
    
    return pd.DataFrame(backup_data)

# --- PROCESSOR DATABASE ---
NOVASTAR_DB = {
    "Novastar MCTRL660 Pro": { "type": "fixed", "capacity_60": 2300000, "ports": 6 }, 
    "Novastar VX1000": { "type": "fixed", "capacity_60": 6500000, "ports": 10 },
    "Novastar MCTRL4K": { "type": "fixed", "capacity_60": 8800000, "ports": 16 },
    "Novastar MX40 Pro": { "type": "fixed", "capacity_60": 8800000, "ports": 20 },
    "Novastar MX2000 Pro (40G Cards)": { "type": "modular", "capacity_60": 35380000, "card_capacity_60": 26200000, "slots": 2 },
    "Novastar MX6000 Pro (40G Cards)": { "type": "modular", "capacity_60": 141000000, "card_capacity_60": 26200000, "slots": 8 }
}

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
    
    use_imperial = st.checkbox("Show Imperial Units (ft/lbs)", value=False)
    
    st.header("1. Configuration")
    
    unique_products = list(df["Product Name"].unique())
    default_prod_index = 0
    if "Bendac Krystl Max" in unique_products:
        default_prod_index = unique_products.index("Bendac Krystl Max")
        
    selected_prod = st.selectbox("Product Series", unique_products, index=default_prod_index)
    
    prod_rows = df[df["Product Name"] == selected_prod]
    available_pitches = sorted(prod_rows["Pitch(mm)"].unique())
    
    default_pitch_index = 0
    if 1.9 in available_pitches:
        default_pitch_index = available_pitches.index(1.9)
        
    selected_pitch = st.selectbox("Pixel Pitch (mm)", available_pitches, index=default_pitch_index)
    spec = prod_rows[prod_rows["Pitch(mm)"].unique() == selected_pitch].iloc[0]
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        panels_w = st.number_input("Panels Wide", min_value=1, value=1)
    with col2:
        panels_h = st.number_input("Panels High", min_value=1, value=1)
        
    st.divider()
    
    st.subheader("2. Curve Geometry")
    curve_radius = st.number_input("Radius (mm)", min_value=0.0, value=0.0, step=100.0)
    curve_angle = st.number_input("Total Angle (deg)", min_value=0.0, value=0.0, step=1.0)
    
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

    st.subheader("3. Processing (Novastar)")
    proc_model = st.selectbox("Processor Model", list(NOVASTAR_DB.keys()))
    
    if "AccuVision" in selected_prod:
        fps_options = [60, 120, 240]
    else:
        fps_options = [60]
    target_fps = st.selectbox("Target Frame Rate (Hz)", fps_options)
    
    st.subheader("4. Visual Context")
    content_file = st.file_uploader("Upload Screen Content", type=["png", "jpg", "jpeg"])
    
    content_img_data = None
    if content_file:
        try:
            image = Image.open(content_file)
            content_img_data = np.array(image)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # --- CALCULATIONS ---
    total_w_mm = panels_w * spec["Width(mm)"]
    total_h_mm = panels_h * spec["Height(mm)"]
    total_res_w = panels_w * spec["ResW(px)"]
    total_res_h = panels_h * spec["ResH(px)"]
    total_pixels = total_res_w * total_res_h
    
    total_power_w = panels_w * panels_h * spec["Power(W)"]
    total_weight_kg = panels_w * panels_h * spec["Weight(kg)"]
    
    btu_hr = total_power_w * 3.412142
    aspect_ratio = total_w_mm / total_h_mm
    
    # --- PROCESSOR CALCULATIONS ---
    fps_scale = 60 / target_fps
    
    port_capacity_60 = 655360 
    port_capacity_real = port_capacity_60 * fps_scale
    total_ports_needed = math.ceil(total_pixels / port_capacity_real)
    
    proc_data = NOVASTAR_DB[proc_model]
    chassis_load_pct = 0 
    reason_str = ""

    if proc_data["type"] == "fixed":
        unit_cap_real = proc_data["capacity_60"] * fps_scale
        procs_by_pixels = math.ceil(total_pixels / unit_cap_real)
        ports_per_unit = proc_data["ports"]
        procs_by_ports = math.ceil(total_ports_needed / ports_per_unit)
        
        total_procs_needed = max(procs_by_pixels, procs_by_ports)
        
        total_sys_cap = total_procs_needed * unit_cap_real
        if total_sys_cap > 0:
            chassis_load_pct = (total_pixels / total_sys_cap) * 100
        
        if procs_by_ports > procs_by_pixels:
            reason_str = "Driven by: Physical Ports"
        else:
            reason_str = "Driven by: Pixel Capacity"
            
        proc_str = f"{total_procs_needed}x {proc_model}"
        
    else:
        card_cap_real = proc_data["card_capacity_60"] * fps_scale
        total_cards_needed = math.ceil(total_pixels / card_cap_real)
        
        slots_per_chassis = proc_data["slots"]
        chassis_by_slots = math.ceil(total_cards_needed / slots_per_chassis)
        
        chassis_cap_real = proc_data["capacity_60"] * fps_scale
        chassis_by_cap = math.ceil(total_pixels / chassis_cap_real)
        
        total_chassis_needed = max(chassis_by_slots, chassis_by_cap)
        
        total_cap_modular = total_chassis_needed * chassis_cap_real
        if total_cap_modular > 0:
            chassis_load_pct = (total_pixels / total_cap_modular) * 100
        
        if chassis_by_cap > chassis_by_slots:
            reason_str = "Driven by: Chassis Bandwidth Limit"
        else:
            reason_str = "Driven by: Slot Count"

        proc_str = f"{total_chassis_needed}x {proc_model.split('(')[0]} ({total_cards_needed}x 40G Cards)"

    if chassis_load_pct > 95:
        st.warning(f"⚠️ High Chassis Load: {chassis_load_pct:.1f}%")

    # --- VIDEO INPUT & IG CALCULATIONS ---
    video_input_cap = 8294400 * fps_scale
    video_inputs_needed = math.ceil(total_pixels / video_input_cap)
    input_str = f"{video_inputs_needed}x 4K Inputs"
    
    is_accuvision = "AccuVision" in selected_prod
    if is_accuvision:
        igs_needed = math.ceil(video_inputs_needed / 4)
        ig_str = f"{igs_needed}x Image Generators (PC)"
    else:
        ig_str = ""

    if is_curved and curve_radius > 0:
        rad_angle = math.radians(curve_angle)
        phys_w = 2 * curve_radius * math.sin(rad_angle/2)
        phys_d = curve_radius * (1 - math.cos(rad_angle/2))
    else:
        phys_w = total_w_mm
        phys_d = 0

    # --- PDF GENERATION FUNCTION ---
    def create_pdf_figure():
        panel_thick = 100 
        
        pdf_fig = plt.figure(figsize=(8.27, 11.69))
        
        if os.path.exists("logo.png"):
            ax_logo = pdf_fig.add_axes([0.35, 0.89, 0.3, 0.08]) 
            img_logo = mpimg.imread("logo.png")
            ax_logo.imshow(img_logo)
            ax_logo.axis('off')
        
        pdf_fig.text(0.5, 0.85, "TECHNICAL SPECIFICATION", ha='center', fontsize=16, weight='bold')
        
        ax_table = pdf_fig.add_axes([0.1, 0.58, 0.8, 0.20])
        ax_table.axis('off')
        
        dim_str = f"{total_w_mm:,.0f} mm (W) x {total_h_mm:,.0f} mm (H)"
        weight_str = f"{total_weight_kg:,.0f} kg"
        if use_imperial:
            dim_str += f"\n{total_w_mm/304.8:,.1f} ft (W) x {total_h_mm/304.8:,.1f} ft (H)"
            weight_str += f" ({total_weight_kg*2.20462:,.0f} lbs)"

        table_data = [
            ["Model Series", f"{selected_prod}"],
            ["Pixel Pitch", f"{selected_pitch} mm"],
            ["Configuration", f"{panels_w} (W) x {panels_h} (H) Panels"],
            ["Dimensions", dim_str],
            ["Resolution", f"{total_res_w} px (W) x {total_res_h} px (H) @ {target_fps}Hz"],
            ["Max Power / Heat", f"{total_power_w/1000:.1f} kW  /  {btu_hr:,.0f} BTU/hr"],
            ["Total Weight", weight_str],
            ["Processing", f"{proc_str}\n({reason_str} - Load: {chassis_load_pct:.1f}%)"],
            ["Data Capacity", f"{total_ports_needed}x 1G Ports required (Ref)"],
            ["Video Inputs", input_str]
        ]
        
        if is_accuvision:
            table_data.append(["Image Generators", ig_str])
        
        if is_curved:
            table_data.append(["Curve", f"R: {curve_radius:,.0f}mm | {curve_angle:.1f}°"])
            # Add Footprint with 600mm buffer on all sides
            buff_w = phys_w + 1200
            buff_d = phys_d + 1200
            table_data.append(["Install Zone (+600mm)", f"{buff_w:,.0f} mm (W) x {buff_d:,.0f} mm (D)"])
        else:
            # Optional: Add footprint for flat too?
            # table_data.append(["Install Zone (+600mm)", f"{total_w_mm+1200:,.0f} mm (W) x {100+1200:,.0f} mm (D)"])
            pass
        
        the_table = ax_table.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.3, 0.7])
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.6)
        
        # Row Height Logic
        for row_idx, row_data in enumerate(table_data):
            content_cell = row_data[1] 
            if "\n" in content_cell:
                current_height = the_table[row_idx, 0].get_height()
                new_height = current_height * 1.8
                the_table[row_idx, 0].set_height(new_height)
                the_table[row_idx, 1].set_height(new_height)

        for (i, j), cell in the_table.get_celld().items():
            if j == 0: cell.set_text_props(weight='bold')
            cell.set_edgecolor('#dddddd')

        ax_front = pdf_fig.add_axes([0.1, 0.30, 0.8, 0.23])
        ax_front.set_title("FRONT VIEW")
        ax_front.set_aspect('equal')
        ax_front.axis('off')
        
        p_color = spec["Color"]
        
        # Transparent face if content loaded
        face_col = 'none' if content_img_data is not None else p_color
        
        rect = patches.Rectangle((0, 0), total_w_mm, total_h_mm, linewidth=1, edgecolor='black', facecolor=face_col)
        ax_front.add_patch(rect)
        
        if content_img_data is not None:
            ax_front.imshow(content_img_data, extent=[0, total_w_mm, 0, total_h_mm], zorder=0)

        if panels_w <= 1000:
            for c in range(int(panels_w) + 1):
                x = c * spec["Width(mm)"]
                ax_front.plot([x, x], [0, total_h_mm], color='black', linewidth=0.1, alpha=0.5, zorder=6)
            for r in range(int(panels_h) + 1):
                y = r * spec["Height(mm)"]
                ax_front.plot([0, total_w_mm], [y, y], color='black', linewidth=0.1, alpha=0.5, zorder=6)

        pdf_px = total_w_mm + 500
        if os.path.exists("person.png"):
            img = mpimg.imread("person.png")
            aspect_ratio = img.shape[1] / img.shape[0]
            ax_front.imshow(img, extent=[pdf_px, pdf_px + (1750*aspect_ratio), 0, 1750], zorder=10)
        else:
             ax_front.add_patch(patches.Rectangle((pdf_px, 0), 600, 1750, color="#ccc"))
        
        ax_front.autoscale_view()
        
        ax_top = pdf_fig.add_axes([0.1, 0.05, 0.8, 0.22])
        ax_top.set_title("TOP VIEW")
        ax_top.set_aspect('equal')
        ax_top.axis('off')
        
        if not is_curved:
            start_x = -(total_w_mm / 2)
            ax_top.add_patch(patches.Rectangle((start_x, 0), total_w_mm, 100, linewidth=1, edgecolor='black', facecolor=p_color))
            pdf_py = 1000
        else:
            center_x = 0
            center_y = curve_radius 
            start_angle = 270 - (curve_angle / 2)
            current_a = math.radians(start_angle)
            angle_step = curve_angle / panels_w
            for i in range(int(panels_w)):
                x1 = center_x + curve_radius * math.cos(current_a)
                y1 = center_y + curve_radius * math.sin(current_a)
                x2 = center_x + curve_radius * math.cos(current_a + math.radians(angle_step))
                y2 = center_y + curve_radius * math.sin(current_a + math.radians(angle_step))
                r_out = curve_radius + panel_thick
                x3 = center_x + r_out * math.cos(current_a + math.radians(angle_step))
                y3 = center_y + r_out * math.sin(current_a + math.radians(angle_step))
                x4 = center_x + r_out * math.cos(current_a)
                y4 = center_y + r_out * math.sin(current_a)
                ax_top.add_patch(patches.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, edgecolor='black', facecolor=p_color))
                current_a += math.radians(angle_step)
            pdf_py = min(curve_radius, phys_d + 3000)

        if os.path.exists("top_person.png"):
            img_top = mpimg.imread("top_person.png")
            t_aspect = img_top.shape[1] / img_top.shape[0]
            t_w = 600
            t_h = t_w / t_aspect
            ax_top.imshow(img_top, extent=[-t_w/2, t_w/2, pdf_py - t_h/2, pdf_py + t_h/2])
        else:
            ax_top.add_patch(patches.Circle((0, pdf_py), 200, color='#333'))

        ax_top.autoscale_view()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="pdf", bbox_inches='tight')
        buf.seek(0)
        return buf

    st.divider()
    pdf_buffer = create_pdf_figure()
    st.download_button(
        label="📄 Download Spec Sheet (PDF)",
        data=pdf_buffer,
        file_name=f"Bendac_Spec_{selected_prod}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    # --- COPY DATA (SIDEBAR) ---
    st.divider()
    with st.expander("Show Text Summary"):
        summary_txt = f"""
PRODUCT: {selected_prod} ({selected_pitch}mm)
CONFIG: {panels_w} x {panels_h}
DIMENSIONS: {total_w_mm}mm x {total_h_mm}mm
RESOLUTION: {total_res_w} x {total_res_h} @ {target_fps}Hz
POWER: {total_power_w/1000:.2f} kW
WEIGHT: {total_weight_kg:.0f} kg
HEAT: {btu_hr:.0f} BTU/hr
PROCESSORS: {proc_str}
DATA PORTS: {total_ports_needed}x 1G
VIDEO INPUTS: {input_str}
"""
        if is_accuvision:
            summary_txt += f"IMAGE GENERATORS: {ig_str}\n"
        st.code(summary_txt, language="text")

# --- 4. MAIN DISPLAY ---
st.subheader(f"{selected_prod} ({selected_pitch}mm)")

# -- TOP METRICS --
m1, m2, m3, m4 = st.columns(4)

dim_val = f"{total_w_mm:,.0f} x {total_h_mm:,.0f} mm"
if use_imperial:
    dim_val += f"\n({total_w_mm/304.8:,.1f}' x {total_h_mm/304.8:,.1f}')"
m1.metric("Dimensions", dim_val)

m2.metric("Resolution", f"{total_res_w} x {total_res_h} px")
m3.metric("Max Power", f"{total_power_w/1000:.1f} kW")

w_val = f"{total_weight_kg:,.0f} kg"
if use_imperial:
    w_val += f" ({total_weight_kg*2.20462:,.0f} lbs)"
m4.metric("Total Weight", w_val)

# -- SECONDARY METRICS (Engineering) --
e1, e2, e3, e4 = st.columns(4)
e1.metric("Video Inputs", input_str)
e2.metric("Processors", proc_str, help=f"Logic: {reason_str}")
e3.metric("Data Ports", f"{total_ports_needed}x (1G)")

# Swap 4th Metric based on product type
if is_accuvision:
    e4.metric("Image Generators", ig_str)
else:
    e4.metric("Heat Output", f"{btu_hr:,.0f} BTU/hr")

# -- PROCESSOR UTILIZATION BAR --
st.caption(f"Processor Load ({chassis_load_pct:.1f}%)")
st.progress(min(chassis_load_pct/100, 1.0))

if is_curved:
    st.caption(f"**Curve Stats:** Radius: {curve_radius:,.0f}mm | Angle: {curve_angle:.1f}° | Footprint: {phys_w:,.0f}mm (W) x {phys_d:,.0f}mm (D)")

# --- 5. PLOTTING ---
plot_col1, plot_col2 = st.columns(2)

# --- Define Common Settings ---
person_h = 1750
panel_color = spec["Color"]

# --- FRONT VIEW (AX1) ---
with plot_col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5)) 
    ax1.set_title("FRONT VIEW (Unfolded)")
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Content Logic for Display
    face_col = 'none' if content_img_data is not None else panel_color
    
    rect = patches.Rectangle((0, 0), total_w_mm, total_h_mm, linewidth=1, edgecolor='white', facecolor=face_col, zorder=1)
    ax1.add_patch(rect)
    
    if content_img_data is not None:
        ax1.imshow(content_img_data, extent=[0, total_w_mm, 0, total_h_mm], zorder=0)

    for c in range(int(panels_w) + 1):
        x = c * spec["Width(mm)"]
        ax1.plot([x, x], [0, total_h_mm], color='white', linewidth=0.5, zorder=3)
    for r in range(int(panels_h) + 1):
        y = r * spec["Height(mm)"]
        ax1.plot([0, total_w_mm], [y, y], color='white', linewidth=0.5, zorder=3)

    p_x = total_w_mm + 500 
    if os.path.exists("person.png"):
        try:
            img = mpimg.imread("person.png")
            img_h_px, img_w_px = img.shape[:2]
            aspect_ratio = img_w_px / img_h_px
            target_h = person_h
            target_w = target_h * aspect_ratio
            ax1.imshow(img, extent=[p_x, p_x + target_w, 0, target_h], zorder=10)
        except:
            ax1.add_patch(patches.Rectangle((p_x, 0), 600, 1750, color="#ccc"))
    else:
        ax1.add_patch(patches.Rectangle((p_x, 0), 600, 1750, color="#888"))

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
        start_x = -(total_w_mm / 2)
        rect_top = patches.Rectangle((start_x, 0), total_w_mm, panel_thick, linewidth=1, edgecolor='black', facecolor=panel_color)
        ax2.add_patch(rect_top)
        person_y = 1000
    else:
        center_x = 0
        center_y = curve_radius 
        start_angle = 270 - (curve_angle / 2)
        current_a = math.radians(start_angle)
        angle_step = curve_angle / panels_w
        for i in range(int(panels_w)):
            x1 = center_x + curve_radius * math.cos(current_a)
            y1 = center_y + curve_radius * math.sin(current_a)
            x2 = center_x + curve_radius * math.cos(current_a + math.radians(angle_step))
            y2 = center_y + curve_radius * math.sin(current_a + math.radians(angle_step))
            r_out = curve_radius + panel_thick
            x3 = center_x + r_out * math.cos(current_a + math.radians(angle_step))
            y3 = center_y + r_out * math.sin(current_a + math.radians(angle_step))
            x4 = center_x + r_out * math.cos(current_a)
            y4 = center_y + r_out * math.sin(current_a)
            poly = patches.Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], closed=True, edgecolor='white', facecolor=panel_color)
            ax2.add_patch(poly)
            current_a += math.radians(angle_step)
        person_y = min(curve_radius, phys_d + 3000) 

    if os.path.exists("top_person.png"):
        try:
            img_top = mpimg.imread("top_person.png")
            t_h_px, t_w_px = img_top.shape[:2]
            t_aspect = t_w_px / t_h_px
            target_top_w = 600
            target_top_h = target_top_w / t_aspect
            extent = [
                -target_top_w / 2, 
                target_top_w / 2, 
                person_y - target_top_h / 2, 
                person_y + target_top_h / 2
            ]
            ax2.imshow(img_top, extent=extent)
        except:
            ax2.add_patch(patches.Circle((0, person_y), 200, color='#333'))
    else:
        ax2.add_patch(patches.Circle((0, person_y), 200, color='#333'))

    ax2.autoscale_view()
    ax2.margins(0.1)
    st.pyplot(fig2, use_container_width=True)
