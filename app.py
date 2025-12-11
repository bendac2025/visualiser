import streamlit as st
import pandas as pd
import math
import os
import io
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import urllib.parse

# --- MATPLOTLIB HEADLESS FIX ---
import matplotlib
matplotlib.use("Agg") # <--- THIS IS THE KEY FIX
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# --- 1. CONFIG & DATABASE ---
st.set_page_config(page_title="Bendac Visualiser", layout="wide")

def load_data():
    csv_file = "bendac_database.csv"
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
            '#4a90e2', '#4a90e2', '#50e3c2', '#50e3c2', '#50e3c2', 
            '#9013fe', '#9013fe', '#9013fe', '#f5a623', '#f5a623', '#f5a623'
        ]
    }

    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'Weight(kg)' not in df.columns: df['Weight(kg)'] = 10.0
            if 'MaxFPS(Hz)' not in df.columns: df['MaxFPS(Hz)'] = 60
            return df
        except: pass
    return pd.DataFrame(backup_data)

NOVASTAR_DB = {
    "Novastar MCTRL660 Pro": { "type": "fixed", "capacity_60": 2300000, "ports": 6 },
    "Novastar VX1000": { "type": "fixed", "capacity_60": 6500000, "ports": 10 },
    "Novastar MCTRL4K": { "type": "fixed", "capacity_60": 8800000, "ports": 16 },
    "Novastar MX40 Pro": { "type": "fixed", "capacity_60": 8800000, "ports": 20 },
    "Novastar MX2000 Pro (40G Cards)": { "type": "modular", "capacity_60": 35380000, "card_capacity_60": 26200000, "slots": 2 },
    "Novastar MX6000 Pro (40G Cards)": { "type": "modular", "capacity_60": 141000000, "card_capacity_60": 26200000, "slots": 8 }
}

if 'db' not in st.session_state:
    st.session_state.db = load_data()
df = st.session_state.db

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("logo.png"): st.image("logo.png", width=200)
    else: st.title("BENDAC")
    st.caption("VISUALISER TOOL")
    
    use_imperial = st.checkbox("Show Imperial Units (ft/lbs)", value=False)
    
    st.header("1. Configuration")
    unique_products = list(df["Product Name"].unique())
    idx_p = unique_products.index("Bendac Krystl Max") if "Bendac Krystl Max" in unique_products else 0
    selected_prod = st.selectbox("Product Series", unique_products, index=idx_p)
    
    prod_rows = df[df["Product Name"] == selected_prod]
    available_pitches = sorted(prod_rows["Pitch(mm)"].unique())
    idx_pt = available_pitches.index(1.9) if 1.9 in available_pitches else 0
    selected_pitch = st.selectbox("Pixel Pitch (mm)", available_pitches, index=idx_pt)
    spec = prod_rows[prod_rows["Pitch(mm)"].unique() == selected_pitch].iloc[0]
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1: panels_w = st.number_input("Panels Wide", 1, value=1)
    with c2: panels_h = st.number_input("Panels High", 1, value=1)
    
    st.divider()
    st.subheader("2. Curve Geometry")
    curve_radius = st.number_input("Radius (mm)", 0.0, step=100.0)
    curve_angle = st.number_input("Total Angle (deg)", 0.0, step=1.0)
    
    is_curved = False
    if curve_radius > 0 and curve_angle > 0:
        is_curved = True
        panels_w = max(1, round((2 * math.pi * curve_radius * (curve_angle / 360)) / spec["Width(mm)"]))
        st.info(f"Auto-set Width to **{panels_w} panels** based on geometry.")
    elif curve_radius > 0:
        is_curved = True
        curve_angle = math.degrees((panels_w * spec["Width(mm)"]) / curve_radius)
    elif curve_angle > 0:
        is_curved = True
        curve_radius = (panels_w * spec["Width(mm)"]) / math.radians(curve_angle)

    st.subheader("3. Processing")
    proc_model = st.selectbox("Processor Model", list(NOVASTAR_DB.keys()))
    fps_opts = [60, 120, 240] if "AccuVision" in selected_prod else [60]
    target_fps = st.selectbox("Frame Rate (Hz)", fps_opts)
    
    st.subheader("4. Electrical")
    voltage = st.selectbox("Voltage", [110, 208, 230, 240], index=2) 
    phase_type = st.radio("Phase", ["Single Phase", "3-Phase"], index=0)

    st.subheader("5. Visual Context")
    content_file = st.file_uploader("Upload Screen Content", type=["png", "jpg", "jpeg"])
    content_img_data = None
    if content_file:
        try: content_img_data = np.array(Image.open(content_file))
        except: pass

    # --- CALCULATIONS ---
    total_w_mm = panels_w * spec["Width(mm)"]
    total_h_mm = panels_h * spec["Height(mm)"]
    total_res_w = panels_w * spec["ResW(px)"]
    total_res_h = panels_h * spec["ResH(px)"]
    total_pixels = total_res_w * total_res_h
    total_power_w = panels_w * panels_h * spec["Power(W)"]
    total_weight_kg = panels_w * panels_h * spec["Weight(kg)"]
    btu_hr = total_power_w * 3.412142
    
    # Electrical Logic
    pdu_rec = ""
    if phase_type == "Single Phase":
        total_amps = total_power_w / voltage
        elec_str = f"{total_amps:.1f}A @ {voltage}V (1-Phase)"
        if total_amps <= 13: pdu_rec = "13A Standard / 16A CEE"
        elif total_amps <= 16: pdu_rec = "1x 16A 1-Phase CEE"
        elif total_amps <= 32: pdu_rec = "1x 32A 1-Phase CEE"
        elif total_amps <= 63: pdu_rec = "1x 63A 1-Phase CEE"
        else: pdu_rec = "Multiple Feeds / Switch to 3-Phase"
    else:
        total_amps = total_power_w / (voltage * 1.732)
        elec_str = f"{total_amps:.1f}A/Line @ {voltage}V (3-Phase)"
        if total_amps <= 32: pdu_rec = "1x 32A 3-Phase CEE"
        elif total_amps <= 63: pdu_rec = "1x 63A 3-Phase CEE"
        elif total_amps <= 125: pdu_rec = "1x 125A 3-Phase CEE"
        else: pdu_rec = "Powerlock / Multiple 125A Feeds"

    fps_scale = 60 / target_fps
    total_ports_needed = math.ceil(total_pixels / (655360 * fps_scale))
    proc_data = NOVASTAR_DB[proc_model]
    
    if proc_data["type"] == "fixed":
        unit_cap = proc_data["capacity_60"] * fps_scale
        req_pix = math.ceil(total_pixels / unit_cap)
        req_port = math.ceil(total_ports_needed / proc_data["ports"])
        total_procs = max(req_pix, req_port)
        load_pct = (total_pixels / (total_procs * unit_cap)) * 100
        proc_str = f"{total_procs}x {proc_model}"
        reason_str = "Ports" if req_port > req_pix else "Pixels"
    else:
        card_cap = proc_data["card_capacity_60"] * fps_scale
        req_cards = math.ceil(total_pixels / card_cap)
        chassis_cap = proc_data["capacity_60"] * fps_scale
        req_chassis_slots = math.ceil(req_cards / proc_data["slots"])
        req_chassis_bw = math.ceil(total_pixels / chassis_cap)
        total_chassis = max(req_chassis_slots, req_chassis_bw)
        load_pct = (total_pixels / (total_chassis * chassis_cap)) * 100 if total_chassis > 0 else 0
        proc_str = f"{total_chassis}x {proc_model.split('(')[0]} ({req_cards}x 40G Cards)"
        reason_str = "Bandwidth" if req_chassis_bw > req_chassis_slots else "Slots"

    if chassis_load_pct := load_pct:
        if load_pct > 95: st.warning(f"⚠️ High Chassis Load: {load_pct:.1f}%")

    video_inputs = math.ceil(total_pixels / (8294400 * fps_scale))
    input_str = f"{video_inputs}x 4K Inputs"
    is_accuvision = "AccuVision" in selected_prod
    ig_str = f"{math.ceil(video_inputs / 4)}x Image Generators" if is_accuvision else ""

    if is_curved and curve_radius > 0:
        rad = math.radians(curve_angle)
        phys_w = 2 * curve_radius * math.sin(rad/2)
        phys_d = curve_radius * (1 - math.cos(rad/2))
    else:
        phys_w, phys_d = total_w_mm, 0

    # --- PDF GENERATOR ---
    def draw_dim_line(ax, start, end, text, offset_dist=1000, color='black'):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length == 0: return
        nx, ny = -dy / length, dx / length
        ox1, oy1 = x1 + nx * offset_dist, y1 + ny * offset_dist
        ox2, oy2 = x2 + nx * offset_dist, y2 + ny * offset_dist
        
        ax.plot([x1, ox1], [y1, oy1], color=color, linestyle=':', linewidth=0.5)
        ax.plot([x2, ox2], [y2, oy2], color=color, linestyle=':', linewidth=0.5)
        ax.annotate("", xy=(ox1, oy1), xytext=(ox2, oy2), arrowprops=dict(arrowstyle="<->", color=color))
        
        text_push = 400 if offset_dist >= 0 else -400
        tx, ty = (ox1 + ox2)/2 + nx*text_push, (oy1 + oy2)/2 + ny*text_push
        angle = math.degrees(math.atan2(dy, dx))
        if 90 < angle <= 270 or -270 < angle <= -90: angle += 180
        ax.text(tx, ty, text, ha='center', va='center', rotation=angle, fontsize=7, bbox=dict(facecolor='white', edgecolor='none', pad=1))

    def create_pdf():
        panel_thick = 100
        fig = plt.figure(figsize=(8.27, 11.69))
        
        if os.path.exists("logo.png"):
            ax_l = fig.add_axes([0.35, 0.90, 0.3, 0.08])
            ax_l.imshow(mpimg.imread("logo.png"))
            ax_l.axis('off')
        
        fig.text(0.5, 0.86, "TECHNICAL SPECIFICATION", ha='center', fontsize=16, weight='bold')
        
        ax_t = fig.add_axes([0.1, 0.55, 0.8, 0.20])
        ax_t.axis('off')
        
        d_s = f"{total_w_mm:,.0f} mm (W) x {total_h_mm:,.0f} mm (H)"
        w_s = f"{total_weight_kg:,.0f} kg"
        if use_imperial:
            d_s += f"\n{total_w_mm/304.8:,.1f} ft (W) x {total_h_mm/304.8:,.1f} ft (H)"
            w_s += f" ({total_weight_kg*2.20462:,.0f} lbs)"

        rows = [
            ["Model Series", f"{selected_prod}"],
            ["Pixel Pitch", f"{selected_pitch} mm"],
            ["Configuration", f"{panels_w} (W) x {panels_h} (H) Panels"],
            ["Dimensions", d_s],
            ["Resolution", f"{total_res_w} px (W) x {total_res_h} px (H) @ {target_fps}Hz"],
            ["Max Power / Heat", f"{total_power_w/1000:.1f} kW  /  {btu_hr:,.0f} BTU/hr"],
            ["Electrical", elec_str],
            ["Recommended Supply", pdu_rec],
            ["Total Weight", w_s],
            ["Processing", f"{proc_str}\n({reason_str} - Load: {load_pct:.1f}%)"],
            ["Data Capacity", f"{total_ports_needed}x 1G Ports required"],
            ["Video Inputs", input_str]
        ]
        if is_accuvision: rows.append(["Image Generators", ig_str])
        if is_curved:
            rows.append(["Curve", f"R: {curve_radius:,.0f}mm | {curve_angle:.1f}°"])
            rows.append(["Install Zone (+600mm)", f"{phys_w+1200:,.0f} mm (W) x {phys_d+1200:,.0f} mm (D)"])

        tbl = ax_t.table(cellText=rows, loc='center', cellLoc='left', colWidths=[0.3, 0.7])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        
        for r, row_d in enumerate(rows):
            if "\n" in row_d[1]:
                tbl[r, 0].set_height(tbl[r, 0].get_height() * 1.8)
                tbl[r, 1].set_height(tbl[r, 1].get_height() * 1.8)
        
        for (i, j), c in tbl.get_celld().items():
            if j == 0: c.set_text_props(weight='bold')
            c.set_edgecolor('#dddddd')

        ax_f = fig.add_axes([0.1, 0.28, 0.8, 0.22])
        ax_f.set_title("FRONT VIEW")
        ax_f.set_aspect('equal')
        ax_f.axis('off')
        
        fc = 'none' if content_img_data is not None else spec["Color"]
        ax_f.add_patch(patches.Rectangle((0, 0), total_w_mm, total_h_mm, fc=fc, ec='black', lw=1))
        if content_img_data is not None:
            ax_f.imshow(content_img_data, extent=[0, total_w_mm, 0, total_h_mm], zorder=0)
        
        if panels_w <= 100:
            for i in range(int(panels_w)+1): ax_f.plot([i*spec["Width(mm)"]]*2, [0, total_h_mm], 'k-', lw=0.1)
            for i in range(int(panels_h)+1): ax_f.plot([0, total_w_mm], [i*spec["Height(mm)"]]*2, 'k-', lw=0.1)
            
        draw_dim_line(ax_f, (0, 0), (total_w_mm, 0), f"{total_w_mm:,.0f}mm", offset_dist=-300)
        draw_dim_line(ax_f, (0, 0), (0, total_h_mm), f"{total_h_mm:,.0f}mm", offset_dist=-1500)

        px = total_w_mm + 500
        if os.path.exists("person.png"):
            im = mpimg.imread("person.png")
            ar = im.shape[1]/im.shape[0]
            ax_f.imshow(im, extent=[px, px+(1750*ar), 0, 1750])
        else: ax_f.add_patch(patches.Rectangle((px, 0), 600, 1750, color='#ccc'))
        
        ax_f.autoscale_view()

        ax_top = fig.add_axes([0.1, 0.05, 0.8, 0.22])
        ax_top.set_title("TOP VIEW")
        ax_top.set_aspect('equal')
        ax_top.axis('off')
        
        if not is_curved:
            sx = -total_w_mm/2
            ax_top.add_patch(patches.Rectangle((sx, 0), total_w_mm, 100, fc=spec["Color"], ec='black'))
            draw_dim_line(ax_top, (sx, 0), (sx+total_w_mm, 0), f"{total_w_mm:,.0f}mm", offset_dist=-500)
            py = 1000
        else:
            cx, cy = 0, curve_radius
            sa = math.radians(270 - curve_angle/2)
            step = math.radians(curve_angle) / panels_w
            curr = sa
            for _ in range(int(panels_w)):
                p1 = (cx + curve_radius*math.cos(curr), cy + curve_radius*math.sin(curr))
                p2 = (cx + curve_radius*math.cos(curr+step), cy + curve_radius*math.sin(curr+step))
                p3 = (cx + (curve_radius+100)*math.cos(curr+step), cy + (curve_radius+100)*math.sin(curr+step))
                p4 = (cx + (curve_radius+100)*math.cos(curr), cy + (curve_radius+100)*math.sin(curr))
                ax_top.add_patch(patches.Polygon([p1,p2,p3,p4], fc=spec["Color"], ec='black'))
                curr += step
            
            c_half = phys_w/2
            bottom_y = cy + curve_radius*math.sin(math.radians(270)) 
            draw_dim_line(ax_top, (-c_half, bottom_y), (c_half, bottom_y), f"Chord: {phys_w:,.0f}mm", offset_dist=-500)
            py = min(curve_radius, phys_d + 3000)

        if os.path.exists("top_person.png"):
            im = mpimg.imread("top_person.png")
            ar = im.shape[1]/im.shape[0]
            w = 600
            ax_top.imshow(im, extent=[-w/2, w/2, py-(w/ar)/2, py+(w/ar)/2])
        else: ax_top.add_patch(patches.Circle((0, py), 200, color='#333'))
        
        ax_top.autoscale_view()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="pdf", bbox_inches='tight')
        buf.seek(0)
        return buf

    st.divider()
    st.download_button("📄 Download Spec Sheet (PDF)", create_pdf(), f"Bendac_Spec_{selected_prod}.pdf", "application/pdf")
    
    # --- LEAD GEN: EMAIL BUTTON ---
    st.divider()
    
    email_subject = f"Quote Request: {selected_prod} ({panels_w}x{panels_h})"
    email_body = f"""Hi Bendac Team,

I would like a quote for the following configuration created in the Visualiser:

Product: {selected_prod} {selected_pitch}mm
Configuration: {panels_w} Wide x {panels_h} High
Dimensions: {total_w_mm}mm x {total_h_mm}mm
Curve: {'Yes' if is_curved else 'No'}
Processing: {proc_model}
Electrical: {elec_str} ({pdu_rec})

Please contact me to discuss.
"""
    safe_subject = urllib.parse.quote(email_subject)
    safe_body = urllib.parse.quote(email_body)
    mailto_link = f"mailto:info@bendac.tech?subject={safe_subject}&body={safe_body}"
    
    st.link_button("📧 Request Quote / Enquire", mailto_link, type="primary")

    # --- DISCLAIMER ---
    st.caption("---")
    st.caption("📝 **Disclaimer:** All dimensions, weights, and power figures are estimates for visualization purposes only. Final engineering drawings should be requested from the Bendac technical team before construction.")

# --- MAIN DISPLAY ---
st.subheader(f"{selected_prod} ({selected_pitch}mm)")

m1, m2, m3, m4 = st.columns(4)
dim_s = f"{total_w_mm:,.0f} x {total_h_mm:,.0f} mm"
if use_imperial: dim_s += f"\n({total_w_mm/304.8:,.1f}' x {total_h_mm/304.8:,.1f}')"
m1.metric("Dimensions", dim_s)
m2.metric("Resolution", f"{total_res_w} x {total_res_h} px")
m3.metric("Max Power", f"{total_power_w/1000:.1f} kW")
w_s = f"{total_weight_kg:,.0f} kg"
if use_imperial: w_s += f" ({total_weight_kg*2.20462:,.0f} lbs)"
m4.metric("Total Weight", w_s)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Video Inputs", input_str)
e2.metric("Processors", proc_str, help=reason_str)
e3.metric("Data Ports", f"{total_ports_needed}x (1G)")
e4.metric("Electrical", f"{elec_str} ({pdu_rec})")

st.caption(f"Processor Load ({chassis_load_pct:.1f}%)")
st.progress(min(chassis_load_pct/100, 1.0))

tab2d, tab3d = st.tabs(["2D Engineering Views", "3D Interactive Model"])

with tab2d:
    pc1, pc2 = st.columns(2)
    with pc1:
        f1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.set_title("FRONT VIEW")
        ax1.axis('off')
        ax1.set_aspect('equal')
        fc = 'none' if content_img_data is not None else spec["Color"]
        ax1.add_patch(patches.Rectangle((0, 0), total_w_mm, total_h_mm, fc=fc, ec='white', lw=1))
        if content_img_data is not None: ax1.imshow(content_img_data, extent=[0, total_w_mm, 0, total_h_mm])
        
        for i in range(int(panels_w)+1): ax1.plot([i*spec["Width(mm)"]]*2, [0, total_h_mm], 'w-', lw=0.2)
        for i in range(int(panels_h)+1): ax1.plot([0, total_w_mm], [i*spec["Height(mm)"]]*2, 'w-', lw=0.2)
        
        px = total_w_mm + 500
        if os.path.exists("person.png"):
            im = mpimg.imread("person.png")
            ar = im.shape[1]/im.shape[0]
            ax1.imshow(im, extent=[px, px+(1750*ar), 0, 1750])
        else: ax1.add_patch(patches.Rectangle((px, 0), 600, 1750, color='#888'))
        
        ax1.autoscale_view()
        st.pyplot(f1)

    with pc2:
        f2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.set_title("TOP VIEW")
        ax2.axis('off')
        ax2.set_aspect('equal')
        
        if not is_curved:
            sx = -total_w_mm/2
            ax2.add_patch(patches.Rectangle((sx, 0), total_w_mm, 100, fc=spec["Color"], ec='black'))
            py = 1000
        else:
            cx, cy = 0, curve_radius
            sa = math.radians(270 - curve_angle/2)
            step = math.radians(curve_angle) / panels_w
            curr = sa
            for _ in range(int(panels_w)):
                p1 = (cx + curve_radius*math.cos(curr), cy + curve_radius*math.sin(curr))
                p2 = (cx + curve_radius*math.cos(curr+step), cy + curve_radius*math.sin(curr+step))
                p3 = (cx + (curve_radius+100)*math.cos(curr+step), cy + (curve_radius+100)*math.sin(curr+step))
                p4 = (cx + (curve_radius+100)*math.cos(curr), cy + (curve_radius+100)*math.sin(curr))
                ax2.add_patch(patches.Polygon([p1,p2,p3,p4], fc=spec["Color"], ec='white'))
                curr += step
            py = min(curve_radius, phys_d + 3000)

        if os.path.exists("top_person.png"):
            im = mpimg.imread("top_person.png")
            ar = im.shape[1]/im.shape[0]
            w = 600
            ax2.imshow(im, extent=[-w/2, w/2, py-(w/ar)/2, py+(w/ar)/2])
        else: ax2.add_patch(patches.Circle((0, py), 200, color='#333'))
        
        ax2.autoscale_view()
        st.pyplot(f2)

with tab3d:
    MAX_POINTS = 100000 
    aspect = total_w_mm / max(1, total_h_mm)
    resolution_z = int(math.sqrt(MAX_POINTS / aspect))
    resolution_x = int(resolution_z * aspect)
    resolution_z = max(10, resolution_z)
    resolution_x = max(10, resolution_x)
    
    if is_curved:
        theta = np.linspace(math.radians(270 - curve_angle/2), math.radians(270 + curve_angle/2), resolution_x)
        x = curve_radius * np.cos(theta)
        y = (curve_radius * np.sin(theta)) + curve_radius
        y = y - (phys_d / 2) 
        
        x_rear = (curve_radius + 50) * np.cos(theta)
        y_rear = ((curve_radius + 50) * np.sin(theta)) + curve_radius - (phys_d / 2)
    else:
        x = np.linspace(-total_w_mm/2, total_w_mm/2, resolution_x)
        y = np.zeros(resolution_x)
        x_rear = x
        y_rear = y + 50

    z = np.linspace(0, total_h_mm, resolution_z)
    X, Z = np.meshgrid(x, z)
    Y = np.tile(y, (resolution_z, 1))
    
    Xr, Zr = np.meshgrid(x_rear, z)
    Yr = np.tile(y_rear, (resolution_z, 1))

    fig3d = go.Figure()

    if content_img_data is not None:
        try:
            pil_img = Image.fromarray(content_img_data).convert("RGB")
            pil_img = pil_img.resize((resolution_x, resolution_z))
            
            img_arr = np.array(pil_img, dtype=np.uint8)
            img_arr = np.flipud(img_arr)
            img_arr = np.fliplr(img_arr) 
            
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            
            color_flat = [f'rgb({r},{g},{b})' for r,g,b in img_arr.reshape(-1, 3)]
            
            fig3d.add_trace(go.Scatter3d(
                x=x_flat, y=y_flat, z=z_flat,
                mode='markers',
                marker=dict(size=3, color=color_flat, symbol='square'),
                name='LED Pixels'
            ))
        except Exception as e:
            st.sidebar.error(f"3D Render Error: {e}")
            c_hex = spec["Color"]
            fig3d.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, c_hex], [1, c_hex]], showscale=False, name='Front'))
    else:
        c_hex = spec["Color"]
        fig3d.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, c_hex], [1, c_hex]], showscale=False, name='Front'))

    fig3d.add_trace(go.Surface(x=Xr, y=Yr, z=Zr, colorscale=[[0, '#4a90e2'], [1, '#4a90e2']], showscale=False, opacity=1.0, name='Back'))

    if is_curved:
        person_y_pos = curve_radius - (phys_d / 2)
    else:
        person_y_pos = 2000

    fig3d.add_trace(go.Scatter3d(
        x=[0, 0], y=[person_y_pos, person_y_pos], z=[0, 1750],
        mode='lines', line=dict(color='red', width=8), name='Viewer'
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[0], y=[person_y_pos], z=[1750],
        mode='markers', marker=dict(size=6, color='red'), showlegend=False
    ))
    
    fig3d.update_layout(
        scene = dict(
            xaxis_title='Width (mm)',
            yaxis_title='Depth (mm)',
            zaxis_title='Height (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=2.5, z=0.5)) 
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)
