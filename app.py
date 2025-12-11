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
import plotly.graph_objects as go

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
    spec = prod_rows[prod_rows["Pitch(mm)"] == selected_pitch].iloc[0]
    
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
    
    # --- ELECTRICAL CALC ---
    st.subheader("4. Electrical")
    voltage = st.selectbox("Voltage", [110, 208, 230, 240], index=2) # Default 230V
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
    
    # Electrical Math
    # Simple engineering formula (PF=1 for max draw estimation)
    if phase_type == "Single Phase":
        total_amps = total_power_w / voltage
        elec_str = f"{total_amps:.1f}A @ {voltage}V (1-Phase)"
    else:
        # 3-Phase formula: Watts / (Volts * sqrt(3))
        total_amps = total_power_w / (voltage * 1.732)
        elec_str = f"{total_amps:.1f}A/Line @ {voltage}V (3-Phase)"

    # Processor Math
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

    # --- CAD DIMENSION HELPER ---
    def draw_dim_line(ax, start, end, text, offset=0, color='black'):
        """Draws a CAD style dimension line with arrows and text"""
        # Midpoint
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Determine angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Offset vector (perpendicular)
        # Normalize
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0: return
        perp_x = -dy / length * offset
        perp_y = dx / length * offset
        
        # New points
        p1 = (start[0] + perp_x, start[1] + perp_y)
        p2 = (end[0] + perp_x, end[1] + perp_y)
        
        # Draw Extension lines
        ax.plot([start[0], p1[0]], [start[1], p1[1]], color=color, linewidth=0.5, linestyle=':')
        ax.plot([end[0], p2[0]], [end[1], p2[1]], color=color, linewidth=0.5, linestyle=':')
        
        # Draw Arrow Line
        ax.annotate('', xy=p1, xytext=p2, arrowprops=dict(arrowstyle='<->', color=color, lw=0.8))
        
        # Draw Text
        t_x = (p1[0] + p2[0]) / 2
        t_y = (p1[1] + p2[1]) / 2
        # Slight bump for text
        text_offset_y = 50 if abs(dx) > abs(dy) else 0
        text_offset_x = 50 if abs(dy) >= abs(dx) else 0
        
        ax.text(t_x + text_offset_x, t_y + text_offset_y, text, ha='center', va='center', fontsize=8, color=color, backgroundcolor='white')

    def create_pdf():
        panel_thick = 100
        fig = plt.figure(figsize=(8.27, 11.69))
        
        if os.path.exists("logo.png"):
            ax_l = fig.add_axes([0.35, 0.89, 0.3, 0.08])
            ax_l.imshow(mpimg.imread("logo.png"))
            ax_l.axis('off')
        
        fig.text(0.5, 0.85, "TECHNICAL SPECIFICATION", ha='center', fontsize=16, weight='bold')
        
        # Table
        ax_t = fig.add_axes([0.1, 0.58, 0.8, 0.20])
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
        
        # Resize Rows
        for r, row_d in enumerate(rows):
            if "\n" in row_d[1]:
                tbl[r, 0].set_height(tbl[r, 0].get_height() * 1.8)
                tbl[r, 1].set_height(tbl[r, 1].get_height() * 1.8)
        
        for (i, j), c in tbl.get_celld().items():
            if j == 0: c.set_text_props(weight='bold')
            c.set_edgecolor('#dddddd')

        # Front View
        ax_f = fig.add_axes([0.1, 0.30, 0.8, 0.23])
        ax_f.set_title("FRONT VIEW")
        ax_f.set_aspect('equal')
        ax_f.axis('off')
        
        fc = 'none' if content_img_data is not None else spec["Color"]
        ax_f.add_patch(patches.Rectangle((0, 0), total_w_mm, total_h_mm, fc=fc, ec='black', lw=1))
        if content_img_data is not None:
            ax_f.imshow(content_img_data, extent=[0, total_w_mm, 0, total_h_mm], zorder=0)
        
        # Grid
        if panels_w <= 100:
            for i in range(int(panels_w)+1): ax_f.plot([i*spec["Width(mm)"]]*2, [0, total_h_mm], 'k-', lw=0.1)
            for i in range(int(panels_h)+1): ax_f.plot([0, total_w_mm], [i*spec["Height(mm)"]]*2, 'k-', lw=0.1)
            
        # DIMS - Width & Height
        draw_dim_line(ax_f, (0, -200), (total_w_mm, -200), f"{total_w_mm:,.0f}mm", offset=0)
        draw_dim_line(ax_f, (-200, 0), (-200, total_h_mm), f"{total_h_mm:,.0f}mm", offset=0)

        # Person
        px = total_w_mm + 500
        if os.path.exists("person.png"):
            im = mpimg.imread("person.png")
            ar = im.shape[1]/im.shape[0]
            ax_f.imshow(im, extent=[px, px+(1750*ar), 0, 1750])
        else: ax_f.add_patch(patches.Rectangle((px, 0), 600, 1750, color='#ccc'))
        
        ax_f.autoscale_view()

        # Top View
        ax_top = fig.add_axes([0.1, 0.05, 0.8, 0.22])
        ax_top.set_title("TOP VIEW")
        ax_top.set_aspect('equal')
        ax_top.axis('off')
        
        if not is_curved:
            sx = -total_w_mm/2
            ax_top.add_patch(patches.Rectangle((sx, 0), total_w_mm, 100, fc=spec["Color"], ec='black'))
            # DIMS
            draw_dim_line(ax_top, (sx, -200), (sx+total_w_mm, -200), f"{total_w_mm:,.0f}mm")
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
            
            # Curve Dims (Chord Width)
            c_half = phys_w/2
            draw_dim_line(ax_top, (-c_half, -200), (c_half, -200), f"Chord: {phys_w:,.0f}mm")
            # Sagitta
            # Line from chord center to arc top
            sagitta_y = phys_d
            ax_top.plot([0,0], [0, sagitta_y], 'k:', lw=0.5)
            ax_top.text(50, sagitta_y/2, f"D: {phys_d:,.0f}mm", fontsize=6)
            
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
    
    st.divider()
    with st.expander("Show Text Summary"):
        t = f"""
PRODUCT: {selected_prod} ({selected_pitch}mm)
CONFIG: {panels_w} x {panels_h}
DIMS: {total_w_mm}x{total_h_mm}mm
RES: {total_res_w}x{total_res_h} @ {target_fps}Hz
POWER: {total_power_w/1000:.1f}kW
ELECTRICAL: {elec_str}
WEIGHT: {total_weight_kg:.0f}kg
HEAT: {btu_hr:.0f} BTU/hr
PROCESSORS: {proc_str}
VIDEO INPUTS: {input_str}
"""
        if is_accuvision: t += f"IGs: {ig_str}"
        st.code(t)

# --- TABS FOR 2D / 3D ---
st.subheader(f"{selected_prod} ({selected_pitch}mm)")

# Top Metrics
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
e4.metric("Electrical", elec_str)

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
        
        # Grid
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
    # 3D MODEL LOGIC
    # Generate points for the screen surface
    theta = np.linspace(math.radians(270 - curve_angle/2), math.radians(270 + curve_angle/2), 50) if is_curved else np.zeros(50)
    
    if is_curved:
        # Cylinder coordinates (Swapped Y/Z for plotting orientation)
        # X = Width, Z = Height, Y = Depth
        # Center of curvature is at Y = curve_radius. Screen face is at (0,0) in flat mode.
        # Let's map: x = r*cos(t), y = r*sin(t).
        # We want screen center at (0,0).
        x = curve_radius * np.cos(theta)
        y = curve_radius * np.sin(theta)
        # Shift so the chord center is near 0
        y = y - curve_radius # Move arc to origin
    else:
        x = np.linspace(-total_w_mm/2, total_w_mm/2, 50)
        y = np.zeros(50) # Flat
    
    # Create Mesh
    z = np.linspace(0, total_h_mm, 10)
    X, Z = np.meshgrid(x, z)
    # Y needs to be tiled
    Y = np.tile(y, (10, 1)).reshape(10, 50)
    
    fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=False)])
    
    # Add a "Person" reference (Cylinder)
    fig3d.add_trace(go.Scatter3d(x=[0,0], y=[2000], z=[0, 1750], mode='lines', line=dict(color='red', width=10), name='Person'))
    
    fig3d.update_layout(
        scene = dict(
            xaxis_title='Width (mm)',
            yaxis_title='Depth (mm)',
            zaxis_title='Height (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)
