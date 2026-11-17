import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_banner():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=180)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 540)
    ax.axis('off')

    # Top border glowing accent
    ax.plot([0, 1400], [538, 538], color='#38bdf8', lw=2.5, alpha=0.9)

    # Subtle ambient grid/depth curves
    for y_offset in [140, 200, 280]:
        ax.plot([0, 350, 700, 1050, 1400], 
                [y_offset, y_offset + 30, y_offset - 20, y_offset + 25, y_offset], 
                color='#1e293b', lw=1, alpha=0.5)

    # 1. Live Deployment Pill Badge
    badge_x, badge_y, badge_w, badge_h = 60, 460, 240, 34
    badge = patches.FancyBboxPatch(
        (badge_x, badge_y), badge_w, badge_h,
        boxstyle="round,pad=0.2",
        facecolor="#0369a1", edgecolor="#38bdf8",
        alpha=0.35, lw=1.2
    )
    ax.add_patch(badge)
    ax.plot(badge_x + 18, badge_y + 17, 'o', color='#38bdf8', markersize=5.5)
    ax.text(badge_x + 32, badge_y + 11, 'LIVE PRODUCTION PLATFORM',
            color='#e0f2fe', fontsize=9.5, weight='bold', fontfamily='Segoe UI')

    # 2. Hero Typography
    ax.text(60, 385, 'Project Varuna',
            color='#f8fafc', fontsize=34, weight='bold', fontfamily='Georgia')
    ax.text(60, 340, 'Hobli-Level Rainfall Deviation & Drought-Risk Prediction for Karnataka',
            color='#94a3b8', fontsize=14, weight='normal', fontfamily='Segoe UI')
    ax.text(60, 305, 'Official KSNDMC Panel: 2020–2025  •  850 Canonical Stations  •  31 Districts  •  Machine Learning Pipeline',
            color='#64748b', fontsize=11, fontfamily='Segoe UI')

    # 3. Action CTA Button (generous width: 480 to fit text comfortably without any overflow or emoji issues)
    btn_x, btn_y, btn_w, btn_h = 60, 215, 480, 50
    btn = patches.FancyBboxPatch(
        (btn_x, btn_y), btn_w, btn_h,
        boxstyle="round,pad=0.25",
        facecolor="#0284c7", edgecolor="#38bdf8",
        lw=1.5
    )
    ax.add_patch(btn)
    # Arrow icon drawn cleanly as a patch triangle
    triangle = patches.Polygon(
        [[btn_x + 22, btn_y + 32], [btn_x + 34, btn_y + 25], [btn_x + 22, btn_y + 18]],
        closed=True, facecolor='#ffffff', edgecolor='none'
    )
    ax.add_patch(triangle)
    # Centered CTA text
    ax.text(btn_x + 48, btn_y + 17, 'Launch Live App  •  varuna-ml.streamlit.app  →',
            color='#ffffff', fontsize=13, weight='bold', fontfamily='Segoe UI')

    # Secondary GitHub / Open Source Tag
    sec_x, sec_y, sec_w, sec_h = 560, 215, 260, 50
    sec = patches.FancyBboxPatch(
        (sec_x, sec_y), sec_w, sec_h,
        boxstyle="round,pad=0.25",
        facecolor="#1e293b", edgecolor="#475569",
        lw=1.2
    )
    ax.add_patch(sec)
    ax.text(sec_x + 20, sec_y + 17, 'Open Access  •  ODC-BY 1.0 License',
            color='#cbd5e1', fontsize=11.5, fontfamily='Segoe UI')

    # 4. Four Stat Metric Cards
    stats = [
        ('850 Stations', 'Canonical Panel (31 Dists)'),
        ('5 Seasons', '2020–2025 Panel (gap=2)'),
        ('4 Regimes', 'Longitudinal K-Means Clusters'),
        ('79.65% Benchmark', 'Honest Baseline Evaluation')
    ]
    card_w = 295
    card_h = 88
    for i, (title, sub) in enumerate(stats):
        cx = 60 + i * 325
        cy = 55
        card = patches.FancyBboxPatch(
            (cx, cy), card_w, card_h,
            boxstyle="round,pad=0.25",
            facecolor="#1e293b", edgecolor="#334155",
            lw=1
        )
        ax.add_patch(card)
        ax.text(cx + 20, cy + 48, title,
                color='#38bdf8', fontsize=15, weight='bold', fontfamily='Segoe UI')
        ax.text(cx + 20, cy + 20, sub,
                color='#94a3b8', fontsize=10.5, fontfamily='Segoe UI')

    png_path = os.path.join(out_dir, "app_banner.png")
    jpg_path = os.path.join(out_dir, "app_banner.jpg")

    plt.tight_layout(pad=0)
    plt.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    # Save high quality JPEG version
    with Image.open(png_path) as img:
        img.convert("RGB").save(jpg_path, "JPEG", quality=95, optimize=True)

    print(f"Generated {png_path} ({os.path.getsize(png_path)} bytes)")
    print(f"Generated {jpg_path} ({os.path.getsize(jpg_path)} bytes)")

if __name__ == "__main__":
    generate_banner()
