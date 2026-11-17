import os
import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

def capture_all_tabs():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")
    os.makedirs(out_dir, exist_ok=True)

    edge_options = Options()
    edge_options.add_argument("--headless=new")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--window-size=1600,1050")

    print("Launching Edge headless browser (1600x1050)...")
    driver = webdriver.Edge(options=edge_options)

    try:
        url = "https://varuna-ml.streamlit.app/"
        print(f"Navigating to {url}...")
        driver.get(url)

        print("Waiting 12 seconds for Streamlit initialization...")
        time.sleep(12)

        # Locate app iframe
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        app_iframe = None
        for f in iframes:
            src = f.get_attribute("src") or ""
            if "~/+" in src or "streamlit" in src:
                app_iframe = f
                break
        if not app_iframe and len(iframes) > 0:
            app_iframe = iframes[0]

        if not app_iframe:
            print("ERROR: Could not locate app iframe.")
            return

        print("Switching into Streamlit app iframe...")
        driver.switch_to.frame(app_iframe)
        time.sleep(2)

        # Find tab buttons by role='tab'
        tabs = driver.find_elements(By.XPATH, '//*[@role="tab"]')
        print(f"Located {len(tabs)} tabs inside iframe:")
        for i, t in enumerate(tabs):
            print(f"  [{i}] {t.text}")

        tab_filenames = [
            ("screenshot_tab1_hobli_report", 0, "Hobli Report"),
            ("screenshot_tab2_risk_zones", 1, "Climatological Risk Zones"),
            ("screenshot_tab3_district_heatmap", 2, "District Trends Heatmap"),
            ("screenshot_tab4_data_quality", 3, "Data Quality Audit"),
            ("screenshot_tab5_model_limits", 4, "Model Limits & Baselines"),
        ]

        for filename, tab_idx, label in tab_filenames:
            if tab_idx < len(tabs):
                tab_elem = tabs[tab_idx]
                print(f"\nActivating Tab [{tab_idx}]: {label}...")
                driver.execute_script("arguments[0].scrollIntoView(true);", tab_elem)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", tab_elem)
                print(f"Waiting 5s for {label} charts & components to render...")
                time.sleep(5)

                # Capture full viewport
                png_path = os.path.join(out_dir, f"{filename}.png")
                jpg_path = os.path.join(out_dir, f"{filename}.jpg")
                
                # Switch to default content briefly to capture full layout with sidebar & Streamlit frame
                driver.switch_to.default_content()
                time.sleep(0.5)
                driver.save_screenshot(png_path)
                
                # Also save JPEG with high quality
                with Image.open(png_path) as img:
                    img.convert("RGB").save(jpg_path, "JPEG", quality=95, optimize=True)
                
                print(f"Saved: {filename}.png ({os.path.getsize(png_path)} bytes)")
                print(f"Saved: {filename}.jpg ({os.path.getsize(jpg_path)} bytes)")

                # Switch back to app iframe for next tab
                driver.switch_to.frame(app_iframe)
                time.sleep(0.5)
                # Re-fetch tabs inside iframe to avoid stale reference
                tabs = driver.find_elements(By.XPATH, '//*[@role="tab"]')

        # Also ensure app_banner.jpg exists
        banner_png = os.path.join(out_dir, "app_banner.png")
        banner_jpg = os.path.join(out_dir, "app_banner.jpg")
        if os.path.exists(banner_png):
            with Image.open(banner_png) as img:
                img.convert("RGB").save(banner_jpg, "JPEG", quality=95, optimize=True)
            print(f"Saved: app_banner.jpg ({os.path.getsize(banner_jpg)} bytes)")

        print("\nAll 5 live screenshots successfully captured and saved in both PNG and JPG format!")

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_all_tabs()
