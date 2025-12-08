# Visualizing DEM-GPU Data in ParaView

This guide explains how to visualize the `.vtp` output files from the DEM-GPU simulation using ParaView.

## 1. Load Data
1.  Open ParaView.
2.  File -> **Open...**
3.  Navigate to the `output/` directory and select `particles_0000.vtp` (or the `particles_*.vtp` group for animation).
4.  Click **Apply**.

## 2. Render Spheres (Method A: Fast / View Only)
*Best for large datasets or quick visual checks.*

1.  **Ensure Usage**:
    -   In the **Pipeline Browser** (left panel), make sure the **Eye Icon** next to `particles_...` is **OPEN** (visible).
    -   If it is closed (gray), click it to make the data visible.
2.  **Find the Representation Dropdown**:
    -   Look at the **Toolbar** at the top of the screen (usually the second row).
    -   It typically defaults to **"Surface"** or "Points".
    -   *Alternative*: You can also change this in the **Properties** panel (bottom left) under the **Display** section (search for "Representation").
2.  Change it to **"Point Gaussian"**.
3.  In the **Properties** panel (bottom left):
    -   Find **Gaussian Radius**.
    -   Check **Use Scale Array**.
    -   Set **Gaussian Scale Array** to **"Radius"**.
    -   Set **Gaussian Radius** parameter to **1.0**.
    -   **Important**: Uncheck **"Use Scale Transfer Function"** if it is enabled (it can make spheres invisible).
3.  You should now see the particles rendered as spheres.

## 3. Render Spheres (Method B: Geometry / Analytic)
*Required for Cross-Sections, Clipping, and high-quality rendering.*

1.  **Select the Object**: In the **Pipeline Browser** (left panel), single-click the `particles_...` item so it is highlighted in blue/grey.
    -   **Important**: If the **Apply** button (properties panel) is green, click it! Filters are disabled until data is loaded.
2.  Go to **Filters** -> **Common** -> **Glyph**.
3.  In the **Properties** panel for Glyph:
    -   **Glyph Type**: Select **Sphere**.
    -   **Orientation**: Can be left as is (or mapped to Velocity/Rotation if available).
    -   **Scale**:
        -   **Scale Array**: Select **"Radius"**.
        -   **Scale Factor**: Set to **1.0**.
    -   **Masking**:
        -   **Glyph Mode**: Change to **"All Points"** (Default is usually "Uniform Spatial", which hides particles).
    -   **Smoothing (Fix "Raspberry" look)**:
        -   Scroll down to **Glyph Source** (or just "Sphere" section).
        -   Increase **Theta Resolution** to **20-40** (Default is 6).
        -   Increase **Phi Resolution** to **20-40** (Default is 6).
4.  Click **Apply**.
5.  Hide the original source object by clicking the "Eye" icon next to it in the Pipeline Browser.

## 4. Transparency
1.  Select the **Glyph** (or Point Gaussian) object in the Pipeline.
2.  In the **Properties** panel, scroll down to the **Styling** or **Display** section.
3.  Find **Opacity**.
4.  Set it to a value between **0.0** and **1.0** (e.g., **0.5**).

## 5. Cross-Sections (Slicing & Clipping)
*Note: This works best with Method B (Glyph).*

### Slicing (Planar Cut)
1.  Select the **Glyph** object.
2.  Click the **Slice** icon (or Filters -> Common -> Slice).
3.  In Properties, select **Plane** as the Slice Type.
4.  Adjust **Origin** and **Normal** to position the cut.
5.  Click **Apply**.

### Clipping (Volume Cutout)
1.  Select the **Glyph** object.
2.  Click the **Clip** icon (or Filters -> Common -> Clip).
3.  In Properties, select **Plane** (or Box).
4.  Check **Inside Out** if you want to see the *interior* of the packing.
5.  Click **Apply**.

## 6. Troubleshooting / Inspecting Values
If you cannot see the particles, try these steps:

1.  **Check Data Values (Spreadsheet View)**:
    -   Split the view (buttons in top right of the 3D view area) or create a new layout.
    -   Select **Spreadsheet View**.
    -   Make sure your data source (`particles_...`) is visible (Eye icon).
    -   Look at the "Attribute" dropdown: Select **Point Data**.
    -   You should see columns for `Points`, `Radius`, `Velocity`, etc.
    -   If `Radius` values are tiny (e.g., 0.05), you might need to zoom in or increase Scale Factor.

2.  **Reset Camera**:
    -   Click the **Reset Camera** icon (scale with arrows) in the toolbar.
    -   Or press **'R'** on your keyboard.
    -   This centers the view on the data.

3.  **Check Scale Settings**:
    -   In the **Glyph** filter properties, ensure **Scale Factor** is 1.0.
    -   Ensure **Scale Array** is set to **"Radius"**.
    -   **Critical**: detailed **"Use Scale Transfer Function"** MUST be **Unchecked**. This is often on by default and can make particles invisible.

4.  **Filters Greyed Out?**:
    -   Ensure the object (`particles_...`) is **Selected** (highlighted) in the Pipeline Browser.
    -   Ensure the **Apply** button has been clicked (filters only work on applied data).
