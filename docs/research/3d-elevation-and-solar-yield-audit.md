# 3D Elevation Fitting, Roof Tilt & Solar Yield Audit

**Date**: 2026-10-22  
**Component**: `pvextent.cad.elevation`, `pvextent.cad.packing`, `pvextent.cad.vectorize`, `pvextent.cli`

---

## 1. Overview & Mathematical Formulation

Standard 2D aerial computer vision models compute flat projected surface area $A_{\text{proj}}$ from orthogonal orthophotos. However, actual rooftop photovoltaic installations are mounted on inclined roof planes with specific compass orientations (azimuth).

### 1.1. True 3D Physical Surface Area
For a roof surface inclined at pitch / tilt angle $\theta \in [0^\circ, 85^\circ]$:
$$A_{\text{true}} = \frac{A_{\text{proj}}}{\cos(\theta)}$$
- For a typical German residential gable roof with $\theta = 35^\circ$, the expansion factor is:
  $$\frac{1}{\cos(35^\circ)} \approx 1.2208 \quad (+22.1\% \text{ additional physical mounting surface})$$
- For a $\theta = 45^\circ$ steep pitched roof:
  $$\frac{1}{\cos(45^\circ)} \approx 1.4142 \quad (+41.4\% \text{ additional physical mounting surface})$$

### 1.2. 3D Surface Plane Fitting from Digital Surface Models (DSM)
Given an elevation raster $Z(y, x)$ (DOM / DSM, e.g. from Geobasis NRW bDOM / DOM1):
We extract 3D points $(X_i, Y_i, Z_i)$ within the regularized PV array polygon and solve for the optimal 3D plane:
$$Z = a \cdot X + b \cdot Y_{\text{geo}} + c$$
From the plane gradient parameters $(a, b)$:
$$\text{Tilt Angle: } \theta = \arctan\left(\sqrt{a^2 + b^2}\right)$$
$$\text{Azimuth Angle: } \phi = \left(\operatorname{degrees}(\operatorname{atan2}(-a, -b)) + 360^\circ\right) \bmod 360^\circ$$
The module computes fit quality metrics:
- Coefficient of determination ($R^2 \in [0, 1]$)
- Root Mean Squared Error ($\text{RMSE}$ in meters)
- Elevation range ($Z_{\min}, Z_{\max}, Z_{\text{mean}}$)

### 1.3. PVGIS-Grade Annual Solar Yield Transposition Modeling
Annual energy production is modeled using empirical Central European irradiation transposition:
$$\text{Specific Yield } Y_{\text{spec}} = \text{GHI} \cdot f(\theta, \phi) \cdot \text{PR} \quad [\text{kWh}/\text{kWp}/\text{year}]$$
where:
- $\text{GHI} = 1050\text{ kWh}/\text{m}^2/\text{year}$ (DWD / PVGIS standard horizontal irradiation)
- $\text{PR} = 0.82$ (Standard European system performance ratio accounting for inverter efficiency $\approx 97\%$, temperature derating, cabling losses, and soiling)
- $f(\theta, \phi)$ is the transposition factor accounting for surface inclination and compass deviation from South:
  - Optimal South ($\phi = 180^\circ$, $\theta = 35^\circ$): $f \approx 1.08 - 1.12 \implies \approx 930 - 980\text{ kWh}/\text{kWp}$
  - East / West ($\phi = 90^\circ / 270^\circ$, $\theta = 30^\circ$): $f \approx 0.88 - 0.92 \implies \approx 760 - 800\text{ kWh}/\text{kWp}$
  - Flat horizontal ($\theta \le 3^\circ$): $f = 1.00 \implies \approx 861\text{ kWh}/\text{kWp}$
  - North ($\phi = 0^\circ$, $\theta = 35^\circ$): $f \approx 0.70 \implies \approx 600\text{ kWh}/\text{kWp}$

$$\text{Annual Energy Yield: } E_{\text{annual}} = P_{\text{fitted\_kwp}} \cdot Y_{\text{spec}} \quad [\text{kWh}/\text{year}]$$
$$\text{Annual Carbon Displacement: } \Delta \text{CO}_2 = \frac{E_{\text{annual}} \cdot 0.380\text{ kg}}{1000} \quad [\text{t CO}_2/\text{year}]$$

---

## 2. Sloped 3D Physical BIM Module Grid Packing

On a pitched roof, the physical slope dimension along the pitch is foreshortened on the 2D orthophoto by $\cos(\theta)$. 
[`src/pvextent/cad/packing.py`](file:///Users/paul/Valeford/spacement/spacement-pvextent/src/pvextent/cad/packing.py) applies this foreshortening to module dimensions along the pitch axis:
$$\text{step}_{v,\text{proj}} = (\text{mod}_{v,\text{phys}} + \text{gap}_{\text{phys}}) \cdot \cos(\theta)$$
This accurately models how more module rows physically fit on the inclined roof slope than appear in horizontal 2D projections.

---

## 3. Verification & Test Suite

All 7 new unit tests in [`tests/test_elevation.py`](file:///Users/paul/Valeford/spacement/spacement-pvextent/tests/test_elevation.py) passed:
1. `test_calculate_3d_surface_area`: Verified expansion at 0°, 30° (+15.5%), and 45° (+41.4%).
2. `azimuth_to_cardinal`: Verified 16-point cardinal compass conversion and flat roof handling.
3. `test_calculate_annual_solar_yield`: Verified South vs. North yield disparity, East/West symmetry.
4. `test_fit_roof_plane_from_dsm_south_slope`: Verified exact 30° South plane recovery ($R^2 > 0.99$, $\text{RMSE} < 0.01\text{ m}$).
5. `test_fit_roof_plane_from_dsm_east_slope`: Verified exact 45° East plane recovery ($R^2 > 0.99$).
6. `test_sloped_module_grid_packing`: Verified sloped roofs accommodate more physical panels.
7. `test_vectorize_mask_with_dsm_and_yield`: Verified end-to-end integration into GeoJSON and summary dictionary.

**Full test suite status**: **489 / 489 tests passing** in 21.4s on Apple M2 MPS.
