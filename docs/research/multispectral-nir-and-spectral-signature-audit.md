# Multispectral NIR & Silicon Photovoltaic Spectral Signature Audit

**Date**: 2026-10-22  
**Component**: `pvextent.cad.spectral`, `pvextent.models.deeplabv3plus`, `pvextent.models.resnet_unet`, `pvextent.cad.vectorize`, `pvextent.cli`

---

## 1. Overview & Operational Problem

Standard optical aerial computer vision models operate strictly on 3-channel visible RGB imagery ($400 - 700\text{ nm}$). In high-resolution RGB imagery, high-efficiency monocrystalline solar panels with anti-reflective coatings (ARC) absorb $\approx 95\%$ of visible photons, rendering them dark navy, charcoal, or deep black.

Consequently, in pure visible RGB space, solar panels present strong visual ambiguities with non-photovoltaic surfaces:
1. **Intensive Green Roofs (Dachbegrünung) & Tree Canopy Overhangs**: Dark foliage beneath building eaves.
2. **Bitumen Roof Membranes & Torch-Down Felt (Dachpappe)**: Dark matte surfaces with rectilinear seams.
3. **Dark Slate & Mineral Composite Shingles (Schieferdächer)**: Dark gray/black tiles with specular facets.
4. **Water Ponding & Roof Puddle Reflections**: Standing rainwater reflecting ambient sky glare.

Official aerial surveying campaigns in Germany (e.g. Geobasis NRW / BKG DOP20) acquire 4-band imagery including **Near-Infrared (NIR, $750 - 920\text{ nm}$)**. Hebel 3 equips `spacement-pvextent` with multispectral radiometry and silicon spectral physics to reduce false positives on complex roofscapes.

---

## 2. Radiometric Physics & Spectral Decision Engine

### 2.1. Biophysical & Radiometric Indices

1. **Normalized Difference Vegetation Index (NDVI)**:
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + \epsilon}$$
   - **Vegetation / Dachbegrünung**: Chlorophyll cellular structures reflect strongly in the near-infrared ("Red Edge" effect), yielding $\text{NDVI} \in [0.35, 0.85]$.
   - **Silicon Solar Panels**: Display no biological red edge, exhibiting $\text{NDVI} \in [-0.25, 0.25]$.
   - **Water Puddles**: Strong dielectric water absorption across NIR yields negative $\text{NDVI} < 0.0$.

2. **Photovoltaic Spectral Index (PVSI)**:
   $$\text{PVSI} = \frac{\text{NIR} - \text{Blue}}{\text{NIR} + \text{Blue} + \epsilon}$$
   Exploits the fundamental silicon semiconductor bandgap cutoff ($E_g \approx 1.12\text{ eV}, \lambda \approx 1100\text{ nm}$) between visible short wavelengths and infrared reflectance.

3. **NIR-to-Red Band Reflectance Ratio**:
   $$\text{Ratio}_{\text{NIR/Red}} = \frac{\text{NIR}}{\text{Red} + \epsilon}$$

### 2.2. Spectral Decision Matrix

| Spectral Verdict | Decision Rule | Physical Interpretation | Action / Veto |
|---|---|---|---|
| `vegetated_roof_veto` | $\text{NDVI} > 0.35$ | Living vegetation / Green roof (Dachbegrünung) | **Suppressed** if `--veto-vegetation`, or flagged `review_recommended` |
| `water_puddle_veto` | $\text{NIR} < 0.04 \land \text{Red} < 0.05$ | Deep infrared absorption of standing puddle water | **Suppressed** / flagged `review_recommended` |
| `silicon_photovoltaic_confirmed` | $\text{NDVI} \le 0.25 \land \text{Ratio}_{\text{NIR/Red}} \ge 0.70$ | Semiconductor bandgap reflectance signature | Confirmed high-confidence physical PV |
| `standard_spectral_response` | Neutral spectral reflectance | Inert roof material (slate, tile, sheet metal) | Evaluated via standard spatial CNN probabilities |

---

## 3. Deep Learning 4-Band Conv1 Weight Expansion

To support 4-channel multispectral ingestion ($R, G, B, \text{NIR}$) in DeepLabV3+ ResNet50 and ResNet18 U-Net without discarding proven ImageNet classification priors:

The `expand_conv1_for_nir` function expands the first convolutional layer $W_{\text{conv1}} \in \mathbb{R}^{C_{\text{out}} \times 3 \times 7 \times 7} \to \mathbb{R}^{C_{\text{out}} \times 4 \times 7 \times 7}$:
$$W_{\text{conv1}}[:, 3, :, :] = \frac{1}{2}\left(W_{\text{conv1}}[:, 0, :, :] + W_{\text{conv1}}[:, 1, :, :]\right)$$
This balances activation energy conservation between visible Red/Green channels and Near-Infrared, maintaining backward compatibility while accepting full 4-band GeoTIFFs.

---

## 4. Color Infrared (CIR) False-Color Composite Rendering

The `compose_cir_composite` function renders standardized Color Infrared false-color imagery matching the specifications of German state surveying offices:
- **Output Channel 1 (Red)** $\leftarrow$ NIR band (highlights living vegetation in intense crimson/magenta)
- **Output Channel 2 (Green)** $\leftarrow$ Red band
- **Output Channel 3 (Blue)** $\leftarrow$ Green band

In CIR composite imagery, vegetated surfaces glow bright magenta-red, while crystalline silicon solar modules appear as dark slate/cyan rectangles, enabling instant visual confirmation.

---

## 5. Verification & Unit Test Suite

All 9 unit tests in `tests/test_spectral.py` passed with 100% green coverage:
1. `test_calculate_ndvi`: Verified vegetation ($\text{NDVI} > 0.35$), silicon PV ($\text{NDVI} \le 0.25$), and water ($\text{NDVI} < 0$).
2. `test_calculate_pvsi`: Verified semiconductor shortwave-to-NIR contrast index.
3. `test_calculate_ndre`: Verified Normalized Difference Red Edge index.
4. `test_compose_cir_composite`: Verified proper band routing (NIR $\to$ R, Red $\to$ G, Green $\to$ B).
5. `test_extract_polygon_spectral_stats_silicon_pv`: Verified detection of confirmed silicon photovoltaic profile.
6. `test_extract_polygon_spectral_stats_green_roof_veto`: Verified automated veto of vegetated roofs ($\text{NDVI} = 0.76 > 0.35$).
7. `test_extract_polygon_spectral_stats_water_puddle_veto`: Verified automated veto of standing water puddles.
8. `test_vectorize_mask_with_nir_and_veto`: Verified end-to-end filtering with `--veto-vegetation`.
9. `test_expand_conv1_for_nir`: Verified 4-channel tensor expansion preserving Red/Green balanced energy.

**Full Test Suite**: **505 / 505 unit tests passing** on Apple Silicon M2 MPS in 21.28 seconds.
