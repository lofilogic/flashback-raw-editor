import argparse
import numpy as np
import rawpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate DNG color matrices from a ColorChecker shot.")
parser.add_argument(
    "--illuminant",
    choices=["d50", "tungsten"],
    default="d50",
    help="Calibration illuminant: 'd50' for daylight/D50 (default), "
         "'tungsten' for Standard Illuminant A (~2856K, e.g. Bellight 75W)",
)
args = parser.parse_args()

# DNG CalibrationIlluminant tag values
ILLUMINANT_TAG = {"d50": 23, "tungsten": 17}
ILLUMINANT_LABEL = {"d50": "D50 (tag value 23)", "tungsten": "Standard Light A / Tungsten (tag value 17)"}

# --- Configuration Variables ---
RAW_FILE = "/Users/julian/Pictures/Flashback_Output/colormatch/match-look-to-flashback/tungsten.dng"
BLACK_LEVEL = 64

# Standard CIE XYZ values for ColorChecker Classic under D50 (0-1 scale)
REF_XYZ_D50 = np.array([
    [0.1150, 0.1009, 0.0336], [0.3920, 0.3582, 0.2505], [0.1834, 0.1927, 0.3347],
    [0.1086, 0.1345, 0.0639], [0.2644, 0.2452, 0.4485], [0.3168, 0.4357, 0.4411],
    [0.3797, 0.3013, 0.0526], [0.1417, 0.1200, 0.3541], [0.2936, 0.1963, 0.1287],
    [0.0867, 0.0655, 0.1171], [0.3583, 0.4443, 0.1165], [0.4729, 0.4326, 0.0768],
    [0.0861, 0.0583, 0.2185], [0.1506, 0.2359, 0.0767], [0.2039, 0.1193, 0.0381],
    [0.5739, 0.6033, 0.1044], [0.2987, 0.1947, 0.3477], [0.1450, 0.1952, 0.3794],
    [0.9069, 0.9634, 0.7951], [0.5898, 0.6277, 0.5215], [0.3622, 0.3848, 0.3207],
    [0.1952, 0.2069, 0.1741], [0.0898, 0.0950, 0.0805], [0.0322, 0.0332, 0.0287]
])

def extract_patch_values(image, points, patch_size=20):
    camera_rgb = []
    for (x, y) in points:
        x, y = int(x), int(y)
        patch = image[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
        avg_color = np.mean(patch, axis=(0, 1))
        camera_rgb.append(avg_color)
    return np.array(camera_rgb)

# 1. Load RAW Image
with rawpy.imread(RAW_FILE) as raw:
    img = raw.postprocess(
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        user_wb=[1.0, 1.0, 1.0, 1.0],
        use_camera_wb=False,
        user_black=BLACK_LEVEL 
    )

img_float = img.astype(np.float32) / 65535.0
img_float = np.clip(img_float, 0.0, 1.0)

# 2. GUI Selection
print("Click the center of all 24 patches. Start at Dark Skin (top-left) and read left-to-right, top-to-bottom.")
plt.imshow(np.power(img_float, 1/2.2)) 
points = plt.ginput(24, timeout=0)
plt.close()

if len(points) != 24:
    raise ValueError("Exactly 24 points must be selected.")

# 3. Process Values
camera_rgb = extract_patch_values(img_float, points)

# Derive ASN from grey patches for tungsten; use the known value for D50
if args.illuminant == "tungsten":
    # Patches 20-23 (0-indexed 19-22): Neutral 8, 6.5, 5, 3.5 — mid-greys, skip white/black
    grey_avg = np.mean(camera_rgb[19:23], axis=0)
    ASN = grey_avg / grey_avg[1]
    print(f"ASN from grey patches: [{ASN[0]:.7f}, 1.0, {ASN[2]:.7f}]")
else:
    ASN = np.array([0.6883759, 1.0, 0.7963592])

# --- REVISED STEP 4 & 5 (Adobe Strict) ---

# 1. Forward Matrix (WB RGB -> XYZ)
# Rows must sum to the D50 White Point coordinates
wb_multipliers = 1.0 / ASN
camera_rgb_wb = camera_rgb * wb_multipliers
F, _, _, _ = np.linalg.lstsq(camera_rgb_wb, REF_XYZ_D50, rcond=None)
ForwardMatrix = F.T

# D50 White Point: X=0.9642, Y=1.0000, Z=0.8249
d50_wp = np.array([0.9642, 1.0000, 0.8249])
row_sums = np.sum(ForwardMatrix, axis=1)
# Scale each row independently to match the D50 white point
ForwardMatrix = ForwardMatrix * (d50_wp / row_sums)[:, np.newaxis]

# 2. Color Matrix (XYZ -> Native Raw)
# Maps the standard XYZ space to your camera's unbalanced raw space
M, _, _, _ = np.linalg.lstsq(REF_XYZ_D50, camera_rgb, rcond=None)
ColorMatrix = M.T

# CRITICAL: Normalize so the green channel (row 1) determines the scale.
# This prevents the matrix from fighting the AsShotNeutral tag.
cm_scale = np.sum(ColorMatrix[1, :]) 
ColorMatrix = ColorMatrix / cm_scale

illuminant = args.illuminant
idx = {"d50": 1, "tungsten": 2}[illuminant]

print(f"\n--- CALIBRATION ILLUMINANT: {ILLUMINANT_LABEL[illuminant]} ---")
print(f"ForwardMatrix{idx} (Camera_WB -> XYZ D50):")
print(np.array2string(ForwardMatrix.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.5f" % x}))

print(f"\nColorMatrix{idx} (XYZ D50 -> Camera_Native):")
print(np.array2string(ColorMatrix.flatten(), separator=', ', formatter={'float_kind':lambda x: "%.5f" % x}))

test_wb_neutral = np.array([1.0, 1.0, 1.0])
result_xyz = ForwardMatrix @ test_wb_neutral
print(f"Verification (Should be ~0.964, 1.0, 0.824): {result_xyz}")

print("\n--- TAG REMINDER ---")
print("In your export script, set:")
print(f"  Tag 50721/50722 (ColorMatrix{idx})   = ColorMatrix{idx}")
print(f"  Tag 50964/50965 (ForwardMatrix{idx}) = ForwardMatrix{idx}")
print(f"  CalibrationIlluminant{idx} tag       = {ILLUMINANT_TAG[illuminant]}  ({ILLUMINANT_LABEL[illuminant]})")
print(f"  AsShotNeutral (asn)                  = [{ASN[0]:.7f}, 1.0, {ASN[2]:.7f}]")