import rawpy
import numpy as np
from scipy.optimize import minimize

# --- 1. CONFIGURATION ---
RAW_FILE = '/Users/julian/Pictures/Flashback_Output/Testbilder/_dng/SN559959799_00215.dng'
TRUE_WB = [2.0333, 1.0000, 1.6796, 1.0000]
PATCH_RADIUS = 20

PATCH_CENTERS = [
    (1835, 1252), (1848, 1345), (1865, 1448), (1875, 1535), (1881, 1633), (1903, 1731),
    (1748, 1257), (1763, 1355), (1773, 1453), (1783, 1548), (1793, 1643), (1805, 1744),
    (1655, 1262), (1660, 1358), (1677, 1458), (1688, 1553), (1700, 1648), (1713, 1756),
    (1552, 1265), (1580, 1365), (1575, 1460), (1592, 1561), (1600, 1663), (1615, 1759)
]

# Standard ColorChecker Classic Linear sRGB Reference Values
REFERENCE_LINEAR_SRGB = np.array([
    [0.031, 0.021, 0.015], [0.402, 0.231, 0.160], [0.081, 0.106, 0.207], 
    [0.035, 0.063, 0.021], [0.066, 0.057, 0.158], [0.100, 0.222, 0.176], 
    [0.320, 0.090, 0.020], [0.029, 0.038, 0.156], [0.264, 0.046, 0.044], 
    [0.024, 0.015, 0.049], [0.159, 0.252, 0.040], [0.551, 0.201, 0.031], 
    [0.013, 0.018, 0.101], [0.031, 0.111, 0.030], [0.177, 0.020, 0.020], 
    [0.584, 0.449, 0.027], [0.250, 0.038, 0.144], [0.014, 0.109, 0.210], 
    [0.871, 0.884, 0.852], [0.585, 0.596, 0.588], [0.354, 0.362, 0.359], 
    [0.187, 0.191, 0.189], [0.087, 0.089, 0.088], [0.031, 0.031, 0.031]
])

# --- 2. PROCESS RAW & EXTRACT SENSOR DATA ---
print("Processing RAW file with calibrated white balance...")
with rawpy.imread(RAW_FILE) as raw:
    rgb_linear = raw.postprocess(
        user_wb=TRUE_WB,
        gamma=(1, 1),
        output_color=rawpy.ColorSpace.raw,
        no_auto_bright=True,
        output_bps=16,
        user_black=64
    ).astype(np.float32) / 65535.0

print("Extracting patch data...")
sensor_data = []
for x, y in PATCH_CENTERS:
    patch = rgb_linear[y - PATCH_RADIUS : y + PATCH_RADIUS, 
                       x - PATCH_RADIUS : x + PATCH_RADIUS]
    
    r = np.median(patch[:, :, 0])
    g = np.median(patch[:, :, 1])
    b = np.median(patch[:, :, 2])
    sensor_data.append([r, g, b])

S = np.array(sensor_data)
R = REFERENCE_LINEAR_SRGB

# --- 3. COMPUTE THE CONSTRAINED CCM ---
print("Optimizing Color Correction Matrix...")

def objective(matrix_flat, S, R):
    M = matrix_flat.reshape(3, 3)
    mapped_data = S @ M.T
    return np.mean((mapped_data - R)**2)

def constraint(matrix_flat):
    M = matrix_flat.reshape(3, 3)
    return np.sum(M, axis=1) - 1.0

initial_guess = np.eye(3).flatten()
cons = {'type': 'eq', 'fun': constraint}

result = minimize(
    objective, 
    initial_guess, 
    args=(S, R), 
    constraints=cons,
    method='SLSQP'
)

CALIBRATED_CCM = result.x.reshape(3, 3)

print("\n--- CALIBRATION COMPLETE ---")
print("np.array([")
for row in CALIBRATED_CCM:
    print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}],")
print("])")
print("\nRow Sums Verification (should be exactly 1.0):", np.sum(CALIBRATED_CCM, axis=1))