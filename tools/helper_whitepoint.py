import rawpy
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and process the raw file
raw = rawpy.imread('/Users/julian/Pictures/Flashback_Output/Testbilder/_dng/SN559959799_00215.dng') # Replace with your file

rgb_linear = raw.postprocess(
    user_wb=[1.0, 1.0, 1.0, 1.0],
    gamma=(1, 1),
    output_color=rawpy.ColorSpace.raw,
    no_auto_bright=True,
    output_bps=16,
    user_black=64
).astype(np.float32) / 65535.0

# 2. Display the image to find coordinates
# Note: Linear raw images are very dark. We apply a temporary 
# gamma of 1/2.2 purely for the matplotlib preview so you can see the chart.
plt.imshow(rgb_linear ** (1/2.2))
plt.title("Hover over the N5 (Middle Gray) patch to find x, y coordinates")
plt.show()

# 3. Define the bounding box for the N5 patch
# Replace these with the coordinates you found in Step 1
x1, x2 = 1570, 1590 
y1, y2 = 1540, 1560

# 4. Slice the linear array to isolate the patch
gray_patch = rgb_linear[y1:y2, x1:x2]

# 5. Calculate the median value for Red, Green, and Blue channels
# gray_patch[:, :, 0] is Red, [:, :, 1] is Green, [:, :, 2] is Blue
r_val = np.median(gray_patch[:, :, 0])
g_val = np.median(gray_patch[:, :, 1])
b_val = np.median(gray_patch[:, :, 2])

print(f"Raw Patch Values -> R: {r_val:.5f}, G: {g_val:.5f}, B: {b_val:.5f}")

# 6. Calculate the White Balance Multipliers
# The Green channel remains 1.0. Red and Blue are scaled to match Green.
wb_red = g_val / r_val
wb_green = 1.0
wb_blue = g_val / b_val

print(f"Calculated Multipliers -> [Red: {wb_red:.4f}, Green: {wb_green:.4f}, Blue: {wb_blue:.4f}, Green2: {wb_green:.4f}]")