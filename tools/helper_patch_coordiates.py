import rawpy
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RAW_FILE = '/Users/julian/Pictures/Flashback_Output/Testbilder/_dng/SN559959799_00215.dng' # Ensure this matches your file name
TRUE_WB = [2.0333, 1.0000, 1.6796, 1.0000]

print("Loading RAW file for coordinate mapping...")
with rawpy.imread(RAW_FILE) as raw:
    rgb_linear = raw.postprocess(
        user_wb=TRUE_WB,
        gamma=(1, 1),
        output_color=rawpy.ColorSpace.raw,
        no_auto_bright=True,
        output_bps=16,
        user_black=64
    ).astype(np.float32) / 65535.0

# Apply a strong gamma curve purely for the visual preview.
# Linear raw data is extremely dark; this makes the black patches visible.
preview_image = np.clip(rgb_linear ** (1 / 2.4), 0, 1)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(preview_image)
ax.set_title("Click the exact center of each of the 24 patches.\n"
             "IMPORTANT ORDER: Left-to-Right, Top-to-Bottom.\n"
             "(Start at Dark Skin top-left, end at Black bottom-right)")

click_count = 0
coordinates = []

def onclick(event):
    global click_count
    # Ensure the click was inside the image bounds
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        coordinates.append(f"({x}, {y})")
        click_count += 1
        
        # Log the individual click
        print(f"Patch {click_count}/24 captured at X:{x}, Y:{y}")
        
        # Draw a visible marker on the image so you know where you clicked
        ax.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
        fig.canvas.draw()
        
        # When all 24 are clicked, format and print the final array
        if click_count == 24:
            print("\n--- COPY THE BLOCK BELOW ---")
            print("PATCH_CENTERS = [")
            for i in range(0, 24, 6):
                row_str = ", ".join(coordinates[i:i+6])
                if i < 18:
                    print(f"    {row_str},")
                else:
                    print(f"    {row_str}")
            print("]")
            print("----------------------------\n")
            print("You can close the window now.")

# Attach the click listener
cid = fig.canvas.mpl_connect('button_press_event', onclick)

print("Waiting for clicks...")
plt.show()