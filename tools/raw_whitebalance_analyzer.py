import rawpy
import numpy as np

def get_wb_from_patch(raw_path: str, x: int, y: int, size: int = 50):
    with rawpy.imread(raw_path) as raw:
        # 1. Force even coordinates and size to guarantee RGGB phase alignment
        x = int(x) & ~1
        y = int(y) & ~1
        size = int(size) & ~1
        
        raw_data = raw.raw_image_visible.astype(np.float32)
        black_levels = raw.black_level_per_channel
        white_level = raw.white_level
        
        patch = raw_data[y:y+size, x:x+size]
        
        # 2. Extract RGGB channels with guaranteed phase
        r  = patch[0::2, 0::2] - black_levels[0]
        g1 = patch[0::2, 1::2] - black_levels[1]
        b  = patch[1::2, 1::2] - black_levels[2]
        g2 = patch[1::2, 0::2] - black_levels[3]
        
        g = (g1 + g2) / 2.0
        
        # 3. Create a valid mask to exclude clipped/saturated pixels
        valid_mask = (patch[0::2, 0::2] < white_level) & \
                     (patch[0::2, 1::2] < white_level) & \
                     (patch[1::2, 1::2] < white_level) & \
                     (patch[1::2, 0::2] < white_level)
        
        if not np.any(valid_mask):
            raise ValueError("The selected patch is completely clipped. Choose a darker patch.")
            
        r_valid = r[valid_mask]
        g_valid = g[valid_mask]
        b_valid = b[valid_mask]
        
        # 4. Calculate valid averages
        avg_r = np.mean(r_valid)
        avg_g = np.mean(g_valid)
        avg_b = np.mean(b_valid)
        
        # 5. Multipliers (Used for raw processing / rendering)
        m_r = avg_g / avg_r
        m_g = 1.0
        m_b = avg_g / avg_b
        
        # 6. AsShotNeutral (Used for writing into DNG metadata tags)
        asn_r = avg_r / avg_g
        asn_g = 1.0
        asn_b = avg_b / avg_g
        
        return {
            "multipliers": (m_r, m_g, m_b),
            "as_shot_neutral": (asn_r, asn_g, asn_b)
        }

# Usage
path = "/Users/julian/Pictures/Flashback_Output/colormatch/match-look-to-flashback/fb_5000K.dng"
try:
    results = get_wb_from_patch(path, 500, 500)
    print(f"WB Multipliers:  {results['multipliers']}")
    print(f"AsShotNeutral:   {results['as_shot_neutral']}")
except Exception as e:
    print(f"Error: {e}")