from sampex_microburst_widths.microburst_id import identify_microbursts
from sampex_microburst_widths.microburst_id import merge_attitude
from sampex_microburst_widths.microburst_id import merge_ae

# Identify and fit microbursts
m = identify_microbursts.Identify_SAMPEX_Microbursts(
    baseline_width_s=0.5, foreground_width_s=0.1,
    prominence_rel_height=0.5
    )
m.loop(debug=False)
cat_path = m.save_catalog()

# Merge the Attitude data
m = merge_attitude.Merge_Attitude(cat_path)
m.loop()
m.save_catalog()

# Merge the AE data
m = merge_ae.Merge_AE(cat_path)
try:
    m.loop()
finally:
    m.save_catalog()