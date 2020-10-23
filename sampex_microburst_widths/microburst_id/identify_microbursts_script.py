from sampex_microburst_widths.microburst_id import identify_microbursts
from sampex_microburst_widths.microburst_id import merge_attitude

m = identify_microbursts.Identify_SAMPEX_Microbursts(
    baseline_width_s=0.5, foreground_width_s=0.1,
    prominence_rel_height=0.5
    )
try:
    m.loop(debug=False)
finally:
    cat_path = m.save_catalog()

m = merge_attitude.Merge_Attitude(cat_path)
m.loop()
m.save_catalog()