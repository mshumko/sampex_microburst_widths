from sampex_microburst_widths.microburst_id import identify_microbursts
from sampex_microburst_widths.microburst_id import append_attitude

m = identify_microbursts.Identify_SAMPEX_Microbursts(
    baseline_width_s=1, foreground_width_s=0.1,
    prominence_rel_height=0.5
    )
# try:
m.loop()
# finally:
#     cat_path = m.save_catalog()

# a = append_attitude.Append_Attitude(cat_path)
# a.loop()
# a.save_catalog()