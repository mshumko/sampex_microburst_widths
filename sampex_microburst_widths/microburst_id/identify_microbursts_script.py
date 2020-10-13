from sampex_microburst_widths.microburst_id import identify_microbursts
from sampex_microburst_widths.microburst_id import append_attitude

m = identify_microbursts.Identify_SAMPEX_Microbursts()
try:
    m.loop()
except:
    raise
finally:
    cat_path = m.save_catalog()

a = append_attitude.Append_Attitude(cat_path)
a.loop()
a.save_catalog()