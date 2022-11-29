# sampex_microburst_widths
Identify and quantify the duration of relativistic microbursts in the SAMPEX/HILT 20 ms data. The results of this study are written up for publication in Journal of Geophysical Research Letters and is titled "Duration of Individual Relativistic Electron Microbursts: A Probe Into Their Scattering Mechanism". I will update the README when the paper is published.

## Installation
Run these shell commands to install the dependencies into a virtual 
environment and configure the SAMPEX data paths:

```bash
git clone https://github.com/mshumko/sampex_microburst_widths
cd sampex_microburst_widths
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt # or python3 -m pip install -e .
python3 -m sampex_microburst_widths init # to answer the prompts and configure paths
```

I developed and tested this code with Python 3.9.0 (tags/v3.9.0:9cf6752276e, Nov  2 2020, 09:08:52)

## User Guide
Many of these scripts are disjoint---I dove into many research rabbit holes---but I hope this guide will help you reproduce these results.