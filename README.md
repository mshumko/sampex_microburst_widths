# sampex_microburst_widths
Identify and estimate widths of microbursts in the SAMPEX/HILT 20 ms data.

## Installation
Run these shell commands to install the dependencies into a virtual 
environment and configure the SAMPEX data paths:

```
# cd into the top project directory
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 -m sampex_microburst_widths init # and answer the promps.
```