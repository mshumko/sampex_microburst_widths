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

## Tasks
- [ ] Validate the catalog against Lauren's data
- [ ] Add microburts browser
- [ ] Add prominence-based peak width estimator
- [ ] Add fit-based peak width estimator
- [ ] Insorporate a goodness of fit statistic to fits
- [ ] Look at the distribution of peak widths. 
- [ ] Check by eye a subset of the dataset. One idea is to pick N examples from each peak width bin and find the error?
- [x] Move my dependencies into this repo and make a requirements.txt file.
