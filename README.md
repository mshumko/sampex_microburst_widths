# sampex_microbursts
Identify and estimate widths of microbursts in the SAMPEX/HILT 20 ms data.

## Installation
To install the dependencies, run these three shell commands:

```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Tasks
- [ ] Validate the catalog against Lauren's data
- [ ] Move my dependencies into this repo and make a requirements.txt file.
- [ ] Add microburts browswer
- [ ] Add prominence-based peak width estimator
- [ ] Add fit-based peak width estimator
- [ ] Insorporate a goodness of fit statistic to fits
- [ ] Look at the distribution of peak widths. 
- [ ] Check by eye a subset of the dataset. One idea is to pick N examples from each peak width bin and find the error?
