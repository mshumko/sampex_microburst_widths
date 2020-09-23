import sys
import pathlib

# Run the configuration script when the user runs 
# python3 -m sampex_microburst_widths [init, config, or configure]

here = pathlib.Path(__file__).parent.resolve()


if (len(sys.argv) > 1) and (sys.argv[1] in ['init', 'config', 'configure']):
    print('Running the configuration script.')
    HILT_DIR = input('What is the SAMPEX/HILT data directory? ')
    # Check that the directory exists
    if not pathlib.Path(HILT_DIR).exists():
        raise OSError(f'The HILT path "{HILT_DIR}" does not exist. Exiting.')
    with open(pathlib.Path(here, 'config.py'), 'w') as f:
        f.write('import pathlib\n\n')
        f.write(f'HILT_DIR = pathlib.Path("{HILT_DIR}")\n')
        f.write(f'PROJECT_DIR = pathlib.Path("{here}")')

else:
    print('This is a configuration script to set up config.py file. The config '
        'file will contain the SAMPEX/HILT data directory, and the base project '
        'directory (here). To get the prompt, run '
        'python3 -m sampex_microburst_widths init')