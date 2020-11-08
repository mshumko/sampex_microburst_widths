import sys
import pathlib

# Run the configuration script when the user runs 
# python3 -m sampex_microburst_widths [init, config, or configure]

here = pathlib.Path(__file__).parent.resolve()


if (len(sys.argv) > 1) and (sys.argv[1] in ['init', 'config', 'configure']):
    print('Running the configuration script.')
    # SAMPEX Data dir
    s = (f'What is the SAMPEX data directory? (It must contain the '
         f'attitude and hilt sub-directories) ')
    HILT_DIR = input(s)
    # AE Dir
    s2 = (f'What is the AE data directory?')
    AE_DIR = input(s2)
    
    # Check that the SAMPEX directory exists
    if not pathlib.Path(HILT_DIR).exists():
        raise OSError(f'The HILT diretory "{HILT_DIR}" does not exist. Exiting.')
    # Check that the AE directory exists
    if not pathlib.Path(AE_DIR).exists():
        raise OSError(f'The AE data directory "{HILT_DIR}" does not '
                        'exist. Exiting.')
    
    with open(pathlib.Path(here, 'config.py'), 'w') as f:
        f.write('import pathlib\n\n')
        f.write(f'SAMPEX_DIR = pathlib.Path("{HILT_DIR}")\n')
        f.write(f'AE_DIR = pathlib.Path("{AE_DIR}")\n')
        f.write(f'PROJECT_DIR = pathlib.Path("{here}")')

else:
    print('This is a configuration script to set up config.py file. The config '
        'file will contain the SAMPEX/HILT data directory, the base project '
        'directory (here), and the auroral electroject directory. To see the '
        'prompt after this package is installed, run '
        'python3 -m sampex_microburst_widths init')
