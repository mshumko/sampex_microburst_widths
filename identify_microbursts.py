import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import dirs
import microburst_detection.signal_to_background.signal_to_background \
        as signal_to_background
import mission_tools.sampex.load_hilt_data as load_hilt_data

class Identify_SAMPEX_Microbursts:
    def __init__(self, baseline_width_s=0.500, threshold=10, 
                spin_file_name='spin_times.csv'):
        self.hilt_dir = pathlib.Path(load_hilt_data.hilt_dir, 'State4')
        self.spin_file_name = spin_file_name
        self.baseline_width_s = baseline_width_s
        self.threshold = threshold
        return

    def loop(self):
        """
        Loop over the HILT files and run the following steps:
        - get the date from the file name,
        - verify that SAMPEX was not in spin mode,
        - load and resolve the 20 ms HILT file,
        - Run the microburst detection code on it, and
        - save to self.microbursts.
        """
        self.get_file_names()
        self.load_spin_times()
        self.microburst_times = pd.DataFrame(data=np.zeros((0, 2)), 
                                            columns=['dateTime', 'burst_param'])

        for hilt_file in self.hilt_files[2:]:
            date = self.get_filename_date(hilt_file)

            # Skip if the file name date was during the SAMPEX spin.
            # OR the data is from 1996
            if self.date_during_spin(date) or (date.year == 1996):
                continue
            
            # If the file is zipped set the zipped kwarg to True
            if 'zip' in hilt_file.suffix:
                zipped = True
            else:
                zipped = False

            # Load the data
            self.hilt_obj = load_hilt_data.Load_SAMPEX_HILT(date, 
                            zipped=zipped)
            # Resolve the 20 ms data
            self.hilt_obj.resolve_counts_state4()

            # Use the hilt data to id microbursts  
            try:
                self.id_microbursts()
            except ValueError as err:
                if str(err) == 'No detections found':
                    print(err, hilt_file.name)
                continue
            # self.test_detections()
        return

    def get_file_names(self):
        """
        Get a sorted list of file names in the State4 directory
        """
        hilt_files_generator = self.hilt_dir.glob('*')
        self.hilt_files = sorted(list(hilt_files_generator))
        return

    def date_during_spin(self, date):
        """ 
        This method checks if the file name taken on date
        was taken when SAMPEX was in the spin mode.
        """
        # This returns True if 
        in_spin_time = ((date >= self.spin_times['start']) & 
                        (date <= self.spin_times['end']))
        return any(in_spin_time) # True if the date was during the spin time.

    def load_spin_times(self):
        """ Load and parse the spin time file """
        spin_file_path = dirs.MISSION_TOOLS_SAMEX / self.spin_file_name
        self.spin_times = pd.read_csv(spin_file_path)
        for column in self.spin_times.columns:
            # Remove the time of day that SAMPEX entered spin mode 
            # since we will exclide the entire day.
            self.spin_times[column] = self.spin_times[column].apply(
                                        lambda s: s.split('T')[0])
            self.spin_times[column] = pd.to_datetime(self.spin_times[column])
        return

    def get_filename_date(self, file_path):
        """ Given a filename find the date using regex and pd.to_datetime"""
        file_name = file_path.name
        # Pick off the numbers out of the filename.
        year_doy_str = re.findall(r'\d+', str(file_name))[0]
        # Parse the date assuming a YYYYDOY format.
        return pd.to_datetime(year_doy_str, format='%Y%j')

    def id_microbursts(self):
        """ Use SignalToBackground class to identify microbursts """
        self.stb = signal_to_background.SignalToBackground(
                                    self.hilt_obj.counts, 20E-3, 
                                    self.baseline_width_s)
        self.stb.significance()
        self.stb.find_microburst_peaks(std_thresh=self.threshold)
        df = pd.DataFrame(
            data={
                'dateTime':pd.Series(self.hilt_obj.times[self.stb.peak_idt]),
                'burst_param':np.round(self.stb.n_std.values[self.stb.peak_idt].flatten(), 1)
                }
            )
        self.microburst_times = self.microburst_times.append(df)
        return 

    def save_catalog(self, save_name):
        """ Saves the microburst_times DataFrame to a csv file. """
        save_path = pathlib.Path('.', 'data', save_name) 
        self.microburst_times.to_csv(save_path, index=False)

    def test_detections(self):
        """ This method plots the microburst detections """
        # plt.plot(pd.Series(self.hilt_obj.times), self.hilt_obj.counts, 'k') 
        plt.plot(pd.Series(self.hilt_obj.times), self.hilt_obj.counts, 'k') 
        plt.scatter(pd.Series(self.hilt_obj.times[self.stb.peak_idt]), 
                    self.hilt_obj.counts[self.stb.peak_idt], 
                    c='r', marker='D')
        plt.show()

if __name__ == '__main__':
    m = Identify_SAMPEX_Microbursts()
    m.loop()