import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sampex_microburst_widths.microburst_id import index_intervals

class SignalToBackground:
    def __init__(self, counts, cadence, background_width_s, 
                foreground_width_s=0.1):
        """ 
        This class implements the signal to background 
        microburst detection. This method is a generalization 
        of the O'Brien 2003 burst parameter that uses a 0.5 
        second baseline instead of the longer baselines used
        in the examples here.

        Parameters
        ----------
        counts : array
            Array of counts. Should be continuous
        background_width_s : float
            The baseline width in time to calculate the running mean
        """
        self.counts = counts

        # Check that counts is a DataFrame
        if not isinstance(self.counts, pd.DataFrame):
            self.counts = pd.DataFrame(self.counts)

        self.cadence = cadence
        self.background_width_s = background_width_s
        self.foreground_width_s = foreground_width_s
        return

    def significance(self):
        """
        Calculate the number of background standard deviations, 
        assuming Poisson statistics, that a count value is above
        a rolling average background of length self.background_width_s.

        Returns a pandas DataFrame object that can be 
        converted to numpy using the .to_numpy() method.
        """
        self.rolling_microburst_counts = self._running_average(self.counts, self.foreground_width_s)
        self.rolling_background_counts = self._running_average(self.counts, self.background_width_s)
        self.n_std = ((self.rolling_microburst_counts-self.rolling_background_counts)/\
                    np.sqrt(self.rolling_background_counts+1))
        return self.n_std

    def find_microburst_peaks(self, std_thresh=2):
        """
        This method finds the data intervals where the 
        microburst criteria is satisfied. Then for
        every interval, calculate the time of the highest
        peak.

        Parameters
        ----------
        std_thresh : float
            The baseline standard deviation threshold above the baseline
            that the data point must be to satisfy the microburst criteria
        """
        self.criteria_idt = np.where(self.n_std >= std_thresh)[0]

        if len(self.criteria_idt) <= 1:
            raise ValueError('No detections found')

        interval_start, interval_end = index_intervals.get_index_intervals(self.criteria_idt)
        self.peak_idt = np.nan*np.ones(interval_start.shape[0])

        # Loop over each continous interval and find the peak index.
        for i, (start, end) in enumerate(zip(interval_start, interval_end)):
            if start == end:
                end+=1
            offset = self.criteria_idt[start]
            self.peak_idt[i] = offset + np.argmax(self.counts.loc[self.criteria_idt[start:end]])
        self.peak_idt = self.peak_idt.astype(int)
        return self.peak_idt

    def _running_average(self, counts, window_width):
        """
        Calculate the running average of the counts array.
        """
        n_samples = int(window_width/self.cadence)
        return counts.rolling(n_samples, center=True).mean()