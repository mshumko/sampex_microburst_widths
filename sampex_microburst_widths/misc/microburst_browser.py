import pathlib
import dateutil.parser
from datetime import date, datetime
import inspect

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib.widgets import Button, TextBox

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import load_hilt_data

plot_save_dir = pathlib.Path(config.PROJECT_DIR, '/plots/')
matplotlib.rcParams["savefig.directory"] = plot_save_dir

class Browser:
    def __init__(self, plot_width_s=5, catalog_name=None,
                catalog_save_name=None, filterDict={}, 
                jump_to_latest=True):
        """
        This class plots the AC6 microbursts and allows the user to browse
        detections in the future and past with buttons. Also there is a button
        to mark the event as a microburst.
        """
        self.plot_half_width_s=pd.Timedelta(seconds=plot_width_s/2)
        self.catalog_name = catalog_name
        # self.filter_catalog(filterDict=filterDict, defaultFilter=defaultFilter)

        if catalog_save_name is None:
            catalog_name_split =self.catalog_name.split('.')
            self.catalog_save_name = (catalog_name_split[0]+'_sorted'+
                                    catalog_name_split[-1])
        else:
            self.catalog_save_name = catalog_save_name
        self.catalog_path = pathlib.Path(config.PROJECT_DIR, 
                                            'data', self.catalog_name)
        self.catalog_save_path = pathlib.Path(config.PROJECT_DIR, 
                                            'data', self.catalog_save_name)

        # Load the original catalog
        self.load_microburst_catalog()

        # Load the filtered catalog if it already exists. This is
        # userful if you can't sort all microbursts at once!
        if pathlib.Path(self.catalog_save_path).exists():
            self.load_filtered_catalog()
        else:
            self.microburst_idx = np.array([])

        self.prev_date = date.min
        self._init_plot()
        if jump_to_latest and len(self.microburst_idx):
            self.index = self.microburst_idx[-1]
        else:
            # Start at row 0 in the dataframe.
            self.index = 0 
        self.plot()
        return

    def next(self, event):
        """ Plots the next detection """
        # Just return if at the end of the dataframe.
        if self.index + 1 >= self.catalog.shape[0]:
            return
        self.index += 1
        self.plot()
        return

    def prev(self, event):
        """ Plots the previous detection """
        # Just return if at the end of the dataframe.
        if self.index == 0:
            return
        self.index -= 1
        self.plot()
        return

    def append_remove_microburst(self, event):
        """ 
        Appends or removes the current catalog row to 
        self.filtered_catalog which will then
        be saved to a file for later processing.
        """
        if self.index not in self.microburst_idx:
            self.microburst_idx = np.append(self.microburst_idx, self.index)
            self.bmicroburst.color = 'g'
            print('Curtain saved at', self.catalog.iloc[self.index].dateTime)
        else:
            self.microburst_idx = np.delete(self.microburst_idx, 
                np.where(self.microburst_idx == self.index)[0])
            self.bmicroburst.color = '0.85'
            print('Curtain removed at', self.catalog.iloc[self.index].dateTime)
        return

    def key_press(self, event):
        """
        Calls an appropriate method depending on what key was pressed.
        """
        if event.key == 'm':
            # Mark as a curtain (can't use the "c" key since it is 
            # the clear command)
            self.append_remove_microburst(event)
        elif event.key == 'a':
            # Move self.index back and replot.
            self.prev(event)
        elif event.key =='d':
            # Move the self.index forward and replot.
            self.next(event)
        return
       
    def change_index(self, index):
        try:
            self.index = int(index)
            print('in change_index()')
        except ValueError:
            # Assume the passed value is a time.
            t = dateutil.parser.parse(index)
            n_time_index = date2num(t)
            n_time_catalog = date2num(self.catalog['dateTime'])
            self.index = np.argmin(np.abs(n_time_index - n_time_catalog))
        self.plot()
        return

    def plot(self):
        """ 
        Given a self.current_row in the dataframe, make a space-time plot 
        """

        # DEBUG
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('caller name:', calframe[1][3])

        current_row = self.catalog.iloc[self.index]
        current_time = self.catalog.index[self.index]
        print(f'Index position = {self.index}/{self.catalog.shape[0]-1} | '
            f'at time = {current_time}')
        self.index_box.set_val(self.index)
        self._clear_ax()

        if current_time.date() != self.prev_date:
            # Load current day AC-6 data if not loaded already
            print('Loading data from {}...'.format(current_time.date()), 
                    end=' ', flush=True)
            l = load_hilt_data.Load_SAMPEX_HILT(current_time)
            l.resolve_counts_state4()
            self.hilt = l.hilt_resolved
            self.prev_date = current_time.date()
            print('done.')

        # Turn microburst button green if this index has been marked as a microburst.
        if self.index in self.microburst_idx:
            self.bmicroburst.color = 'g'
        else:
            self.bmicroburst.color = '0.85'

        start_time = current_time-self.plot_half_width_s
        end_time = current_time+self.plot_half_width_s
        filtered_hilt = self.hilt.loc[start_time:end_time, :] 
        self.ax.plot(filtered_hilt.index, filtered_hilt.counts, c='k')
        self.ax.axvline(current_time, ls=':', c='g')

        # Plot the peak width in red
        print(self.catalog.columns)
        if 'left_peak_base' in self.catalog.columns:
            print(current_row[['width_height', 'left_peak_base', 'right_peak_base']])
            self.ax.hlines(current_row['width_height'], 
                        current_row['left_peak_base'],
                        current_row['right_peak_base'],
                        colors='r')

        self.ax.set_title('SAMPEX Microburst Browser\n {} {}'.format(
                        current_time.date(), 
                        current_time.time()))
        self.ax.set_ylabel('[counts/s]')
        self.ax.set_xlabel('UTC')
        
        # self._print_aux_info(current_row)

        t = num2date(self.ax.get_xlim()[0]).replace(tzinfo=None).replace(microsecond=0)
        save_datetime = t.strftime('%Y%m%d_%H%M')
        self.fig.canvas.get_default_filename = lambda: (
            f'{save_datetime}_sampex_microburst.png'
            )
        plt.draw()
        return

    def _print_aux_info(self, current_row):
        """ Print separation info as well as peak width info to the canvas. """
        self.textbox.clear()
        self.textbox.axis('off')

        current_row = current_row.copy()

        # Replace a few default values if they don't exist.
        if not hasattr(current_row, 'peak_width_A'):
            current_row['peak_width_A'] = np.nan
            current_row['peak_width_B'] = np.nan
        if not hasattr(current_row, 'time_cc'):
            current_row['time_cc'] = np.nan
            current_row['space_cc'] = np.nan

        col1 = ('Lag_In_Track = {} s\nDist_In_Track = {} km\n'
                    'Dist_total = {} km\npeak_width_A = {} s\n'
                    'peak_width_B = {} s'.format(
                    round(current_row['Lag_In_Track'], 1), 
                    round(current_row['Dist_In_Track'], 1), 
                    round(current_row['Dist_Total'], 1), 
                    round(current_row['peak_width_A'], 2), 
                    round(current_row['peak_width_B'], 2)))
        col2 = ('time_cc = {}\nspace_cc = {}\n'.format(
                    round(current_row['time_cc'], 2), 
                    round(current_row['space_cc'], 1)
                    ))
        self.textbox.text(0, 1, col1, va='top')
        self.textbox.text(1.3, 1, col2, va='top')
        return

    def _clear_ax(self):
        self.ax.clear()
        return 

    def _init_plot(self):
        """
        Initialize subplot objects and text box.
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(bottom=0.2)

        # Define button axes.
        self.axprev = plt.axes([0.54, 0.06, 0.12, 0.075])
        self.axburst = plt.axes([0.67, 0.06, 0.13, 0.075])
        self.axnext = plt.axes([0.81, 0.06, 0.12, 0.075])

        # Define buttons and their actions.
        self.bnext = Button(self.axnext, 'Next (d)', hovercolor='g')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous (a)', hovercolor='g')
        self.bprev.on_clicked(self.prev)
        self.bmicroburst = Button(self.axburst, 'Curtain (m)', hovercolor='g')
        self.bmicroburst.on_clicked(self.append_remove_microburst)

        # Define the textbox axes.
        self.textbox = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.textbox.axis('off')
        # Define index box.
        self.axIdx = plt.axes([0.59, 0.01, 0.32, 0.04])
        self.index_box = TextBox(self.axIdx, 'Index')
        self.index_box.on_submit(self.change_index)

        # Initialise button press
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        return

    def load_microburst_catalog(self):
        """
        Loads the original microburst catalog and saves to self.catalog.
        """
        self.catalog = pd.read_csv(self.catalog_path, 
                                index_col=0, parse_dates=True)
        if 'left_peak_base' in self.catalog.columns:
            self.catalog['left_peak_base'] = pd.to_datetime(self.catalog['left_peak_base'])
            self.catalog['right_peak_base'] = pd.to_datetime(self.catalog['right_peak_base'])
        return

    def load_filtered_catalog(self):
        """
        Load a filtered catalog and populate the self.microbirst_idx array
        with existing detections. This method exists to help the user resume
        the 
        """
        filtered_catalog = pd.read_csv(self.catalog_save_path, 
                                    index_col=0, parse_dates=True)
        # Convert times to numeric for faster computation of what times 
        flt_times_numeric = date2num(filtered_catalog.index)
        times_numeric = date2num(self.catalog.index)
        # Find the start microburst index
        self.microburst_idx = np.where(np.in1d(times_numeric, flt_times_numeric, 
                                    assume_unique=True))[0]
        return

    def save_filtered_catalog(self):
        """
        For every index that a user clicked microburst on, save
        those rows from the catalog into a new catalog with the
        name of self.catalog_save_name.
        """
        # Return if there are no micriobursts to save.
        if not hasattr(self, 'curtain_idx'):
            return
        # Remove duplicates indicies
        self.microburst_idx = np.unique(self.microburst_idx)
        print('Saving filtered catalog to {}'.format(self.catalog_save_path))
        df = self.catalog.iloc[self.microburst_idx]
        # Remove duplicate times (different than indicies since the same time
        # from the other sc may be assigned to a different index. 
        #df.drop_duplicates(subset='dateTime', inplace=True)

        # Save to csv file.
        df.to_csv(self.catalog_save_path, index=True)
        return


callback = Browser(catalog_name='microburst_test_catalog.csv',            
                    filterDict={}, plot_width_s=5)
# Initialize the GUI
plt.show()
# Save the catalog.
callback.save_filtered_catalog()
