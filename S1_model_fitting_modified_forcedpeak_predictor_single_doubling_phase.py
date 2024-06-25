import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def analyze_light_curve(time, mag, magerr):
    
    
    import pandas as pd
    import numpy as np
    from astropy.timeseries import LombScargle
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import matplotlib.cm as cm
    from scipy.optimize import curve_fit
    
    
    
    # Normalize the time values
    time_normalized = time - np.min(time)
    

    # Calculate Lomb-Scargle periodogram
    frequency, power = LombScargle(time_normalized, mag).autopower()

    # Plot the periodogram
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequency, power, color='blue')
    plt.xlabel('Period (time units)')
    plt.ylabel('Lomb-Scargle Power')
    plt.title('Lomb-Scargle Periodogram')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)

    # Find the peaks
    peaks, _ = find_peaks(power, height=0.1)  # Adjust the height as needed

    # Zoom in on the peaks
    plt.figure(figsize=(12, 6))
    plt.plot(1/frequency, power, color='blue')
    plt.xlim(1, 10)
    plt.ylim(0.01, 0.42)
    plt.xlabel('Period (time units)')
    plt.ylabel('Lomb-Scargle Power')
    plt.title('Lomb-Scargle Periodogram using astropy (Zoomed In)')
    plt.grid(True)
    plt.scatter(1/frequency[peaks], power[peaks], color='red', marker='x', label='Peaks')

    # Print and annotate the first 12 peaks
    print("First 12 peaks:")
    for i, period in enumerate(1 / frequency[peaks][:12]):
        label = f'P{i+1} ({period:.2f})'  # Include period information in the label
        print(f"{label}: Lomb-Scargle Power = {power[peaks[i]]:.4f}")
        plt.annotate(label, xy=(period, power[peaks[i]]), xytext=(period, power[peaks[i]] + 0.06),
                     arrowprops=dict(facecolor='black', arrowstyle='->', mutation_scale=10), fontsize=8)

    # Find the best peak based on power
    best_peak_index = None
    best_peak_power = -float('inf')
    for peak in peaks:
        period = 1 / frequency[peak]
        if 3 < period < 20:  # Adjust the range if needed
            if power[peak] > best_peak_power:
                best_peak_index = peak
                best_peak_power = power[peak]

# If a best peak is found, print and annotate it
    if best_peak_index is not None:
        best_period = 1 / frequency[best_peak_index]
        best_power = power[best_peak_index]
        print(f"Best Peak: P ({best_period:.2f}), Lomb-Scargle Power = {best_power:.4f}")
        plt.annotate("Best Peak", xy=(best_period, best_power), xytext=(best_period+0.5, best_power - 0.1),
                 arrowprops=dict(facecolor='green', arrowstyle='->', mutation_scale=10), color='green', fontsize=8)
    else:
        print("No peak found with period between 3 and 5")
        
    plt.legend()
    plt.show()

    period = best_period
    print(period)
    
    
    # Create colormap with four colors
    cmap = cm.get_cmap('Spectral_r', 7)

    # Define the x-axis value ranges and corresponding colors
    color_values = [(0,6),(92,100),(116,124)]

    # Map x-axis values to colors
    colors = [cmap(i) for i in range(len(color_values))]

    # Define colors for each range
    colors_scatter = ['#2F4F4F', '#FF7F50', '#DB7093', 'mediumvioletred', 'lightgreen']

    # Plot
    plt.figure(figsize=(10, 10))

    omega1 = 2 * np.pi / period

    # Define the one sine function with modified frequency
    def single_sine_function(x, A1, phi1, offset):
        return (A1 * np.sin(omega1 * x + phi1)) + offset

    # Fit the one sin function to the data with modified frequency, including magerr_7
    params, covariance = curve_fit(single_sine_function, time, mag, sigma=magerr)

    # Extract the fitted parameters
    A1_fit, phi1_fit, offset_fit = params
    
    # A1_fit = 0.085
    # Generate points for the fitted curve
    x_fit = np.linspace(time.min(), time.max(), 1000)
    y_fit = single_sine_function(x_fit, A1_fit, phi1_fit,offset_fit)

    # Define a list of colors for scatter points based on the x-axis values
    scatter_colors = []

    # Plot the data points with colors based on the x-axis values and annotate each point
    for i, (x, y) in enumerate(zip(time, mag)):
        for j, (xmin, xmax) in enumerate(color_values):
            if xmin <= x <= xmax:
                plt.errorbar(x, y, yerr=magerr[i],
                             fmt='o', color=colors[j], markersize=4, capsize=3, label="Data")
    #             plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    # Plot the fitted curve
    
    # Read data of the sixth night
   
    # sixth_night_data = pd.read_csv('tno_star1.csv', skiprows=[3,9]+list(range(12,97)))# 1.3 sigma [9] away point

    # # Extract time, delta mag, and delta error for the sixth night
    # sixth_night_time = sixth_night_data['OTIME']
    # sixth_night_delta_mag = sixth_night_data['DELTAMAG7']
    # sixth_night_magerr = sixth_night_data['DELTAERR7']

    # # Plot data of the sixth night
    # plt.errorbar(sixth_night_time, sixth_night_delta_mag, yerr=sixth_night_magerr, color='OliveDrab', label='1st Night Data')
    # plt.scatter(sixth_night_time, sixth_night_delta_mag, c='OliveDrab', marker='D', s=40, edgecolors='black', linewidths=0.5, label="1st Night Data", zorder=3)
    
    
    # Plot the data points with custom colors and set the zorder
    plt.scatter(time, mag, c=scatter_colors, marker='o', s=40, edgecolors='black', linewidths=0.5, label="Data", zorder=3)
    # Extend the time range till 155 hours
    extended_time = np.linspace(0, time.max(), 1000)

    # Generate synthetic data using the fitted model parameters
    synthetic_mag = single_sine_function(extended_time, A1_fit, phi1_fit, offset_fit)

    # Plot the synthetic data along with the actual data
    plt.plot(extended_time, synthetic_mag, color='navy', label='Predicted Model')
    plt.xlabel('Time (hours)')
    plt.ylabel('Delta mag')
    plt.title(f'Prediction of Delta mag beyond the observed range w period:{period:.2f}', fontsize=30)
  
    

    #plt.gca().invert_yaxis()
    plt.grid(True)

    print("A1:", A1_fit)

    print("phi1:", phi1_fit)
   
    print("offset:", offset_fit)

    period1 = period
 

    print("Period of the the sine function:", period)
    
#     plt.xlim(0,7)
    plt.show()
    
    
    


    # Calculate the number of data points
    N = len(time)

    # Calculate the chi-square value
    expected_values = single_sine_function(time, A1_fit, phi1_fit, offset_fit)
    squared_differences = (expected_values - mag)**2
    squared_differences /= magerr**2
    chi_square = np.sum(squared_differences)
    parameters = 4
    # Calculate the degrees of freedom
    degrees_of_freedom = N - parameters 

    # Calculate the reduced chi-square
    reduced_chi_square_b = chi_square / degrees_of_freedom

    print(f'Reduced Chi-Square Value before (parameters={parameters}):', reduced_chi_square_b)
    
    # Phase fold the extended time
    extended_phase = (extended_time % period) / period

    # # Concatenate data from the sixth night
    # extended_time_concat = np.concatenate((time, sixth_night_time))
    # extended_mag_concat = np.concatenate((mag, sixth_night_delta_mag))
    # extended_magerr_concat = np.concatenate((magerr, sixth_night_magerr))

     # Concatenate data from the sixth night
   
    
    # Calculate the number of data points
    # N = len(extended_time_concat)

    # # Calculate the chi-square value
    # expected_values = single_sine_function(extended_time_concat, A1_fit, phi1_fit, offset_fit)
    # squared_differences = (expected_values - extended_mag_concat)**2
    # squared_differences /= extended_magerr_concat**2
    # chi_square = np.sum(squared_differences)
    # parameters = 4
    # # Calculate the degrees of freedom
    # degrees_of_freedom = N - parameters 

    # # Calculate the reduced chi-square
    # reduced_chi_square_a = chi_square / degrees_of_freedom
    
    
    # print(f'Reduced Chi-Square Value after (parameters={parameters}):', reduced_chi_square_a)
    
    
    # Calculate the differences between original mag values and the fitted curve
    differences = mag - single_sine_function(time, A1_fit, phi1_fit,  offset_fit)
    plt.figure(figsize=(8, 5))
    # Plot histogram
    plt.hist(differences, bins=20, color='MediumPurple', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals around model fitting')
    plt.show()


#     import pandas as pd
#     import matplotlib.pyplot as plt


#     intervals = [
#         (0, 5.0548296),   
#         (24.2964552, 28.76268),  
#         (47.5757736, 52.1618544), 
#         (72.0810648, 76.5702432)  ]


#     # Calculate the phase
#     df['Phase'] = (df[time_column] % period) / period

#     # Sort the dataframe by the phase values
#     df.sort_values(by='Phase', inplace=True)

#     # Plot the phase-folded lightcurve with different colors for each interval
#     plt.figure(figsize=(8, 5))

#     for i, (start, end) in enumerate(intervals):
#         interval_mask = (df[time_column] >= start) & (df[time_column] <= end)
#         plt.scatter(
#             df.loc[interval_mask, 'Phase'],
#             df.loc[interval_mask, magnitude_column],
#             s=10,
#             label=f'Interval {i + 1}',
#             alpha=0.7
#         )
    
# #     # Annotate points with numbers
# #     for idx, (x, y) in enumerate(zip(df.loc[interval_mask, 'Phase'], df.loc[interval_mask, magnitude_column])):
# #         plt.text(x, y, str(idx + 1), color='blue', fontsize=12, ha='center', va='center')

# # Define a sinusoidal function
#     def sinusoidal(x, amplitude, frequency, phase, offset):
#         return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

#     # Fit the sinusoidal curve to the data
#     p0 = [1, 1, 0, np.mean(df[magnitude_column])]  # Initial guess for parameters
#     popt, _ = curve_fit(sinusoidal, df['Phase'], df[magnitude_column], p0=p0)

#     # Generate points for the fitted curve
#     fit_curve = sinusoidal(df['Phase'], *popt)

#     # Plot the fitted sinusoidal curve
#     plt.plot(df['Phase'], fit_curve, color='black', linestyle='--', label='Fitted Sinusoidal Curve')

#     plt.title(f'Phase-folded Lightcurve for period: {period:.2f} hrs')
#     plt.xlabel('Phase')
#     plt.ylabel('Delta mag')
#     plt.gca().invert_yaxis()  # Invert y-axis for magnitude
#     plt.legend()
#     plt.show()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Define the intervals for phase folding
    intervals = [
        (0,6),(92,100),(116,124)]
#     ]
    
    # Calculate the phase
    period = period1  # Use the period derived from the previous fitting
    phase = (time % period) / period
    
   # Sort the data by the phase values
    sorted_indices = np.argsort(phase)
    phase_sorted = phase[sorted_indices]
    mag_sorted = mag[sorted_indices]
    time_sorted = time[sorted_indices]
    magerr_sorted = magerr[sorted_indices]

    
    # Extend phase to 1.1 by repeating 0.0 to 0.1 range
    extra_phase_mask = (phase_sorted <= 0.3)
    extra_phase = phase_sorted[extra_phase_mask] + 1.0
    extended_phase_sorted = np.concatenate([phase_sorted, extra_phase])
    extended_mag_sorted = np.concatenate([mag_sorted, mag_sorted[extra_phase_mask]])
    extended_magerr_sorted = np.concatenate([magerr_sorted, magerr_sorted[extra_phase_mask]])
    extended_time_sorted = np.concatenate([time_sorted, time_sorted[extra_phase_mask]])
    
# Plot the phase-folded light curve with different colors for each interval
    plt.figure(figsize=(10, 8))
    cmap_phase = plt.get_cmap('jet_r')  # Choose a colormap

    for i, (start, end) in enumerate(intervals):
        interval_mask = (extended_time_sorted >= start) & (extended_time_sorted <= end)
        plt.errorbar(
            extended_phase_sorted[interval_mask],
            extended_mag_sorted[interval_mask] - np.mean(extended_mag_sorted),  # Deviation from the mean magnitude
            yerr=extended_magerr_sorted[interval_mask],  # adding error bars
            fmt='o',
            color=cmap_phase(i / len(intervals)),
            label=f'Night {i + 2}',
            alpha=0.7
        )
    # Generate synthetic data using the parameters from the fitted model
    synthetic_time = np.linspace(time_sorted.min(), time_sorted.max(), 1000)
    synthetic_mag = single_sine_function(synthetic_time, A1_fit, phi1_fit, offset_fit)
    
    # Phase fold the synthetic data
    synthetic_phase = (synthetic_time % period) / period
    
    # Extend synthetic phase to 2.1 by repeating 0.0 to 0.1 range
    extra_synthetic_phase_mask = (synthetic_phase <= 0.3)
    extra_synthetic_phase = synthetic_phase[extra_synthetic_phase_mask] + 1.0
    extended_synthetic_phase = np.concatenate([synthetic_phase, extra_synthetic_phase])
    extended_synthetic_mag = np.concatenate([synthetic_mag, synthetic_mag[extra_synthetic_phase_mask]])

    
   # Plot the synthetic data
    plt.scatter(
        extended_synthetic_phase,
        extended_synthetic_mag - np.mean(extended_synthetic_mag),  # Centering at 0
        marker='+',
        color='indigo',
        label='Data using model parameters',
        alpha=0.5
    )
    plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=1)  # Add vertical dotted line at 2.0

   
     # Phase fold the sixth night data
    # sixth_night_phase = (sixth_night_time % period) / period

    # # Plot data of the sixth night
    # plt.errorbar(
    #     sixth_night_phase,
    #     sixth_night_delta_mag - np.mean(sixth_night_delta_mag),  # Deviation from the mean magnitude
    #     yerr=sixth_night_magerr, 
    #     fmt='D',# adding error bars
    #     color='RebeccaPurple',
    #     label='1st Night Data predicted',
    #     alpha = 0.7
    # )
    
    #  # Phase fold the sixth night data
    

    
    plt.title(f'Phase-folded (Once) Lightcurve for period: {period:.2f} hrs')
    
    plt.xlabel('Phase')
    plt.ylabel(f'Delta mag-{np.mean(mag_sorted):.2f}')
    plt.ylim(-np.max(np.abs(mag - np.mean(mag)))-0.15, np.max(np.abs(mag - np.mean(mag)))+0.15)
 # Set y-axis limits
    plt.legend()
    plt.show()
    
    
    #new method

# Calculate the phase with halved period
    period = 2*period1  # Use the period derived from the previous fitting
    phase = (time % period) / (period/2)
    
   # Sort the data by the phase values
    sorted_indices = np.argsort(phase)
    phase_sorted = phase[sorted_indices]
    mag_sorted = mag[sorted_indices]
    time_sorted = time[sorted_indices]
    magerr_sorted = magerr[sorted_indices]

    
    # Extend phase to 1.1 by repeating 0.0 to 0.1 range
    extra_phase_mask = (phase_sorted <= 0.3)
    extra_phase = phase_sorted[extra_phase_mask] + 2.0
    extended_phase_sorted = np.concatenate([phase_sorted, extra_phase])
    extended_mag_sorted = np.concatenate([mag_sorted, mag_sorted[extra_phase_mask]])
    extended_magerr_sorted = np.concatenate([magerr_sorted, magerr_sorted[extra_phase_mask]])
    extended_time_sorted = np.concatenate([time_sorted, time_sorted[extra_phase_mask]])
    
# Plot the phase-folded light curve with different colors for each interval
    plt.figure(figsize=(10, 8))
    cmap_phase = plt.get_cmap('jet_r')  # Choose a colormap

    for i, (start, end) in enumerate(intervals):
        interval_mask = (extended_time_sorted >= start) & (extended_time_sorted <= end)
        plt.errorbar(
            extended_phase_sorted[interval_mask],
            extended_mag_sorted[interval_mask] - np.mean(extended_mag_sorted),  # Deviation from the mean magnitude
            yerr=extended_magerr_sorted[interval_mask],  # adding error bars
            fmt='o',
            color=cmap_phase(i / len(intervals)),
            label=f'Night {i + 2}',
            alpha=0.7
        )
    # Generate synthetic data using the parameters from the fitted model
    synthetic_time = np.linspace(time_sorted.min(), time_sorted.max(), 1000)
    synthetic_mag = single_sine_function(synthetic_time, A1_fit, phi1_fit, offset_fit)
    
    # Phase fold the synthetic data
    synthetic_phase = (synthetic_time % period) / (period/2)
    
    # Extend synthetic phase to 2.1 by repeating 0.0 to 0.1 range
    extra_synthetic_phase_mask = (synthetic_phase <= 0.3)
    extra_synthetic_phase = synthetic_phase[extra_synthetic_phase_mask] + 2.0
    extended_synthetic_phase = np.concatenate([synthetic_phase, extra_synthetic_phase])
    extended_synthetic_mag = np.concatenate([synthetic_mag, synthetic_mag[extra_synthetic_phase_mask]])

    
   # Plot the synthetic data
    plt.scatter(
        extended_synthetic_phase,
        extended_synthetic_mag - np.mean(extended_synthetic_mag),  # Centering at 0
        marker='+',
        color='indigo',
        label='Data using model parameters',
        alpha=0.5
    )
    plt.axvline(x=2.0, color='gray', linestyle='--', linewidth=1)  # Add vertical dotted line at 2.0

   
    #  # Phase fold the sixth night data
    # sixth_night_phase = (sixth_night_time % period) / (period/2)

    # # Plot data of the sixth night
    # plt.errorbar(
    #     sixth_night_phase,
    #     sixth_night_delta_mag - np.mean(sixth_night_delta_mag),  # Deviation from the mean magnitude
    #     yerr=sixth_night_magerr, 
    #     fmt='D',# adding error bars
    #     color='RebeccaPurple',
    #     label='1st Night Data predicted',
    #     alpha = 0.7
    # )
    
     # Phase fold the sixth night data
    

    
    plt.title(f'Phase-folded (Twice) Lightcurve for period: {period:.2f} hrs')
    
    plt.xlabel('Phase')
    plt.ylabel(f'Delta mag-{np.mean(mag_sorted):.2f}')
    plt.ylim(-np.max(np.abs(mag - np.mean(mag)))-0.15, np.max(np.abs(mag - np.mean(mag)))+0.15)
 # Set y-axis limits
    plt.legend()
    plt.show()
    
    
    
    # new method
    # Phase fold once
    phase_once = (time / period1) % 1
    sorted_indices_once = np.argsort(phase_once)
    phase_sorted_once = phase_once[sorted_indices_once]
    magnitude_sorted_once = mag[sorted_indices_once]
    
    # Phase fold twice
    phase_twice = (time / (2 * period1)) % 1
    sorted_indices_twice = np.argsort(phase_twice)
    phase_sorted_twice = phase_twice[sorted_indices_twice]
    magnitude_sorted_twice = mag[sorted_indices_twice]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Single-phase folded light curve
    plt.subplot(1, 2, 1)
    plt.errorbar(phase_sorted_once, magnitude_sorted_once, yerr=magerr, fmt='o')
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.title('Single-phase Folded Light Curve')
    
    # Double-phase folded light curve
    plt.subplot(1, 2, 2)
    plt.errorbar(phase_sorted_twice, magnitude_sorted_twice, yerr=magerr, fmt='o')
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.title('Double-phase Folded Light Curve')
    
    plt.tight_layout()
    plt.show()
    
        
        # old method
        
#     # Calculate the phase, folding twice (i.e., divide period by 2)
#     period = 2*period1  # Use the period derived from the previous fitting
#     phase = (time % period) / (period / 2)

# # Sort the data by the phase values
#     sorted_indices = np.argsort(phase)
#     phase_sorted = phase[sorted_indices]
#     mag_sorted = mag[sorted_indices]
#     time_sorted = time[sorted_indices]
#     magerr_sorted = magerr[sorted_indices]

#     # Extend phase to 2.1 by repeating 0.0 to 0.1 range
#     extra_phase_mask = (phase_sorted <= 0.3)
#     extra_phase = phase_sorted[extra_phase_mask] + 2.0
#     extended_phase_sorted = np.concatenate([phase_sorted, extra_phase])
#     extended_mag_sorted = np.concatenate([mag_sorted, mag_sorted[extra_phase_mask]])
#     extended_magerr_sorted = np.concatenate([magerr_sorted, magerr_sorted[extra_phase_mask]])
#     extended_time_sorted = np.concatenate([time_sorted, time_sorted[extra_phase_mask]])

# # Plot the phase-folded light curve with different colors for each interval
#     plt.figure(figsize=(10, 8))
#     cmap_phase = plt.get_cmap('jet_r')  # Choose a colormap

#     for i, (start, end) in enumerate(intervals):
#         interval_mask = (extended_time_sorted >= start) & (extended_time_sorted <= end)
#         plt.errorbar(
#             extended_phase_sorted[interval_mask],
#             extended_mag_sorted[interval_mask] - np.mean(extended_mag_sorted),  # Deviation from the mean magnitude
#             yerr=extended_magerr_sorted[interval_mask],  # adding error bars
#             fmt='o',
#             color=cmap_phase(i / len(intervals)),
#             label=f'Night {i + 2}',
#             alpha=0.7
#         )

# # Generate synthetic data using the parameters from the fitted model
#     synthetic_time = np.linspace(time_sorted.min(), time_sorted.max(), 1000)
#     synthetic_mag = single_sine_function(synthetic_time, *params)

# # Phase fold the synthetic data, folding twice
#     synthetic_phase = (synthetic_time % period) / (period / 2)

# # Extend synthetic phase to 2.1 by repeating 0.0 to 0.1 range
#     extra_synthetic_phase_mask = (synthetic_phase <= 0.3)
#     extra_synthetic_phase = synthetic_phase[extra_synthetic_phase_mask] + 2.0
#     extended_synthetic_phase = np.concatenate([synthetic_phase, extra_synthetic_phase])
#     extended_synthetic_mag = np.concatenate([synthetic_mag, synthetic_mag[extra_synthetic_phase_mask]])

# # Plot the synthetic data
#     plt.scatter(
#         extended_synthetic_phase,
#         extended_synthetic_mag - np.mean(extended_synthetic_mag),  # Centering at 0
#         marker='+',
#         color='indigo',
#         label='Data using model parameters',
#         alpha=0.5
#     )
#     plt.axvline(x=2.0, color='gray', linestyle='--', linewidth=1)  # Add vertical dotted line at 2.0
    
    
#       # Phase fold the sixth night data
#     sixth_night_phase = (sixth_night_time % period) / period

#     # Plot data of the sixth night
#     plt.errorbar(
#         sixth_night_phase,
#         sixth_night_delta_mag - np.mean(sixth_night_delta_mag),  # Deviation from the mean magnitude
#         yerr=sixth_night_magerr, 
#         fmt='D',# adding error bars
#         color='RebeccaPurple',
#         label='1st Night Data predicted',
#         alpha = 0.7
#     )

#     plt.title(f'Phase-folded (Twice) Lightcurve for period: {period:.2f} hrs (Single peak model folded twice)')
#     plt.xlabel('Phase')
#     plt.ylabel('Delta mag')
#     plt.legend()
#     plt.show()

