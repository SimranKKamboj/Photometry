import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.stats import chi2

def analyze_light_curve(time, mag, magerr):
    
    import pandas as pd
    import numpy as np
    from astropy.timeseries import LombScargle
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import matplotlib.cm as cm
    from scipy.optimize import curve_fit
    # Normalize the time values
    time -= np.min(time)

    # Calculate Lomb-Scargle periodogram
    frequency, power = LombScargle(time, mag).autopower()

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
    plt.xlim(1, 25)
    plt.ylim(0.01, 0.7)
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

    # Print and annotate the best peak
    best_peak = None
    for peak in peaks:
        period = 1 / frequency[peak]
        if 3 < period < 60:
            best_peak = peak
            break
    if best_peak is not None:
        best_period = 1 / frequency[best_peak]
        best_power = power[best_peak]
        print(f"Best Peak: P ({best_period:.2f}), Lomb-Scargle Power = {best_power:.4f}")
        plt.annotate("Best Peak", xy=(best_period, best_power), xytext=(best_period+0.5, best_power - 0.1),
                     arrowprops=dict(facecolor='green', arrowstyle='->', mutation_scale=10), color='green', fontsize=8)
    else:
        print("No peak found with period between 3 and 60")
        
    plt.legend()
    plt.show()

    period = best_period
    print(period)

    # Create colormap with four colors
    cmap = cm.get_cmap('Spectral_r', 6)

    # Define the x-axis value ranges and corresponding colors
    color_values = [(0, 6),(92,100),(116,124)]

    # Map x-axis values to colors
    colors = [cmap(i) for i in range(len(color_values))]

    # Define colors for each range
    colors_scatter = ['#2F4F4F', '#FF7F50', '#DB7093', 'mediumvioletred', 'lightgreen']

    # Plot
    plt.figure(figsize=(15, 6))

    omega1 = 2 * np.pi / period

    # Define the sum of two sine functions with modified frequencies
    def double_sine_function(x, A1, phi1, A2, phi2, offset):
        return (A1 * np.sin(omega1 * x + phi1) + A2 * np.sin((1/2) * omega1 * x + phi2)) + offset

    # Fit the sum of two sine functions to the data with modified frequencies, including magerr_7
    params, covariance = curve_fit(double_sine_function, time, mag, sigma=magerr)

    # Extract the fitted parameters
    A1_fit, phi1_fit, A2_fit, phi2_fit, offset_fit = params

    # Generate points for the fitted curve
    x_fit = np.linspace(time.min(), time.max(), 1000)
    y_fit = double_sine_function(x_fit, A1_fit, phi1_fit, A2_fit, phi2_fit, offset_fit)

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
    plt.plot(x_fit, y_fit, color='navy', label="Fitted Double Sine Curve")

    # Plot the data points with custom colors and set the zorder
    plt.scatter(time, mag, c=scatter_colors, marker='o', s=40, edgecolors='black', linewidths=0.5, label="Data", zorder=3)

    # Labels, title, legend, etc
    plt.xlabel('Hours', fontsize=25)
    plt.ylabel('Delta mag', fontsize=25)
    plt.title(f'Photometry of Jan23 with a fitted light curve w period: {period:.2f} hrs', fontsize=30)

    print("A1:", A1_fit)
    print("A2:", A2_fit)
    print("phi1:", phi1_fit)
    print("phi2:", phi2_fit)
    print("offset:", offset_fit)

    period1 = period
    period2 = 2 * 2 * np.pi / (omega1)  # Period of the second sine component

    print("Period of the first sine function:", period)
    print("Period of the second sine function:", period2)

    plt.show()


    # Calculate the number of data points
    N = len(time)

    # Calculate the chi-square value
    expected_values = double_sine_function(time, A1_fit, phi1_fit, A2_fit, phi2_fit, offset_fit)
    squared_differences = (expected_values - mag)**2
    squared_differences /= magerr**2
    chi_square = np.sum(squared_differences)

    parameters  = 6
    # Calculate the degrees of freedom
    degrees_of_freedom = N - parameters 

    # Calculate the reduced chi-square
    reduced_chi_square_b = chi_square / degrees_of_freedom

    print(f'Reduced Chi-Square Value before plantation (parameters={parameters}):', reduced_chi_square_b)
    
    
    
    
    # Calculate the differences between original mag values and the fitted curve
    differences = mag - double_sine_function(time, A1_fit, phi1_fit, A2_fit, phi2_fit, offset_fit)
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
        (0, 6),(92,100),(116,124)]#,
#         (142.7628, 146.75904)
#     ]
    
    # Calculate the phase
    period = 2*period1  # Use the period derived from the previous fitting
    phase = (time % period) / period
    
    # Sort the data by the phase values
    sorted_indices = np.argsort(phase)
    phase_sorted = phase[sorted_indices]
    mag_sorted = mag[sorted_indices]
    time_sorted = time[sorted_indices]
    
# Plot the phase-folded light curve with different colors for each interval
    plt.figure(figsize=(9, 8))
    cmap_phase = plt.get_cmap('jet_r')  # Choose a colormap
    
    for i, (start, end) in enumerate(intervals):
        interval_mask = (time_sorted >= start) & (time_sorted <= end)
        plt.errorbar(
            phase_sorted[interval_mask],
            mag_sorted[interval_mask] - np.mean(mag_sorted),  # Deviation from the mean magnitude
            yerr=magerr[interval_mask],  # adding error bars
            fmt='o',
            color=cmap_phase(i/len(intervals)),
            label=f'Interval {i + 1}',
            alpha=0.7
        )
    
    # Generate synthetic data using the parameters from the fitted model
    synthetic_time = np.linspace(time_sorted.min(), time_sorted.max(), 1000)
    synthetic_mag = double_sine_function(synthetic_time, *params)
    
    # Phase fold the synthetic data
    synthetic_phase = (synthetic_time % period) / period
    
    # Plot the synthetic data
    plt.scatter(
        synthetic_phase,
        synthetic_mag - np.mean(synthetic_mag),  # Centering at 0
        marker='+',
        color='indigo',
        label='Data using model parameters',
        alpha=0.5
    )
    
   
    
    
    plt.title(f'Phase-folded Lightcurve for period: {period:.2f} hrs')
    plt.xlabel('Phase')
    plt.ylabel(f'Delta mag-{np.mean(mag_sorted):.2f}')
    plt.ylim(-np.max(np.abs(mag - np.mean(mag)))-0.15, np.max(np.abs(mag - np.mean(mag)))+0.15)
 # Set y-axis limits
    plt.legend()
    plt.show()
