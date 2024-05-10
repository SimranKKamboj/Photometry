import numpy as np
import matplotlib.pyplot as plt

def plot_difference(datasets, use_mag):
    for i, (data_i, label_i) in enumerate(datasets):
        if use_mag == 7:
            mag_i_mean = data_i['MAG7'].mean()
            mag_i = data_i['MAG7']
            merr_i = data_i['MAGUNCERT7']
        elif use_mag == 5:
            mag_i_mean = data_i['MAG5'].mean()
            mag_i = data_i['MAG5']
            merr_i = data_i['MAGUNCERT5']

        for j, (data_j, label_j) in enumerate(datasets[i+1:], start=i+1):
            if use_mag == 7:
                mag_j = data_j['MAG7']
                merr_j = data_j['MAGUNCERT7']
            elif use_mag == 5:
                mag_j = data_j['MAG5']
                merr_j = data_j['MAGUNCERT5']

            mjd_i = (data_i['MJDATE'] - data_i['MJDATE'].iloc[0]) * 24
            mjd_j = (data_j['MJDATE'] - data_j['MJDATE'].iloc[0]) * 24

            mean_i = mag_i.mean()
            mean_j = mag_j.mean()

            delta_mag_ij = mag_j - mag_i
            mean_delta_mag_ij = delta_mag_ij.mean()

            err_delta = np.sqrt((merr_i**2)+(merr_j**2))
            
#             # Determine which error bars to use based on which magnitude is larger
#             max_mag = max(np.max(mag_i), np.max(mag_j))
#             if np.max(mag_i) == max_mag:
#                 error_bars = merr_i
#             else:
#                 error_bars = merr_j

            fit_ij = np.polyfit(mjd_j, delta_mag_ij, 2)
            poly_func_ij = np.poly1d(fit_ij)

            # Calculate residuals
            residuals = (delta_mag_ij - mean_delta_mag_ij)/err_delta

            # Calculate chi-square
            chi_square = np.sum(residuals ** 2)

            # Calculate degrees of freedom
            dof = len(delta_mag_ij) - len(fit_ij)

            # Calculate reduced chi-square
            reduced_chi_square = chi_square / dof

            plt.figure(figsize=(15, 6))
            # Plot the difference between MAG5 or MAG7 of j and MAG5 or MAG7 of i
            plt.plot(mjd_j, delta_mag_ij, 'o', label=f'ΔMAG{use_mag} ({label_j} - {label_i})')
            plt.errorbar(mjd_j, delta_mag_ij, yerr=err_delta, fmt='o', capsize=5)
            plt.axhline(y=mean_delta_mag_ij, color='r', linestyle='--', label=f'Mean ΔMAG{use_mag}: {mean_delta_mag_ij:.2f}')
            #plt.plot(mjd_j, poly_func_ij(mjd_j), 'b--', label='Polyfit Curve')
            plt.xlabel('MJD (hours)')
            plt.ylabel(f'ΔMAG{use_mag}')
            plt.title(f'Difference between MAG{use_mag} mean of {mean_i:.2f} star and MAG{use_mag} mean of {mean_j:.2f} star\nChi-square: {chi_square:.2f}, Reduced Chi-square: {reduced_chi_square:.2f}')
    #         plt.xlim(94, 100)
            # Annotate points with numbers
    #         for idx, (x, y) in enumerate(zip(mjd_j, delta_mag_ij)):
    #             plt.annotate(str(idx + 1), (x, y), textcoords="offset points", xytext=(0,10), ha='center', va='center', fontsize=12, color='blue')

            plt.legend()

            plt.show()
