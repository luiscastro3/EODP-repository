import os
import numpy as np
import netCDF4 as nc


def validate_equalised_results(ref_dir, my_dir, threshold_pct=1e-3):
    # Convert percentage to decimal (1e-3 % = 1e-5)
    threshold = threshold_pct / 100.0

    # Filter for NetCDF files based on your directory structure
    files_to_check = [f for f in os.listdir(my_dir) if f.startswith('l1b_toa') and f.endswith('.nc')]

    if not files_to_check:
        print("No matching equalised .nc files found in the target directory.")
        return

    all_passed = True

    for filename in files_to_check:
        ref_path = os.path.join(ref_dir, filename)
        my_path = os.path.join(my_dir, filename)

        if not os.path.exists(ref_path):
            print(f"Skipping {filename}: Not found in teacher's output directory.")
            continue

        print(f"Validating: {filename}...")

        # Open both datasets
        with nc.Dataset(ref_path, 'r') as ds_ref, nc.Dataset(my_path, 'r') as ds_my:
            # Iterate through all variables in the NetCDF file
            for var_name in ds_ref.variables:
                if var_name not in ds_my.variables:
                    print(f"  [ERROR] Variable '{var_name}' missing in your output.")
                    all_passed = False
                    continue

                # Extract data as numpy arrays
                ref_data = ds_ref.variables[var_name][:]
                my_data = ds_my.variables[var_name][:]

                # Ensure the data types are numeric before calculating differences
                if not np.issubdtype(ref_data.dtype, np.number):
                    continue

                # Handle masked arrays (common in NetCDF) by filling with NaNs for safe comparison
                if np.ma.isMaskedArray(ref_data):
                    ref_data = ref_data.filled(np.nan)
                if np.ma.isMaskedArray(my_data):
                    my_data = my_data.filled(np.nan)

                # Calculate relative difference: |(my - ref) / ref|
                # Suppress division by zero warnings to handle them cleanly
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_diff = np.abs((my_data - ref_data) / ref_data)

                # Handle edge cases: Where ref_data is 0, check if my_data is also 0
                rel_diff[np.isnan(rel_diff)] = 0.0  # Ignore regions where both are NaN
                zero_mask = (ref_data == 0)
                rel_diff[zero_mask] = np.abs(my_data[zero_mask])  # Absolute diff if ref is 0

                # Get the maximum relative difference
                max_rel_diff = np.nanmax(rel_diff)

                if max_rel_diff >= threshold:
                    print(f"  [FAIL] {var_name} | Max Rel Diff: {max_rel_diff * 100:.6e}% >= {threshold_pct}%")
                    all_passed = False
                else:
                    print(f"  [PASS] {var_name} | Max Rel Diff: {max_rel_diff * 100:.6e}% < {threshold_pct}%")

    print("\n--- Final Verdict ---")
    if all_passed:
        print(f"SUCCESS: All equalised results demonstrated a relative difference lower than {threshold_pct}%")
    else:
        print("FAILED: One or more variables exceeded the maximum allowed relative difference.")


# Define your paths here
teacher_output = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-L1B\output"
my_output = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-L1B\myOutput"

# Run validation
validate_equalised_results(teacher_output, my_output)

import netCDF4 as nc
import matplotlib.pyplot as plt


def recreate_equalization_plot(file_eq, file_no_eq, file_isrf, alt_line_idx=50):
    """
    Reads 3 NetCDF files and plots a single across-track line to compare Equalization effects.
    """
    # Open datasets
    ds_eq = nc.Dataset(file_eq, 'r')
    ds_no_eq = nc.Dataset(file_no_eq, 'r')
    ds_isrf = nc.Dataset(file_isrf, 'r')

    # Extract the target variable (handling 'toaw' or 'toa' naming variations)
    var_eq = ds_eq.variables['toaw'][:] if 'toaw' in ds_eq.variables else ds_eq.variables['toa'][:]
    var_no_eq = ds_no_eq.variables['toaw'][:] if 'toaw' in ds_no_eq.variables else ds_no_eq.variables['toa'][:]
    var_isrf = ds_isrf.variables['toaw'][:] if 'toaw' in ds_isrf.variables else ds_isrf.variables['toa'][:]

    # The files are 2D (alt_lines x act_columns). Extract a single along-track line.
    # Adjust alt_line_idx if a specific line was used to generate the reference image.
    data_eq = var_eq[alt_line_idx, :]
    data_no_eq = var_no_eq[alt_line_idx, :]
    data_isrf = var_isrf[alt_line_idx, :]

    # Close datasets
    ds_eq.close()
    ds_no_eq.close()
    ds_isrf.close()

    # --- Plotting section ---
    plt.figure(figsize=(10, 6))

    # Plot lines with colors and labels matching the reference image
    plt.plot(data_eq, color='black', label='TOA L1B with eq')
    plt.plot(data_no_eq, color='red', label='TOA L1B no eq')
    plt.plot(data_isrf, color='blue', label='TOA - ISRF')

    # Apply identical titles, labels, and styling
    plt.title('Effect of the Equalization for VNIR-0', fontsize=14)
    plt.xlabel('ACT pixel [-]', fontsize=11)
    plt.ylabel('TOA [mW/m2/sr]', fontsize=11)

    plt.xlim(-5, 155)  # Match the slightly padded x-axis view
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save the plot as a PNG image in the current directory
    plt.savefig('Equalization_model.png', dpi=300, bbox_inches='tight')

    plt.show()


# Define your file paths based on the 3 NetCDF files you have
file_with_eq = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-L1B\myOutput\l1b_toa_VNIR-0.nc"
file_no_eq = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-L1B\myOutput_notEqualized\l1b_toa_VNIR-0.nc"
file_isrf = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc"

# Generate the plot
recreate_equalization_plot(file_with_eq, file_no_eq, file_isrf, alt_line_idx=50)