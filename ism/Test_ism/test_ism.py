import os
import numpy as np
import netCDF4 as nc


def validate_ism_results(ref_dir, my_dir, threshold_pct=1e-3):
    # Convert percentage to decimal (1e-3 % = 1e-5)
    threshold = threshold_pct / 100.0

    # Filter for NetCDF files (adapted for ISM outputs)
    files_to_check = [f for f in os.listdir(my_dir) if f.endswith('.nc')]

    if not files_to_check:
        print("No matching .nc files found in the target directory.")
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

                # Handle edge cases
                rel_diff[np.isnan(rel_diff)] = 0.0  # Ignore regions where both are NaN

                # If ref is exactly 0, check absolute diff instead
                zero_mask = (ref_data == 0)
                rel_diff[zero_mask] = np.abs(my_data[zero_mask])

                # Floating-point noise safeguard for near-zero values
                near_zero_mask = (np.abs(ref_data) < 1e-12) & (np.abs(my_data - ref_data) < 1e-12)
                rel_diff[near_zero_mask] = 0.0

                # Get the maximum relative difference
                max_rel_diff = np.nanmax(rel_diff)

                if max_rel_diff >= threshold:
                    print(f"  [FAIL] {var_name} | Max Rel Diff: {max_rel_diff * 100:.6e}% >= {threshold_pct}%")
                    all_passed = False
                else:
                    print(f"  [PASS] {var_name} | Max Rel Diff: {max_rel_diff * 100:.6e}% < {threshold_pct}%")

    print("\n--- Final Verdict ---")
    if all_passed:
        print(f"SUCCESS: All results demonstrated a relative difference lower than {threshold_pct}%")
    else:
        print("FAILED: One or more variables exceeded the maximum allowed relative difference.")


# Define your paths here
teacher_output = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-ISM\output"
my_output = r"C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-ISM\myOutput"

# Run validation
validate_ism_results(teacher_output, my_output)