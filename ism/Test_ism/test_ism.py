import os
import numpy as np
import netCDF4 as nc


def cross_validate_outputs(my_out_dir, teacher_out_dir, tol_pct=1e-3):
    """
    Compares .nc files in my_out_dir with those in teacher_out_dir.
    Passes if the maximum relative difference is < tol_pct.
    """
    # Convert percentage tolerance to fractional tolerance
    tol_frac = tol_pct / 100.0

    my_files = [f for f in os.listdir(my_out_dir) if f.endswith('.nc')]

    if not my_files:
        print(f"No .nc files found in {my_out_dir}")
        return

    print(f"Starting cross-validation with tolerance: {tol_pct}% ({tol_frac})\n"
          f"{'-' * 60}")

    for filename in my_files:
        my_path = os.path.join(my_out_dir, filename)
        teacher_path = os.path.join(teacher_out_dir, filename)

        if not os.path.exists(teacher_path):
            print(f"[SKIP] {filename:<30} -> Not found in teacher output.")
            continue

        try:
            ds_my = nc.Dataset(my_path, 'r')
            ds_teacher = nc.Dataset(teacher_path, 'r')

            # Iterate through all variables in the netCDF file
            for var_name in ds_my.variables:
                # Skip dimension variables if any
                if var_name not in ds_teacher.variables:
                    continue

                my_data = ds_my.variables[var_name][:]
                teacher_data = ds_teacher.variables[var_name][:]

                # Suppress divide-by-zero warnings for safe masking
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Calculate relative difference
                    rel_diff = np.abs((my_data - teacher_data) / teacher_data)

                    # Handle exact zeros in teacher data (avoid NaN)
                    zeros_mask = (teacher_data == 0)
                    rel_diff[zeros_mask] = np.abs(my_data[zeros_mask])

                max_rel_diff = np.nanmax(rel_diff)
                max_rel_diff_pct = max_rel_diff * 100.0

                if max_rel_diff_pct <= tol_pct:
                    print(f"[PASS] {filename:<30} | Var: {var_name:<5} | Max Diff: {max_rel_diff_pct:.8f}%")
                else:
                    print(f"[FAIL] {filename:<30} | Var: {var_name:<5} | Max Diff: {max_rel_diff_pct:.8e}%")

        except Exception as e:
            print(f"[ERROR] Processing {filename}: {e}")
        finally:
            if 'ds_my' in locals() and ds_my.isopen(): ds_my.close()
            if 'ds_teacher' in locals() and ds_teacher.isopen(): ds_teacher.close()


if __name__ == "__main__":
    # Update these paths to point to the directories containing the .nc files
    MY_OUTPUT_FOLDER = r'C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-ISM\myOutput'
    TEACHER_OUTPUT_FOLDER = r'C:\Users\Luis Castro\Downloads\EODP\EODP_TER_2021\EODP-TS-ISM\output'

    cross_validate_outputs(MY_OUTPUT_FOLDER, TEACHER_OUTPUT_FOLDER, tol_pct=1e-3)