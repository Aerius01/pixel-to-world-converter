"""
Validation of the input tables shared by every pipeline entry point.

The pipeline functions accept DataFrames directly, so the checks live here rather
than in the file-based wrappers: whichever entry point is used, the same checks run.
"""

from .errors import DataLoadError
from .schema import DRONE_LOG_SCHEMA, IMAGE_TRACKS_SCHEMA


def _require_columns(df, schema, description):
    missing_columns = schema.validate(df)
    if missing_columns:
        error_msg = f"{description} is missing required columns: {', '.join(missing_columns)}\n"
        error_msg += f"Required columns: {', '.join(schema.get_required_columns())}"
        raise DataLoadError(error_msg)


def prepare_image_tracks(image_tracks_df):
    """
    Check the image tracks table and keep only its required columns.

    Args:
        image_tracks_df: DataFrame with at least the IMAGE_TRACKS_SCHEMA columns

    Returns:
        pd.DataFrame: The required columns only, indexed from 0

    Raises:
        DataLoadError: If required columns are missing
    """
    _require_columns(image_tracks_df, IMAGE_TRACKS_SCHEMA, "Image tracks")
    return image_tracks_df[IMAGE_TRACKS_SCHEMA.get_required_columns()].reset_index(drop=True)


def prepare_drone_log(sensor_data_df):
    """
    Check the drone log table and return its required columns as floats.

    The table may come straight from a CSV read as text; every required column is
    converted to float here, with empty cells treated as missing (as a CSV read gives).

    Args:
        sensor_data_df: DataFrame with at least the DRONE_LOG_SCHEMA columns

    Returns:
        pd.DataFrame: The required columns as floats, indexed from 0

    Raises:
        DataLoadError: If required columns are missing, are not numeric, or a
            velocity column has missing values
    """
    _require_columns(sensor_data_df, DRONE_LOG_SCHEMA, "Drone log")
    try:
        required = sensor_data_df[DRONE_LOG_SCHEMA.get_required_columns()]
        required = required.where(required != "")
        prepared = required.astype(float).reset_index(drop=True)
    except (TypeError, ValueError) as e:
        raise DataLoadError(f"Drone log has non-numeric values in a required column: {e}")

    velocity_columns = [DRONE_LOG_SCHEMA.vel_x_col, DRONE_LOG_SCHEMA.vel_y_col, DRONE_LOG_SCHEMA.vel_z_col]
    for col in velocity_columns:
        if prepared[col].isna().any():
            missing_count = prepared[col].isna().sum()
            raise DataLoadError(
                f"Drone log contains {missing_count} missing values in {col}. "
                f"All velocity measurements are required for GPS conversion."
            )
    return prepared
