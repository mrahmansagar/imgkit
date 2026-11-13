import glob
import os
import imageio.v3 as iio
from natsort import natsorted
import numpy as np
import tifffile


def _read_single_image(path: str) -> np.ndarray:
  """Route TIFF files to tifffile and standard formats to imageio.

  Args:
      path: Path to a single image file.

  Returns:
      Image data as a NumPy array.
  """
  if path.lower().endswith((".tif", ".tiff")):
    return tifffile.imread(path)
  return iio.imread(path)


def load_image_data(
    path: str,
    sort_files: bool = True,
    file_extension: str = "*.tif",
) -> np.ndarray:
  """Load a single 2D/3D image file or a folder of image slices into an array.

  Args:
      path: Path to a single image file or a directory containing slices.
      sort_files: Whether to naturally sort filenames when loading a folder.
        Defaults to True.
      file_extension: Glob pattern for matching files in a directory (e.g.,
        '*.tif', '*.png'). Defaults to '*.tif'.

  Returns:
      NumPy array containing the loaded 2D image or 3D volume.

  Raises:
      ValueError: If path does not exist or no matching files are found.
  """
  if not os.path.exists(path):
    raise ValueError(f"Data directory or file '{path}' does not exist.")

  if os.path.isdir(path):
    file_list = glob.glob(os.path.join(path, file_extension))
    if not file_list:
      raise ValueError(
          f"No files matching '{file_extension}' found in '{path}'."
      )

    if sort_files:
      file_list = natsorted(file_list)

    if file_extension.lower().endswith((".tif", ".tiff")):
      return tifffile.imread(file_list)
    return np.stack([_read_single_image(f) for f in file_list], axis=0)

  return _read_single_image(path)


def _validate_dtype(extension: str, dtype: np.dtype) -> None:
  """Validate that the array dtype is compatible with the target extension.

  Args:
      extension: Normalized lowercase file extension starting with a dot.
      dtype: NumPy data type of the array to be saved.

  Raises:
      ValueError: If the dtype is incompatible with .jpg/.jpeg or .png.
  """
  if extension in (".jpg", ".jpeg") and dtype != np.dtype(np.uint8):
    raise ValueError(
        f"JPEG format requires 'uint8' data, got '{dtype}'. "
        "Please scale and cast the data before saving."
    )
  if extension == ".png" and dtype != np.dtype(np.uint16):
    raise ValueError(
        f"PNG format requires 'uint16' data, got '{dtype}'. "
        "Please scale and cast the data before saving."
    )


def _write_image(path: str, image: np.ndarray) -> None:
  """Write an image array to disk using tifffile or imageio.

  Args:
      path: Destination file path including extension.
      image: 2D or 3D NumPy array to write.
  """
  if path.lower().endswith((".tif", ".tiff")):
    tifffile.imwrite(path, image)
  else:
    iio.imwrite(path, image)


def save_image_data(
    data: np.ndarray,
    output_path: str,
    file_extension: str = ".tif",
    as_sequence: bool = False,
) -> None:
  """Save a 2D image or 3D volume to disk as a single file or slice sequence.

  TIFF files (.tif, .tiff) preserve raw scientific data types and support
  multi-page 3D volumes. Standard formats enforce strict data types without
  automatic scaling: .jpg/.jpeg requires uint8, and .png requires uint16.

  Args:
      data: Input 2D (Y, X) or 3D (Z, Y, X) NumPy array.
      output_path: Destination file path or target directory for sequences.
      file_extension: Desired file extension (e.g., '.tif', '.png', '.jpg').
        Applied when saving a sequence or if output_path lacks an extension.
        Defaults to '.tif'.
      as_sequence: If True and data is 3D, saves individual 2D slices inside
        output_path named 'slice_0000.ext' through 'slice_nnnn.ext'.
        Defaults to False.

  Raises:
      ValueError: If data is not 2D or 3D, if a 3D volume is saved as a
          single file with a non-TIFF extension, or if the array dtype does
          not meet format requirements (uint8 for JPEG, uint16 for PNG).
  """
  if data.ndim not in (2, 3):
    raise ValueError(f"Expected a 2D or 3D array, got {data.ndim}D.")

  ext = (
      file_extension
      if file_extension.startswith(".")
      else f".{file_extension}"
  )
  ext = ext.lower()

  if data.ndim == 3 and as_sequence:
    _validate_dtype(ext, data.dtype)
    os.makedirs(output_path, exist_ok=True)
    num_slices = data.shape[0]
    pad_width = max(4, len(str(num_slices)))

    for idx in range(num_slices):
      filename = f"slice_{idx:0{pad_width}d}{ext}"
      slice_path = os.path.join(output_path, filename)
      _write_image(slice_path, data[idx])
    return

  if not os.path.splitext(output_path)[1]:
    output_path = f"{output_path}{ext}"

  final_ext = os.path.splitext(output_path)[1].lower()

  if data.ndim == 3 and final_ext not in (".tif", ".tiff"):
    raise ValueError(
        "Saving a 3D volume as a single file requires a '.tif' or '.tiff' "
        "extension."
    )

  _validate_dtype(final_ext, data.dtype)

  parent_dir = os.path.dirname(output_path)
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

  _write_image(output_path, data)
