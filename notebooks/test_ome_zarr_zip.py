import asyncio
import os
import tempfile
import zipfile
import numpy as np
import psutil

import zarr
from zarr.storage import MemoryStore, LocalStore
from zarr.buffer import default_buffer_prototype

from bioio_ome_zarr.writers import OMEZarrWriter, Channel


def estimate_memory_bytes(data: np.ndarray, num_levels: int = 2) -> int:
    """
    Estimates total memory needed to write full-resolution data, multi-resolution 
    pyramids, and uncompressed chunk buffers during writing.
    """
    base_bytes = data.nbytes
    # Downsampling Y/X by 2 per level adds ~33% extra total array volume
    pyramid_scale = 1.33 if num_levels > 1 else 1.0
    # Safety factor (2.5x) for intermediate numpy/dask chunk buffers
    overhead_multiplier = 2.5
    
    return int(base_bytes * pyramid_scale * overhead_multiplier)


def should_use_memory_store(data: np.ndarray, num_levels: int, max_ram_fraction: float = 0.7) -> bool:
    """
    Checks if available RAM can comfortably hold the Zarr 3 store in memory.
    """
    required_bytes = estimate_memory_bytes(data, num_levels)
    available_ram = psutil.virtual_memory().available
    usable_ram = available_ram * max_ram_fraction
    
    print(f"Data Raw Size: {data.nbytes / (1024**2):.2f} MB")
    print(f"Estimated Peak RAM Needed: {required_bytes / (1024**2):.2f} MB")
    print(f"Available System RAM: {available_ram / (1024**2):.2f} MB (Usable limit: {usable_ram / (1024**2):.2f} MB)")
    
    return required_bytes < usable_ram


async def _extract_memory_store_contents(store: MemoryStore) -> dict[str, bytes]:
    """
    Asynchronously extracts keys and raw byte values from a Zarr 3 MemoryStore.
    """
    contents = {}
    async for key in store.list():
        buf = await store.get(key, prototype=default_buffer_prototype())
        if buf is not None:
            contents[key] = buf.to_bytes()
    return contents


def save_memory_store_to_zip(store: MemoryStore, zip_path: str):
    """
    Packages a Zarr 3 MemoryStore directly into a ZIP archive.
    """
    contents = asyncio.run(_extract_memory_store_contents(store))
    
    # ZIP_STORED avoids double-compressing chunks that Zarr already compressed
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for key, data_bytes in contents.items():
            zf.writestr(key, data_bytes)


def save_directory_to_zip(dir_path: str, zip_path: str):
    """
    Packages a local directory store into a ZIP archive.
    """
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dir_path)
                # POSIX path separators inside ZIP archive
                zip_rel_path = rel_path.replace(os.sep, "/")
                zf.write(full_path, arcname=zip_rel_path)


def write_ome_zarr_zip(
    data: np.ndarray,
    output_zip_path: str,
    axes_names: list[str] = ["t", "c", "z", "y", "x"],
    channels: list[Channel] | None = None,
    num_levels: int = 2,
    max_ram_fraction: float = 0.7,
):
    """
    Writes multi-dimensional image data to an OME-Zarr (Zarr 3) ZIP archive.
    
    Parameters
    ----------
    data : np.ndarray
        Input array (e.g. 5D TCZYX or matching axes_names).
    output_zip_path : str
        Target output file path ending in `.ome.zarr.zip`.
    axes_names : list[str]
        Dimension labels matching input data shape.
    channels : list[Channel], optional
        BioIO OME channel metadata specs.
    num_levels : int
        Number of resolution pyramid levels to generate (halving Y/X per level).
    max_ram_fraction : float
        Fraction of available RAM to allow before falling back to disk temp store.
    """
    # 1. Generate pyramid level shapes (halving spatial Y and X axes)
    level_shapes = [data.shape]
    curr_shape = list(data.shape)
    for _ in range(1, num_levels):
        curr_shape[-2] = max(1, curr_shape[-2] // 2)  # Y
        curr_shape[-1] = max(1, curr_shape[-1] // 2)  # X
        level_shapes.append(tuple(curr_shape))

    # 2. Decide between MemoryStore vs LocalStore fallback
    use_memory = should_use_memory_store(data, num_levels, max_ram_fraction)

    if use_memory:
        print("--> Decision: Using In-Memory Store (zarr.storage.MemoryStore)")
        store = MemoryStore()
        
        writer = OMEZarrWriter(
            store=store,
            level_shapes=level_shapes,
            dtype=data.dtype,
            zarr_format=3,  # Enforces Zarr v3 (NGFF 0.5)
            axes_names=axes_names,
            channels=channels,
        )
        writer.write_full_volume(data)
        
        print(f"--> Writing in-memory store to ZIP: {output_zip_path}")
        save_memory_store_to_zip(store, output_zip_path)

    else:
        print("--> Decision: RAM constrained. Falling back to Temporary Disk Store (zarr.storage.LocalStore)")
        with tempfile.TemporaryDirectory(prefix="ome_zarr_v3_") as temp_dir:
            store = LocalStore(temp_dir)
            
            writer = OMEZarrWriter(
                store=store,
                level_shapes=level_shapes,
                dtype=data.dtype,
                zarr_format=3,  # Enforces Zarr v3 (NGFF 0.5)
                axes_names=axes_names,
                channels=channels,
            )
            writer.write_full_volume(data)
            
            print(f"--> Packaging temporary disk store into ZIP: {output_zip_path}")
            save_directory_to_zip(temp_dir, output_zip_path)

    print(f"Successfully generated OME-Zarr archive: {output_zip_path}")


# =====================================================================
# Example Usage
# =====================================================================
if __name__ == "__main__":
    # Generate mock 5D image volume: (T=1, C=2, Z=4, Y=512, X=512)
    shape = (1, 2, 4, 512, 512)
    mock_data = np.random.randint(0, 255, size=shape, dtype=np.uint8)

    # Optional channel metadata
    channels_meta = [
        Channel(label="DAPI", color="0000FF"),
        Channel(label="GFP", color="00FF00"),
    ]

    output_file = "sample_dataset.ome.zarr.zip"
    
    write_ome_zarr_zip(
        data=mock_data,
        output_zip_path=output_file,
        axes_names=["t", "c", "z", "y", "x"],
        channels=channels_meta,
        num_levels=2,
        max_ram_fraction=0.7,
    )