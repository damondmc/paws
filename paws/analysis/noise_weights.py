"""Per-detector noise weight of a 1 Hz band, for the noise-normalized H1/L1 excess ratio.

W_X = sum over detector X's SFTs of 1 / S_X, S_X = the SFT's median |X|^2 over the band / ln 2. A signal's expected
2F - 4 in detector X scales with W_X (noise level and amount of data), so R_HL = r_HL * W_L1 / W_H1 does not depend
on the detectors' noise.
"""

from pathlib import Path

import numpy as np

from paws.definitions import SECONDS_PER_DAY

CHUNK_DAYS = 50  # SFTs are loaded this many days at a time


def band_noise_weight(sft_dir: Path, band: int) -> tuple[float, int]:
    """(W, number of SFTs) of the SFTs in sft_dir over the 1 Hz band [band, band + 1)."""
    import lal
    import lalpulsar

    (path,) = sorted(sft_dir.glob("*.sft"))
    catalog = lalpulsar.SFTdataFind(str(path), None)
    first = catalog.data[0].header.epoch.gpsSeconds
    last = catalog.data[catalog.length - 1].header.epoch.gpsSeconds
    inverse_noise = []
    for start in range(first, last + 1, CHUNK_DAYS * SECONDS_PER_DAY):
        chunk = lalpulsar.ReturnSFTCatalogTimeslice(
            catalog,
            lal.LIGOTimeGPS(start),
            lal.LIGOTimeGPS(start + CHUNK_DAYS * SECONDS_PER_DAY),
        )
        if chunk.length == 0:
            continue
        sfts = lalpulsar.LoadSFTs(chunk, band, band + 1)
        power = (
            np.abs(np.array([sfts.data[i].data.data for i in range(sfts.length)])) ** 2
        )
        inverse_noise.append(np.log(2) / np.median(power, axis=1))
    inverse_noise = np.concatenate(inverse_noise)
    return inverse_noise.sum(), len(inverse_noise)
