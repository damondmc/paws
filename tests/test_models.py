import numpy as np

from paws.definitions import SECONDS_PER_YEAR
from paws.params.models import PowerLawModel, UniformModel
from paws.params.search import SearchParamGenerator


def search_cells(model, freq, df1, df2):
    generator = SearchParamGenerator(model, 2)
    return generator.generate_search_ranges(
        1.0, 0.0, -0.5, 0.0, freq, freq + 1, df=0.5, df1=df1, df2=df2
    )[freq].data


def test_power_law_cells_tile_the_ranges():
    model = PowerLawModel(nc_min=2, nc_max=7, tau=300 * SECONDS_PER_YEAR)
    cells = search_cells(model, 100, 1e-10, 1e-19)
    f1_min, f1_max = model.f1_range(np.array([100.0]), 0.5)
    first = cells[cells["freq"] == 100.0]
    assert first["f1dot"].min() == f1_min[0]
    assert (first["f1dot"] + first["df1dot"]).max() == f1_max[0] == 0.0
    assert np.all(first["df2dot"] > 0)


def test_uniform_cells_tile_the_ranges():
    model = UniformModel(f1_lim=(-1e-8, 0.0), f2_lim=(0.0, 4e-18))
    cells = search_cells(model, 100, 2.5e-9, 1e-18)
    first = cells[cells["freq"] == 100.0]
    assert len(first) == 4 * 4
    assert first["f1dot"].min() == -1e-8
    assert (first["f1dot"] + first["df1dot"]).max() == 0.0
    assert first["f2dot"].min() == 0.0
    assert np.isclose((first["f2dot"] + first["df2dot"]).max(), 4e-18)


def test_uniform_values_per_point():
    model = UniformModel(f1_lim=(-1e-8, 0.0), f2_lim=(0.0, 4e-18), f3_lim=(0.0, 2e-27))
    f0 = np.array([100.0, 200.0])
    f1_min, f1_max = model.f1_range(f0, 0.0)
    assert np.array_equal(f1_min, [-1e-8, -1e-8]) and np.array_equal(
        f1_max - f1_min, [1e-8, 1e-8]
    )
    assert np.array_equal(model.f3_value(f0, f1_min, f1_max), [1e-27, 1e-27])
