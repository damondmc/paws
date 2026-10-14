"""Type checker the tests apply to every paws function (jaxtyping import hook in conftest.py): beartype, with an int
accepted where a float is annotated (PEP 484)."""

from beartype import BeartypeConf, beartype

checker = beartype(conf=BeartypeConf(is_pep484_tower=True))
