from house_price_model.Config.core import PACKAGE_ROOT

with open(PACKAGE_ROOT / "VERSION", 'r') as file:
    __version__ = file.read().strip()

