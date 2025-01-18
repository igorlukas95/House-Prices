from pathlib import Path
from setuptools import setup, find_packages


setup(
    name="house_price_model",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "house_price_model": ["VERSION"]
    },

    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pydantic",
        ""
    ]
)