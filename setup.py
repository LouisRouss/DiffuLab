from setuptools import find_packages, setup  # type: ignore

setup(
    name="diffulab",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
