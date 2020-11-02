import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformer-htr",
    version="0.0.1",
    author="Rafael d'Arce",
    author_email="rd17647@bristol.ac.uk",
    description="HTR with Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaeljcdarce/transfomer_htr",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)