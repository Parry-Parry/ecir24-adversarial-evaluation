import setuptools

setuptools.setup(
    name="advseq2seq",
    version="0.0.1",
    author="Andrew Parry, Maik Fröbe",
    author_email="a.parry.1@research.gla.ac.uk",
    description="Implementation of experiments",
    packages=setuptools.find_packages(exclude=['tests', 'eval']),
)