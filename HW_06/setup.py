from setuptools import find_packages, setup

__version__ = None

setup(
    name='prod_stories_spell_checker',
    version=__version__,
    author='Dmitry Leonov',
    install_requires=['numpy', 'hunspell', 'textdistance'],
    packages=find_packages(),
    license='MIT',
    entry_points={'console_scripts': ['spell_check=spell_checker:spell_check']}
)