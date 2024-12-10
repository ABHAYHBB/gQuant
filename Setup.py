from setuptools import setup, find_packages

setup(
    name='gQuant',
    version='0.1.0',
    author='Abhay Kumar Pathak',
    author_email='pathakabhay@bhu.ac.in',
    description='A Python package for preprocessing, analysis, and visualization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ABHAYHBB/gQuant.git',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

