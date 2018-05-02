from distutils.core import setup, Extension
from setuptools import setup, Extension

config = {
    'include_package_data': True,
    'description': 'deep learning model training and tracking framework',
    'url': 'NA',
    'download_url': 'https://github.com/kundajelab/momma_dragonn',
    'version': '0.2.3',
    'packages': ['momma_dragonn', 'momma_dragonn.model_creators',
                 'momma_dragonn.model_trainers',
                 'momma_dragonn.model_wrappers',
                 'momma_dragonn.data_loaders'],
    'setup_requires': [],
    'install_requires': ['numpy', 'avutils'],
    'dependency_links': ['https://github.com/kundajelab/avutils/tarball/master#egg=avutils-0.2'],
    'scripts': ['scripts/momma_dragonn_train'],
    'name': 'momma_dragonn'
}

if __name__== '__main__':
    setup(**config)
