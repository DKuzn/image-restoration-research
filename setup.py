from setuptools import setup
from image_restoration import __version__

setup(
    name='image_restoration',
    version=__version__,
    packages=['image_restoration'],
    url='https://github.com/DKuzn/image-restoration-research',
    license='LGPLv3',
    author='Dmitry Kuznetsov',
    author_email='DKuznetsov2000@outlook.com',
    description='Library to image restoration with neural network',
    install_requires=['torch', 'torchvision']
)