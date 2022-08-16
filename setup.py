import setuptools

setuptools.setup(
    name="messenger",
    version="0.1.1",
    author="Austin Wang Hanjie",
    author_email="hjwang@cs.princeton.edu",
    description="Implements EMMA model and Messenger environments.",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'gym',
        'numpy',
        'vgdl @ git+https://github.com/ahjwang/py-vgdl',
        'pygame==1.9.6'
    ],
    extras_require={
        'models': ['torch>=1.3', 'transformers>=4.2']
    }
)