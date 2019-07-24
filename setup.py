from distutils.core import setup

setup(
    name='phaunos_ml',
    version='0.1',
    packages=['phaunos_ml', 'phaunos_ml.utils', 'phaunos_ml.models', 'challenge_utils'],
    license='MIT',
    description = 'Tools for machine learning with audio data.',
    long_description=open('README.md').read(),
    author='Julien Ricard',
    author_email='julien.ricard@gmail.com',
    install_requires=[
        'numpy>=1.16.3',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.1',
        'scikit-multilearn>=0.2.0',
        'matplotlib>=3.0.3',
        'tensorflow-gpu==1.14.0', # quite strict but versions change a lot around here
        'pandas>=0.24.2',
        'pysoundfile>=0.9.0',
        'librosa>=0.6.3',
        'tqdm>=4.31.1'
    ]
)
