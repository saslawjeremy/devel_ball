import setuptools

setuptools.setup(
    name='devel_ball',
    version='0.1',
    description='An NBA analysis package',
    url='#',
    author='Jeremy',
    install_requires=[],
    author_email='',
    packages=['devel_ball'],
    zip_safe=False,
    scripts=[
        'scripts/load-data',
        'scripts/post-process',
        'scripts/create-pandas',
        'scripts/load-season',
        'scripts/create-model',
        'scripts/predict-date',
        'scripts/add-dk-player',
        'scripts/devel-ball-update',
    ],
)
