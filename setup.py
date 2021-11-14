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
    scripts=['scripts/load_models', 'scripts/post_process', 'scripts/create_pandas', 'scripts/load_season'],
)
