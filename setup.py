from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


setup(name='aarms',
      version='0.0.1',
      description='Attribute-Aware Recommender ModelS',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      keywords='Attribute-Aware Recommender ModelS',
      url='http://github.com/eldrin/aarms',
      author='Jaehun Kim',
      author_email='j.h.kim@tudelft.nl',
      license='MIT',
      # packages=['aarms'],
      packages=find_packages('.'),
      install_requires=requirements(),
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
