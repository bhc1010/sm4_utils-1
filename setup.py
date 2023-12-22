from distutils.core import setup

setup(
  name = 'sm4_utils',         # How you named your package folder (MyLib)
  packages = ['sm4_utils'],   # Chose the same as "name"
  version = '1.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = "Simple wrapper over spym package for quick plotting of STM data in RHK's .sm4 format.",   # Give a short description about your library
  author = 'Ben Campbell',                   # Type in your name
  author_email = 'ben.campbell@unh.edu',      # Type in your E-Mail
  url = 'https://github.com/hollenlab/sm4_utils',   # Provide either the link to your github or to your website
  keywords = ['STM', 'spym', 'sm4'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'matplotlib',
          'spym',
      ],
  classifiers=[
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.11',
  ],
)