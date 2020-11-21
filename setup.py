from setuptools import setup, find_packages
with open('./Yolov3/requirements.txt') as f:
    required = f.read().splitlines()
setup(
   name='Yolov3',
   version='1.0',
   description='',
   author='HugPhat',
   author_email='hug.phat.vo@gmail.com',
   license="MIT",
   packages=find_packages(),  # same as name
   include_package_data=True,
   install_requires=[
       required
   ], #external packages as dependencies
   scripts=[
            
           ],
    python_requires='>=3.6',
)
