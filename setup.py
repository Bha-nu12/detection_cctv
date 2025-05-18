from setuptools import setup, find_packages

setup(
    name="face_recognition_system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pyttsx3>=2.90',
    ],
    python_requires='>=3.8',
    author="Your Name",
    description="A multi-camera face recognition system",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
) 