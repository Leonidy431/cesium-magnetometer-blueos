"""
Setup script for Cesium Magnetometer BlueOS Extension.

Installation:
    pip install -e .

Author: НПО Лаборатория К
"""

from pathlib import Path

from setuptools import find_packages, setup


# Read README for long description
readme_path = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="magnit-magnetometer",
    version="1.0.0",
    author="НПО Лаборатория К",
    author_email="lab767@gmail.com",
    description="Cesium Magnetometer BlueOS Extension with Hailo 8L AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Leonidy431/magnit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pyserial>=3.5",
        "aiohttp>=3.8.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "hailo": ["hailo-platform>=4.15.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "magnit-magnetometer=magnetometer.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
