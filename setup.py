from setuptools import setup, find_packages

setup(
    name="tiktok-mcp",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "mcp>=1.6.0",
        "aiohttp>=3.9.3",
        "python-dotenv>=1.0.0"
    ],
    entry_points={
        'console_scripts': [
            'tiktok-mcp=tiktok_mcp_service.main:main',
        ],
    },
) 