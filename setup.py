from setuptools import setup

setup(
    name="cover-score",
    version="1.0.0",
    description="CLI tool for scoring children's book covers",
    py_modules=["cover_score"],
    install_requires=["Pillow"],
    entry_points={
        "console_scripts": [
            "cover-score=cover_score:main",
        ],
    },
    python_requires=">=3.8",
)
