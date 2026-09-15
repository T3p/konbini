from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


setup(
    name="konbini",
    version="0.1.0",
    description="Combinatorial bandit environments",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["konbini", "konbini.*"]),
    install_requires=REQUIREMENTS,
    python_requires=">=3.10",
    license="CC0-1.0",
)
