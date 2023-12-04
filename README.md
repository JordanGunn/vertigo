# Vertigo User Guide

## Introduction
Vertigo (Vertical Error Regression Tool for Independent Ground Observations) is a Python-based tool designed for processing and analyzing LiDAR data. Particularly, it is used for evaluating the Absolute Vertical Accuracy of LiDAR data, against independently measured ground observation points.

## Getting Started
### Prerequisites
- Python environment with libraries: `os`, `csv`, `tqdm`, `laspy`, `shapefile`, `numpy`, `pandas`, `geopandas`.
- LiDAR data files in LAS/LAZ format.

### Installation
1. Ensure all required libraries are installed.
2. Download `Vertigo.py` to your local system.

## Usage
### Loading Data
- Use `laspy` to load your LAS/LAZ files.
- Import the `Vertigo` class from `Vertigo.py`.

```python
from vertigo.Vertigo import Vertigo
# Example of loading a LAS file
import laspy

las_data = laspy.read("path/to/your/file.las")
```

### Data Analysis
- Instantiate the `Vertigo` class with your LiDAR data.
- Call methods provided by Vertigo for data analysis like calibration, image compositing, and change detection.

```python
# Instantiate Vertigo
vertigo_instance = Vertigo(las_data)
# Call methods for processing
# vertigo_instance.some_method()
```

### Reporting
- Use Vertigo's reporting features to generate insights and visualizations from your data.

```python
# Generate report
report = vertigo_instance.generate_report()
print(report)
```

## Advanced Usage
- For advanced users, delve into custom workflows for specific data processing needs.
- Explore integration with other data sources and machine learning algorithms for enhanced analysis.

## Conclusion
Vertigo offers a comprehensive suite of tools for LiDAR data analysis, catering to both basic and advanced use cases.
