from typing import Union
from dataclasses import dataclass
from collections import namedtuple

# -----------------------------------------------------------
# -- Type definitions
# -----------------------------------------------------------
Point3D = namedtuple("Point", "x y z")
ComputationType = namedtuple("ComputationType", "tin grid idw types")
ControlPointFormat = namedtuple("ControlPointFormat", "shp csv gpkg")
DistanceMetric = namedtuple("DistanceMetric", "manhattan euclidean distant_points very_distant_points")
AssessResult = namedtuple("AssessResult", "las gcp distance surface flagged")


# -----------------------------------------------------------
# -- Data class definitions
# -----------------------------------------------------------
@dataclass
class PlumbDistance:

    """
    Provides a container, as well as type restriction for storage of measured
    plumb distances.
    """

    tin: float
    idw: float
    grid: float


@dataclass
class PointInterpolated:

    """
    Provides a container, as well as type restriction for storage of
    interpolated 3D points.
    """

    tin: Union[Point3D, None]
    grid: Union[Point3D, None]


@dataclass
class ColumnFormat:

    """
    Defines name or indices for name, x, y, and z columns or fields within input
    data sources such as shapefile, geopackage, or csv. \n
    >>> col_fmt_idx = ColumnFormat(x=1, y=2, z=3, name=0)
    >>> col_fmt_names = ColumnFormat(x="Easting", y="Northing", z="Height", name="GCP_ID")
    """

    x: Union[str, int]
    y: Union[str, int]
    z: Union[str, int]
    name: Union[str, int]


# defines Minkowski p-norm distance type parameters for cKDTree.query_ball_point()
DISTANCE_METRIC = DistanceMetric(
    euclidean=1,
    manhattan=2,
    distant_points=3,
    very_distant_points=4
)

# Valid input formats for control points
CONTROL_FORMAT = ControlPointFormat("shp", "csv", "gpkg")

# Constants for plumb distances computation vertical_accuracy
COMPUTATION_TYPE = ComputationType(
    tin=0, grid=1, idw=2,
    types=["tin", "grid", "idw"]
)
