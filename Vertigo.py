"""
==================================================================
~~~~~~~~~~~~~~~~~~~~~~ V E R T I G O ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vertical Error Regression Tool for Independent Ground Observations
------------------------------------------------------------------
==================================================================
"""

import os
import csv
import tqdm
import laspy
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd
from math import ceil, isnan
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import cKDTree
from collections import namedtuple
from typing import Union, Tuple, List, Optional
from scipy.interpolate import bisplrep, bisplev, RectBivariateSpline

from rsge_toolbox.lidar.lidar_const import LidarClass
from rsge_toolbox.lidar.Laszy import Laszy, POINT_FILTER_TYPE
from rsge_toolbox.lidar.LidarSurface import PointsXYZA, Not3DDataError, LidarSurface

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


# ------------------------------------------------------
# -- Constants
# ------------------------------------------------------

# Plotting constants
PLOT_OFFSET = 0.100
PLOT_OFFSET_AXIS = 0.025
PLOT_OFFSET_NONE = 0.000
PLOT_OFFSET_LABEL = 0.010
PLOT_OFFSET_THRESHOLD = 0.010

# Computational constants
SURFACE_MIN_POINTS = 3
IDW_MIN_POINTS = 3
BICUBIC_SMOOTH_0_01 = 0.01
BICUBIC_KY = 3
BICUBIC_KX = 3
BISQUARE_KX = 2
BISQUARE_KY = 2
BILINEAR_KY = 1
BILINEAR_KX = 1


# defines Minkowski p-norm distance type parameters for cKDTree.query_ball_point()
DISTANCE_METRIC = DistanceMetric(
    euclidean=1,
    manhattan=2,
    distant_points=3,
    very_distant_points=4
)

# Valid input formats for control points
CONTROL_FORMAT = ControlPointFormat("shp", "csv", "gpkg")

# Constants for plumb distances computation types
COMPUTATION_TYPE = ComputationType(
    tin=0, grid=1, idw=2,
    types=["tin", "grid", "idw"]
)


# ------------------------------------------------------
# -- Class definitions and main logic
# ------------------------------------------------------
class GroundControlPoint:

    def __init__(self, coord_xyz: Point3D, std_xyz: Point3D = None, name: str = ""):

        if len(coord_xyz) != 3:
            raise Not3DDataError

        self.name = name
        self.x = coord_xyz.x
        self.y = coord_xyz.y
        self.z = coord_xyz.z
        if std_xyz:
            if len(std_xyz) != 3:
                raise Not3DDataError
            self.std_x = std_xyz.x
            self.std_y = std_xyz.y
            self.std_z = std_xyz.z

        self._proj_cs = None
        self._vert_cs = None


class VerticalAccuracy:

    def __init__(self, gcp: GroundControlPoint = None):

        """
        Initialize VerticalAccuracyTest object.

        Initializes a VerticalAccuracyTest object. Accepts a cluster of points and
        a ground control point you want to compare between. The class has two attributes
        it uses for Vertical Accuracy testing:
            - self.tin: A TIN model generated from the input points
            - self.grid: A gridded surface generated from the input points

        By default, neither ``self.tin`` or ``self.grid`` will be generated on initialization.
        These two derived surfaces can be generated on initialization through the
        constructor parameters:
            - cell_size: When this value is greater than 0
            - tin: When this value is set to True

        :param gcp: GroundControlPoint object containing valid XYZ coordinates.
        """

        self.gcp = gcp
        self._src = None
        self._tin = None
        self._grid = None
        self._data = None   # tuple containing PointsXYZ at index 0, and scan angles at index 1
        self.surface = None
        self.point_interpolated = PointInterpolated(tin=None, grid=None)
        self.distance = PlumbDistance(tin=float("nan"), grid=float("nan"), idw=float("nan"))

    def set_source_data(self, src: Union[np.array, Laszy, str]):

        """
        Set the source data for the accuracy test.

        Note the that 'source' parameter may be a:
            - valid path to a LAS/LAZ file.
            - laspy.LasData object (returned from either laspy.read() or laspy.open())
            - 2D numpy array containing elements X, Y, and Z coordinates, respectively.

        :param src: A LAS/LAZ file, laspy.LasData object, or 2D numpy array containing XYZ coordinates.
        """

        if isinstance(src, str) and os.path.isfile(src):
            self._data = self.__from_file(src)

        elif isinstance(src, Laszy):
            self._data = self.__from_laszy(src)

        elif isinstance(np.array, src):
            self._data = PointsXYZA(*(np.split(src, 3, axis=1)))

        self._src = src

    def set_gcp(self, gcp: Union[GroundControlPoint, Point3D, tuple]):

        """
        `Set the gcp attribute.`\n

        Sets to gcp attribute. Argument may be a:
        - GroundControlPoint namedtuple.
        - Point3D namedtuple
        - tuple containing XYZ coordinates.

        **When passing a GroundControlPoint namedtuple:**
            1: from Vertigo import GroundControlPoint, Point3D \n
            2: gcp = GroundControlPoint(Point(x=1, y=2, z=3), name="my_gcp") \n
            3: vt = Vertigo() \n
            4: vt.set_gcp(gcp) \n
        \n
        **When passing a Point3D namedtuple:**
            1: from Vertigo import GroundControlPoint, Point3D \n
            2: point = Point3D(x=1, y=2, z=3) \n
            3: vt = Vertigo() \n
            4: vt.set_gcp(point) \n
        \n
        **When passing a tuple:**
            1: x, y, z = 1, 2, 3 \n
            2: coords = (x, y, z) \n
            3: vt = Vertigo() \n
            4: vt.set_gcp(coords) \n
        \n
        Note that when passing a tuple or Point3D object as an argument,
        no name will be assigned to the gcp name property.

        :param gcp: GroundControlPoint namedtuple
        """

        if isinstance(gcp, tuple):
            point = Point3D(x=gcp[0], y=gcp[1], z=gcp[2])
            self.gcp = GroundControlPoint(coord_xyz=point, name="")
        elif isinstance(gcp, Point3D):
            self.gcp = GroundControlPoint(coord_xyz=gcp, name="")
        elif isinstance(gcp, GroundControlPoint):
            self.gcp = gcp
        else:
            self.gcp = None

    def set_surface(self, nn_dist: float):

        """
        Set the 'self.surface' attribute.

        Initializes a MeasurementSurface object with points collected from a Nearest Neighbour search and
        assigns it to the 'self.surface' attribute. Nearest neighbour points are determined as a function of
        the nn_distance parameter.

        Further regarding the 'nn_dist' parameter, note that:
            - Smaller values will improve runtime, but may yield sparse results depending on the terrain.
            - Larger values will improve NN search results, but cause an increase in runtime.

        :param nn_dist: The distance to be applied to nearest neighbour search against self.gcp
        :exception ValueError:
        """

        if bool(self._data):
            points = self.proximity_filter(self._data, nn_dist)
            nn_points = self.__nearest_neighbours(points, self.gcp, nn_dist)
            if len(nn_points.z) < SURFACE_MIN_POINTS:
                raise InsufficientSurfacePointsError
            else:
                self.surface = LidarSurface(nn_points, points.a)

    def proximity_filter(self, points: PointsXYZA, proximity: float) -> PointsXYZA:
        """
        Filter numpy array of XYZ coordinates and angles by proximity to GCP.

        Filters an input point cloud represented as a numpy array of XYZ coordinates and corresponding angles
        by its proximity to the object's gcp attribute 'self.gcp' and returns the filtered coordinates and angles.

        :param points: Numpy array of XYZ coordinates.
        :param proximity: Distance units from 'self.gcp.x' and 'self.gcp.y'
        :return: namedtuple PointsXYZ containing pts.x, pts.y, and pts.z.
        """
        # Create xy proximity masks for numpy array indexing
        x_mask = (np.abs(points.x - self.gcp.x) < proximity)
        y_mask = (np.abs(points.y - self.gcp.y) < proximity)
        mask = (x_mask & y_mask)

        # Filter points and angles based on their proximity to self.gcp
        points_filtered = PointsXYZA(
            x=points.x[mask],
            y=points.y[mask],
            z=points.z[mask],
            a=points.a[mask]
        )

        return points_filtered

    def tin_compare(self) -> float:

        """
        Interpolates a point on a plane created by the smallest triangle surrounding the gcp and
        calculates the distance along the z-axis between the interpolated point and gcp.z.

        :return: Distance along the z-axis between the interpolated point and gcp.z.
        """

        # reference to nn points
        points = self.surface.points

        if points.size > 0:
            # Find the smallest triangle surrounding gcp
            gcp_xy = (self.gcp.x, self.gcp.y)
            dist, ind = cKDTree(points[:, :2]).query(gcp_xy, k=3)
            tri_points = points[ind]

            # Create plane from triangle
            v1 = tri_points[1] - tri_points[0]
            v2 = tri_points[2] - tri_points[0]
            cp = np.cross(v1, v2)
            a, b, c = cp
            d = np.dot(cp, tri_points[0])

            # Interpolate point on plane directly above self.gcp
            interpolated_z = (d - a * self.gcp.x - b * self.gcp.y) / c

            # Calculate distance along z-axis between interpolated point and self.gcp.z
            plumb_dist = self.gcp.z - interpolated_z
            self.distance.tin = float(round(plumb_dist, 3))
            self.point_interpolated.tin = Point3D(self.gcp.x, self.gcp.y, interpolated_z)

        return self.distance.tin

    def grid_compare(self, cell_size: float = 0.30) -> Union[float, None]:

        """
        Calculate the vertical plumbline distance between the gcp and the derived gridded surface.

        :param cell_size: Size of each cell in the gridded surface.
        :return: Absolute value of the vertical plumbline distance (float).
        """

        self.__grid_create(cell_size)
        if self._grid is not None and self._grid.size > 0:
            # Compute index of the closest point in _grid to gcp
            x_idx = np.argmin(np.abs(self._grid[0, 0, :] - self.gcp.x))
            y_idx = np.argmin(np.abs(self._grid[1, :, 0] - self.gcp.y))

            # Get the coordinates of the closest point on the grid
            gx, gy, gz = self._grid[0][x_idx, y_idx], self._grid[1][x_idx, y_idx], self._grid[2][y_idx, x_idx]

            # Compute plumbline distance
            plumb_dist = round(self.gcp.z - gz, 3)
            self.distance.grid = float(plumb_dist)
            self.point_interpolated.grid = Point3D(gx, gy, gz)

        return self.distance.grid

    def idw_compare(self, n: int) -> float:
        """
        Compare gcp against 'self.surface.points' using Inverse Distance Weighting (IDW).
        :param n: Number of point to use in IDW computation.
        :return: Weighted distance from IDW computation.
        """
        # reference to object attributes
        gcp = self.gcp
        points = self.surface.points

        if points.size > 0:
            x_dists = gcp.x - points[:, 0]
            y_dists = gcp.y - points[:, 1]
            z_dists = gcp.z - points[:, 2]
            dists = np.sqrt(
                np.square(x_dists) + np.square(y_dists) + np.square(z_dists)
            )
            n_closest_indices = np.argpartition(dists, n)[:n]
            dists = dists[n_closest_indices]

            weights = 1 / dists
            weights_normalized = weights / np.sum(weights)

            if np.sum(weights_normalized) != 0:
                dists_weighted = dists * weights_normalized
                plumb_dist = np.sum(dists_weighted) / np.sum(weights_normalized)

                if plumb_dist:
                    self.distance.idw = round(float(plumb_dist), 3)

        return self.distance.idw

    def grid_plot(self) -> None:

        """
        Plot the gridded surface as a 3D grid surface.
        """

        # extract the x, y, and z values from the grid property
        x, y, z = self._grid

        # reference to class attributes
        gcp = self.gcp
        grid_interp = self.point_interpolated.grid

        # create a figure and axis object
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # plot the surface as a wireframe
        ax.plot_surface(x, y, z)

        # plot the gcp
        offset_z = PLOT_OFFSET_NONE
        if gcp is not None:  # assign visual plotting elements and plot gcp
            if grid_interp:  # if the gcp is very close to the tin surface, we need to add a "visual offset"
                offset_z = PLOT_OFFSET if abs(grid_interp.z - gcp.z) < PLOT_OFFSET_THRESHOLD else offset_z

            label_z = gcp.z + offset_z + PLOT_OFFSET_LABEL
            label_x, label_y = gcp.x + PLOT_OFFSET_LABEL, gcp.y + PLOT_OFFSET_LABEL
            ax.text(label_x, label_y, label_z, f"{gcp.name}\nElev: {round(gcp.z, 3)}", color='black', fontsize=10)
            ax.scatter(gcp.x, gcp.y, (gcp.z + offset_z), c='orange', marker='^', s=100, label=gcp.name, zorder=3)

        # Plot interp_point and dashed line
        ax.scatter(
            grid_interp.x, grid_interp.y, grid_interp.z,
            s=100, c='magenta', label=f"TIN Elev: {round(grid_interp.z, 3)}", zorder=1
        )
        ax.plot(
            [gcp.x, grid_interp.x], [gcp.y, grid_interp.y], [gcp.z + (offset_z * 0.95), grid_interp.z],
            '--', color='black', zorder=4, label=f"Plumb Dist.: {round(self.distance.grid, 3)}"
        )

        # set the x, y, and z axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Grid Surface')

        # show the plot
        plt.show()

    def tin_plot(self) -> None:

        # reference to class attributes
        gcp = self.gcp
        points = self.surface.points
        tin_interp = self.point_interpolated.tin

        # create a figure and axis object
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Set x, y, and z limits to ensure gcp is always visible
        ax.set_xlim([points[:, 0].min() - PLOT_OFFSET_AXIS, points[:, 0].max() + PLOT_OFFSET_AXIS])
        ax.set_ylim([points[:, 1].min() - PLOT_OFFSET_AXIS, points[:, 1].max() + PLOT_OFFSET_AXIS])
        ax.set_zlim([points[:, 2].min() - PLOT_OFFSET_AXIS, points[:, 2].max() + PLOT_OFFSET_AXIS])

        # plot the points as a scatter plot
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

        # create a Triangulation object from the points
        triang = mtri.Triangulation(points[:, 0], points[:, 1])

        # create a TIN surface from the Triangulation
        ax.plot_trisurf(triang, points[:, 2], cmap='Blues', edgecolor='lightblue', zorder=0)

        offset_z = PLOT_OFFSET_NONE
        if gcp is not None:  # assign visual plotting elements and plot gcp
            if tin_interp:  # if the gcp is very close to the tin surface, we need to add a "visual offset"
                offset_z = PLOT_OFFSET if abs(tin_interp.z - gcp.z) < PLOT_OFFSET_THRESHOLD else offset_z

            label_z = gcp.z + offset_z + PLOT_OFFSET_LABEL
            label_x, label_y = gcp.x + PLOT_OFFSET_LABEL, gcp.y + PLOT_OFFSET_LABEL
            ax.text(label_x, label_y, label_z, f"{gcp.name}\nElev: {round(gcp.z, 3)}", color='black', fontsize=10)
            ax.scatter(gcp.x, gcp.y, (gcp.z + offset_z), c='orange', marker='^', s=100, label=gcp.name, zorder=3)

        # Plot interp_point and dashed line
        ax.scatter(
            tin_interp.x, tin_interp.y, tin_interp.z,
            s=100, c='magenta', label=f"TIN Elev: {round(tin_interp.z, 3)}", zorder=1
        )
        ax.plot(
            [gcp.x, tin_interp.x], [gcp.y, tin_interp.y], [gcp.z + (offset_z * 0.95), tin_interp.z],
            '--', color='black', zorder=4, label=f"Plumb Dist.: {round(self.distance.tin, 3)}"
        )

        # set the labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('TIN Surface')

        # add a legend
        ax.legend()

        # show the plot
        plt.show()

    def reset(self):

        """
        Reset all computed properties to their initial values.

        Resets any properties derived from computations. This method
        is used to provide safety when recomputing certain properties
        with new input data.

        Note that this includes any surfaces, such as TINs, grids,
        or derived planes.
        """

        self._tin = None
        self._grid = None
        self.surface = None
        self.point_interpolated = PointInterpolated(tin=None, grid=None)
        self.distance = PlumbDistance(tin=float("nan"), grid=float("nan"), idw=float("nan"))

    def __grid_create(self, cell_size: float) -> None:
        """
        Create an interpolated grid surface and assign it to the 'grid' property.

        :param cell_size: Float value indicating size per unit of a single grid cell.
        """

        if bool(self.surface):
            x = self.surface.points[:, 0]
            y = self.surface.points[:, 1]
            z = self.surface.points[:, 2]

            # Check if constraints satisfied for target grid interpolation.
            # Grid interpolation must satisfy constraint: 'm >= (kx + 1)(ky + 1)'
            # Where:
            #   m = total number of points to interpolate over
            #   k = degree of polynomial for interpolation
            #   x = number of cells along x-axis
            #   y = number of cells along y-axis
            m = z.size
            if m < (BILINEAR_KX + 1) * (BILINEAR_KY + 1):
                raise ValueError("Grid interpolation not possible for the given data")

            # generate grid for interpolated surface
            bicubic = bisplrep(x, y, z, kx=BILINEAR_KX, ky=BILINEAR_KY, s=0.08)
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            dim_x, dim_y = max_x - min_x, max_y - min_y
            x = np.linspace(min_x, max_x, ceil(dim_x / cell_size))
            y = np.linspace(min_y, max_y, ceil(dim_y / cell_size))
            grid_x, grid_y = np.meshgrid(x, y)

            # evaluate interpolated surface at grid points
            grid_z = np.transpose(bisplev(x, y, bicubic))

            self._grid = np.empty((3, grid_x.shape[0], grid_y.shape[1]))
            self._grid[0, :, :] = grid_x
            self._grid[1, :, :] = grid_y
            self._grid[2, :, :] = grid_z

    def __interpolate_grid_point(self) -> Union[Point3D, None]:

        """
        Interpolate a point on derived grid {xi, yi}, such that:
        xi == gcp.x, yi == gcp.y
        """

        if self._grid is not None and self._grid.size > 0:

            # Get x, y coordinates of input point
            x, y, z = self.gcp.x, self.gcp.y, self.gcp.z

            if not hasattr(self, "_grid"):
                raise ValueError("The gridded surface has not been created yet.")

            x_grid, y_grid, z_grid = self._grid
            x_idx = np.searchsorted(x_grid[0], x)
            y_idx = np.searchsorted(y_grid[:, 0], y)

            if x_idx >= x_grid.shape[1] or y_idx >= y_grid.shape[0]:
                raise ValueError("The given coordinates are out of range of the gridded surface.")

            # Sort the coordinates to ensure they are in strictly increasing order
            x_grid_sorted = np.sort(x_grid[0])
            y_grid_sorted = np.sort(y_grid[:, 0])
            z_grid_sorted = np.sort(z_grid, axis=0)

            f = RectBivariateSpline(y_grid_sorted, x_grid_sorted, z_grid_sorted)
            z = f(y, x)[0][0]

            self.point_interpolated.grid = Point3D(x, y, z)
            return self.point_interpolated.grid

    def __find_smallest_triangle(self) -> Tuple[int, int, int]:
        """
        Finds the three points in `points` that form the smallest triangle containing the
        XY coordinates of `gcp`.

        :return: tuple containing the indices points that form the smallest triangle containing the `gcp`.
        """

        # reference to nn points and gcp coords
        points = self.surface.points

        # extract the XY coordinates of the points and gcp
        xy_points = points[:, :2]
        xy_gcp = np.array([self.gcp.x, self.gcp.y])

        # calculate the distances from gcp to each point in xy_points
        distances = np.linalg.norm(xy_points - xy_gcp, axis=1)

        # sort the points by distance from gcp
        sorted_indices = np.argsort(distances)

        # loop through the points and find the smallest triangle containing gcp
        for i in range(3, len(points)):
            triangle_indices = sorted_indices[:i]
            triangle_points = xy_points[triangle_indices]

            # calculate the area of the triangle
            area = 0.5 * abs(np.cross(triangle_points[1] - triangle_points[0], triangle_points[2] - triangle_points[0]))

            # calculate the barycentric coordinates of gcp in the triangle
            total_area = area
            v0, v1, v2 = triangle_points
            w0 = (v1[1] - v2[1]) * (xy_gcp[0] - v2[0]) + (v2[0] - v1[0]) * (xy_gcp[1] - v2[1])
            w1 = (v2[1] - v0[1]) * (xy_gcp[0] - v2[0]) + (v0[0] - v2[0]) * (xy_gcp[1] - v2[1])
            w2 = total_area - w0 - w1

            # check if gcp is inside the triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                return tuple(triangle_points)

        raise ValueError("GCP is not bounded by surface points")

    @staticmethod
    def __nearest_neighbours(points: PointsXYZA, gcp: GroundControlPoint, nn_dist: float) -> PointsXYZA:

        """
        Find a set of nearest neighbour points surrounding the gcp.

        :param points: A set of XYZ coordinates stored in a PointsXYZ namedtuple.
        :param gcp: A GroundControlPoint object.
        :param nn_dist: The distance for the spherical ball search.
        :return: PointsXYZ namedtuple containing nearest neighbour points.
        """

        # cast PointsXYZ obj over to numpy array
        arr = np.vstack((points.x, points.y, points.z)).T

        # create a KDTree from the input points
        tree = cKDTree(arr)

        # query the KDTree to find the indices of the points within distance 'd' of 'p'
        # --
        # NOTE: the parameter 'p' in tree_query_ball() defines which 'minkowski p-norm' to use.
        # Effectively, this defines the distance metric you will to use. Based on some research
        # both manhattan (p=1) and euclidean (p=2) distance types are appropriate for clustered
        # points (like we have in LiDAR). This has been hard-coded into the following call
        # however it is  WORTH REVISITING the 'p' argument. For now, euclidean distances will be used.
        indices = tree.query_ball_point([gcp.x, gcp.y, gcp.z], nn_dist, p=DISTANCE_METRIC.euclidean)

        # return the nearest neighbors as PointsXYZ object
        nn_points = np.split(arr[indices], 3, axis=1)
        nn_points = [pts.flatten() for pts in nn_points]
        angles = [None for _ in nn_points[0]]  # create placeholder for angles ('a' attribute in PointsXYZA object)

        return PointsXYZA(*nn_points, np.array(angles))

    @staticmethod
    def __from_file(path: str) -> PointsXYZA:

        """
        Extract XYZ coordinates and scan angles from LAS/LAZ file.

        :param path: Path to LAS/LAZ file.
        :return: Tuple of PointsXYZ object and numpy array of scan angles.
        """

        las = Laszy(path, read_points=True)

        las_points = VerticalAccuracy.__class_return_filter(las)
        las_angles = las_points["scan_angle"]

        # Apply offset and keep the corresponding indices
        x, y, z = VerticalAccuracy.__apply_offset(las, las_points)
        a = las_angles / 100

        points = PointsXYZA(x, y, z, a)

        return points

    @staticmethod
    def __apply_offset(las: Laszy, las_points: laspy.ScaleAwarePointRecord) -> tuple:

        """
        Apply Las Header offset values to point record data.

        :param las: Laszy object.
        :param las_points: laspy.ScalePointAwareRecord
        :return: Tuple of x, y, and z coordinates, respectively.
        """

        x = las_points.x + las.public_header_block.x_offset
        y = las_points.y + las.public_header_block.y_offset
        z = las_points.z + las.public_header_block.z_offset

        return x, y, z

    @staticmethod
    def __class_return_filter(las: Laszy) -> laspy.ScaleAwarePointRecord:

        """
        Apply appropriate filter depending on classification of data.

        :param las: Laszy object.
        :return: laspy.ScalePointAwareRecord data.
        """

        # filter points (ground for classified data, last_return for unclassified)
        classes = las.get_classes()
        if LidarClass.GROUND.number in classes:
            las_points = las.filter_points(
                class_num=LidarClass.GROUND.number,
                return_num=POINT_FILTER_TYPE.IGNORE_RETURN
            )
        else:
            las_points = las.filter_points(
                class_num=POINT_FILTER_TYPE.IGNORE_CLASS,
                return_num=POINT_FILTER_TYPE.LAST_RETURN
            )

        return las_points

    @staticmethod
    def __from_laszy(las: Laszy) -> PointsXYZA:

        """
        Extract XYZ coordinates for laspy.LasData object.

        :param las: laspy.LasData object.
        :return: Numpy array of three numpy arrays containing X, Y, and Z coordinates.
        """

        las_points = VerticalAccuracy.__class_return_filter(las)

        # Apply offset. note that accessing xyz coords on LasData object (Stored in Laszy.points attribute)
        # using lowercase xyz will yield pre-scaled coordinates.
        x, y, z = VerticalAccuracy.__apply_offset(las, las_points)

        return PointsXYZA(x, y, z, None)


class Vertigo:

    VERTICAL_THRESHOLD = 0.30

    def __init__(self, flist: List[str], control_points: Optional[List[GroundControlPoint]] = None):

        """
        Initialize a Vertigo object.

        Constructs a Vertigo object. Constructor requires a list of input las or laz files. \n

        Additionally, the user may optionally pass a list of GroundControlPoint namedtuples
        to be assigned to the control property.

        :param flist: A list of input LAS/LAZ filepaths.
        :param control_points: A list of GroundControlPoint(s) or None (Optional)
        """

        self._err = []
        self.results = []
        self.flagged = []
        self.flist = flist
        self.ctrl_src = None
        self.control_map = None
        self.control_table = None
        self._column_format = None
        self.ctrl_attr_table = None
        self.stats = self.__init_stats()
        self.control = control_points if control_points else []

    def set_column_format(self, x: Union[str, int], y: Union[str, int], z: Union[str, int], name: Union[str, int]) -> None:

        """
        Set the column format by individual names or indices.

        Sets the potential columns names or indices for which control points may be read from source data.
        Arguments may be a sting or integer depending on whether the external source from which control
        points are read from contains string fields (e.g. x="utm_easting") or indices (e.g. x=1).

        :param x: X coordinate field name or index.
        :param y: Y coordinate field name or index.
        :param z: Z coordinate field name or index.
        :param name: control point identifier name or index.
        """

        self._column_format = ColumnFormat(x=x, y=y, z=z, name=name)

    def set_control(self, src: str, column_format: ColumnFormat = None) -> None:

        """
        Set the control property from src data.

        Sets the control property (list[GroundControlPoint]) from source dataset. Note that the 'src' may be:
            - Shapefile (.shp)
            - Comma Delimited (.csv)
            - Geopackage (.gpkg)
        Failure to provide one of the above-mentioned input formats will raise a ValueError exception. \n

        User may optionally pass a ColumnFormat dataclass as an argument to explicitly define the
        column names or indices for 'x', 'y', 'z' and 'name'. For example: \n
        >>> col_fmt_names = ColumnFormat(x="utm_e", y="utm_n", z="height", name="gcp_id")
        >>> vt = Vertigo(["file1.las", "file2.las"])
        >>> vt.set_control("control.shp", col_fmt_names)
        \n
        Additionally, if no 'column_format' is provided, the method will attempt to set
        the control attribute using the following default field names depending on whether
        string fields or indices are required by the input data source: \n
        >>> indices = ColumnFormat(x=1, y=2, z=3, name=0)
        >>> strings = ColumnFormat(x="x", y="y", z="z",name="name")

        :param column_format: A ColumnFormat dataclass.
        :param src: Input source data (valid formats: [shp, csv, gpkg])
        :exception ValueError:
        """

        self.ctrl_src = src
        if src.endswith(CONTROL_FORMAT.csv):
            self.__from_csv(src, column_format)
        elif src.endswith(CONTROL_FORMAT.shp):
            self.__from_shp(src, column_format)
        elif src.endswith(CONTROL_FORMAT.gpkg):
            self.__from_gpkg(src, column_format)
        else:
            raise InvalidVectorFormatError

    def map_control(self) -> dict:

        """
        Map control points to input data set in flist.

        Finds which GCPs belong to which input LAS/LAZ files and returns
        a dictionary of the mapping.

        :return: Map of LAS/LAZ files to GCPs
        """

        gcp_count = 0
        control_map = {}
        control_copy = self.control.copy()

        # loop through files and check for GCPs within geo bounds
        for f in self.flist:
            if gcp_count == len(self.control):
                break  # if we find them all, stop looping
            laszy = Laszy(f, read_points=False)
            x_min, x_max = laszy.get_x_minmax()
            y_min, y_max = laszy.get_y_minmax()

            control_map[f] = []
            for gcp in control_copy:
                if x_min <= gcp.x <= x_max and y_min <= gcp.y <= y_max:
                    control_map[f].append(gcp)
                    gcp_count += 1

            if control_map[f]:  # if GCPs mapped, remove them from our search
                control_found = set(control_map[f])
                control_copy = list(set(control_copy).difference(control_found))
            else:  # otherwise, no GCPs in that file. Delete them from our map
                del control_map[f]

        self.control_map = control_map
        return control_map

    def results_to_string(self) -> str:

        """
        Dump results array to formatted string.

        :return: Results array as string.
        """

        results = ""
        for result in self.results:

            results += (result.gcp + "\n")
            results += f"\ttin: {result.distance.tin}"
            results += f"\tgrid: {result.distance.grid}"
            results += f"\tidw: {result.distance.idw}"
            results += "\n"

        return results

    def assess(self, tin: bool = True, grid: bool = False, idw: Union[int, bool] = False, nn_dist: int = 1.2, verbose: bool = False) -> None:

        """
        Evaluate vertical accuracy between control points and input lidar data.

        Performs vertical accuracy assessment and reports results. User may optionally
        enable various methods of measurement via input parameters.
            - tin: Plumbline distance measurement between derived TIN surface and GCPs.
            - grid: Plumbline distance measurement between derived bicubic interpolated grid and GCPs.
            - idw: Inverse Distance Weighting (idw) measurement between Nearest Neighbour points and GCPs.

        :param tin: Enable TIN to GCP distance computation [default=True].
        :param grid: Enable Grid to GCP distance computation [default=False].
        :param idw: Enabled when set to a value greater than IDW_MIN (value=3) [default=0].
        :param nn_dist: Nearest Neighbour distance to each GCP (in distance units) [default=1].
        """

        if not self.control_map:
            self.map_control()

        for file in self.control_map.keys():
            vat = VerticalAccuracy()
            vat.set_source_data(file)
            gcps = self.control_map[file]
            if verbose:
                base = os.path.basename(file).split(".")[0]
                gcps = tqdm.tqdm(gcps, desc=f"{base}: ")

            for gcp in gcps:
                vat.set_gcp(gcp)
                try:  # try to generate surface from gcp nearest neighbors
                    vat.set_surface(nn_dist=nn_dist)
                except (np.linalg.LinAlgError, InsufficientSurfacePointsError):
                    self._err.append(f"{file} :: \n\tSurface not suitable for computations")
                    continue

                if tin:
                    self.__assess_handler(file, gcp, vat, method=COMPUTATION_TYPE.tin, arg=None)
                if grid:
                    self.__assess_handler(file, gcp, vat, method=COMPUTATION_TYPE.grid, arg=None)
                if idw >= IDW_MIN_POINTS:
                    self.__assess_handler(file, gcp, vat, method=COMPUTATION_TYPE.idw, arg=idw)
                dist_copy = PlumbDistance(tin=vat.distance.tin, idw=vat.distance.idw, grid=vat.distance.grid)

                result_surface = vat.surface
                result_surface.points = None  # Release actual points from memory, keep stats (saving some memory)
                result = AssessResult(
                    las=os.path.basename(file), gcp=vat.gcp.name,
                    distance=dist_copy, surface=result_surface,
                    flagged=self.__should_flag_gcp(dist_copy)
                )

                self.results.append(result)
                vat.reset()

            if self._err:
                with open("./vertigo_errors.log", "w") as e:
                    for err in self._err:
                        e.write(err)

    def __should_flag_gcp(self, dist_copy):
        tin_exceeds = (dist_copy.grid >= self.VERTICAL_THRESHOLD)
        grid_exceeds = (dist_copy.tin >= self.VERTICAL_THRESHOLD)
        idw_exceeds = (dist_copy.idw >= (self.VERTICAL_THRESHOLD + 0.20))

        return tin_exceeds or grid_exceeds or idw_exceeds

    def get_dists(self) -> tuple:

        dists = ([], [], [])
        for result in self.results:
            if not isnan(result.distance.tin):
                dists[0].append(result.distance.tin)
            if not isnan(result.distance.grid):
                dists[1].append(result.distance.grid)
            if not isnan(result.distance.idw):
                dists[2].append(result.distance.idw)

        return dists

    def get_stats(self):

        """
        Compute summary statistics from _results

        :return:
        """

        dists = self.get_dists()

        types = (
            COMPUTATION_TYPE.tin,
            COMPUTATION_TYPE.grid,
            COMPUTATION_TYPE.idw
        )

        for dist, typ in zip(dists, types):
            self.__stats_compute(dist, typ)

        return self.stats

    def set_control_attribute_table(self) -> List[List]:

        """
        Read from the control point src file and store contents of
        the attribute table in a list of lists.

        :param ctrl_src: path to src file containing control point info.
        :return: List of lists, where each list corresponds to a row in the attr. table.
        """

        # Open the shapefile
        sf = shapefile.Reader(self.ctrl_src)

        # Get the attribute table records
        records = sf.records()

        # Get the field names (column names)
        fields = sf.fields[1:]  # Exclude the first element (DeletionFlag)
        field_names = [field[0] for field in fields]

        # Create a list of lists to hold the attribute data
        attribute_data = [field_names]  # Start with the column names as the first row

        # Append the attribute values for each record
        for record in records:
            attribute_data.append(list(record))

        self.ctrl_attr_table = attribute_data

        return attribute_data

    def __stats_compute(self, dists: Union[list, tuple], dist_type: int):

        """
        Compute statistics from list of plumb distances.

        :param dists: A list of distance values.
        :param dist_type: Integer constant for computation type [tin=0, grid=1, idw=2]
        """

        total = len(dists)
        key = COMPUTATION_TYPE.types[dist_type]
        if len(dists) > 0:
            dists = np.array(dists)
            mask_nan = ~np.isnan(dists)
            dists = dists[mask_nan]

            self.stats[key]["min"] = round(float(np.min(dists)), 3)
            self.stats[key]["max"] = round(float(np.max(dists)), 3)
            self.stats[key]["std"] = round(float(np.std(dists)), 3)
            self.stats[key]["mean"] = round(float(dists.mean()), 3)
            self.stats[key]["median"] = round(float(np.median(dists)), 3)
            self.stats[key]["rmse"] = round(float((np.sum(dists ** 2) / len(dists)) ** (1 / 2)), 3)
            self.stats[key]["computed_from"] = f"{len(dists)} / {total}"

    def __assess_handler(self, file: str, gcp: GroundControlPoint, vat: VerticalAccuracy, method: int, arg: Union[float, int, None]) -> None:

        """
        Wrap distance computation with simpel error handling.

        :param file: Input file being processed
        :param gcp: Input GroundControlPoint
        :param vat: VerticalAccuracyTest object.
        :param method: constant integer [METHOD_TIN, METHOD_GRID, METHOD_IDW]
        :param arg: Argument for computation method (if necessary).
        """

        try:
            if method == COMPUTATION_TYPE.tin:
                vat.tin_compare()
            elif method == COMPUTATION_TYPE.idw:
                vat.idw_compare(arg)
            elif method == COMPUTATION_TYPE.grid:
                vat.grid_compare(arg) if arg else vat.grid_compare()
        except Exception as e:
            self._err.append(f"{file} :: {COMPUTATION_TYPE.types[method]} :: {gcp.x}, {gcp.y}\n\t{e}")

    def __from_gpkg(self, gpkg_file: str, col_fmt: ColumnFormat = None) -> None:

        """
        Read gpkg into control attribute.

        :param gpkg_file:
        """

        if col_fmt is None:
            col_fmt = ColumnFormat(x="x", y="y", z="z", name="name")

        gdf = gpd.read_file(gpkg_file)
        for i in range(len(gdf)):
            x, y, z = (gdf.iloc[i][c] for c in [col_fmt.x, col_fmt.y, col_fmt.z])
            name = gdf.iloc[i][col_fmt.name]
            gcp = GroundControlPoint(coord_xyz=Point3D(x, y,  z), name=name)
            self.control.append(gcp)

    def __from_shp(self, shp_file: str, col_fmt: ColumnFormat = None) -> None:

        """
        Read shp into control attribute.

        :param shp_file:
        """

        if col_fmt is None:
            col_fmt = ColumnFormat(x="x", y="y", z="z", name="name")

        with shapefile.Reader(shp_file) as shp:
            for record in shp.records():
                x, y, z = record[col_fmt.x], record[col_fmt.y], record[col_fmt.z]
                gcp = GroundControlPoint(coord_xyz=Point3D(x, y, z), name=record[col_fmt.name])
                self.control.append(gcp)

    def __from_csv(self, csv_file: str, col_fmt: ColumnFormat = None) -> None:

        """
        Read csv into control attribute.

        :param csv_file:
        """

        if col_fmt is None:
            col_fmt = ColumnFormat(x=1, y=2, z=3, name=0)

        with open(csv_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                x = float(row[col_fmt.x])
                y = float(row[col_fmt.y])
                z = float(row[col_fmt.z])
                point_3d = Point3D(x=x, y=y, z=z)
                point = GroundControlPoint(name=row[col_fmt.name], coord_xyz=point_3d)
                self.control.append(point)

        self.control_table = pd.read_csv(csv_file)

    @staticmethod
    def __init_stats() -> dict:

        """
        Encapsulate initialization of stats dictionary.
        """

        stats = {
            "tin": {
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
                "rmse": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "computed_from": "",
            },
            "idw": {
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
                "rmse": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "computed_from": "",
            },
            "grid": {
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
                "rmse": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "computed_from": "",
            }
        }

        return stats


class InsufficientSurfacePointsError(Exception):

    def __init__(self, message: str = "Insufficient data to generate surface."):
        self.message = message
        super().__init__(self.message)


class InvalidVectorFormatError(Exception):

    def __init__(self, message: str = "Supported Formats: SHP, CSV, GPKG"):
        self.message = message
        super().__init__(self.message)


def main():
    pass


if __name__ == "__main__":
    main()
