# system imports
import os
import laspy
import numpy as np
from math import ceil
import matplotlib.tri as mtri
from typing import Union, Tuple
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import bisplrep, bisplev, RectBivariateSpline

# user imports
from gcp import GroundControlPoint
from rsge_toolbox.lidar.lidar_const import LidarClass
from rsge_toolbox.lidar.Laszy import Laszy, POINT_FILTER_TYPE
from vertigo.VertigoError import InsufficientSurfacePointsError
from rsge_toolbox.lidar.LidarSurface import PointsXYZA, Not3DDataError, LidarSurface
from vertigo.VertigoType import PointInterpolated, PlumbDistance, Point3D, DISTANCE_METRIC

# ------------------------------------------------------
# -- Constants
# ------------------------------------------------------

# Plotting constants
PLOT_OFFSET = 0.100
PLOT_OFFSET_AXIS = 0.025
PLOT_OFFSET_NONE = 0.000
PLOT_OFFSET_LABEL = 0.010
PLOT_OFFSET_THRESHOLD = 0.010

# computational constants
SURFACE_MIN_POINTS = 3
IDW_MIN_POINTS = 3
BICUBIC_SMOOTH_0_01 = 0.01
BICUBIC_KY = 3
BICUBIC_KX = 3
BISQUARE_KX = 2
BISQUARE_KY = 2
BILINEAR_KY = 1
BILINEAR_KX = 1


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
        # both manhattan (p=1) and euclidean (p=2) distance vertical_accuracy are appropriate for clustered
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
    def __class_return_filter(las: Laszy) -> Union[laspy.ScaleAwarePointRecord, np.ndarray]:

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
