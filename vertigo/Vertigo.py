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
import shapefile
import numpy as np
import pandas as pd
from math import isnan
import geopandas as gpd
from typing import List, Optional

from VertigoType import *
from vertical_accuracy.gcp import GroundControlPoint
from rsge_toolbox.lidar.Laszy import Laszy, POINT_FILTER_TYPE
from vertical_accuracy.vertical_accuracy import VerticalAccuracy
from VertigoError import InvalidVectorFormatError, InsufficientSurfacePointsError


# Computational constants
IDW_MIN_POINTS = 3
BICUBIC_SMOOTH_0_01 = 0.01
BICUBIC_KY = 3
BICUBIC_KX = 3
BISQUARE_KX = 2
BISQUARE_KY = 2
BILINEAR_KY = 1
BILINEAR_KX = 1


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
        :param verbose: Show progress bar during processing.
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


def main():
    pass


if __name__ == "__main__":
    main()
