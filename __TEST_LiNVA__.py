import os
import sys
import glob
import subprocess
from collections import namedtuple
from random import random, randint
from rsge_toolbox.lidar.Laszy import Laszy

from VertigoReport import VertigoReport
from Vertigo import GroundControlPoint, Point3D, Vertigo, ColumnFormat

GCP_ID = "793158"
FLIST_DIR_LOCAL = "/home/jordan/work/geobc/test_data/laz/v14"
FLIST_DIR_DRIVE = "/media/jordan/BCTS_OP23BMRS014_Eagle/LiDAR_data/Stuart_Nechako/lidar_laz/UTM10"
F = "/media/jordan/BCTS_OP23BMRS014_Eagle/LiDAR_data/Stuart_Nechako/lidar_laz/UTM10/bc_093n034_1_3_1_xyes_8_utm10_2022.laz"
FILE_LAS = "/home/jordan/work/geobc/test_data/laz/v14/bc_092b044_1_2_3_xyes_8_utm10_2019.laz"
FILE_LAS_PROBLEM1 = "/media/jordan/BCTS_OP23BMRS014_Eagle/LiDAR_data/Stuart_Nechako/lidar_laz/UTM10/bc_093n016_3_1_3_xyes_8_utm10_2022.laz"
FILE_LAS_PROBLEM2 = "/media/jordan/BCTS_OP23BMRS014_Eagle/LiDAR_data/Stuart_Nechako/lidar_laz/UTM10/bc_093n010_2_1_4_xyes_8_utm10_2022.laz"
FILE_CTRL_GCM = "/home/jordan/work/geobc/test_data/gcm/bc_092b044_1_2_3_xyes_8_utm10_2019_CGVD2013.csv"
FILE_CTRL_SHP = "/home/jordan/work/geobc/test_data/ground_control/OP23BMRS014/Ground_Control/Shapefiles/OP23BMRS014_Ground_Control_UTM10.shp"


IS_LINUX = (sys.platform == "linux") or (sys.platform == "linux2")
SHELL_EXEC = "/usr/bin/zsh" if IS_LINUX else None

TestArea = namedtuple("TestArea", "llx, lly, size")


class TestData:
    def __init__(self, filename: str):
        self.file = filename
        self.control_points = []
        self.las = Laszy(filename, read_points=False)
        self.file_basename = os.path.basename(self.file)

        self.test_file = None
        self.test_area = None

    def set_test_area(self, size: int, llx: float = -1, lly: float = -1):

        """
        Set the bounds for the sample area.

        If arguments llx or lly are less than the minimum XY coordinates
        of the input file, the xmin, ymin of input file will be used instead.

        :param llx: Lower left corner X
        :param lly: Lower left corner Y
        :param size: Tile side length (i.e. 'size=100' yields 100x100 size tile.)
        """

        xmin, xmax = self.las.get_x_minmax()
        ymin, ymax = self.las.get_y_minmax()

        # if bounds are invalid, overwrite input value with minimum x coord
        if llx < xmin:
            llx = xmin

        # if bounds are invalid, overwrite input value with minimum y coord
        if lly < ymin:
            lly = ymin

        self.test_area = TestArea(llx, lly, size)

    def write_tile(self, odir: str = ".", suffix: str = "", olaz=False):

        """Generate a smaller file from input filepath to LiDAR data (LAS/LAZ)"""

        if TestArea:
            out_format = "laz" if olaz else "las"
            las2las = "las2las64" if IS_LINUX else "las2las"
            suffix = suffix if bool(suffix) else f"tile_{self.test_area.size}m"
            keep_tile = f"{self.test_area.llx} {self.test_area.lly} {self.test_area.size}"
            out = os.path.join(odir, f"{self.file_basename.split('.')[0]}_{suffix}.{out_format}")
            if not os.path.exists(out):
                cmd = f"{las2las} -i {self.file} -keep_tile {keep_tile} -o {out} -v"
                subprocess.call(cmd, shell=True, executable=SHELL_EXEC)

            self.test_file = out

    def create_control_points(self, npoints: int, vertical_offset: int = 0, odir: str = "") -> list:

        """
        Create a list of XYZ coordinates.

        Creates a list of tuples containing X, Y, Z coordinates.
        Points in each tuple are generated within the bounds defined
        by self.test_area, and make no attempt at being accurate relative
        to the input lidar.

        :param odir: When not an empty string, points will be written to CSV file in x, y, z format
        :param npoints: Number of points to create.
        :param vertical_offset: A vertical offset to apply to the points.
        :return: list[tuple(x, y, z), ..]
        """

        xmin, xmax = self.las.get_x_minmax()
        ymin, ymax = self.las.get_y_minmax()
        zmin, zmax = self.las.get_z_minmax()

        z_avg = 30
        zmin, zmax = z_avg - 10, z_avg + 10

        points = []
        for _ in range(npoints):
            rand_x = randint(int(xmin), int(xmax)) + random()
            rand_y = randint(int(ymin), int(ymax)) + random()
            rand_z = randint(int(zmin), int(zmax)) + random() + vertical_offset
            points.append(
                (rand_x, rand_y, rand_z)
            )

        if bool(odir) and bool(points):
            out_csv = self.file_basename.split(".")[0] + f"_{self.test_area.size}m_control.csv"
            out = os.path.join(odir, out_csv)
            with open(out, "w") as csv:
                csv.write("X, Y, Z\n")
                for p in points:
                    line = str(p).replace("(", "").replace(")", "") + "\n"
                    csv.write(line)

        return points

    @staticmethod
    def read_control(file):

        control = []
        with open(file, "r") as ctrl:
            ctrl.readline()
            for line in ctrl:
                coords_ls = line.split(",")
                coords = [float(coord.strip()) for coord in coords_ls[1:4]]
                p = Point3D(x=coords[1], y=coords[0], z=coords[2])
                gcp = GroundControlPoint(coord_xyz=p, std_xyz=None, name=coords_ls[0])
                control.append(gcp)

        return control


def main():
    test_data = TestData(FILE_LAS)
    flist = glob.glob(
        os.path.join(FLIST_DIR_DRIVE, "*.laz")
    )

    # flist = [FILE_LAS_PROBLEM1]
    fifth = flist[:int(len(flist) / 5)]
    vt = Vertigo(flist=fifth, control_points=None)
    col_fmt = ColumnFormat(x="coord_e", y="coord_n", z="coord_ht", name="point_id")
    vt.set_control(src=FILE_CTRL_SHP, column_format=col_fmt)
    vt.set_control_attribute_table()
    vt.assess(tin=True, grid=True, idw=6, verbose=True)
    vt.get_stats()
    res = vt.results_to_string()
    print(res)

    vtr = VertigoReport(vt, "vertigo_sample_report.pdf")
    vtr.write(title="OMP_BSR_024 Accuracy Report")


if __name__ == "__main__":
    main()
