from vertigo.VertigoType import Point3D
from rsge_toolbox.lidar.LidarSurface import Not3DDataError


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