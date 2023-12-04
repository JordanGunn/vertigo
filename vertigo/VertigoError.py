class InsufficientSurfacePointsError(Exception):

    def __init__(self, message: str = "Insufficient data to generate surface."):
        self.message = message
        super().__init__(self.message)


class InvalidVectorFormatError(Exception):

    def __init__(self, message: str = "Supported Formats: SHP, CSV, GPKG"):
        self.message = message
        super().__init__(self.message)
