# specific lib functions
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# system libs
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# user libs
from rsge_toolbox.util.time_tools import today_str
from Vertigo import Vertigo


DISCLAIMER_TEXT = (
    "This software is supplied as an evaluation tool for data providers "
    "to check for systematic errors in Lidar point cloud data sets. "
    "The final output report does in no way lead to the acceptance of a "
    "related Lidar data submission to the Province. GeoBC assumes no "
    "responsibility for errors or omissions in the software or "
    "documentation available.\n\n"
    "In no event shall GeoBC be liable to the user or any third parties "
    "for any special, punitive, incidental, indirect or consequential "
    "damages of any kind, or any damages whatsoever, including, without "
    "limitation, those resulting from loss of use, data or profits, "
    "whether or not GeoBC has been advised of the possibility of such "
    "damages, and on any theory of liability, arising out of or in "
    "connection with the use of this software.\n\n"
    "The use of the software downloaded through the provincial website "
    "is done at your own discretion and risk and with agreement that you "
    "will be solely responsible for any damage to your computer system "
    "or loss of data that results from such activities. No advice or "
    "information, whether oral or written, obtained from GeoBC or from "
    "the provincial website shall create any warranty for the software.\n\n"
    "By clicking OK you confirm that you have read and agree to the above "
    "disclaimer."
)


class VertigoGlossary:
    GLOSSARY = {
        "TIN (Triangulated Irregular Network)": [
            "A TIN is a digital representation of a terrain or surface, where the surface is divided into a set of non-overlapping triangles. TINs are commonly",
            " used in geospatial analysis for representing elevation or other continuous surfaces."
        ],

        "Grid": [
            "In the context of geospatial data, a grid refers to a regular or irregular arrangement of cells or pixels covering an area. Grids are commonly used, ",
            "for representing raster data, such as elevation grids or satellite imagery."
        ],

        "IDW (Inverse Distance Weighting)": [
            "IDW is a spatial interpolation technique that estimates unknown values at specific locations based on the known values of surrounding points. It ",
            "assigns weights to neighboring points based on their distance and uses these weights to calculate the interpolated value."
        ],

        "Surface RMSE (Root Mean Square Error)": [
            "Surface RMSE is a measure of the vertical variability of the derived plane from a set of nearest neighbour points. It represents the root mean square ",
            "difference between the LiDAR-derived surface and the reference surface, such as ground control points (GCPs). It quantifies the average vertical ",
            "deviation between the two surfaces."
        ],

        "Kolmogorov-Smirnov (KS) p-value": [
            "The KS p-value is a statistical measure resulting from the Kolmogorov-Smirnov test. It indicates the probability of obtaining the observed data or ",
            "data more extreme under the null hypothesis. In the context of vertical accuracy testing, the KS p-value can assess the agreement between observed ",
            "surface distances and a reference distribution."
        ],

        "Kolmogorov-Smirnov (KS) statistic": [
            "The KS statistic is the maximum vertical distance between the empirical cumulative distribution function (CDF) of the observed data and the theoretical ",
            "CDF of a reference distribution. It measures the discrepancy between the observed data and the reference distribution."
        ],

        "Quality Level (QL)": [
            "Quality Level refers to a metric or indicator that assesses the quality or reliability of LiDAR data or derived products. It can reflect the precision, ",
            "accuracy, completeness, or other characteristics of the data, often based on specified criteria or standards."
        ],

        "TIN Distance": [
            "TIN Distance refers to the plumbline distance measured between a given Ground Control Point (GCP) and a cluster of nearest neighbor LiDAR points",
            " surrounding the GCP within the Triangulated Irregular Network (TIN)."
        ],

        "Grid Distance": [
            "Grid Distance refers to the plumbline distance measured between a given Ground Control Point (GCP) and a cluster of nearest neighbor LiDAR points ",
            "surrounding the GCP within a grid or raster dataset."
        ],

        "IDW Distance": [
            "IDW Distance refers to the weighted Euclidean distance used in the Inverse Distance Weighting interpolation method. It determines the influence of",
            " neighboring points based on their distance in the interpolation process."
        ],

        "GCP (Ground Control Point)": [
            "GCP is a known point on the Earth's surface with accurately measured coordinates. GCPs are used as reference points to assess the accuracy and",
            " georeferencing of remote sensing data, including LiDAR data. They provide a reliable framework for aligning and validating the data."
        ]
    }


class VertigoReport:

    HIST_X_INCHES = 5.5
    HIST_Y_INCHES = 8.0
    DEFAULT_BIN_SIZE = 30
    RESULT_TABLE_SIZE = 35
    CONTROL_MAP_TABLE_SIZE = 20
    THRESHOLD_VERTICAL_RECESS = 0.20
    THRESHOLD_VERTICAL_OUTLIERS = 0.30

    COMPUTED_FROM = "Computed from"
    DEFAULT_TITLE = "VERTIGO REPORT"
    DEFAULT_NAME = "vertigo_report.pdf"
    COMPUTED_FROM_KEY = "computed_from"

    SUMMARY_COLUMNS = ["   ", "tin", "grid", "idw"]
    STATS_COLUMNS = ["min", "max", "std", "mean", "median", "rmse", "computed_from"]
    RESULT_COLUMNS = [
        "gcp name", "point count", "density", "surface rmse", "curvature", "ks p-value",
        "ks statistic", "QL", "tin distance", "grid distance", "idw distance"
    ]

    def __init__(self, vertigo: Vertigo, name: str = DEFAULT_NAME):
        self._name = name
        self._vertigo = vertigo

    def write(self, title: str = DEFAULT_TITLE):

        """
        Write the Vertigo computation results to a PDF file.

        :param title: Title to be drawn at the top of the PDF.
        """


        # =======================================================
        # First Page
        # =======================================================

        # Create a new PDF canvas with the specified page size
        c = canvas.Canvas(self._name, pagesize=letter)
        c = canvas.Canvas(self._name, pagesize=letter)

        # draw main page and summary information
        self.__cover_page_draw(c)

        # =======================================================
        # First Page
        # =======================================================

        # draw main page and summary information
        self.__main_page_draw(c, title)

        # Draw the summary table with table title
        self.__summary_table_draw(c)

        # --------------------------------------------------------

        # =======================================================
        # Histograms
        # =======================================================

        # Add a new page to start the histograms
        c.showPage()

        # draw the histograms
        self.__histograms_draw(c)

        # =======================================================
        # Result Table Pages
        # =======================================================

        # Draw the Control Map Title centered at the top of the page
        self.__results_table_draw(c)

        # --------------------------------------------------------

        # =======================================================
        # Control Map Pages
        # =======================================================

        # Draw the Control Map Title centered at the top of the page
        self.__control_map_table_draw(c)

        # --------------------------------------------------------

        # =======================================================
        # Control Attribute Table Pages
        # =======================================================

        # Draw the Control Map Title centered at the top of the page
        self.__attr_table_draw(c)

        # --------------------------------------------------------

        # =======================================================
        # Glossary Page
        # =======================================================

        # Draw the Control Map Title centered at the top of the page
        self.__glossary_draw(c)

        # --------------------------------------------------------

        c.save()

    def __histograms_draw(self, c: canvas):

        """
        Draw histograms for the report.

        :param c: reportlab canvas object.
        """

        tin_data, grid_data, idw_data = self._vertigo.get_dists()
        stats = self._vertigo.stats

        # Assuming your data is stored in the 'data' variable
        key_index = 0
        stat_keys = ["tin", "grid", "idw"]
        for data, title in zip([tin_data, grid_data, idw_data], ["TIN", "Grid", "IDW"]):

            hist_stats = stats[stat_keys[key_index]]

            if len(data) <= 0:
                continue

            # draw the title
            c.setFont("Helvetica", 20)
            c.drawCentredString(letter[0] / 2, letter[1] - inch, f"{title} Histogram")

            # Create a histogram using matplotlib
            n = len(data)
            bins = n if n < self.DEFAULT_BIN_SIZE else self.DEFAULT_BIN_SIZE
            fig, ax = plt.subplots()
            ax.set_xlabel(f"{title} Distances (m)")
            fig.set_size_inches(self.HIST_X_INCHES, self.HIST_Y_INCHES)  # Adjust the dimensions as needed
            ax.hist(data, bins=bins, edgecolor="black")

            # Adding text to top right and creating dummy lines for legend
            lines = [
                mlines.Line2D(
                    [], [], color='none', markersize=10, markeredgewidth=1.5,
                    label=f"{k}: {v}"
                ) for k, v in hist_stats.items()
            ]

            # Create legend
            plt.legend(handles=lines[:-1], loc='upper right', frameon=False)

            # Save the figure to a BytesIO buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)

            # Embed the histogram image onto the canvas
            image = ImageReader(buffer)
            c.drawImage(image, x=40, y=10)  # Adjust the position as needed

            # Advance to the next page
            c.showPage()
            key_index += 1

    @staticmethod
    def __cover_page_draw(c):

        # Draw the report title centered at the top of the page
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(letter[0] / 2, letter[1] - inch, "VERTIGO")

        # Draw the date
        c.setFont("Helvetica", 16)
        c.drawCentredString(
            letter[0] / 2, letter[1] - 1.4 * inch,
            "Vertical Error Regression Tool for Independent Ground Observations"
        )

        # Create an ImageReader object for the image
        image_path = "." + os.sep + "img" + os.sep + "geobc_logo.jpeg"
        img = ImageReader(image_path)

        # Specify the position and size of the image
        img_width = 379
        img_height = 160

        # Draw the image on the canvas
        # Adjust the y-position to move the image up or down
        c.drawImage(img, letter[0] / 2 - img_width / 2, letter[1] - 4.5 * inch, width=img_width, height=img_height)

        # Draw the disclaimer text
        styles = getSampleStyleSheet()
        disclaimer_style = styles["BodyText"]
        disclaimer_paragraph = Paragraph(DISCLAIMER_TEXT, disclaimer_style)
        disclaimer_wrapper = [disclaimer_paragraph, Spacer(1, 12)]

        # Define a padding for the paragraph
        padding = letter[0] * 0.05  # Adjust the padding as needed

        c.setFont("Helvetica", 4)
        x = padding
        y = letter[1] * 0.05  # place the paragraph near the bottom of the page
        for flowable in disclaimer_wrapper:
            # define the width and height for wrapping
            # Width is set to the page width minus padding on both sides
            width = letter[0] - 2 * padding
            height = letter[1] - 1.0
            flowable.wrapOn(c, width, height)
            flowable.drawOn(c, x, y)

        c.showPage()

    def __main_page_draw(self, c: canvas, title: str):

        """
        Draw the first page of the report.

        :param c: reportlab canvas object.
        :param title: Title to be drawn at the top of PDF.
        """

        # Draw the report title centered at the top of the page
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(letter[0] / 2, letter[1] - inch, title)
        # Draw the date
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(letter[0] / 2, letter[1] - 1.4 * inch, today_str())
        c.setFont("Helvetica", 16)
        c.drawCentredString(letter[0] / 2, letter[1] - 2.5 * inch, self.__number_files_str())
        c.drawCentredString(letter[0] / 2, letter[1] - 3.0 * inch, self.__number_gcp_str())
        c.drawCentredString(letter[0] / 2, letter[1] - 3.5 * inch, self.__tin_outliers_str())

    def __summary_table_draw(self, c):

        """
        Draw the first page of the report.

        :param c: reportlab canvas object.
        """

        c.setFont("Helvetica", 20)
        c.drawCentredString(letter[0] / 2, letter[1] - 5.2 * inch, "Results Summary")
        t = self.__summary_table(self._vertigo.stats)
        # Center the table horizontally
        table_width, table_height = t.wrapOn(c, inch, inch)
        x = (letter[0] - table_width) / 2
        y = (letter[1] - table_height) / 3
        t.drawOn(c, x=x, y=y)

    def __control_map_table_draw(self, c):

        """
        Draw the control map table to the pdf.

        :param c: reportlab canvas object.
        """

        # Rows of statistics for each computation method
        control_map_tables = self.__control_map_table()

        # Draw each table on a new page
        for i, control_map_table in enumerate(control_map_tables):
            # Draw the title for this table
            c.setFont("Helvetica", 20)
            c.drawCentredString(letter[0] / 2, letter[1] - inch, f"GCPs Per File (Pg. {i + 1})")

            # Calculate the height of the title
            title_height = inch + 20

            # Calculate the available height for the table
            start_y = letter[1] - inch - (title_height * .50)
            avail_height = start_y - inch

            # Calculate the available width for the table
            avail_width = letter[0] - 2 * inch

            # Draw the table on the canvas
            table_height = control_map_table.wrapOn(c, avail_width, avail_height)[1]
            control_map_table.drawOn(c, x=inch, y=start_y - table_height)

            # Show a new page after the table is drawn
            c.showPage()

    def __results_table_draw(self, c: canvas):

        """
        Draw the full results table on the PDF.

        :param c: reportlab canvas object.
        """

        # Rows of statistics for each computation method
        result_tables = self.__full_results_table()

        # Draw each part of the table on the canvas
        for i, result_table in enumerate(result_tables):
            # Draw the results table title on the canvas
            c.setFont("Helvetica", 20)
            c.drawCentredString(letter[0] / 2, letter[1] - inch, f"Results Table Pg. {i + 1}")

            # Calculate the height of the title
            title_height = inch + 20

            # Calculate the available height for the table
            start_y = letter[1] - inch - (title_height * .50)
            avail_height = start_y - inch

            # Split the table into parts that fit on the page
            table_parts = result_table.split(availWidth=letter[0] - 2 * inch, availHeight=avail_height)

            avail_width = letter[0] - inch - inch

            # Draw each part of the table on the canvas
            total_table_height = 0
            for table_part in table_parts:
                table_part_height = table_part.wrapOn(c, avail_width, avail_height)[1]
                total_table_height += table_part_height

                # If this part of the table won't fit on the page, start a new page
                if total_table_height > avail_height:
                    c.showPage()
                    total_table_height = table_part_height
                    start_y = letter[1] - inch - (title_height * .50)

                table_part.drawOn(c, x=inch * .2, y=start_y - total_table_height + 0.5 * inch)
                start_y -= table_part_height - 0.5 * inch

            c.showPage()

    def __attr_table_draw(self, c: canvas):

        """
        Draw the full results table on the PDF.

        :param c: reportlab canvas object.
        """

        # Rows of statistics for each computation method
        result_tables = self.__control_attribute_table()

        # Draw each part of the table on the canvas
        for i, result_table in enumerate(result_tables):
            # Draw the results table title on the canvas
            c.setFont("Helvetica", 20)
            c.drawCentredString(letter[0] / 2, letter[1] - inch, f"Control Table Pg. {i + 1}")

            # Calculate the height of the title
            title_height = inch + 20

            # Calculate the available height for the table
            start_y = letter[1] - inch - (title_height * .50)
            avail_height = start_y - inch

            # Split the table into parts that fit on the page
            table_parts = result_table.split(availWidth=letter[0] - 2 * inch, availHeight=avail_height)

            avail_width = letter[0] - inch - inch

            # Draw each part of the table on the canvas
            total_table_height = 0
            for table_part in table_parts:
                table_part_height = table_part.wrapOn(c, avail_width, avail_height)[1]
                total_table_height += table_part_height

                # If this part of the table won't fit on the page, start a new page
                if total_table_height > avail_height:
                    c.showPage()
                    total_table_height = table_part_height
                    start_y = letter[1] - inch - (title_height * .50)

                table_part.drawOn(c, x=inch * .2, y=start_y - total_table_height + 0.5 * inch)
                start_y -= table_part_height - 0.5 * inch

            c.showPage()

    @staticmethod
    def __glossary_draw(c: canvas):

        c.setFont("Helvetica", 20)
        c.drawCentredString(letter[0] / 2, letter[1] - inch, f"Glossary")

        # Set initial coordinates for drawing
        x = 50
        y = 700

        # Set spacing between glossary items
        line_spacing = 14
        glossary = VertigoGlossary.GLOSSARY

        for key in glossary.keys():
            # Draw the term
            c.setFont("Helvetica-Bold", 8)
            c.drawString(x, y, key)

            # Draw a line separator
            line_y = y - 5  # Adjust the position of the line
            c.line(x, line_y, x + 550, line_y)

            # Draw the description
            description_x = x + 20  # Adjust the position of the description
            c.setFont("Helvetica", 8)

            # split definition into chunks that fit on the page
            def_chunks = glossary[key]
            description_y = line_y - line_spacing
            for chunk in def_chunks:
                c.drawString(description_x, description_y, chunk)
                description_y -= line_spacing  # Update y-coordinate for the next line

            # Update the y-coordinate for the next glossary item
            y = description_y - line_spacing

    def __summary_table(self, vertigo_stats: dict) -> Table:

        """
        Create the summary table for PDF report.

        :param vertigo_stats: stats dictionary from vertigo object.
        :return: Table object.
        """

        # rows of statistics for each computation method
        stat_rows = []
        for stat in VertigoReport.STATS_COLUMNS:
            stat_row = [stat if stat != VertigoReport.COMPUTED_FROM_KEY else "GCPs Used: "]
            for col in VertigoReport.SUMMARY_COLUMNS[1:]:
                if stat != VertigoReport.COMPUTED_FROM_KEY:
                    stat_row.append(f"{vertigo_stats[col][stat]:.3f} m")
                else:
                    stat_row.append(f"{vertigo_stats[col][stat]}")
            stat_rows.append(stat_row)
        data = [VertigoReport.SUMMARY_COLUMNS, *stat_rows]

        # set column widths
        col_widths = [1.5 * inch] + [(6.5 / 3) * inch] * 3
        summary_table = Table(data, colWidths=col_widths)
        summary_table.setStyle(self.__summary_table_style())
        return summary_table

    def __full_results_table(self) -> list[Table]:

        """
        Create the full results table.

        Note that the table gets split into several tables of 30 rows.

        :return: list of Table objects.
        """

        # Create a sample stylesheet for paragraphs
        stylesheet = getSampleStyleSheet()

        # Rows of statistics for each computation method
        result_tables = []

        # split results into N size chunks for multiple Tables
        result_chunks = [self._vertigo.results[i:i + 30] for i in range(0, len(self._vertigo.results), 30)]
        for result_chunk in result_chunks:  # use each chunk to create a table
            result_table = [self.RESULT_COLUMNS]
            for result in result_chunk:  # loop through each result in the chunks of size N
                # normality_deviation = float(np.mean(result.surface.normality_deviation))
                result_row = [
                    Paragraph(result.gcp, stylesheet["BodyText"]),  # Convert string to Paragraph object
                    f"{result.surface.point_count} pts",
                    f"{round(result.surface.density, 2)} pts/m^2",
                    f"{round(result.surface.rmse, 3)} m",
                    f"{round(result.surface.curvature, 3)}",
                    # f"{round(normality_deviation, 2)} deg",
                    f"{round(result.surface.ks_test.p, 2)}",
                    f"{round(result.surface.ks_test.statistic, 2)}",
                    result.surface.quality_level,
                    f"{result.distance.tin} m",
                    f"{result.distance.grid} m",
                    f"{result.distance.idw} m"
                ]
                result_table.append(result_row)
            result_tables.append(result_table)

        # Set column widths
        for i in range(len(result_tables)):
            col_widths = [0.7 * inch] + [0.7 * inch] * 3 + [0.8 * inch] + [0.7 * inch] * 3
            result_tables[i] = Table(result_tables[i], splitByRow=1, colWidths=col_widths)
            result_tables[i].setStyle(self.__results_table_style())
            result_tables[i].width = letter[0] - 0.5 * inch
            result_tables[i].height = letter[1] - 1.5 * inch

        return result_tables

    def __control_attribute_table(self) -> list[Table]:

        """
        Create the full results table.

        Note that the table gets split into several tables of 30 rows.

        :return: list of Table objects.
        """

        if not self._vertigo.ctrl_attr_table:
            return []

        attr_tables = []

        # split results into N size chunks for multiple Tables
        attr_chunks = [self._vertigo.ctrl_attr_table[i:i + 30] for i in range(1, len(self._vertigo.ctrl_attr_table), 30)]
        for attr_chunk in attr_chunks:  # use each chunk to create a table
            attr_table = [self._vertigo.ctrl_attr_table[0]]  # column names
            for attr_row in attr_chunk:  # loop through each result in the chunks of size N
                row = [
                    item[0:10] if isinstance(item, str) and len(item) > 10
                    else item for item in attr_row
                ]
                attr_table.append(row)
            attr_tables.append(attr_table)

        # Set column widths
        results = self._vertigo.results
        flagged_results = [(result.gcp, result.flagged) for result in results if result.flagged]
        for i in range(len(attr_tables)):
            style = self.__attr_table_style()
            table = attr_tables[i]
            for j in range(len(table)):
                row = table[j]
                for res in flagged_results:
                    name = res[0]
                    if name == row[0]:
                        style.add('BACKGROUND', (0, j), (-1, j), colors.yellow)

            col_widths = [0.5 * inch] + [0.5 * inch] * 3 + [0.6 * inch] + [0.5 * inch] * 3
            attr_tables[i] = Table(attr_tables[i], splitByRow=1, colWidths=col_widths)
            attr_tables[i].setStyle(style)
            attr_tables[i].width = letter[0] - 0.5 * inch
            attr_tables[i].height = letter[1] - 1.5 * inch

        return attr_tables

    def __control_map_table(self) -> list[Table]:

        """
        Create the control map table.

        Note that the result gets split into a list of Table(s) of 20 rows each.

        :return: A list of Table objects.
        """

        # Rows of statistics for each computation method
        control_map_tables = []

        # Convert the control map dictionary to a list of tuples
        control_map_list = [(os.path.basename(filename), len(gcps)) for filename, gcps in
                            self._vertigo.control_map.items()]

        # Split the control map list into parts that fit on the page
        data_chunks = [control_map_list[i:i + 20] for i in range(0, len(control_map_list), 25)]

        # Loop through each chunk and create a table
        for chunk in data_chunks:
            control_table = [["Filename", "Number of GCPs"]]  # Add the headers to the beginning of the rows list
            for tup in chunk:
                fname = tup[0]
                num_gcps = tup[1]
                control_table.append(
                    [fname, num_gcps]
                )
            control_map_table = Table(control_table, colWidths=[5 * inch, 2 * inch])
            control_map_table.setStyle(self.__control_map_table_style())
            control_map_table.width = letter[0] - 1.5 * inch
            control_map_table.height = letter[1] - 2.5 * inch
            control_map_tables.append(control_map_table)

        return control_map_tables

    @staticmethod
    def __results_table_style() -> TableStyle:

        """
        Get the TableStyle for the results table.

        :return: TableStyle object.
        """

        ts = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTWEIGHT", (0, 0), (-1, 0), "BOLD"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 5),  # set the font size to 10 for the first row
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ]
        )

        return ts

    @staticmethod
    def __attr_table_style() -> TableStyle:

        """
        Get the TableStyle for the results table.

        :return: TableStyle object.
        """

        ts = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTWEIGHT", (0, 0), (-1, 0), "BOLD"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 5),  # set the font size to 10 for the first row
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ]
        )

        return ts

    @staticmethod
    def __summary_table_style() -> TableStyle:

        """
        Get the table style for the summary table.

        :return: TableStyle object.
        """

        ts = TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, 0), 16),
                ('FONTWEIGHT', (0, 0), (0, -1), 'BOLD'),  # Set font weight to bold for cells in the first column
                ('FONTWEIGHT', (0, 0), (-1, 0), 'BOLD'),  # Set font weight to bold for the first row
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                ('BACKGROUND', (0, -1), (-1, -1), colors.grey),
                ('TEXTCOLOR', (0, -1), (-1, -1), colors.whitesmoke),
                ('ALIGN', (0, -1), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, -1), (-1, -1), 16),
                ('TOPPADDING', (0, -1), (-1, -1), 10),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),  # Right-justify the first column
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Center all cells vertically
                ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),  # Add line above the first row
                ('FONTSIZE', (0, 0), (-1, 0), 14),  # set font size of first row to 14
                ('FONTSIZE', (0, -1), (-1, -1), 14),  # set font size of last row to 14
            ]
        )

        return ts

    @staticmethod
    def __control_map_table_style() -> TableStyle:

        """
        Get the table style for the control map table.

        :return: A TableStyle object.
        """

        # Create a TableStyle object to format the table
        ts = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTWEIGHT", (0, 0), (-1, 0), "BOLD"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ]
        )

        return ts

    def __number_files_str(self):

        """
        String constant for number of files
        """

        file_str = f"# of LiDAR Files Used: {len(self._vertigo.control_map.keys())}"
        return file_str

    def __number_gcp_str(self):

        """
        String constant for number of GCPs
        """

        gcp_count = 0
        for key in self._vertigo.control_map.keys():
            gcp_count += len(self._vertigo.control_map[key])
        gcp_str = f"# of GCPs Used: {gcp_count}"
        return gcp_str

    def __tin_outliers(self) -> int:

        """
        Compute number of TIN outliers.
        """

        outlier_count = 0
        for result in self._vertigo.results:
            lower_bound = result.distance.tin < (-1 * VertigoReport.THRESHOLD_VERTICAL_OUTLIERS)
            upper_bound = result.distance.tin > VertigoReport.THRESHOLD_VERTICAL_OUTLIERS
            if lower_bound or upper_bound:
                outlier_count += 1

        return outlier_count

    def __tin_outliers_str(self) -> str:

        """
        Constant for Outlier count string.
        """

        return f"# of Outliers: {self.__tin_outliers()}"


def main():
    pass


if __name__ == "__main__":
    main()
