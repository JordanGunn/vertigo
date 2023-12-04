import os
import csv
import fiona
import tkinter as tk
from tkinter import filedialog, ttk

from Vertigo import Vertigo, ColumnFormat
from VertigoReport import VertigoReport


class VertigoGUI:

    def __init__(self):

        self.fieldnames = []
        self.window = tk.Tk()
        self.window.title("VertigoGUI")

        # Set ttk Style to use rounded corners
        self.style = ttk.Style()
        self.style.configure(".", corner_radius=8)

        # Las/Laz file selection
        self.__las_file_selection()

        # Control point file selection
        self.__control_point_file_selection()

        # XYZ Column Name selection
        self.__xyz_coordinate_fields()

        # computation method selection
        self.__computation_methods()

        # set output names
        self.__report_settings()

        # run button
        self.__run_button()

        self.window.mainloop()

    def __las_file_selection(self):

        """
        Create GUI components related to LAS/LAZ file selection.
        """

        # Label for LAS/LAS file selection
        self.las_listbox_label = tk.Label(self.window, text="Select LAS/LAZ files:")
        self.las_listbox_label.grid(row=0, column=0, sticky="w")

        # Container that lists out LAS/LAZ files selected
        self.las_listbox = tk.Listbox(self.window)
        self.las_listbox.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")

        # Add files button
        self.add_las_button = tk.Button(self.window, text="Add Files", command=self.__add_las_files)
        self.add_las_button.grid(row=2, column=0, ipadx=70, padx=10, pady=5)

        # remove files button
        self.remove_las_button = tk.Button(self.window, text="Remove Files", command=self.__remove_las_files)
        self.remove_las_button.grid(row=2, column=1, ipadx=60, padx=2, pady=5)
        self.canvas = tk.Canvas(self.window, height=1, background="gray")

        # horizontal seperator
        self.canvas.grid(row=3, column=0, columnspan=3, sticky="we", pady=10)
        self.canvas.create_line(0, 0, 500, 0, fill="gray", width=2)

    def __add_las_files(self):

        """
        Add LAS/LAZ files to GUI data.
        """

        filetypes = (("LAS Files", "*.las"), ("LAZ Files", "*.laz"))
        files = filedialog.askopenfilenames(filetypes=filetypes)
        for file in files:
            self.las_listbox.insert(tk.END, file)

    def __remove_las_files(self):

        """
        Remove LAS/LAZ files from GUI data.
        """

        selection = self.las_listbox.curselection()
        for index in selection[::-1]:
            self.las_listbox.delete(index)

    def __control_point_file_selection(self):

        """
        Create GUI components for selection of Control Point files.
        """

        # Label for Control Point files selection
        self.control_file_label = tk.Label(self.window, text="Select Control Point File:")
        self.control_file_label.grid(row=4, column=0, pady=5, sticky="w")

        # Textbox to enter and display selected control point file.
        self.control_file_entry = tk.Entry(self.window)
        self.control_file_entry.grid(row=5, column=0, ipadx=2, pady=3)
        self.control_file_entry.bind("<<Modified>>", self.__populate_fields)

        # browse button for selecting control point file
        self.control_file_button = tk.Button(self.window, text="Browse", command=self.__select_control_file)
        self.control_file_button.grid(row=5, column=1, ipadx=50, pady=3)

    def __populate_fields(self, event=None):

        """
        Trigger population of XYZ and name column/fields into dropdown lists.
        """

        filename = self.control_file_entry.get()
        extension = os.path.splitext(filename)[1].lower()
        if extension in (".csv",):
            with open(filename) as f:
                reader = csv.reader(f)
                self.fieldnames = next(reader)
        elif extension in (".gpkg", ".shp"):
            with fiona.open(filename) as src:
                self.fieldnames = [prop for prop in src.schema["properties"]]
        else:
            return

    def __select_control_file(self):

        """
        Trigger filedialog for selection of control point files.
        """

        filetypes = (("CSV files", "*.csv"), ("GPKG files", "*.gpkg"), ("SHP files", "*.shp"))
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.control_file_entry.delete(0, tk.END)
            self.control_file_entry.insert(0, filename)
            extension = os.path.splitext(filename)[1].lower()
            if extension in (".csv",):
                with open(filename) as f:
                    reader = csv.reader(f)
                    self.fieldnames = next(reader)
            elif extension in (".gpkg", ".shp"):
                with fiona.open(filename) as src:
                    self.fieldnames = [prop for prop in src.schema["properties"]]
            else:
                return
            self.__populate_fields()

    def __xyz_coordinate_fields(self):

        """
        Create GUI components for XYZ and Name columns in control point data file.

        :return:
        """

        # GCP ID field selection dropdown
        self.gcp_id_label = tk.Label(self.window, text="GCP ID Field Name:")
        self.gcp_id_label.grid(row=7, column=0, padx=2, pady=3, sticky="w")
        self.gcp_id_input = ttk.Combobox(self.window, values=self.fieldnames)
        self.gcp_id_input.grid(row=7, column=1, padx=2, pady=3)
        self.gcp_id_input.bind("<Button-1>", lambda event: self.__update_dropdown(self.gcp_id_input))

        # X coordinate field selection dropdown
        self.x_label = tk.Label(self.window, text="X Coord Field Name:")
        self.x_label.grid(row=8, column=0, padx=2, pady=3, sticky="w")
        self.x_input = ttk.Combobox(self.window, values=self.fieldnames)
        self.x_input.grid(row=8, column=1, padx=2, pady=3)
        self.x_input.bind("<<ComboboxSelected>>", self.__populate_fields)
        self.x_input.bind("<Button-1>", lambda event: self.__update_dropdown(self.x_input))

        # Y coordinate field selection dropdown
        self.y_label = tk.Label(self.window, text="Y Coord Field Name:")
        self.y_label.grid(row=9, column=0, padx=2, pady=3, sticky="w")
        self.y_input = ttk.Combobox(self.window, values=self.fieldnames)
        self.y_input.grid(row=9, column=1, padx=2, pady=3)
        self.y_input.bind("<<ComboboxSelected>>", self.__populate_fields)
        self.y_input.bind("<Button-1>", lambda event: self.__update_dropdown(self.y_input))

        # Z coordinate field selection dropdown
        self.z_label = tk.Label(self.window, text="Z Coord Field Name:")
        self.z_label.grid(row=10, column=0, padx=2, pady=3, sticky="w")
        self.z_input = ttk.Combobox(self.window, values=self.fieldnames)
        self.z_input.grid(row=10, column=1, padx=2, pady=3)
        self.z_input.bind("<<ComboboxSelected>>", self.__populate_fields)
        self.z_input.bind("<Button-1>", lambda event: self.__update_dropdown(self.z_input))

        # Horizontal seperator
        self.canvas = tk.Canvas(self.window, height=1, background="gray")
        self.canvas.grid(row=11, column=0, columnspan=3, sticky="we", pady=10)
        self.canvas.create_line(0, 0, 500, 0, fill="gray", width=2)

    def __update_dropdown(self, dropdown):

        """
        Update the data stored in the XYZ and Name column dropdowns.

        :param dropdown: Target dropdown list.
        """

        dropdown['values'] = self.fieldnames

    def __computation_methods(self):

        """
        Create GUI components related to computation type (e.g. TIN, Grid, IDW).
        """

        # Computation Type label
        self.method_label = tk.Label(self.window, text="Select computation method:")
        self.method_label.grid(row=12, column=0, pady=5, sticky="w")

        # TIN computation checkbox
        self.tin_var = tk.BooleanVar()
        self.tin_checkbox = tk.Checkbutton(self.window, text="TIN", variable=self.tin_var)
        self.tin_checkbox.grid(row=13, column=0, padx=10, pady=5, sticky="w")

        # Grid computation checkbox
        self.grid_var = tk.BooleanVar()
        self.grid_checkbox = tk.Checkbutton(self.window, text="Grid", variable=self.grid_var)
        self.grid_checkbox.grid(row=13, column=1, padx=10, pady=5, sticky="w")

        # IDW computation checkbox
        self.idw_var = tk.BooleanVar()
        self.idw_checkbox = tk.Checkbutton(self.window, text="IDW", variable=self.idw_var, command=self.__toggle_idw_input)
        self.idw_checkbox.grid(row=14, column=0, padx=10, pady=5, sticky="w")

        # Number of IDW computation points entry box
        self.idw_input_label = tk.Label(self.window, text="# of IDW Points:")
        self.idw_input_label.grid(row=15, column=0, padx=20, pady=5, sticky="w")
        self.idw_input = tk.Entry(self.window, state=tk.DISABLED)
        self.idw_input.grid(row=15, column=1, padx=10, pady=5, sticky="w")

        # horizontal seperator
        self.canvas = tk.Canvas(self.window, height=1, background="gray")
        self.canvas.grid(row=16, column=0, columnspan=3, sticky="we", pady=10)
        self.canvas.create_line(0, 0, 500, 0, fill="gray", width=2)

    def __report_settings(self):

        """
        Create GUI components related to pdf output.
        """

        # report settings label
        self.precision_label = tk.Label(self.window, text="System Precision (metres):")
        self.precision_label.grid(row=17, column=0, pady=5, sticky="w")
        self.precision_label_entry = tk.Entry(self.window)
        self.precision_label_entry.grid(row=17, column=1, padx=2, pady=3)

        # output report filename entry box
        self.report_name_label = tk.Label(self.window, text="Output Report Name:")
        self.report_name_label.grid(row=18, column=0, padx=2, pady=3, sticky="w")
        self.report_name_entry = tk.Entry(self.window)
        self.report_name_entry.grid(row=18, column=1, padx=2, pady=3)

        # output report title entry box
        self.report_title_label = tk.Label(self.window, text="Report Title:")
        self.report_title_label.grid(row=19, column=0, padx=2, pady=3, sticky="w")
        self.report_title_entry = tk.Entry(self.window)
        self.report_title_entry.grid(row=19, column=1, padx=2, pady=3)

        # output report directory entry box
        self.report_directory_label = tk.Label(self.window, text="Output Directory:")
        self.report_directory_label.grid(row=20, column=0, padx=2, pady=3, sticky="w")
        self.report_directory_entry = tk.Entry(self.window)
        self.report_directory_entry.grid(row=20, column=1, padx=2, pady=3)

        # output directory browse button
        self.report_directory_button = tk.Button(self.window, text="Browse", command=self.__select_report_directory)
        self.report_directory_button.config()
        self.report_directory_button.grid(row=20, column=2, ipadx=20, padx=2, pady=3)

        # hoizontal seperator
        self.canvas = tk.Canvas(self.window, height=1, background="gray")
        self.canvas.grid(row=21, column=0, columnspan=3, sticky="we", pady=10)
        self.canvas.create_line(0, 0, 500, 0, fill="gray", width=2)

    def __run_button(self):

        """
        GUI component for the 'run' button.
        """

        self.run_button = tk.Button(self.window, text="Run", command=self.run_vertigo)
        self.run_button.grid(row=22, column=0, ipadx=250, pady=10, columnspan=3)

    def run_vertigo(self):

        """
        Execute Vertigo and VertigoReport.

        Drives the program.
        """

        # Get the selected LAS/LAZ files
        las_files = self.las_listbox.get(0, tk.END)

        # Get the control point file
        control_file = self.control_file_entry.get()

        # Get the XYZ coordinate fields
        gcp_id_field = self.gcp_id_input.get()
        x_field = self.x_input.get()
        y_field = self.y_input.get()
        z_field = self.z_input.get()
        fmt = ColumnFormat(x=x_field, y=y_field, z=z_field, name=gcp_id_field)

        # Get the computation method(s)
        methods = [None, None, None, None]
        if self.tin_var.get():
            methods[0] = 'tin'
        if self.grid_var.get():
            methods[1] = 'grid'
        if self.idw_var.get():
            methods[2] = 'idw'
            methods[3] = int(self.idw_input.get())

        # define booleans and input values for Vertigo execution
        idw_points = -1
        is_tin = "tin" == methods[0]
        is_grid = "grid" == methods[1]
        is_idw = "idw" == methods[2]
        if is_idw:
            idw_points = methods[3]

        # Get the output report name and title
        report_name = self.report_name_entry.get()
        report_title = self.report_title_entry.get()

        # Get the output report directory
        report_directory = self.report_directory_entry.get()

        # execute Vertigo
        vt = Vertigo(las_files)
        vt.set_control(src=control_file, column_format=fmt)
        vt.assess(tin=is_tin, grid=is_grid, idw=idw_points, verbose=True)
        vt.get_stats()

        # execute VertigoReport
        out = os.path.join(report_directory, report_name)
        report = VertigoReport(vt, name=out)
        report.write(report_title)

    def __select_report_directory(self):

        """
        Set report directory selection event.
        """

        directory = filedialog.askdirectory()
        self.report_directory_entry.delete(0, tk.END)
        self.report_directory_entry.insert(0, directory)

    def __toggle_idw_input(self):

        """
        Enable/Disable IDW computation input textbox.
        """

        if self.idw_var.get():
            self.idw_input.config(state='normal')
        else:
            self.idw_input.delete(0, tk.END)
            self.idw_input.config(state='disabled')


if __name__ == '__main__':
    gui = VertigoGUI()
