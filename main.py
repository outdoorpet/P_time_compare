from PyQt4 import QtCore, QtGui, QtWebKit, QtNetwork
from obspy import read_inventory, read_events, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
import pandas as pd
import functools
import os
import pyqtgraph as pg
import numpy as np
from DateAxisItem import DateAxisItem


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, cat_nm=None, pick_nm=None, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = np.array(data.values)
        self._cols = data.columns
        self.r, self.c = np.shape(self._data)

        self.cat_nm = cat_nm
        self.pick_nm = pick_nm

        # Column headers for tables
        self.cat_col_header = ['Event ID', 'Lat (dd)', 'Lon  (dd)', 'Depth (km)', 'Mag', 'Time (UTC)']
        self.pick_col_header = ['Station', 'Event ID', 'Arr Time Residual (s)', 'P Arr Time (UTC)',
                                'P_as Arr Time (UTC)']

    def rowCount(self, parent=None):
        return self.r

    def columnCount(self, parent=None):
        return self.c

    def data(self, index, role=QtCore.Qt.DisplayRole):

        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return self._data[index.row(), index.column()]
        return None

    def headerData(self, p_int, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if not self.cat_nm == None:
                    return self.cat_col_header[p_int]
                elif not self.pick_nm == None:
                    return self.pick_col_header[p_int]
            elif orientation == QtCore.Qt.Vertical:
                return p_int
        return None


class TableDialog(QtGui.QDialog):
    """
    Class to create a separate child window to display the event catalogue and picks
    """

    def __init__(self, parent=None, cat_df=None, pick_df=None):
        super(TableDialog, self).__init__(parent)

        self.cat_df = cat_df
        self.pick_df = pick_df

        self.initUI()

    def initUI(self):
        self.layout = QtGui.QVBoxLayout(self)

        self.cat_event_table_view = QtGui.QTableView()
        self.pick_table_view = QtGui.QTableView()

        self.cat_event_table_view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.pick_table_view.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        self.layout.addWidget(self.cat_event_table_view)
        self.layout.addWidget(self.pick_table_view)

        self.setLayout(self.layout)

        # Populate the tables using the custom Pandas table class
        self.cat_model = PandasModel(self.cat_df, cat_nm=True)
        self.pick_model = PandasModel(self.pick_df, pick_nm=True)

        self.cat_event_table_view.setModel(self.cat_model)
        self.pick_table_view.setModel(self.pick_model)

        self.setWindowTitle('Tables')
        self.show()


class MainWindow(QtGui.QWidget):
    """
    Main Window for Timing Error QC Application
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi()
        self.show()
        self.raise_()
        QtGui.QApplication.instance().focusChanged.connect(self.changed_widget_focus)

    def setupUi(self):
        main_vbox = QtGui.QVBoxLayout()
        self.setLayout(main_vbox)

        buttons_hbox = QtGui.QHBoxLayout()
        self.open_pick_button = QtGui.QPushButton('Open Picks')
        openPick = functools.partial(self.open_pick_file)
        self.open_pick_button.released.connect(openPick)
        buttons_hbox.addWidget(self.open_pick_button)

        self.open_cat_button = QtGui.QPushButton('Open Catalogue')
        openCat = functools.partial(self.open_cat_file)
        self.open_cat_button.released.connect(openCat)
        self.open_cat_button.setEnabled(False)
        buttons_hbox.addWidget(self.open_cat_button)

        self.open_xml_button = QtGui.QPushButton('Open StationXML')
        openXml = functools.partial(self.open_xml_file)
        self.open_xml_button.released.connect(openXml)
        self.open_xml_button.setEnabled(False)
        buttons_hbox.addWidget(self.open_xml_button)

        self.sort_drop_down_button = QtGui.QPushButton('Sort')
        self.sort_drop_down_button.setEnabled(False)
        buttons_hbox.addWidget(self.sort_drop_down_button)

        # Button to Bring all events together onto a closer X axis
        self.gather_events_checkbox = QtGui.QCheckBox('Gather Events')
        Gather = functools.partial(self.gather_events_checkbox_selected)
        self.gather_events_checkbox.stateChanged.connect(Gather)
        self.gather_events_checkbox.setEnabled(False)
        buttons_hbox.addWidget(self.gather_events_checkbox)

        main_vbox.addLayout(buttons_hbox)

        centre_hbox = QtGui.QHBoxLayout()

        left_grid_lay = QtGui.QGridLayout()

        self.graph_view = pg.GraphicsLayoutWidget()

        left_grid_lay.addWidget(self.graph_view, 0, 0, 3, 6)

        self.col_grad_w = pg.GradientWidget(orientation='bottom')
        self.col_grad_w.loadPreset('spectrum')
        self.col_grad_w.setEnabled(False)
        self.col_grad_w.setToolTip("""
                - Click a triangle to change its color
                - Drag triangles to move
                - Click in an empty area to add a new color
                - Right click a triangle to remove
                """)

        self.max_lb = QtGui.QLabel("Max")
        self.min_lb = QtGui.QLabel("Min")

        self.reset_view_tool_button = QtGui.QPushButton()
        self.reset_view_tool_button.setIcon(QtGui.QIcon('eLsS8.png'))
        self.reset_view_tool_button.released.connect(self.reset_plot_view)
        self.reset_view_tool_button.setToolTip("Reset the scatter plot zoom and sort method")

        left_grid_lay.addWidget(self.col_grad_w, 4, 1, 1, 2)
        left_grid_lay.addWidget(self.min_lb, 4, 0, 1, 1)
        left_grid_lay.addWidget(self.max_lb, 4, 4, 1, 1)
        left_grid_lay.addWidget(self.reset_view_tool_button, 4, 5, 1, 1)

        centre_hbox.addLayout(left_grid_lay)

        # Open StreetMAP view
        view = self.view = QtWebKit.QWebView()
        cache = QtNetwork.QNetworkDiskCache()
        cache.setCacheDirectory("cache")
        view.page().networkAccessManager().setCache(cache)
        view.page().networkAccessManager()

        view.page().mainFrame().addToJavaScriptWindowObject("MainWindow", self)
        view.page().setLinkDelegationPolicy(QtWebKit.QWebPage.DelegateAllLinks)
        view.load(QtCore.QUrl('map.html'))
        view.loadFinished.connect(self.onLoadFinished)
        view.linkClicked.connect(QtGui.QDesktopServices.openUrl)

        centre_hbox.addWidget(view)

        main_vbox.addLayout(centre_hbox)

    def onLoadFinished(self):
        with open('map.js', 'r') as f:
            frame = self.view.page().mainFrame()
            frame.evaluateJavaScript(f.read())

    @QtCore.pyqtSlot(float, float, str, str, int)
    def onMap_marker_selected(self, lat, lng, event_id, df_id, row_index):
        self.table_view_highlight(self.tbl_view_dict[str(df_id)], row_index)

    def changed_widget_focus(self):
        try:
            if not QtGui.QApplication.focusWidget() == self.graph_view:
                self.scatter_point_deselect()
        except AttributeError:
            pass

    def sort_method_selected(self, sort_pushButton, value, prev_view):
        # Method to plot information on the scatter plot and to provide sort functionality
        # All calls to update the plot area should pass through here rather than calling update_plot
        if prev_view:
            try:
                self.saved_state = self.plot.getViewBox().getState()
            except AttributeError:
                # Plot does not exist, i.e. it is the first time trying to call update_graph
                self.saved_state = None
        elif not prev_view:
            self.saved_state = None
        # if no sort:
        if value[1] == "no_sort":
            sort_pushButton.setText("Sort")
            self.axis_station_list = self.picks_df['sta'].unique()
            self.update_graph()
        # if sort by station:
        elif value[1] == 0:
            sort_pushButton.setText(value[0])
            self.axis_station_list = np.sort(self.picks_df['sta'].unique())  # numpy array
            self.update_graph()
        # if sort by gcarc
        elif value[1] == 1:
            sort_pushButton.setText("Sorted by GCARC: " + value[0])
            self.axis_station_list = self.spatial_dict[value[0]].sort_values(by='gcarc')['station'].tolist()
            self.update_graph()
        # if sort by azimuth
        elif value[1] == 2:
            sort_pushButton.setText("Sorted by AZ: " + value[0])
            self.axis_station_list = self.spatial_dict[value[0]].sort_values(by='az')['station'].tolist()
            self.update_graph()
        # if sort by ep dist
        elif value[1] == 3:
            sort_pushButton.setText("Sorted by Ep Dist: " + value[0])
            self.axis_station_list = self.spatial_dict[value[0]].sort_values(by='ep_dist')['station'].tolist()
            self.update_graph()

    def reset_plot_view(self):
        self.sort_method_selected(self.sort_drop_down_button, ('no_sort', 'no_sort'), False)

    def dispMousePos(self, pos):
        # Display current mouse coords if over the scatter plot area as a tooltip
        if self.gather_events_checkbox.isChecked():
            # Dont display the tooltip
            pass
        elif not self.gather_events_checkbox.isChecked():
            x_coord = UTCDateTime(self.plot.vb.mapSceneToView(pos).toPoint().x()).ctime()
            self.time_tool = self.plot.setToolTip(x_coord)

    def gather_events_checkbox_selected(self):
        self.sort_method_selected(self.sort_drop_down_button, ('no_sort', 'no_sort'), False)

    def update_graph(self):
        # List of colors for individual scatter poinst based on the arr time residual
        col_list = self.picks_df['col_val'].apply(lambda x: self.col_grad_w.getColor(x)).tolist()

        self.graph_view.clear()

        # generate unique stationID integers
        # unique_stations = self.picks_df['sta'].unique()
        enum_sta = list(enumerate(self.axis_station_list))

        # rearrange dict
        sta_id_dict = dict([(b, a) for a, b in enum_sta])

        def get_sta_id(x):
            return (sta_id_dict[x['sta']])

        # add column with sta_id to picks df
        self.picks_df['sta_id'] = self.picks_df.apply(get_sta_id, axis=1)

        y_axis_string = pg.AxisItem(orientation='left')
        y_axis_string.setTicks([enum_sta])

        if not self.gather_events_checkbox.isChecked():

            # Set up the plotting area
            self.plot = self.graph_view.addPlot(0, 0, title="Time Difference",
                                                axisItems={'bottom': DateAxisItem(orientation='bottom',
                                                                                  utcOffset=0), 'left': y_axis_string})
            self.plot.setMouseEnabled(x=True, y=False)

            # When Mouse is moved over plot print the data coordinates
            self.plot.scene().sigMouseMoved.connect(self.dispMousePos)

            # Re-establish previous view if it exists
            if self.saved_state:
                self.plot.getViewBox().setState(self.saved_state)

            # Plot Error bar showing P and Pas arrivals
            diff_frm_mid = (self.picks_df['tt_diff'].abs() / 2).tolist()
            err = pg.ErrorBarItem(x=self.picks_df['time_mid'], y=self.picks_df['sta_id'], left=diff_frm_mid,
                                  right=diff_frm_mid, beam=0.05)

            self.plot.addItem(err)

            # Plot extra scatter points showing P_as picked (crosses)
            p_as_scatter_plot = pg.ScatterPlotItem(pxMode=True)
            p_as_scatter_plot.addPoints(self.picks_df['P_as_pick_time_UTC'],
                                        self.picks_df['sta_id'], symbol="+", size=10,
                                        brush=col_list)
            self.plot.addItem(p_as_scatter_plot)

            # Plot midway scatter points between time diff
            self.time_diff_scatter_plot = pg.ScatterPlotItem(pxMode=True)
            self.lastClicked = []
            self.time_diff_scatter_plot.sigClicked.connect(self.scatter_point_clicked)
            self.time_diff_scatter_plot.addPoints(self.picks_df['time_mid'],
                                                  self.picks_df['sta_id'], size=9,
                                                  brush=col_list)
            self.plot.addItem(self.time_diff_scatter_plot)

        elif self.gather_events_checkbox.isChecked():

            rearr_midpoint_dict = [(b, a) for a, b in self.midpoint_dict.iteritems()]

            x_axis_string = pg.AxisItem(orientation='bottom')
            x_axis_string.setTicks([rearr_midpoint_dict])

            # Set up the plotting area
            self.plot = self.graph_view.addPlot(0, 0, title="Time Difference",
                                                axisItems={'left': y_axis_string, 'bottom': x_axis_string})
            self.plot.setMouseEnabled(x=True, y=False)

            # Re-establish previous view if it exists
            if self.saved_state:
                self.plot.getViewBox().setState(self.saved_state)

            # Plot Error bar showing P and Pas arrivals
            diff_frm_mid = (self.picks_df['tt_diff'].abs() / 2).tolist()
            err = pg.ErrorBarItem(x=self.picks_df['alt_midpoints'], y=self.picks_df['sta_id'], left=diff_frm_mid,
                                  right=diff_frm_mid, beam=0.05)

            self.plot.addItem(err)

            # Plot extra scatter points showing P picked (crosses)
            p_as_scatter_plot = pg.ScatterPlotItem(pxMode=True)
            p_as_scatter_plot.addPoints(self.picks_df['alt_p_as'],
                                        self.picks_df['sta_id'], symbol="+", size=10,
                                        brush=col_list)
            self.plot.addItem(p_as_scatter_plot)

            # Plot midway scatter points between time diff
            self.time_diff_scatter_plot = pg.ScatterPlotItem(pxMode=True)
            self.lastClicked = []
            self.time_diff_scatter_plot.sigClicked.connect(self.scatter_point_clicked)
            self.time_diff_scatter_plot.addPoints(self.picks_df['alt_midpoints'],
                                                  self.picks_df['sta_id'], size=9,
                                                  brush=col_list)
            self.plot.addItem(self.time_diff_scatter_plot)

    def scatter_point_deselect(self):
        try:
            for p in self.lastClicked:
                p.resetPen()
        except AttributeError:
            pass

    def scatter_point_select(self, scatter_index):
        # Select the point on the scatter plot
        selected_point_tbl = [self.time_diff_scatter_plot.points()[scatter_index]]
        for p in self.lastClicked:
            p.resetPen()
        for p in selected_point_tbl:
            p.setPen('r', width=2)
        self.lastClicked = selected_point_tbl

    def scatter_point_clicked(self, plot, points):
        if self.gather_events_checkbox.isChecked():
            self.select_scatter_pick = self.picks_df.loc[(self.picks_df['alt_midpoints'] == points[0].pos()[0]) &
                                                         (self.picks_df['sta_id'] == points[0].pos()[1]), :]
        elif not self.gather_events_checkbox.isChecked():
            self.select_scatter_pick = self.picks_df.loc[self.picks_df['time_mid'] == points[0].pos()[0]]
        pick_row_index = self.select_scatter_pick.index.tolist()[0]

        # if there is a catalogue loaded in attempt to highlight it on the list
        try:
            self.table_view_highlight(self.tbld.pick_table_view, pick_row_index)
        except AttributeError:
            print('Error: No Earthquake Catalogue is Loaded')
            pass

    def build_tables(self):

        self.table_accessor = None

        if self.gather_events_checkbox.isChecked():
            # drop some columns from the dataframes
            dropped_picks_df = self.picks_df.drop(['col_val', 'sta_id', 'time_mid', 'alt_midpoints', 'alt_p_as'],
                                                  axis=1)
        elif not self.gather_events_checkbox.isChecked():
            dropped_picks_df = self.picks_df.drop(['col_val', 'sta_id', 'time_mid'],
                                                  axis=1)

        # make string rep of time
        def mk_picks_UTC_str(row):
            return (pd.Series([UTCDateTime(row['P_pick_time_UTC']).ctime(),
                               UTCDateTime(row['P_as_pick_time_UTC']).ctime()]))

        dropped_picks_df[['P_time', 'P_as_time']] = dropped_picks_df.apply(mk_picks_UTC_str, axis=1)

        dropped_picks_df = dropped_picks_df.drop(['P_pick_time_UTC', 'P_as_pick_time_UTC'], axis=1)

        dropped_cat_df = self.cat_df

        # make UTC string from earthquake cat
        def mk_cat_UTC_str(row):
            return (UTCDateTime(row['qtime']).ctime())

        dropped_cat_df['Q_time'] = dropped_cat_df.apply(mk_cat_UTC_str, axis=1)

        dropped_cat_df = dropped_cat_df.drop(['qtime'], axis=1)

        self.tbld = TableDialog(parent=self, cat_df=dropped_cat_df, pick_df=dropped_picks_df)

        # Lookup Dictionary for table views
        self.tbl_view_dict = {"cat": self.tbld.cat_event_table_view, "picks": self.tbld.pick_table_view}

        # Create a new table_accessor dictionary for this class
        self.table_accessor = {self.tbld.cat_event_table_view: [dropped_cat_df, range(0, len(dropped_cat_df))],
                               self.tbld.pick_table_view: [dropped_picks_df, range(0, len(dropped_picks_df))]}

        self.tbld.cat_event_table_view.clicked.connect(self.table_view_clicked)
        self.tbld.pick_table_view.clicked.connect(self.table_view_clicked)

        # If headers are clicked then sort
        self.tbld.cat_event_table_view.horizontalHeader().sectionClicked.connect(self.headerClicked)
        self.tbld.pick_table_view.horizontalHeader().sectionClicked.connect(self.headerClicked)

    def table_view_highlight(self, focus_widget, row_index):

        if focus_widget == self.tbld.cat_event_table_view:
            self.tbld.pick_table_view.clearSelection()
            self.selected_row = self.cat_df.loc[row_index]

            # Find the row_number of this index
            cat_row_number = self.table_accessor[focus_widget][1].index(row_index)
            focus_widget.selectRow(cat_row_number)

            # Highlight the marker on the map
            js_call = "highlightEvent('{event_id}');".format(event_id=self.selected_row['event_id'])
            self.view.page().mainFrame().evaluateJavaScript(js_call)

        elif focus_widget == self.tbld.pick_table_view:
            self.selected_row = self.picks_df.loc[row_index]

            # Select the point on the scatter plot
            self.scatter_point_select(row_index)

            pick_row_number = self.table_accessor[focus_widget][1].index(row_index)
            focus_widget.selectRow(pick_row_number)

            # get the quake selected
            pick_tbl_event_id = self.selected_row['pick_event_id']
            cat_row_index = self.cat_df[self.cat_df['event_id'] == pick_tbl_event_id].index.tolist()[0]

            # Find the row_number of this index on the earthquake cat table
            cat_row_number = self.table_accessor[self.tbld.cat_event_table_view][1].index(cat_row_index)
            self.tbld.cat_event_table_view.selectRow(cat_row_number)

            # Highlight the marker on the map
            js_call = "highlightEvent('{event_id}');".format(event_id=self.selected_row['pick_event_id'])
            self.view.page().mainFrame().evaluateJavaScript(js_call)

    def headerClicked(self, logicalIndex):
        focus_widget = QtGui.QApplication.focusWidget()
        table_df = self.table_accessor[focus_widget][0]

        header = focus_widget.horizontalHeader()

        self.order = header.sortIndicatorOrder()
        table_df.sort_values(by=table_df.columns[logicalIndex],
                             ascending=self.order, inplace=True)

        self.table_accessor[focus_widget][1] = table_df.index.tolist()

        if focus_widget == self.tbld.cat_event_table_view:
            self.model = PandasModel(table_df, cat_nm=True)
        elif focus_widget == self.tbld.pick_table_view:
            self.model = PandasModel(table_df, pick_nm=True)

        focus_widget.setModel(self.model)
        focus_widget.update()

    def table_view_clicked(self):
        focus_widget = QtGui.QApplication.focusWidget()
        row_number = focus_widget.selectionModel().selectedRows()[0].row()
        row_index = self.table_accessor[focus_widget][1][row_number]
        # Highlight/Select the current row in the table
        self.table_view_highlight(focus_widget, row_index)

    def plot_inv(self):
        # plot the stations
        print(self.inv)
        for i, station in enumerate(self.inv[0]):
            if station.code in self.picks_df['sta'].unique():
                js_call = "addStation('{station_id}', {latitude}, {longitude});" \
                    .format(station_id=station.code, latitude=station.latitude,
                            longitude=station.longitude)
                self.view.page().mainFrame().evaluateJavaScript(js_call)

    def plot_events(self):
        # Plot the events
        for row_index, row in self.cat_df.iterrows():
            js_call = "addEvent('{event_id}', '{df_id}', {row_index}, " \
                      "{latitude}, {longitude}, '{a_color}', '{p_color}');" \
                .format(event_id=row['event_id'], df_id="cat", row_index=int(row_index), latitude=row['lat'],
                        longitude=row['lon'], a_color="Red",
                        p_color="#008000")
            self.view.page().mainFrame().evaluateJavaScript(js_call)

    def open_pick_file(self):
        pick_filenames = QtGui.QFileDialog.getOpenFileNames(
            parent=self, caption="Choose File",
            directory=os.path.expanduser("~"),
            filter="Pick Files (*.pick)")

        # dictionary to contain pandas merged array for each event
        self.event_df_dict = {}

        # iterate through selected files
        for _i, pick_file in enumerate(pick_filenames):
            pick_file = str(pick_file)
            event_id = os.path.basename(pick_file).split('_')[0]

            # read pick file into dataframe
            df = pd.read_table(pick_file, sep=' ', header=None, names=['sta', 'phase', 'date', 'hrmin', 'sec'],
                               usecols=[0, 4, 6, 7, 8], dtype=str)

            df = df.drop(df[df['phase'] == 'To'].index)

            df[df['phase'].iloc[0] + '_pick_time'] = df['date'].astype(str) + 'T' + df['hrmin'].astype(str) \
                                                     + df['sec'].astype(str)
            df['pick_event_id'] = event_id

            df = df.drop(['phase', 'date', 'hrmin', 'sec'], axis=1)

            dict_query = event_id in self.event_df_dict

            if not dict_query:
                # save the df to the dictionary
                self.event_df_dict[event_id] = df

            elif dict_query:
                # merge the dataframes for same events
                new_df = pd.merge(self.event_df_dict.get(event_id), df, how='outer', on=['sta', 'pick_event_id'])
                self.event_df_dict[event_id] = new_df

        # now concat all dfs
        self.picks_df = pd.concat(self.event_df_dict.values())

        def calc_diff(x):
            if x['P_pick_time'] == np.nan or x['P_as_pick_time'] == np.nan:
                return (np.nan)

            P_UTC = UTCDateTime(x['P_pick_time']).timestamp
            P_as_UTC = UTCDateTime(x['P_as_pick_time']).timestamp

            time_diff = P_UTC - P_as_UTC

            return (pd.Series([P_UTC, P_as_UTC, time_diff]))

        # calculate diff between theoretical and observed tt, change P_pick_time and P_as_pick_time to
        # UTCDateTime stamps
        # The function called by apply will return a dataframe
        self.picks_df[['P_pick_time_UTC', 'P_as_pick_time_UTC', 'tt_diff']] = \
            self.picks_df.apply(calc_diff, axis=1)  # passes series object row-wise to function

        self.picks_df.reset_index(drop=True, inplace=True)
        self.picks_df = self.picks_df.drop(['P_pick_time', 'P_as_pick_time'], axis=1)

        # Now normalize the tt_diff column to get colors
        max_val = self.picks_df['tt_diff'].max()
        min_val = self.picks_df['tt_diff'].min()

        self.picks_df['col_val'] = (self.picks_df['tt_diff'] - min_val) / (max_val - min_val)

        self.picks_df['time_mid'] = (self.picks_df['P_pick_time_UTC'] - 0.5 * self.picks_df['tt_diff'])

        print('--------')
        print(self.picks_df)

        self.col_grad_w.setEnabled(True)
        col_change = functools.partial(self.sort_method_selected, self.sort_drop_down_button, ('no_sort', 'no_sort'),
                                       True)
        self.col_grad_w.sigGradientChanged.connect(col_change)

        # update min/max labels
        self.max_lb.setText(str("%.2f" % max_val))
        self.min_lb.setText(str("%.2f" % min_val))

        self.open_cat_button.setEnabled(True)

        self.sort_method_selected(self.sort_drop_down_button, ('no_sort', 'no_sort'), False)

    def open_cat_file(self):
        self.cat_filename = str(QtGui.QFileDialog.getOpenFileName(
            parent=self, caption="Choose File",
            directory=os.path.expanduser("~"),
            filter="XML Files (*.xml)"))
        if not self.cat_filename:
            return

        self.cat = read_events(self.cat_filename)

        # create empty data frame
        self.cat_df = pd.DataFrame(data=None, columns=['event_id', 'qtime', 'lat', 'lon', 'depth', 'mag'])

        # iterate through the events
        for _i, event in enumerate(self.cat):
            # Get quake origin info
            origin_info = event.preferred_origin() or event.origins[0]

            # check if the eventID is in the pick files.
            if not str(event.resource_id.id).split('=')[1] in self.picks_df['pick_event_id'].values:
                continue

            try:
                mag_info = event.preferred_magnitude() or event.magnitudes[0]
                magnitude = mag_info.mag
            except IndexError:
                # No magnitude for event
                magnitude = None

            self.cat_df.loc[_i] = [str(event.resource_id.id).split('=')[1], int(origin_info.time.timestamp),
                                   origin_info.latitude, origin_info.longitude,
                                   origin_info.depth, magnitude]

        self.cat_df.reset_index(drop=True, inplace=True)

        print('------------')
        print(self.cat_df)
        self.build_tables()
        self.plot_events()

        def add_quakes_menu(menu_item, action_id):
            for event in self.cat_df['event_id']:
                menu_item.addAction(event, functools.partial(
                    self.sort_method_selected, self.sort_drop_down_button, (event, action_id), True))

        # add the events and sort methods to sort menu dropdown
        sort_menu = QtGui.QMenu()

        # first add top level sort method items
        sort_menu.addAction('by Station', functools.partial(
            self.sort_method_selected, self.sort_drop_down_button, ('Sorted by Station..', 0)))
        by_GCARC = sort_menu.addMenu('by GCARC Dist')
        by_az = sort_menu.addMenu('by Azimuth')
        by_epdist = sort_menu.addMenu('by Ep Dist')

        # now add events as second level
        add_quakes_menu(by_GCARC, 1)
        add_quakes_menu(by_az, 2)
        add_quakes_menu(by_epdist, 3)

        self.sort_drop_down_button.setMenu(sort_menu)

        midpoint_list = []
        # sort the events by time:
        sorted_events_list = self.cat_df.sort_values(by='qtime')['event_id'].tolist()
        for _i, event in enumerate(sorted_events_list):
            print(event)
            if _i == 0:
                midpoint_list.append(0)
                print '...'
                continue

            select_prev_events_df = self.picks_df.loc[self.picks_df['pick_event_id'] == sorted_events_list[_i - 1]]
            select_events_df = self.picks_df.loc[self.picks_df['pick_event_id'] == event]

            max_prev_tt_diff = (select_prev_events_df['tt_diff'].abs() / 2).max()
            max_current_tt_diff = (select_events_df['tt_diff'].abs() / 2).max()

            max_right = midpoint_list[_i - 1] + max_prev_tt_diff

            midpoint = max_right + 0.5 + max_current_tt_diff

            midpoint_list.append(midpoint)

        self.midpoint_dict = dict(zip(sorted_events_list, midpoint_list))
        print(self.midpoint_dict)

        # assign the new midpoint values to the picks dataframe
        def add_x_midpoints(row):
            alt_midpoint = self.midpoint_dict[row['pick_event_id']]

            alt_p_as = alt_midpoint - row['tt_diff'] / 2

            return (pd.Series([alt_midpoint, alt_p_as]))

        self.picks_df[['alt_midpoints', 'alt_p_as']] = self.picks_df.apply(add_x_midpoints, axis=1)

        print('-------------')
        print(self.picks_df)

        self.gather_events_checkbox.setEnabled(True)
        self.open_xml_button.setEnabled(True)

    def open_xml_file(self):
        self.stn_filename = str(QtGui.QFileDialog.getOpenFileName(
            parent=self, caption="Choose File",
            directory=os.path.expanduser("~"),
            filter="XML Files (*.xml)"))
        if not self.stn_filename:
            return

        self.inv = read_inventory(self.stn_filename)
        self.plot_inv()

        # Now create distance to stations dict for each event
        unique_stations = self.picks_df['sta'].unique()

        self.spatial_dict = {}

        def calc_spatial_diff(x):
            temp_df = pd.DataFrame(data=None, columns=['station', 'gcarc', 'az', 'ep_dist'])
            for _i, station in enumerate(unique_stations):
                stn_lat = self.inv.select(station=station)[0][0].latitude
                stn_lon = self.inv.select(station=station)[0][0].longitude
                # first GCARC dist & az
                gcarc, az, baz = gps2dist_azimuth(stn_lat, stn_lon, x['lat'], x['lon'])

                gcarc = gcarc / 1000  # turn it into km's

                # epicentral_dist
                ep_dist = kilometer2degrees(gcarc)

                temp_df.loc[_i] = [station, gcarc, az, ep_dist]

            self.spatial_dict[x['event_id']] = temp_df

        self.cat_df.apply(calc_spatial_diff, axis=1)

        self.sort_drop_down_button.setEnabled(True)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = MainWindow()
    w.raise_()
    app.exec_()
