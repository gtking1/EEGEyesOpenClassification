import argparse
import logging
import time
from collections import deque

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        #print(self.exg_channels)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        print("SAMPLING RATE", self.sampling_rate)
        self.nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600), show=True)

        self._init_timeseries()

        while board_shim.get_board_data_count() <= self.nfft:
            time.sleep(self.nfft / self.sampling_rate) # allow time to gather >=2 data points for nfft
        #print(board_shim.get_board_data_count())

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        self.alphas = deque()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
        
        p = self.win.addPlot(row=16, col=0)
        p.showAxis('left', False)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', False)
        p.setMenuEnabled('bottom', False)
        self.plots.append(p)
        curve = p.plot()
        self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points) # does not delete ring buffer data
        fftdata = data
        for count, channel in enumerate(self.exg_channels):
            #print(count, channel)
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())
            #print(type(data[channel]))
            #print(data[channel].shape)
        
        DataFilter.detrend(fftdata[7], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(fftdata[7], self.nfft, self.nfft // 2, self.sampling_rate,
                                   WindowOperations.BLACKMAN_HARRIS.value)
        band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0) # try get_custom_band_powers
        self.alphas.append(band_power_alpha)
        if len(self.alphas) >= 100:
            self.alphas.popleft()
        self.curves[16].setData(self.alphas)
        print(np.average(self.alphas))
        # self.curves[16].setData(fftdata[16].tolist()) # copies data of bottom curve

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    useRealBoard = False
    
    if useRealBoard == False:
        # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=0)
        parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
        parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                            default=0)
        parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
        parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=False, default=BoardIds.SYNTHETIC_BOARD)
        parser.add_argument('--file', type=str, help='file', required=False, default='')
        parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                            required=False, default=BoardIds.NO_BOARD)

    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()

    if useRealBoard:
        params.serial_port = "/dev/cu.usbserial-DP04VZS3"
        board_id = BoardIds.CYTON_DAISY_BOARD
        board_shim = BoardShim(board_id, params)
    else:
        params.ip_port = args.ip_port
        params.serial_port = args.serial_port
        params.mac_address = args.mac_address
        params.other_info = args.other_info
        params.serial_number = args.serial_number
        params.ip_address = args.ip_address
        params.ip_protocol = args.ip_protocol
        params.timeout = args.timeout
        params.file = args.file
        params.master_board = args.master_board

        board_shim = BoardShim(args.board_id, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params) # 450000 is number of samples in ring buffer 
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()