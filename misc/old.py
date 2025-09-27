import argparse
import logging
import time
from collections import deque
from playsound import playsound

import brainflow
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import pandas as pd

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        #print(self.exg_channels)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 1000
        self.open_segments = 0
        self.closed_segments = 0

        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600), show=True)

        self.eyes_open = True
        self.num_each_seg = 2

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        print(self.board_shim.get_board_data_count())
        self.timer.start(self.update_speed_ms)
        #_ = self.board_shim.get_board_data()
        QtWidgets.QApplication.instance().exec()

    def update(self):
        #print("Writing data and clearing buffer")
        #data = self.board_shim.get_current_board_data(self.num_points) # does not delete ring buffer data
        data = self.board_shim.get_board_data(1000) # does not delete ring buffer data
        #print(data[-2][0])
        #print(data[-2][-1])
        # df = pd.DataFrame(np.transpose(data))
        #print(data.shape)
        labels = np.full((1, data.shape[1]), self.eyes_open)
        data = np.concatenate((data, labels), axis=0)
        DataFilter.write_file(data, 'test.csv', 'a')  # use 'a' for append mode
        self.eyes_open = not self.eyes_open
        if self.eyes_open:
            print("Open eyes")
            self.closed_segments += 1
        else:
            print("Close eyes")
            self.open_segments += 1
        
        if ((self.open_segments == self.closed_segments) and self.open_segments == self.num_each_seg):
            self.app.exit()

        # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()


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
        print("In 5 seconds, prepare to hold eyes open for 10 seconds")
        time.sleep(5)
        print("Starting recording")
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