from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic

import sys
import os
import threading
import socket
import configparser

import mq_server


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


gui_form = uic.loadUiType(resource_path('monitor_gui.ui'))[0]


class StdoutRedirect(QObject):
    print_occur = pyqtSignal(str, str, name="print")

    def __init__(self):
        QObject.__init__(self, None)
        self.daemon = True
        self.sys_stdout = sys.stdout.write
        self.sys_stderr = sys.stderr.write

    def stop(self):
        sys.stdout.write = self.sys_stdout
        sys.stderr.write = self.sys_stderr

    def start(self):
        sys.stdout.write = self.write
        sys.stderr.write = lambda msg: self.write(msg)

    def write(self, s, color='black'):
        sys.stdout.flush()
        self.print_occur.emit(s, color)


class MainWindow(QMainWindow, gui_form):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.config_filename = './setting.ini'
        self.table_th_list = []

        # monitor tab
        monitor_layout = QVBoxLayout()
        monitor_layout.addWidget(self.plain_text_monitor)
        self.tab_monitor.setLayout(monitor_layout)

        central_layout = QGridLayout()
        central_layout.addWidget(self.tabWidget)
        main = QWidget()
        main.setLayout(central_layout)
        self.setCentralWidget(main)

        # setting tab
        setting_layout = QGridLayout()
        setting_layout.addWidget(self.table_threshold, 0, 0, 1, 2)
        setting_layout.addWidget(self.push_btn_apply, 1, 0, 1, 1)
        setting_layout.addWidget(self.push_btn_reset, 1, 1, 1, 1)
        self.group_th.setLayout(setting_layout)

        config = configparser.ConfigParser()
        config.read(self.config_filename, encoding='UTF8')
        self.config_list = []
        self.part_lib_num = int(config.get('part', 'lib_num'))
        for idx in range(0, self.part_lib_num):
            self.config_list.append(config.get('part', f'lib_{idx:02}').split(','))
        tab_lib_num = int(config.get('tab', 'lib_num'))
        for idx in range(0, tab_lib_num):
            self.config_list.append(config.get('tab', f'lib_{idx:02}').split(','))

        self.table_threshold.setColumnWidth(0, 175)
        # self.table_threshold.resizeColumnToContents(1)
        # table_threshold_state_header = self.table_threshold.horizontalHeader()
        # table_threshold_state_header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_threshold.setHorizontalHeaderLabels(['Part Lib', 'Threshold'])
        self.set_table_threshold()

        ip_add = socket.gethostbyname(socket.gethostname())
        self.lb_ip.setText(ip_add)
        self.lb_port.setText('5672')
        self.lb_id.setText('rabbitmq')
        self.lb_pw.setText('rabbitmq')

        # connected function
        self.push_btn_apply.clicked.connect(self.push_btn_apply_clicked)
        self.push_btn_reset.clicked.connect(self.push_btn_reset_clicked)

        # cmd monitor
        self.stdout = StdoutRedirect()
        self.stdout.start()
        self.stdout.print_occur.connect(lambda x: self.append_text(x))

    def append_text(self, msg):
        if not msg == '\n':
            if '[Error]' in msg:
                html = (f'<p style=color:red;>' + msg)
                self.plain_text_monitor.appendHtml(html)
            elif '[~]' in msg:
                self.plain_text_monitor.clear()
                self.plain_text_monitor.appendPlainText(msg)
            elif '[-]' in msg:
                self.plain_text_monitor.clear()
            else:
                html = (f'<p style=color:black;>' + msg)
                self.plain_text_monitor.appendHtml(html)

    def set_table_threshold(self):
        self.table_threshold.setRowCount(0)
        self.table_th_list = []
        for idx in range(0, len(self.config_list)):
            table_file_lib = QTableWidgetItem()
            table_file_lib.setTextAlignment(Qt.AlignCenter)
            table_file_lib.setText(self.config_list[idx][0])
            table_th = QLineEdit()
            table_th.setStyleSheet('border:white')
            self.table_th_list.append(table_th)
            self.table_th_list[idx].setAlignment(Qt.AlignCenter)
            self.table_th_list[idx].setText(self.config_list[idx][1])
            self.table_threshold.insertRow(idx)
            self.table_threshold.setItem(idx, 0, table_file_lib)
            self.table_threshold.setCellWidget(idx, 1, self.table_th_list[idx])

    def push_btn_apply_clicked(self):
        # sample = self.table_threshold.itemAt(0, 0)
        # sample = self.table_th_list[0].text()

        current_th_list = []
        try:
            for idx in range(0, len(self.config_list)):
                float_check = float(self.table_th_list[idx].text())
                if not 0.0 <= float_check <= 1.0:
                    raise Exception
                else:
                    current_th_list.append(float_check)
            for idx in range(0, len(self.config_list)):
                self.config_list[idx][1] = str(current_th_list[idx])
        except:
            QMessageBox.warning(None, 'message', 'Enter a number between 0~1')
        finally:
            self.set_table_threshold()

            config = configparser.ConfigParser()
            config.read(self.config_filename, encoding='UTF8')
            for idx in range(0, self.part_lib_num):
                record_str = f'{self.config_list[idx][0]},{self.config_list[idx][1]},{self.config_list[idx][2]}'
                config.set('part', f'lib_{idx:02}', record_str)
            for idx in range(self.part_lib_num, len(self.config_list)):
                record_str = f'{self.config_list[idx][0]},{self.config_list[idx][1]},{self.config_list[idx][2]}'
                config.set('tab', f'lib_{idx - self.part_lib_num:02}', record_str)
            with open(self.config_filename, 'w', encoding='UTF8') as config_edit:
                config.write(config_edit)

    def push_btn_reset_clicked(self):
        for idx in range(0, len(self.config_list)):
            self.config_list[idx][1] = self.config_list[idx][2]
        self.set_table_threshold()

        config = configparser.ConfigParser()
        config.read(self.config_filename, encoding='UTF8')
        for idx in range(0, self.part_lib_num):
            record_str = f'{self.config_list[idx][0]},{self.config_list[idx][1]},{self.config_list[idx][2]}'
            config.set('part', f'lib_{idx:02}', record_str)
        for idx in range(self.part_lib_num, len(self.config_list)):
            record_str = f'{self.config_list[idx][0]},{self.config_list[idx][1]},{self.config_list[idx][2]}'
            config.set('tab', f'lib_{idx - self.part_lib_num:02}', record_str)
        with open(self.config_filename, 'w', encoding='UTF8') as config_edit:
            config.write(config_edit)

    def closeEvent(self, event):
        close_event_box = QMessageBox.warning(self, "Quit",
                                              "Are you sure you want to quit the [AI Verify] program?",
                                              QMessageBox.Yes | QMessageBox.No)
        if close_event_box == QMessageBox.Yes:
            close_event_box2 = QMessageBox.critical(self, "Quit",
                                                    "Do you really want to end the [AI Verify] program?",
                                                    QMessageBox.Yes | QMessageBox.No)
            if close_event_box2 == QMessageBox.Yes:
                event.accept()
                write_error_log("[Quit] The AI Verify program has ended.")
            else:
                event.ignore()
        else:
            event.ignore()


def start_server():
    verify_server = mq_server.RabbitMQServer()
    try:
        verify_server.set_ai_verify()
    except Exception as ex:
        print(f'[Error] | Failed to load AI weight : {ex}')
        write_error_log(ex)
        os.system('pause')
    else:
        while True:
            try:
                verify_server.channel_connect()
            except Exception as ex:
                print(f'[Error] | Failed to set RabbitMQ server : {ex}')
                import time
                time.sleep(1)
                write_error_log(ex)


def write_error_log(err_msg):
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%H:%M:%S')
    try:
        os.makedirs("./log", exist_ok=True)
        f = open(f'./log/log_{today}.txt', 'a', encoding='utf-8')
        f.write(f'{now} | {err_msg}\n')
        f.close()
    except Exception as ex:
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    # console 감추기
    import win32gui
    import win32console
    win32gui.ShowWindow(win32console.GetConsoleWindow(), 0)
    #
    main_window.show()
    thread_server = threading.Thread(target=start_server)
    thread_server.setDaemon(True)
    thread_server.start()
    app.exec_()
