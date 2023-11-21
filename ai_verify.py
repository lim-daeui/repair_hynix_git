import os
from ftplib import FTP
from PIL import Image
from datetime import datetime
import xml.etree.ElementTree as et
import configparser
import tensorflow as tf
import keras
import numpy as np
from collections import OrderedDict
import json
import math

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import efficientnet
import ai_verify_retina as retina

# gpu 선택
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# gpu 메모리 초기화
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)

# package_label = {
#     0: ['Resistor', 'Capacitor'],
#     1: ['diode', 'tvs_diode'],
#     2: ['mosfet', 'switch'],
#     3: ['voltage_level_translator', 'load_switch']
# }
package_label = dict()


def xml_reader(xml_path, target_idx):
    img_list = []
    part_list = []
    lib_list = []

    try:
        f = open(xml_path, 'r', encoding='utf-8')
        text = f.read()
        f.close()
    except:
        f = open(xml_path, 'r', encoding='euc-kr')
        text = f.read()
        f.close()

    device_root = et.fromstring(text)
    for device in device_root:
        product_idx = device.attrib['ID']
        product_barcode = device.attrib['Barcode']
        part_info = device.find('DEFECT')
        part_id = part_info.attrib['PartUID']
        object_id = part_info.attrib['ObjUID']
        defect_info = part_info.attrib['DefectType']
        # lib_id = part_info.attrib['LibraryName']
        defect_pos = part_info.find('DefectPos')
        pos_left = int(defect_pos.attrib['LEFT'])
        pos_top = int(defect_pos.attrib['TOP'])
        pos_right = int(defect_pos.attrib['RIGHT']) - pos_left
        pos_bottom = int(defect_pos.attrib['BOTTOM']) - pos_top
        absolute_pos = part_info.find('SaveLT')
        abs_pos_x = int(absolute_pos.attrib['X'])
        abs_pos_y = int(absolute_pos.attrib['Y'])

        if product_idx == target_idx.zfill(5):
            img_name = f'[{product_idx}][{product_barcode}]{part_id}#{object_id}#{defect_info}#{pos_left}_{pos_top}_{pos_right}_{pos_bottom}_{abs_pos_x}_{abs_pos_y}.jpg'
            if img_name not in img_list:
                success_find_lib = 0
                ## 수정 전
                # for package_info in package_label.items():
                #     lib_id = part_info.attrib['LibraryName']
                #     # lib_id = 'tab'
                #     if lib_id in package_info[1]:
                #         img_list.append(img_name)
                #         part_list.append(part_id)
                #         lib_list.append(package_info[0])
                #         success_find_lib = 1
                ##
                ## 수정 후
                lib_id = part_info.attrib['LibraryName']
                for package_class in package_label:
                    if lib_id in package_label[package_class]:
                        img_list.append(img_name)
                        part_list.append(part_id)
                        lib_list.append(package_class)
                        success_find_lib = 1
                ##
                if success_find_lib == 0:
                    img_list.append(img_name)
                    part_list.append(part_id)
                    lib_list.append(999)

    return img_list, part_list, lib_list


def json_writer(rot_num, img_info, pred_result, save_path, top_btm, no_insp_check):
    component = []
    folder = OrderedDict()

    for i in range(0, len(img_info)):
        if no_insp_check == 0:
            if pred_result[i] != 'SKIP':
                feature_item = {}
                feature = []
                component_item = {}

                feature_item['AI_overrule_AOI'] = '0'
                feature_item['show_image'] = '1'
                feature_item['predicted_label'] = pred_result[i]

                # for j in range
                feature.append(feature_item)

                # component_item['posNo'] = img_info[i].RefID
                # component_item['pos'] = str(int(img_info[i].ModuleID) + 1)
                component_item['pos'] = img_info[i]
                component_item['posNo'] = img_info[i]
                component_item['features'] = feature
                component.append(component_item)
        else:
            feature_item = {}
            feature = []
            component_item = {}

            feature_item['AI_overrule_AOI'] = '0'
            feature_item['show_image'] = '1'
            if pred_result[i] != 'SKIP':
                feature_item['predicted_label'] = pred_result[i]
            else:
                feature_item['predicted_label'] = 'Undefined'
            # for j in range
            feature.append(feature_item)

            # component_item['posNo'] = img_info[i].RefID
            # component_item['pos'] = str(int(img_info[i].ModuleID) + 1)
            component_item['pos'] = img_info[i]
            component_item['posNo'] = img_info[i]
            component_item['features'] = feature
            component.append(component_item)

    folder['identifier'] = rot_num
    folder['component'] = component

    try:
        with open(save_path + f'{rot_num}_{top_btm}.json', 'w', encoding='utf-8') as make_file:
            json.dump(folder, make_file, indent='\t')
        print(f' Json file is created')
        return True
    except Exception as ex:
        print(f'[Error] | Failed to create json file : {ex}')
        return False


class FTPConnect():

    def __init__(self):
        self.ftp_ip_add = ''
        self.ftp = None

    def login_server(self, ip_add, ftp_id='D', ftp_pw=''):
        self.ftp_ip_add = ip_add
        self.ftp = FTP(self.ftp_ip_add)
        self.ftp.encoding = 'cp949'
        self.ftp.login(ftp_id, ftp_pw)

    def logout_server(self):
        self.ftp.quit()

    def download_file(self, load_path, save_path, index):
        select_file_list = []
        select_part_list = []
        select_lib_list = []
        try:
            self.ftp.cwd(load_path)
        except Exception as ex:
            print(' [Error] | Failed to access the folder. Check if the folder is valid')
        else:
            data = []
            file_list = []
            self.ftp.dir(data.append)
            for datum in data:
                pos = datum.rfind(' ')
                file_list.append(datum[pos + 1:])
            # print(file_list)

            xml_name = 'INSP_RESULT.xml'
            if xml_name in file_list:
                self.ftp.retrbinary('RETR ' + xml_name, open(save_path + xml_name, 'wb').write)

                select_file_list, select_part_list, select_lib_list = xml_reader(save_path + xml_name, index)
                for index, select_file in enumerate(select_file_list):
                    # 시간 측정
                    # download_start = datetime.now()
                    if not select_lib_list[index] == 999:
                        if select_file in file_list:
                            try:
                                self.ftp.retrbinary('RETR ' + select_file, open(save_path + select_file, 'wb').write)
                            except Exception as ex:
                                print(' [Error] | Failed to download file. Check if the ftp server is accessible ')
                    else:
                        if select_file in file_list:
                            try:
                                self.ftp.retrbinary('RETR ' + select_file,
                                                    open(save_path + 'NoTrain_' + select_file, 'wb').write)
                            except Exception as ex:
                                print(' [Error] | Failed to download file. Check if the ftp server is accessible ')
                    # 시간 측정
                    # download_end = datetime.now()
                    # print(f'image download time : {(download_end - download_start).total_seconds()} sec')

        return select_file_list, select_part_list, select_lib_list

    def upload_file(self, load_path, save_path, file):
        # try:
        #     self.ftp.cwd('json_list')
        # except:
        #     self.ftp.mkd('json_list')
        #     self.ftp.cwd('json_list')
        # finally:
        #     self.ftp.storbinary('STOR ' + file, open(load_path + file, 'rb'))
        if not self.ftp.nlst('json_list'):
            self.ftp.mkd('json_list')
        self.ftp.cwd('json_list')
        self.ftp.storbinary('STOR ' + file, open(load_path + file, 'rb'))


def pre_normalize(input_data, normalize_method=0):
    if normalize_method == 0:  # base
        input_data = input_data / 255.0
    elif normalize_method == 1:  # xception imgnet
        input_data = input_data / 127.5 - 1.0

    elif normalize_method == 2:  # resnet imgnet
        mean = [103.939, 116.779, 123.68]  # BGR
        for index in range(0, len(input_data)):
            input_data[index][..., 0] -= mean[0]
            input_data[index][..., 1] -= mean[1]
            input_data[index][..., 2] -= mean[2]

    elif normalize_method == 3:  # efficient imgnet
        input_data = input_data / 255.0
        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]
        channel = input_data[0].shape[2]
        if channel == 3:
            for index in range(0, len(input_data)):
                input_data[index][..., 0] -= mean[0]
                input_data[index][..., 1] -= mean[1]
                input_data[index][..., 2] -= mean[2]
                if std is not None:
                    input_data[index][..., 0] /= std[0]
                    input_data[index][..., 1] /= std[1]
                    input_data[index][..., 2] /= std[2]
        elif channel == 1:
            mean_one = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]
            std_one = 0.299 * std[0] + 0.587 * std[1] + 0.114 * std[2]
            input_data -= mean_one
            if std is not None:
                input_data /= std_one
    return input_data


def efficient_net_b2(input_row, input_col, input_channel):
    input = keras.layers.Input(shape=(input_row, input_col, input_channel))
    conv = efficientnet.EfficientNet(1.1, 1.2, 260, 0.3, model_name='efficientnet-b2',
                                     include_top=False, weights='imagenet', pooling='avg')(input)
    drout = keras.layers.Dropout(0.4)(conv)
    fc = keras.layers.Dense(1, activation=tf.nn.sigmoid)(drout)

    net = keras.models.Model(inputs=input, outputs=fc)
    net.compile(optimizer=keras.optimizers.SGD(),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    return net


class AIVerify():

    def __init__(self):
        self.lib_list = []
        self.model_list = []
        self.threshold_list = []

        self.input_row = 260
        self.input_col = 260
        self.input_channel = 3

        self.home = ''

        self.config_file = './setting.ini'
        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='UTF8')

        mode = int(config.get('server', 'GPU'))
        if mode:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.part_lib_num = int(config.get('part', 'lib_num'))
        for idx in range(0, self.part_lib_num):
            self.lib_list.append(config.get('part', f'lib_{idx:02}').split(',')[0])
            package_label[idx] = config.get('part', f'lib_{idx:02}_ele').split(',')
        self.tab_lib_num = int(config.get('tab', 'lib_num'))
        for idx in range(0, self.tab_lib_num):
            self.lib_list.append(config.get('tab', f'lib_{idx:02}').split(',')[0])
            package_label[idx + self.part_lib_num] = config.get('tab', f'lib_{idx:02}_ele').split(',')
        self.home = config.get('server', 'home') + '/'

    def load_model(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        print(f' [~] Loading(0/{len(self.lib_list)})')
        dummy = np.zeros((self.input_row, self.input_col, self.input_channel))
        dummy = np.expand_dims(dummy, axis=0)
        for index, lib in enumerate(self.lib_list):
            if index < self.part_lib_num:
                model = efficient_net_b2(self.input_row, self.input_col, self.input_channel)
                model.load_weights(f'./weight/{lib}_weight.h5')
                model.predict(dummy)
            else:
                model = models.load_model(f'./weight/{lib}_weight.h5')
                model.predict_on_batch(dummy)
            self.model_list.append(model)
            self.threshold_list.append(0.5)
            comma = '.' * index
            print(f' [~] Loading{comma}({index + 1}/{len(self.lib_list)})')
        print(' [-] Loading Finish!')

    def inspect_ftp(self, msg):
        start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{start} : Receive a message')
        # 시간 측정
        inspect_start = datetime.now()

        split_msg = msg.split('|')
        pcb_idx = split_msg[1]
        path_msg = split_msg[0].split('\\')
        access_ip = path_msg[0]
        top_btm = path_msg[-1].split('_')[-1]
        del path_msg[0]
        server_path = '/'.join(path_msg)

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='UTF8')
        self.home = config.get('server', 'home') + '/'
        download_home = self.home + start.split(' ')[0] + '/'

        detail_path = access_ip
        for idx in range(1, len(path_msg)):
            detail_path = os.path.join(detail_path, path_msg[idx])
        if not (os.path.isdir(download_home + detail_path)):
            os.makedirs(download_home + detail_path)
        target_path = download_home + detail_path + '/'

        no_insp_check = int(config.get('server', 'no_inspect'))
        # library 실시간 체크
        for idx in range(0, self.part_lib_num):
            package_label[idx] = config.get('part', f'lib_{idx:02}_ele').split(',')
        for idx in range(0, self.tab_lib_num):
            package_label[idx + self.part_lib_num] = config.get('tab', f'lib_{idx:02}_ele').split(',')
        
        # ftp 서버 연결
        ftp_id = config.get('ftp', 'id')
        ftp_pw = config.get('ftp', 'pw')
        ftp_server = FTPConnect()
        try:
            ftp_server.login_server(ip_add=access_ip, ftp_id=ftp_id, ftp_pw=ftp_pw)
        except Exception as ex:
            print(' [Error] | Failed to access the IP. Check if the IP is valid')
            inspect_file_list, inspect_part_list, inspect_lib_list = [], [], []
        else:
            inspect_file_list, inspect_part_list, inspect_lib_list = ftp_server.download_file(server_path,
                                                                                              target_path,
                                                                                              pcb_idx)
        predict_list = []

        for idx in range(0, self.part_lib_num):
            self.threshold_list[idx] = float(config.get('part', f'lib_{idx:02}').split(',')[1])
        for idx in range(0, self.tab_lib_num):
            self.threshold_list[idx + self.part_lib_num] = float(config.get('tab', f'lib_{idx:02}').split(',')[1])

        if inspect_file_list:
            for idx, inspect_file in enumerate(inspect_file_list):
                if not inspect_lib_list[idx] == 999:
                    try:
                        # 시간 측정
                        # open_start = datetime.now()
                        img = Image.open(target_path + inspect_file).convert('RGB')
                        # 시간 측정
                        # open_end = datetime.now()
                        # print(f'image open time : {(open_end - open_start).total_seconds()} sec')
                    except Exception as ex:
                        print(' [Error] | Failed to open Image. Check if the image file is valid')
                        predict_list.append('SKIP')
                    else:
                        # 시간 측정
                        # ai_start = datetime.now()
                        if inspect_lib_list[idx] < self.part_lib_num:
                            img = img.resize((self.input_row, self.input_col), resample=Image.BILINEAR)
                            img = np.float32(img)
                            img = np.expand_dims(img, 0)
                            img = pre_normalize(img, 3)
                            prediction = self.model_list[inspect_lib_list[idx]].predict(img)
                            if prediction < self.threshold_list[inspect_lib_list[idx]]:
                                predict_list.append('NOK')
                            else:
                                predict_list.append('OK')
                        elif inspect_lib_list[idx] in [self.part_lib_num, self.part_lib_num+1]:
                            prediction = retina.predict_retina_net_chip(model=self.model_list[inspect_lib_list[idx]],
                                                                        img=img,
                                                                        th=self.threshold_list[inspect_lib_list[idx]])
                            predict_list.append(prediction)
                        elif inspect_lib_list[idx] == self.part_lib_num + 2:
                            prediction = retina.predict_retina_net_tab(model=self.model_list[inspect_lib_list[idx]],
                                                                       img=img,
                                                                       th=self.threshold_list[inspect_lib_list[idx]])
                            predict_list.append(prediction)
                        elif inspect_lib_list[idx] == self.part_lib_num + 3:
                            prediction = retina.predict_retina_net_ground(model=self.model_list[inspect_lib_list[idx]],
                                                                          img=img,
                                                                          th=self.threshold_list[inspect_lib_list[idx]])
                            predict_list.append(prediction)
                        elif inspect_lib_list[idx] == self.part_lib_num + 4:
                            prediction = retina.predict_retina_net_edge(model=self.model_list[inspect_lib_list[idx]],
                                                                        img=img,
                                                                        th=self.threshold_list[inspect_lib_list[idx]])
                            predict_list.append(prediction)
                        # 시간 측정
                        # ai_end = datetime.now()
                        # print(f'{idx} inspect time : {(ai_end - ai_start).total_seconds()} sec')
                else:
                    predict_list.append('SKIP')

            # 검사 결과 대행
            # f = open(target_path + f'{pcb_idx}.json', 'w')
            # for inspect_file in inspect_file_list:
            #     f.write(inspect_file)
            # f.close()
            success = json_writer(pcb_idx, inspect_part_list, predict_list, target_path, top_btm, no_insp_check)

            if success:
                ftp_server.upload_file(target_path, server_path, f'{pcb_idx}_{top_btm}.json')
            ftp_server.logout_server()
        else:
            print('No valid data in INSP_RESULT.xml')
        # 시간 측정
        inspect_end = datetime.now()
        print(f'Total inspect time : {(inspect_end - inspect_start).total_seconds()} sec')

    def inspect_http(self, msg):
        start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{start} : Receive a message')
        split_msg = msg.split('|')
        pcb_idx = split_msg[1]
        path_msg = split_msg[0].split('\\')
        access_ip = path_msg[0]
        # del path_msg[0]
        top_btm = path_msg[-1].split('_')[-1]
        server_path = '/'.join(path_msg)
        download_home = self.home + start.split(' ')[0] + '/'

        detail_path = '\\\\' + access_ip
        for idx in range(1, len(path_msg)):
            detail_path = os.path.join(detail_path, path_msg[idx])
        detail_path = detail_path + '/'
        xml_name = 'INSP_RESULT.xml'
        inspect_file_list, inspect_part_list, inspect_lib_list = xml_reader(detail_path + xml_name, pcb_idx)
        predict_list = []

        if inspect_file_list:
            for idx, inspect_file in enumerate(inspect_file_list):
                if os.path.isfile(detail_path + inspect_file):
                    img = Image.open(detail_path + inspect_file).convert('RGB')
                    if inspect_lib_list[idx] < self.part_lib_num:
                        img = img.resize((self.input_row, self.input_col), resample=Image.BILINEAR)
                        img = np.float32(img)
                        img = np.expand_dims(img, 0)
                        img = pre_normalize(img, 3)
                        prediction = self.model_list[inspect_lib_list[idx]].predict(img)
                        if prediction < self.threshold_list[inspect_lib_list[idx]]:
                            predict_list.append('NOK')
                        else:
                            predict_list.append('OK')
                    elif inspect_lib_list[idx] == self.part_lib_num:
                        prediction = retina.predict_retina_net_tab(model=self.model_list[inspect_lib_list[idx]], img=img,
                                                                   th=self.threshold_list[inspect_lib_list[idx]])
                        predict_list.append(prediction)
                    else:
                        prediction = retina.predict_retina_net_ground(model=self.model_list[inspect_lib_list[idx]], img=img,
                                                                      th=self.threshold_list[inspect_lib_list[idx]])
                        predict_list.append(prediction)
                else:
                    predict_list.append('SKIP')

        save_path = detail_path + 'json_list'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        success = json_writer(pcb_idx, inspect_part_list, predict_list, save_path + '/', top_btm)

    def test1(self, msg):  # keras 외 내용 test
        split_msg = msg.split('|')
        pcb_idx = split_msg[1]
        path_msg = split_msg[0].split('/')
        access_ip = path_msg[0]
        del path_msg[0]
        server_path = '/'.join(path_msg)
        download_home = 'C:/Users/Mirtec/Documents/dullim/sample/ftp 관련/201117/'

        detail_path = access_ip
        for idx in range(1, len(path_msg)):
            detail_path = os.path.join(detail_path, path_msg[idx])
        if not (os.path.isdir(download_home + detail_path)):
            os.makedirs(download_home + detail_path)
        target_path = download_home + detail_path + '/'

        ftp_server = FTPConnect()
        ftp_server.login_server(access_ip)
        inspect_file_list, inspect_lib_list = ftp_server.download_file(server_path, target_path, pcb_idx)

        if inspect_file_list:
            for inspect_file in inspect_file_list:
                if os.path.isfile(target_path + inspect_file):
                    img = Image.open(target_path + inspect_file)
                    # img.show()
                    # 검사 타이밍

        # 검사 결과 대행
        f = open(target_path + 'INSP_RESULT.json', 'w')
        for inspect_file in inspect_file_list:
            f.write(inspect_file)
        f.close()

        ftp_server.upload_file(target_path, server_path, 'INSP_RESULT.json')


if __name__ == '__main__':
    sample = 'ftp://10.196.163.58/IntellisysImage/20191001/EI99N0457I01/HFS6T4GDUFEH-A430A_H_BTM/'
    sample2 = 'ftp://10.196.163.58/IntellisysImage/20191101/EI99N0457IOA0/HFS800GDUFEH-A430A_H_BTM/'
    sample3 = 'ftp://10.196.163.58/IntellisysImage_mirtec/20200712/EJ07N4444101/SC401_9_HFS256G_D_TOP/'
    msg = '10.196.163.58/IntellisysImage/20191001/EI99N0457I01/HFS6T4GDUFEH-A430A_H_BTM|00009'
    msg2 = '10.196.163.58/IntellisysImage/20191101/EI99N0457IOA0/HFS800GDUFEH-A430A_H_BTM|00031'
    msg3 = '10.196.163.58/IntellisysImage_mirtec/20200712/EJ07N4444101/SC401_9_HFS256G_D_TOP|00006'

    # http test
    msg4 = r'192.168.1.170\1TB\DOCUMENT\#DULim\hynix_ai\EJ07N4444101\SC401_9_HFS256G_D_TOP|6'
    msg5 = r'192.168.1.170\1TB\DOCUMENT\#DULim\hynix_ai\LOT_NUM_NONE\HFS256G39TND-N210A_X_BTM|1'
    msg6 = r'192.168.1.119\\IntellisysImage\\20201124\\LOT_NUM_NONE\\HFS256G39TND-N210A_X_BTM|1'

    msg7 = r'10.196.163.58\IntellisysImage_mirtec\20200712\EJ07N4444101\SC401_9_HFS256G_MIR_TOP|6'
    msg8 = r'10.196.164.206\IntellisysImage\20201203\LOT_NUM_NONE\HFS960GD0TEG-6410A_X_TOP|1'

    # edge 추가 test

    updown_test = AIVerify()
    # updown_test.test1(msg3)
    updown_test.load_model()
    updown_test.inspect_ftp(msg7)