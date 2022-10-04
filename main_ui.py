import sys
import os
#os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
import cv2
import pyvista
import pyvistaqt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel, QSizePolicy, QApplication, QHBoxLayout,QVBoxLayout,QSlider, QWidget, QPushButton, QCheckBox
from qtpy import uic
from util import cv2qimage
from functools import partial
from deca_module import DECAInterface

class DemoWindow(QMainWindow, uic.loadUiType(os.path.join(os.path.dirname(__file__), "demo.ui"))[0]):
    def __init__(self, parent=None, app=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.label_image = QLabel()
        self.plot = pyvistaqt.BackgroundPlotter()
        self.plot_window = self.plot.window()


        self.horizontalLayout.addWidget(self.label_image)
        self.horizontalLayout.addWidget(self.plot_window)
        self.horizontalLayout.setStretchFactor(self.label_image, 1)
        self.horizontalLayout.setStretchFactor(self.plot_window, 1)

        self.config = {}

        #flame id pca
        self.id_feat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #5,9 => [4, 8]
        self.id_feat_special = [4, 8]
        self.id_sliders = []
        self.id_value=[0] * len(self.id_feat)
        id_widget = QWidget()
        id_layout = QVBoxLayout()

        #add some widget
        self.checkbox_tex = QCheckBox("texture")
        self.checkbox_tex.setChecked(True)
        self.checkbox_tex.toggled.connect(self.texBoxState)
        self.config['disTex'] = self.checkbox_tex.isChecked()
        id_layout.addWidget(self.checkbox_tex)

        for index, id_f in enumerate(self.id_feat):
            widget = QWidget()
            name = "shape_{}".format(index)
            lable_tmp = QLabel()
            lable_tmp.setText(name)
            lable_tmp.adjustSize()
            if id_f in self.id_feat_special:
                lable_tmp.setStyleSheet("color:red")
            id_slider = QSlider(Qt.Horizontal)
            id_label = QLabel()
            id_label.adjustSize()
            id_label.setText('00.00')
            button_reset = QPushButton('reset')
            h_layout = QHBoxLayout()
            h_layout.addWidget(lable_tmp)
            h_layout.addWidget(id_slider)
            h_layout.addWidget(id_label)
            h_layout.addWidget(button_reset)
            widget.setLayout(h_layout)
            self.id_sliders.append((id_slider, id_label))
            id_layout.addWidget(widget)

            id_slider.setMinimum(0) #6/0.1=60 [0, 60] == > [-3.0, 3.0]  (x-0)/60 = (y+3.0)/6
            id_slider.setMaximum(60)
            id_slider.setSingleStep(1)
            id_slider.setValue(30)
            #id_slider.valueChanged.connect(lambda: self.valueChanged(index))
            id_slider.valueChanged.connect(partial(self.valueChanged, index))
            button_reset.clicked.connect(partial(self.onClicked, index))
        id_widget.setLayout(id_layout)
        self.horizontalLayout.addWidget(id_widget)

        #self.plot.add_mesh(pyvista.Sphere())

        self.actionopen.triggered.connect(self.onOpen)
        self.actionsave.triggered.connect(self.onMeshSave)

        self.build_deca()

    def texBoxState(self):
        chk_stat = self.checkbox_tex.isChecked()
        self.config['disTex'] = chk_stat
        print('Debug checkbox ', chk_stat)

    def set_image(self, image):
        image = cv2qimage(image)
        if self.label_image.height()/image.height() < self.label_image.width() / image.width():
            self.label_h = self.label_image.height()
            self.label_w = image.width() * self.label_image.height() / image.height()
        else:
            self.label_w = self.label_image.width()
            self.label_h = image.height() * self.label_image.width() / image.width()

        self.label_image.setPixmap(QPixmap.fromImage(image.scaled(self.label_w, self.label_h, QtCore.Qt.KeepAspectRatio)))

    def onOpen(self):
        self.image_file, _ = QFileDialog.getOpenFileName(self, 'Select Image', './', 'Image Files (*.jpg *.JPG *.png *.PNG *.bmp *.BMP *.jpeg *.JPEG)')

        if os.path.exists(self.image_file):
            image = cv2.imread(self.image_file)
            self.set_image(image)

            self.deca_process()
        #self.plot.add_mesh(pyvista.Cube())

    def onMeshSave(self):
        if hasattr(self, 'image_file'):
            self.deca_interface.saveMesh(self.image_file)
        else:
            print("Don`t Have Mesh")

    def onClicked(self, index):
        id_slider, id_label = self.id_sliders[index]
        if hasattr(self, 'id_array'):
            id_v = self.id_array[self.id_feat[index]]
            #id_label.setText("{:+.2f}".format(id_v))

            slider_v = self.id_to_slider(id_v)
            id_slider.setValue(slider_v)

    def valueChanged(self, n):
        slider, label = self.id_sliders[n]
        id_v = slider.value() / 10.0 - 3.0  # #(x - 0) / 60 = (y + 3.0) / 6
        print(n, slider.value(), id_v)
        self.id_value[n] = id_v

        id_v_cur = float(label.text())
        if abs(id_v - id_v_cur) > 0.05:
            print('Debug, triger id changed')
            id_hand_dict={}
            for index, id in enumerate(self.id_feat):
                if index == n:
                    id_hand_dict[id] = id_v
                else:
                    id_slider, id_label = self.id_sliders[index]
                    id_hand_dict[id] = float(id_label.text())
            self.deca_interface.plot_with_hand(id_hand_dict, self.plot)

        label.setText("{:+.2f}".format(id_v))


    def id_to_slider(self, v):
        return int((v+3.0) * 10 + 0.5)

    def build_deca(self):
        self.deca_interface = DECAInterface(config=self.config)

    def deca_process(self):
        self.deca_interface(self.image_file)
        self.deca_interface.plot_with_pyvista(self.plot)

        #update id pca part value
        self.id_array = self.deca_interface.getShapeParameter().copy().squeeze()
        for index, id in enumerate(self.id_feat):
            id_v =  self.id_array[id]
            slider_v = self.id_to_slider(id_v)
            id_slider, id_label = self.id_sliders[index]
            id_label.setText("{:+.2f}".format(id_v))
            id_slider.setValue(slider_v) #qslider setValue会引起valueChanged回调

            print('DEBUG, ID {} = {}'.format(id, id_v))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = DemoWindow(app=app)
    w.showMaximized()
    sys.exit(app.exec_())



