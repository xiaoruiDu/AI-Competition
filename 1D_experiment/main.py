#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:20:03 2019

@author: amie
"""

import matplot as mat_plot
import sys
from PyQt5.QtWidgets import QApplication , QMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 创建一个Qdialog，Ui_Dialog（继承）
    ui = mat_plot.MainDialogImgBW()
    # 主窗口循环
    ui.show()
    # 多线程开启
    ui.Init()
    sys.exit(app.exec_())
    
    