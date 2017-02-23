#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, sys


# 从csv数据准备模型数据

if __name__ == "__main__":
    if len(sys.argv)<2:
        print "usage: python %s <csv file name>" % sys.argv[0]
        sys.exit(2)

    csv_file = sys.argv[1]

    # 来自文件
    f=open(csv_file, 'r')
    csv_data = f.readlines()
    f.close()


