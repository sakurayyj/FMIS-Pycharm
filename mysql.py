import pymysql

import pandas as pd

import csv # 导入CSV模块

def MySQL():
    print("打开数据库")
    # 打开数据库连接
    db = pymysql.connect(host="localhost", user="root", passwd="L123456", db="data1", charset='utf8')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    sql = "SELECT * FROM data1"

    try:
        # 创建一个空的list用于读入csv文件
        list_result = []
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        # 1. 创建文件对象（指定文件名，模式，编码方式）
        with open("code_data/data1.csv", "w", encoding="gbk", newline="") as f:
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 3. 构建列表头
            csv_writer.writerow(
                ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "y"])
            # 4. 写入csv文件内容
            for it in results:
                new_it = list(it)
                del (new_it[14])
                list_result = new_it
                csv_writer.writerow(list_result)
                print("写入数据成功1")
            # 5. 关闭文件
            f.close()

        # 1. 创建文件对象（指定文件名，模式，编码方式）
        with open("code_data/data2.csv", "w", encoding="gbk", newline="") as f1:
            # 2. 基于文件对象构建 csv写入对象
            csv_writer1 = csv.writer(f1)
            # 3. 构建列表头
            csv_writer1.writerow(["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "y","year"])
            # 4. 写入csv文件内容
            for it1 in results:
                list_result1 = it1
                csv_writer1.writerow(list_result1)
                print("写入数据成功2")
            # 5. 关闭文件
            f1.close()

    except:
        print("Error: unable to fecth data")

    # 关闭数据库连接
    cursor.close()
    db.close()