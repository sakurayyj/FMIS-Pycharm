import this

import pymysql

def MySQL2(list):
    this.list = list

    # 连接数据库,地址:localhost,账号:root,密码:root,数据库名:school
    db = pymysql.connect(host="localhost", user="root", passwd="123456", db="data1", charset='utf8')
    # 创建一个游标对象
    cursor = db.cursor()

    SNO = '2016081111'
    # 删除数据SQL语句
    DELETE_SQL = '''
    delete  from data1;
    ''' % (SNO)
    # 查询数据SQL语句
    SELECT_SQL = '''
    select * from data1;
    ''' % (SNO)
    try:
        # 数据删除前
        cursor.execute(SELECT_SQL)
        print('删除前:', cursor.fetchall())
        # 删除数据
        cursor.execute(DELETE_SQL)
        # 提交到数据库
        db.commit()
        # 数据删除后
        cursor.execute(SELECT_SQL)
        print('删除后:', cursor.fetchall())

    except:
        # 删除失败,回滚
        db.rollback()
        print('删除失败')


    try:      # 插入数据
        for i in len(list):
            # SQL语句
            SQL = ''' insert into data1('
                      ''' +list[0]+'''\',\''''+list[1]+'''\',\''''+list[2]+'''\',\''''+list[3]+'''\',\''''+list[4]+'''\',\''''+list[5]+'''\',\''''+list[6]+'''\',\''''+list[7]+'''\',\''''+list[8]+'''\',\''''+list[9]+'''\',\''''+list[10]+'''\',\''''+list[11]+'''\',\''''+list[12]+'''\',\''''+list[13]+'''\',\''''+list[14]+'''\');'''
            cursor.execute(SQL)
                  # 提交到数据库
            db.commit()
            print('插入成功')
    except:
        # 插入失败
        db.rollback()
        print('插入失败')

    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    db.close()