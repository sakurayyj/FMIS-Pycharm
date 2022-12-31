import socket
from pretreatment import *
from apripri import *

def getipaddrs(hostname):  # 只是为了显示IP，仅仅测试一下
    result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
    return [x[4][0] for x in result]


if __name__ == '__main__':
    host = ''  # 为空代表为本地host
    hostname = socket.gethostname()
    hostip = getipaddrs(hostname)
    print('host ip', hostip)  # 应该显示为：127.0.1.1
    port = 999  # Arbitrary non-privileged port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(4)
    while True:
        conn, addr = s.accept()
        print('Connected by', addr)
        data = conn.recv(1024)
        print('data',data)
        print('type',type(data))
        if not data:
            break
        Pretrearment()  # 数据处理
        # 初始化数据
        dataSet = loadDataSet()
        # 计算出频繁项集合对应的支持度
        L, suppData = apriori(dataSet, minSupport)
        print("频繁项集：")
        for i in L:
            for j in i:
                print(list(j))
        # 得出强关联规则
        print("关联规则：")
        rules = generateRules(L, suppData, minConf)
        for i in range(len(rules)):
            print(rules[i])
        # conn.sendall(str(len(rules)).encode('utf-8'))
        # print(str(len(rules)).encode('utf-8'))
        # print(data.encode('utf-8'))
        # for i in range(len(rules)):
        #     conn.sendall(rules[i].encode('utf-8'))  # 把接收到数据原封不动的发送回去
        # print('Received', repr(data1))
        conn.close()

