import os
import configparser

conf = configparser.ConfigParser()

version = input("Input Config Version Name: ")
num_threads = input("Input Number of Threads: ")

os.system("cp Conf/"+ version + "/GlobalNodeService.ini GlobalNodeService.ini")

conf.read("GlobalNodeService.ini")

os.system("rm GlobalNodeService.ini")

num_querynodes = conf.getint("querynode", "num_querynodes")

addrs = []

for i in range(num_querynodes):
    addrs.append([
        conf.get("querynode", "a"+str(i)),
        conf.get("querynode", "p"+str(i))
    ])

# print(addrs)

for addr in addrs:
    host, port = addr[0], addr[1]
    failed = os.system("ssh -tt "+ host + " 'cd /home/anns_balancer/cmake/bin/ && ./QueryNodeService " + host + " " + port + " " + num_threads + "' &")
    if failed == 0:
        print("Started processes on " + host + ":" + port)
    else:
        print("Failed to start processes on " + host + ":" + port)
