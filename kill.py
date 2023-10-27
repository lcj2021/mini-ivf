import os
import configparser

conf = configparser.ConfigParser()

version = input("Input Config Version Name: ")

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
    failed = os.system("ssh "+ host + " \"lsof -ti :" + str(port) + " | xargs -I {} kill {}\"")
    if failed == 0:
        print("Killed processes on " + host + ":" + port)
    else:
        print("Failed to kill processes on " + host + ":" + port)
