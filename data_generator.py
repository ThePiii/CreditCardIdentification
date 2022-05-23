

file = open("train.txt")
l = file.readlines()

path_label = []

for line in l:
    temp = line[0:-1].split(" ")
    path_label.append(temp)

