list = [6666,7777,888888,[66,677]]

def write_file(filename, data):
    file = open(filename, 'w')
    file.write(str(data))
    file.close()

write_file('file_name.txt', list)


file = open('file_name.txt','r')

listR = eval(file.read())

file.close()

print(listR)



