# 清洗数据，删除10次以下的项目，再删除只有一个项目或没有项目的用户，再得到项目集
import collections

with open(r'C:\Users\CrazyCat\Desktop\LA-Tips.txt', 'r', encoding='UTF-8') as tipf:
    all_user_list = tipf.readlines()

item_list = []
for user in all_user_list:
    user_list = user.split('\t')
    for i in range(0, len(user_list)):
        if user_list[i] == "null":
            item_list.append(user_list[i - 1])
print(item_list.__len__())

obj = collections.Counter(item_list)
item_li = []
size = 3000
for data in obj.most_common():
    if data[1] >= 10:
        item_li.append(data[0])
        print(data[1])
print(item_li)
print(item_li.__len__())

user_dict = {}
for user in all_user_list:
    user_info = {}
    user_list2 = user.split('\t')
    for i in range(0, len(user_list2)):
        if user_list2[i] == "null":
            item = user_list2[i - 1]
            if item in item_li:
                user_info[user_list2[i + 2]] = item
    if user_info.__len__() > 1:
        user_dict[user_list2[0]] = user_info

print('user_dict.__len__()', user_dict.__len__())
print(user_dict['6756'].__len__())


f = open(r'data\user_dict.txt', 'w')
f.write(str(user_dict))
f.close()

f = open(r'data\user_dict.txt', 'r')
neww = f.read()
dictnew = eval(neww)
print(dictnew.__len__())
print(dictnew['6756'].__len__())
f.close()

item_list = []
item_list.clear()
for user_info in dictnew.values():
    for item in user_info.values():
        item_list.append(item)
obj = collections.Counter(item_list)
item_set = []
item_set.clear()
for data in obj.most_common():
    item_set.append(data[0])
print(item_list.__len__())
print(item_set.__len__())

with open('data\item_set.txt', 'w') as itemsetf:
    itemsetf.write(str(item_set))

with open('data\item_set.txt', 'r') as itemsetf:
    iiiii = itemsetf.read()
print(iiiii)

user_set = []
for user in user_dict.keys():
    user_set.append(user)

with open(r'data\user_set.txt', 'w') as usersetf:
    usersetf.write(str(user_set))

with open('data\item_set.txt', 'r') as usersetf:
    jjjjj = eval(usersetf.read())

print("user_set", jjjjj.__len__())
print(jjjjj)
