with open(r'C:\Users\CrazyCat\Desktop\LA-Venues.txt', 'r', encoding='UTF-8') as f:
    all_item_list = f.readlines()

item_dict = {}
for item in all_item_list:
    item_list = item.split('\t')
    item_dict[item_list[0]] = (eval(item_list[1]), item_list[2], item_list[3])

with open('data/item_dict.txt', 'w', encoding='UTF-8') as f:
    f.write(str(item_dict))

with open('data/item_dict.txt', 'r', encoding='UTF-8') as f:
    test_dict = eval(f.read())
print(test_dict)
print(test_dict.__len__())
