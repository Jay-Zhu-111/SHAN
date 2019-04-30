import random
with open('data/user_input_test.txt', 'r', encoding='UTF-8') as f:
    user_input = eval(f.read())
with open('data/L_input_test.txt', 'r', encoding='UTF-8') as f:
    L_input = eval(f.read())
with open('data/S_input_test.txt', 'r', encoding='UTF-8') as f:
    S_input = eval(f.read())
with open('data/pos_item_input_test.txt', 'r', encoding='UTF-8') as f:
    j_input = eval(f.read())
with open('data/item_set.txt', 'r', encoding='UTF-8') as item_set_f:
    item_set = eval(item_set_f.read())


def write_file(filename, data):
    file = open(filename, 'w')
    file.write(str(data))
    file.close()


k_input = []
for i in range(user_input.__len__()):
    L = L_input[i]
    S = S_input[i]
    j = j_input[i]
    ob_item = [j]
    for item in L:
        ob_item.append(item)
    for item in S:
        ob_item.append(item)
    k_list = []
    for _ in range(0, 99):
        while True:
            k = random.randint(0, item_set.__len__() - 1)
            if k not in ob_item:
                break
        k_list.append(k)
    k_input.append(k_list)
write_file('data/neg_item_input_test.txt', k_input)

with open('data/neg_item_input_test.txt', 'r', encoding='UTF-8') as f:
    kkk = eval(f.read())
print(kkk.__len__())
print(kkk)
print(kkk[0].__len__())
print(kkk[1].__len__())
