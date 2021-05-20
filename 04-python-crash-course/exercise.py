# 1
print(7 ** 4)

# 2
s = "Hi there Sam!"
print(s.split())

# 3

planet = "Earth"
diameter = 12742

print("The diameter of {} is {} kilometers.".format(planet, diameter))

# 4 ** Given this nested list, use indexing to grab the word "hello" **

lst = [1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7]

print(lst[3][1][2][0])

# 5

d = {'k1': [1, 2, 3, {'tricky': ['oh', 'man', 'inception', {'target': [1, 2, 3, 'hello']}]}]}

print(d['k1'][3]['tricky'][3]['target'][3])


# 6

def domain_get(email):
    split = email.split('@')
    return split[1]


print(domain_get('user@domain.com'))


# 7

def find_dog(sentence):
    return 'dog' in sentence


print(find_dog('Is there a dog here?'))


# 8
def count_dog(sentence):
    occ = sentence.split()
    return occ.count('dog')


print(count_dog('This dog runs faster than the other dog dude!'))

# 9

seq = ['soup', 'dog', 'salad', 'cat', 'great']
print(list(filter(lambda word: word[0] == 's', seq)))


# 10

def caught_speeding(speed, is_birthday):
    min_speed = 60
    birthday_allowance = 5
    max_speed = 80
    if speed < min_speed or (is_birthday and (speed < min_speed + birthday_allowance)):
        return 'No ticket'
    elif speed <= max_speed or (is_birthday and (speed <= max_speed + birthday_allowance)):
        return 'Small ticket'
    else:
        return 'Big Ticket'


print(caught_speeding(83, True))
