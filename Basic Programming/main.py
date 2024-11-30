#####hello word in python#####
print('Hello word')

#####ki bhabe comment kore#####
# type 1 - single line comment
"""
type 2 multiline comment
""" 

#####varible#####
roll_no = 123456       
my_name = "Asadzaman"

print(my_name)

#####type casting#####
roll_no_str = str(roll_no) 
roll_no_int = int(roll_no)
roll_no_float = float(roll_no)

print(roll_no_float)
print(type(roll_no_float))

#####operators#####
apple = 10
banana = 20

total_fruit = apple + banana

print(total_fruit)

#####list#####
fruit_list_single = ["apple", "banana", "cherry"]
fruit_list_multiple = [["apple", "banana", "cherry"],["apple", "banana", "cherry"]]

print(fruit_list_multiple[0][1])

print(fruit_list_single[1:3])
fruit_list_single[1] = "blackcurrant"
fruit_list_single.insert(2, "watermelon")
fruit_list_single.append("orange")
fruit_list_single.remove("banana")


for x in fruit_list_single:
  print(x)