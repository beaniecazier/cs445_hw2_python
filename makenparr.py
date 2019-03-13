import numpy
array = numpy.array([[2, 3, 4], [1, 2, 3], [1,1,1]])
scalar = 10
print(numpy.multiply(scalar, array))
split = int(len(array) / 2)
print(split)
quarter_train_data = array[:split, :]
print(quarter_train_data)
# quarter_train_target =
