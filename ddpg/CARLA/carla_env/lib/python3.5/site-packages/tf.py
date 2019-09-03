from random import randint
""" This is the "tf.py" module, and it provides one function called rand_tf()
which prints a random value of true or false """
def rand_tf():
	valuation = randint(0, 1)
	if valuation == 1:
		return True
	else:
		return False