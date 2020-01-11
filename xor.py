import torch as t
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

NUM = 8
LEARNING_RATE = 0.02
EPOCHS = 7000

class XorNet(nn.Module):
	def __init__(self, input_size, hidden_size_1, hideden_size_2, output_size):
		super(XorNet, self).__init__()
		self.fc_1 = nn.Linear(input_size, hidden_size_1)
		self.fc_2 = nn.Linear(hidden_size_1, hideden_size_2)
		self.fc_3 = nn.Linear(hideden_size_2, output_size)

		#initialize
		self.fc_1.weight.data.normal_(0,1)
		self.fc_2.weight.data.normal_(0,1)
		self.fc_3.weight.data.normal_(0,1)

	def forward(self, x):
		out = self.fc_1(x)
		out = t.sigmoid(out)
		out = self.fc_2(out)
		out = t.sigmoid(out)
		out = self.fc_3(out)
		return out

def get_train_data():
	X = [[a, b] for a in range(NUM) for b in range(NUM)]
	Y = [x[0] ^ x[1] for x in X]
	train_X = t.FloatTensor(X)
	train_Y = t.LongTensor(Y)	
	return train_X, train_Y


def train():
	train_X, train_Y= get_train_data()
	#train_Y = t.zeros(train_Y.shape[0], NUM).scatter_(1, train_Y, 1).long()
	train_X = Variable(train_X)
	train_Y = Variable(train_Y)
	model = XorNet(2,6,3,NUM)
	optimizer = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.CrossEntropyLoss()
	
	loss_history = []
	for epoch in range(EPOCHS):
		optimizer.zero_grad()
		output_Y = model(train_X)
		loss = criterion(output_Y, train_Y)
		loss_history.append(loss.item())
		loss.backward()
		optimizer.step()
		print("epoch {}| loss: {}".format(epoch, loss))

	'''
	#batch trainning
	for epoch in range(EPOCHS):
		for i in range(8):
			train_batch_x = Variable(train_X[i*8:(i+1)*8])
			train_batch_y = Variable(train_Y[i*8:(i+1)*8])
			optimizer.zero_grad()
			output_y = model(train_batch_x)
			loss = criterion(output_y, train_batch_y)
			loss.backward()
			optimizer.step()
			print("epoch {} step {}| loss: {}".format(epoch, i, loss))
	'''
	#test
	test_X, test_Y = get_train_data()
	test_X = Variable(test_X)
	output_Y = model(train_X)
	_, predict = t.max(output_Y, 1)
	accuracy = sum(predict == test_Y)
	print("test accuracy: %d" % accuracy.item())

	t.save(model.state_dict(), "model.pth")
	plt.title("Loss History")
	plt.xlabel("loss")
	plt.ylabel("epoch")
	plt.plot(loss_history)
	plt.savefig('loss.jpg')

if __name__ == '__main__':
	train()
