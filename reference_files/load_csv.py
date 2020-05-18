filename = '../data/sign-language-mnist/sign_mnist_train.csv'

labels = []
images = []
with open(filename) as file:
	next(file)
	csv_reader = csv.reader(file)
	for row in csv_reader:
	    row = np.array(row).astype(np.float)
	    labels.append(row[0])
	    images.append(np.array(row[1:].reshape(28, 28)))


np.array(images), np.array(labels)
