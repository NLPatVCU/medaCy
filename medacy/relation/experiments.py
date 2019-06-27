from medacy.relation.NN import Simple_NN
from medacy.relation.NN import CNN
from medacy.relation.NN import Embeddings
from medacy.relation.models import Model


label = ['PIP','TeCP', 'TeRP','TrAP', 'TrCP','TrIP', 'TrNAP','TrWP']

#create dataset
data = Model(label)
#create dataset (padded)
# data_pad = Model(label, True, 10000, 100)
embedding_path = "glove/glove.200d.txt"


# #Model 1 : Train on all the train data and test directly on the test data
nn = Simple_NN(data)
#build model
model = nn.build_Model(8, data.train,64, 'relu', 'softmax', 'adam')
model, loss, acc = nn.fit_Model (model, data.train, data.train_label)
#evaluate on the test
nn.evaluate_Model(model, data.x_test, data.y_test)
#
# #Model 2: Train on partial train and validation data and test on the test data
# nn = Simple_NN()
# #build model
# model = nn.build_Model(8, data.x_train,64, 'sigmoid', 'softmax')
# model, loss, val_loss, acc, val_acc, max_epoch = nn.fit_Model (model, data.x_train, data.y_train, 20, 512, validation=(data.x_val, data.y_val))
# #Train the model with all train data until the epoch that acheived max accuracy with the validation data
# model, loss, acc = nn.fit_Model (model, data.train, data.train_label, max_epoch, 512)
# #evaluate on the test
# nn.evaluate_Model(model, data.x_test, data.y_test)
#
#Model 3 : Train on all the train data and test directly on the test data with embeddings
# nn = Simple_NN()
# #build model
# model = nn.build_Embedding_Model(8, 32)
# model, loss, acc = nn.fit_Model (model, data_pad.train, data_pad.train_label)
# #evaluate on the test
# nn.evaluate_Model(model, data_pad.x_test, data_pad.y_test)

# #Model 4: Train on partial train and validation data and test on the test data with embeddings
# nn = Simple_NN()
# #build model
# model = nn.build_Embedding_Model(8, 64, 'sigmoid', 'softmax', 'adam')
# model, loss, val_loss, acc, val_acc, max_epoch = nn.fit_Model (model, data_pad.x_train, data_pad.y_train, 20, 512, validation=(data_pad.x_val, data_pad.y_val))
# #Train the model with all train data until the epoch that acheived max accuracy with the validation data
# model, loss, acc = nn.fit_Model (model, data_pad.train, data_pad.train_label, max_epoch, 512)
# #evaluate on the test
# nn.evaluate_Model(model, data_pad.x_test, data_pad.y_test)

#Model 5: Train CNN on all the train data and test directly on the test data with padded sequences

# cnn = CNN()
# #build model
# model = cnn.build_Model(8)
# model, loss, acc = cnn.fit_Model (model, data_pad.train, data_pad.train_label)
# #evaluate on the test
# cnn.evaluate_Model(model, data_pad.x_test, data_pad.y_test)

#Model 6: Train CNN on partial train and validation data and test on the test data with padded sequences
# cnn = CNN()
# #build model
# model = cnn.build_Model(8, 32)
# model, loss, val_loss, acc, val_acc, max_epoch = cnn.fit_Model (model, data_pad.x_train, data_pad.y_train, 20, 512, validation=(data_pad.x_val, data_pad.y_val))
# #Train the model with all train data until the epoch that acheived max accuracy with the validation data
# model, loss, acc = cnn.fit_Model (model, data_pad.train, data_pad.train_label, max_epoch, 512)
# #evaluate on the test
# cnn.evaluate_Model(model, data_pad.x_test, data_pad.y_test)




#Model
# embedding = Embeddings(embedding_path)
# nn = Simple_NN()
# #build model
# model = nn.build_external_Embedding_Model(8)
# model, loss, acc = nn.fit_Model (model, data_pad.train, data_pad.train_label)
# #evaluate on the test
# nn.evaluate_Model(model, data_pad.x_test, data_pad.y_test)