from medacy.relation.NN import Simple_NN
from medacy.relation.NN import CNN
from medacy.relation.NN import Segment_CNN
from medacy.relation.NN import Embeddings
from medacy.relation.models import Model

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# embedding_path = "mm3_vectors.nc.200.txt"
# embedding_path = "mimic3_d400.txt"
embedding_path = "glove.6B.200d.txt"

#create dataset
data = Model(False, True)
data_pad = Model(True, True, maxlen = 100)
embedding = Embeddings(embedding_path, data_pad, 200)
nn = Simple_NN(data)

# Model 1 : Train on all the train data and test directly on the test data using one-hot vector inputs

# model = nn.build_Model_OneHot(data.train_onehot, 64, 'relu', 'softmax', 'adam')
# model, loss, acc = nn.fit_Model (model, data.train_onehot, data.train_label)
# nn.evaluate_Model(model, data.x_test_onehot, data.y_test)

# Model 2 : Train on all the train data and test directly on the test data using sentence - vector inputs

# model = nn.build_Model(data.train, 64, 'relu', 'softmax', 'adam')
# model, loss, acc = nn.fit_Model (model, data.train, data.train_label)
# y_pred, y_true = nn.predict(model, data.x_test, data.y_test)
# nn.evaluate_Model(y_pred, y_true )

# Model 3: Train on partial train and validation data and test on the test data using one-hot vector inputs

# model = nn.build_Model_OneHot(data.train_onehot, 64, 'sigmoid', 'softmax')
# #Train the model with all train data until the epoch that acheived max accuracy with the validation data
# model, loss, val_loss, acc, val_acc, max_epoch = nn.fit_Model (model, data.x_train_onehot, data.y_train, 20, 512, validation=(data.x_val_onehot, data.y_val))
# nn.evaluate_Model(model, data.x_test_onehot, data.y_test)

# Model 4: Train on partial train and validation data and test on the test data using sentence - vector inputs

# model = nn.build_Model(data.x_train,64, 'sigmoid', 'softmax')
# model, loss, val_loss, acc, val_acc, max_epoch = nn.fit_Model (model, data.x_train, data.y_train, 20, 512, validation=(data.x_val, data.y_val))
# model, loss, acc = nn.fit_Model (model, data.train, data.train_label, max_epoch, 512)
# nn.evaluate_Model(model, data.x_test, data.y_test)

# Model 5: Train CNN on all the train data and test directly on the test data using sentence - vector inputs

# cnn = CNN(data)
# model = cnn.build_Model()
# model, loss, acc = cnn.fit_Model (model, data.train, data.train_label)
# cnn.evaluate_Model(model, data.x_test, data.y_test)

# Model 6: Train CNN on all the train data and test directly on the test data using one-hot vector inputs

# cnn = CNN(data_pad, embedding)
# model_cnn = cnn.define_Embedding_Model()
# model, loss, acc = cnn.fit_Model (model_cnn, data_pad.train, data_pad.train_label)
# y_pred, y_true = cnn.predict(model, data_pad.x_test, data_pad.y_test)
# cnn.evaluate_Model(y_pred, y_true)

# Model 7: Train NN on all the train data and test directly on the test data using sentence - vector inputs

# nn.cross_validate( data.train, data.train_label)

# Model 8: Train CNN on all the train data and test directly on the test data using sentence - vector inputs
cnn = CNN(data_pad, embedding)
cnn.cross_validate(data_pad.train, data_pad.train_label)

# Model 9: Train seg_CNN on all the train data and test directly on the test data using sentence - vector inputs
# seg_cnn = Segment_CNN(data_pad, embedding)
# seg_cnn.cross_validate(data_pad.preceding, data_pad.middle, data_pad.succeeding, data_pad.concept1, data_pad.concept2, data_pad.train_label)
