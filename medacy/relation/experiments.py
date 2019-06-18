from medacy.relation.NN import Simple_NN
from medacy.relation.models import Model


#create dataset
data = Model()

#Model 1 : Train on all the train data and test directly on the test data
nn = Simple_NN()
#build model
model = nn.build_Model(8, data.train)
model, loss, acc = nn.fit_Model (model, data.train, data.train_label, 20, 512)
#evaluate on the test
nn.evaluate_Model(model, data.x_test, data.y_test)


#Model 2: Train on partial train and validation data and test on the test data
nn = Simple_NN()
#build model
model = nn.build_Model(8, data.x_train)
model, loss, val_loss, acc, val_acc, max_epoch = nn.fit_Model (model, data.x_train, data.y_train, 20, 512, validation=(data.x_val, data.y_val))

model, loss, acc = nn.fit_Model (model, data.train, data.train_label, max_epoch, 512)
#evaluate on the test
nn.evaluate_Model(model, data.x_test, data.y_test)
