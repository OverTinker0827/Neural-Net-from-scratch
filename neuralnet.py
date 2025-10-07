import numpy as np
import pickle
class NeuralNet:
  def __init__(self,layer_conf):
    if layer_conf is None:
      layer_conf=[]
      pass
    self.layers=[]
    for i in range(1,len(layer_conf)):
      self.layers.append(
          [np.random.randn(layer_conf[i-1],layer_conf[i])*0.1,
           np.random.rand(1,layer_conf[i])-0.5

          ]
      )

  def forward(self,inputs):
    self.hidden=[inputs.copy()]
    for i in range(len(self.layers)):
      inputs=inputs@self.layers[i][0]+self.layers[i][1]
      if i<len(self.layers)-1:
          inputs=np.tanh(inputs)
      self.hidden.append(inputs.copy())
    return inputs




  def backward(self,grad,learning_rate):
    for i in range(len(self.layers)-1,-1,-1):
        h=self.hidden[i]
        if i!=len(self.layers)-1:
          grad=grad*(1-self.hidden[i+1]**2)

        weight_grad=h.T@grad
        bias_grad=np.mean(grad,axis=0,keepdims=True)
        self.layers[i][0]-=learning_rate*weight_grad
        self.layers[i][1]-=learning_rate*bias_grad
        grad=grad@self.layers[i][0].T

  def train(self,lr,epochs,batch_size,x_train,y_train,valid_x,valid_y):
    for epoch in range(epochs):
      epoch_loss=0
      for i in range(0,len(x_train),batch_size):
        x_batch=x_train[i:i+batch_size]
        y_batch=y_train[i:i+batch_size]
        y_pred=self.forward(x_batch)
        grad=y_pred-y_batch
        self.backward(grad,lr)
        epoch_loss+=np.mean((y_pred-y_batch)**2)
      if valid_x is not None and valid_y is not None:
        valid_predictions=self.forward(valid_x)
        valid_loss=np.mean((valid_predictions-valid_y)**2)
        print(f"Epoch {epoch+1}/{epochs},Training Loss: {epoch_loss}, Validation Loss: {valid_loss}")
      else:
        print(f"Epoch {epoch+1}/{epochs},Training Loss: {epoch_loss}")


  def predict(self,x_test):
    return self.forward(x_test)

  def save(self, model_path="model.pkl"):

      with open(model_path, "wb") as f:
          pickle.dump(self.layers, f)
      print(f"Model saved to {model_path}")

  def load(self, model_path="model.pkl"):

      with open(model_path, "rb") as f:
          self.layers = pickle.load(f)
      print(f"Model loaded from {model_path}")