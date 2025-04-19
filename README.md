# -PJ1-CNN
### Start Up

First look into the `dataset_explore.ipynb` and get familiar with the data.

### Codes need your implementation

1. `op.py` 
   Implement the forward and backward function of `class Linear`
   Implement the `MultiCrossEntropyLoss`. Note that the `Softmax` layer could be included in the `MultiCrossEntropyLoss`.
   Try to implement `conv2D`, do not worry about the efficiency.
   You're welcome to implement other complicated layer (e.g.  ResNet Block or Bottleneck)
2. `models.py` You may freely edit or write your own model structure.
3. `mynn/lr_scheduler.py` You may implement different learning rate scheduler in it.
4. `MomentGD` in `optimizer.py`
5. Modifications in `runner.py` if needed when your model structure is slightly different from the given example.


### Train the model.

Open test_train.py, modify parameters and run it.

If you want to train the model on your own dataset, just change the values of variable *train_images_path* and *train_labels_path*

### Test the model.

Open test_model.py, specify the saved model's path and the test dataset's path, then run the script, the script will output the accuracy on the test dataset.



###下面是我写的备注：
train.ipynb MLP的训练文件
train_CNN.ipynb CNN的训练文件
test仍然使用test_model.py

best_model_1:初始模型784-600-10 no weight_decay no schelduer  准确率0.9496
best_model_2:784-512-512-10（最后一行）以及exponentialLR 准确率：0.9424

best_model_CNN1---best_model_CNNwithpool2  这些是一些中间模型
best_model_CNNwithpoolanddropout  dropout 0.8  0.9441
best_model_CNNwithpoolanddropout0_2 dropout 0.2 0.9466
best_modelCNNwithpool_dropout_argu 这是最终模型，使用了cnn pool dropout argu方法 准确率0.9381。一些训练细节可能不足
