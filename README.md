# Symbol Detection Network
    本项目是一个基于深度学习的浅层神经网络，用于检测输入身份证照片中的党徽。网络包含两部分输出，第一部分是模型预测的党徽存在的概率，第二部分为党徽位置的坐标。党徽不存在的情况包含两种：1）输入图片为身份证正面; 2）输入的图片为背面，但是党徽处被部分或全部遮挡。

### 项目结构
- data_processing.ipynb:
    - 原始的真实数据储存在original_data文件夹中，如果未来有扫描更多真实身份证图片，可以放进这个文件夹中，并调整代码来生成更多的训练数据。请勿删除这个文件夹中的图片，因为这是唯一的真实数据来源。
    - 这个文件负责训练数据的生成，因为现存的身份证数据太少，所以需要通过数据增强的方式来生成训练数据，包括图片的旋转，色彩调整和模糊处理。
    - 原始数据存储在labelimg.json中，现在只有两张身份证背面的图片以及对应的党徽坐标。正面和被遮挡背面不需要党徽坐标，所以并不需要储存数据。但是本文件还是会根据总共四张图片生成训练数据。如果未来有更多的数据，请使用labelimg或其他工具标注数据，并且生成新的labelimg.json文件，然后根据新的labelimg.json调整data_processing.ipynb中的代码生成更为丰富的训练数据。
    - 运行完第8个cell后，会生成一个resized_data文件夹，里面包含了调整大小后的训练数据和标签数据，可以用于训练模型。默认每一张原始图片模拟生成100张图片，总共生成400张图片。数据标签储存在resized_data/annotations.json中。
    - visualize()函数可以用来可视化生成的数据，可以通过调整参数来查看不同的数据。
    - 运行完第10个cell之后，会生成一个resized_validation_data文件夹，里面包含了调整大小后的验证数据和标签数据，可以用于验证模型。默认每一张原始图片模拟生成20张图片，总共生成80张图片。数据标签储存在resized_validation_data/annotations.json中。
    - 运行完第12个cell之后，会生成一个resized_test_data或者test_data文件夹，里面包含了调整大小后或者原始大小的测试数据和标签数据，可以用于测试模型。
    - 也可以直接run all cells，生成所有的数据。

- model.py & data.py:
    - model.py定义了模型的结构，包括两个输出，一个是党徽存在的概率，另一个是党徽的坐标。模型的结构是一个简单的卷积神经网络，包含两个输出层，分别是一个sigmoid激活的输出层和一个线性激活的输出层。
    - data.py定义了数据的读取和处理，包括数据的读取，预处理和增强。

- main.py:
    - 运行这个文件可以训练模型，训练完成后会在当前目录下生成两个model文件，分别是resized_model_{epoch}.pth和onnx_model.onnx，分别是pytorch模型和onn模型。可以通过调整参数来调整模型的训练。

- test.py:
    - 运行这个文件可以测试模型。


### 可提升的地方
- 数据增强的方式可以更加丰富，可以尝试更多的方式来生成数据。
- 模型的结构可以更加复杂，可以尝试更多的结构来提升模型的性能。
- 可以尝试更多的训练方式，调证训练参数来提升模型的性能。