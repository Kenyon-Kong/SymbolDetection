from absl import app, flags
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import *
from data import *
import tqdm
import torch.onnx
import matplotlib.pyplot as plt

def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'resized_data'
    val_path = 'resized_validation_data'
    annotation_file = 'resized_data/annotations.json'
    val_annotation_file = 'resized_validation_data/annotations.json'


    batch_size = 32
    num_classes = 4
    num_epochs = 50
    learning_rate = 0.001
    
    image_size = 256

    transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = IDCardDataProcessor(image_dir=data_path, annotations_file=annotation_file, 
                                        image_size=image_size, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = IDCardDataProcessor(image_dir=val_path, annotations_file=val_annotation_file, 
                                      image_size=image_size, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SymbolDetector(num_classes=num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, train_loss, val_loss = training_model(model, data_loader, val_loader, optimizer, num_epochs=num_epochs, device=device)

    # model_path = f'model_{num_epochs}.pth'
    model_path = f'resized_model_{num_epochs}.pth'
    torch.save(model.state_dict(), model_path)

    # plot the training and validation loss
    # plt.plot(train_loss, label='Training loss')
    # plt.plot(val_loss, label='Validation loss')
    # plt.legend()
    # plt.title('Loss metrics')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()


    # model.eval()
    # batch_size = 1
    # dummy_input = torch.randn(batch_size, 3, image_size, image_size)
    # dummy_input = dummy_input.to('cuda')
    # torch.onnx.export(model,
    #                 dummy_input,
    #                 "onnx_model.onnx",
    #                 export_params=True,
    #                 opset_version=10, # the ONNX version to export the model to
    #                 do_constant_folding=True, # whether to execute constant folding for optimization
    #                 input_names=['image'],
    #                 output_names=['presence_pred', 'bbox_pred'], 
    #                 dynamic_axes={'image' : {0 : 'batch_size'},    # variable lenght axes
    #                                 'presence_pred' : {0 : 'batch_size'},
    #                                 'bbox_pred' : {0 : 'batch_size'}})



    # onnx_model_path = f'model_{num_epochs}.onnx'
    # dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    # torch.onnx.export(model, dummy_input, onnx_model_path)


if __name__ == '__main__':
    app.run(main)

