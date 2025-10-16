from src.models.eegnet import EEGNet
from torchinfo import summary

model = EEGNet(
    num_electrodes=32,
)
batch_size = 512
summary(model, input_size=(batch_size, 32, 128))
