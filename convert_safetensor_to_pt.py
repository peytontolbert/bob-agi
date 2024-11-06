import torch
from ultralytics.nn.tasks import DetectionModel
from safetensors.torch import load_file

tensor_dict = load_file("OmniParser/icon_detect/model.safetensors")

model = DetectionModel('OmniParser/icon_detect/model.yaml')
model.load_state_dict(tensor_dict)
torch.save({'model':model}, 'OmniParser/icon_detect/best.pt')
