import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Kullanılan cihaz: {device}")

model = YOLO('yolo11m.pt')
model.to(device)

data_path = 'C:/Users/mehme/OneDrive/Masaüstü/UÇAN_ÖRDEK/YOLO_11_BASKET/data.yaml'

model.train(
    data=data_path,
    epochs=100,
    batch=16,
    imgsz=640,
    device=device,
    name='basketball_custom_model_gpu',
    pretrained=True
)
