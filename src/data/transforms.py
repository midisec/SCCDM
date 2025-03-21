from torchvision import transforms

def get_train_transforms(image_size=512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])