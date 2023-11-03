import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import argparse
from tqdm import tqdm

class RemoveAlphaChannel:
    def __call__(self, image):
        return image.convert("RGB")
    

def evaluate_model(model_path, test_data_dir, image_size):
    width, height = map(int, image_size.split('x'))
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    model = torch.jit.load(model_path)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return precision, recall, f1, auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ML Model")
    parser.add_argument("model_path", type=str, help="Path to the saved PyTorch model")
    parser.add_argument("test_data_path", type=str, help="Path to the test data directory")
    parser.add_argument('image_size', type=str, help='Image size in format WIDTHxHEIGHT, e.g., 244x244.')
    args = parser.parse_args()

    precision, recall, f1, auc = evaluate_model(args.model_path, args.test_data_path, args.image_size)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')