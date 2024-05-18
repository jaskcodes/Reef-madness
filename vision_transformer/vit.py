import os
import datetime
import torch
import matplotlib.pyplot as plt
import mlflow
from transformers import AutoImageProcessor, ViTForImageClassification
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from huggingface_hub import login
from torch.utils.tensorboard import SummaryWriter

# Set Hugging Face token
HUGGINGFACE_HUB_TOKEN = "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"  # Replace with your actual Hugging Face token
os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_HUB_TOKEN

# Log in to Hugging Face Hub
login(token=HUGGINGFACE_HUB_TOKEN)

# Constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3  # For early stopping

# Define the model and image processor
model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout layer
    nn.Linear(model.classifier.in_features, 2)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and prepare data
def load_data(directory_path, batch_size=32, val_split=0.2):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]),
    }

    dataset = datasets.ImageFolder(directory_path, transform=data_transforms['train'])
    
    # Split the dataset into training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply the validation transform to the validation dataset
    val_dataset.dataset.transform = data_transforms['val']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Training and evaluation functions
def train_model(model, train_loader, val_loader, epochs, device, patience=3, log_dir=None, checkpoint_dir=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Apply weight decay
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            inputs = image_processor(images, return_tensors="pt").to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss}, Accuracy: {train_acc}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_custom_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping")
            break

    writer.close()
    return history

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            inputs = image_processor(images, return_tensors="pt").to(device)
            labels = labels.to(device)
            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total

    return val_loss, val_acc

# Plot the training history
def plot_training_history(history):
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'bo', label='Training accuracy')
    plt.plot(epochs, history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    plot_path = "training_history.png"
    plt.savefig(plot_path)
    return plot_path

# Create directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load the saved model
def load_model(checkpoint_path, model_name="google/vit-base-patch16-224", num_labels=2):
    # Define the model
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout layer
        nn.Linear(model.classifier.in_features, num_labels)
    )
    # Load the saved state_dict
    model.load_state_dict(torch.load(checkpoint_path))
    return model

# Main
if __name__ == "__main__":
    # Setup experiment directories
    experiment_name = "experiment_03"
    base_dir = f"/Users/jaskiratkaur/Documents/ACV/Reef-madness/logs/checkpoints/healthy/vit/{experiment_name}"
    log_dir = f"{base_dir}/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = f"{base_dir}/checkpoints"

    # Ensure directories are created
    ensure_dir(base_dir)
    ensure_dir(log_dir)
    ensure_dir(checkpoint_dir)

    # Parameters
    coral_health_directory = '/Users/jaskiratkaur/Documents/ACV/Reef-madness/data/Coral Health'

    # Set the MLflow tracking URI and experiment
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/jaskiratk@uchicago.edu/Reef Madness ViT")

    # MLflow Logging
    mlflow.autolog(disable=True)
    with mlflow.start_run():
        # Load data
        train_loader, val_loader = load_data(coral_health_directory, BATCH_SIZE)

        # Training
        history = train_model(model, train_loader, val_loader, EPOCHS, device, patience=PATIENCE, log_dir=log_dir, checkpoint_dir=checkpoint_dir)

        # Evaluate the model
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        # Extract training metrics from the last epoch
        train_loss = history['loss'][-1]
        train_acc = history['accuracy'][-1]
        val_loss = history['val_loss'][-1]
        val_acc = history['val_accuracy'][-1]

        # Log parameters and results
        model_params = {
            'model_name': model_name,
            'num_labels': 2,
            'dropout_rate': 0.5
        }
        mlflow.log_params(model_params)
        mlflow.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc
        })

        # Plot and save training history
        plot_path = plot_training_history(history)
        mlflow.log_artifact(plot_path, "plots")

    print("Training completed successfully.")

    # Load the best model
    best_model_path = f"{checkpoint_dir}/best_custom_model.pth"
    best_model = load_model(best_model_path)

    print("Best model loaded successfully.")
