import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import BasicDataset
from net import SVAnet


def load_model(model_path, device):
    model = SVAnet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def normalize_signal(signal):
    min_val = signal.min()
    max_val = signal.max()
    return 2 * (signal - min_val) / (max_val - min_val) - 1


def predict_and_visualize(model, data_loader, device, num_samples_to_show=3):
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            for j in range(min(num_samples_to_show, inputs.size(0))):
                sample_input = inputs[j].cpu().numpy().flatten()
                sample_target = targets[j].cpu().numpy().flatten()
                sample_output = outputs[j].cpu().numpy().flatten()

                x = range(sample_input.shape[0])

                plt.figure(figsize=(15, 6))

                plt.subplot(1, 4, 1)
                plt.plot(x, sample_input)
                plt.title("Input Signal")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")

                plt.subplot(1, 4, 2)
                plt.plot(x, sample_target)
                plt.title("Ground Truth")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")

                plt.subplot(1, 4, 3)
                plt.plot(x, sample_output)
                plt.title("Predicted Signal")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")

                normalized_output = normalize_signal(sample_output)
                plt.subplot(1, 4, 4)
                plt.plot(x, normalized_output)
                plt.title("Normalized Prediction (-1 to 1)")
                plt.xlabel("Time (samples)")
                plt.ylabel("Amplitude")

                plt.tight_layout()
                plt.show()
            break


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/best_denoising_model.pth'
    model = load_model(model_path, device)

    dataset = BasicDataset(is_train=False)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    predict_and_visualize(model, data_loader, device, num_samples_to_show=10)
