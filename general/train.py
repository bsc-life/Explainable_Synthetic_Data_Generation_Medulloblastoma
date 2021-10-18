# def libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


# Setup Learning Rate, optimizer, criterion... of the model
def setup(model, learning_rate=0.0001, option='mse'):
    if option == 'bce':
        criterion = nn.BCELoss(reduction='sum')
    else:
        criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return optimizer, criterion, device


# Final loss function with beta as hyperparameters
def final_loss(mu, logvar, reconstruction_loss, beta):
    kl_divergence = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
    reconstruction = reconstruction_loss

    return beta * kl_divergence + reconstruction, kl_divergence


# Fitting function
def fit(model, dataloader, beta, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    rec_loss = 0.0
    kl_loss = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar, _ = model(data)
        reconstruction_loss = criterion(reconstruction, data)
        loss, kl = final_loss(mu, logvar, reconstruction_loss, beta)
        rec_loss += reconstruction_loss.item()
        kl_loss += kl.item()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    kl_div_loss = kl_loss/len(dataloader.dataset)
    reconstructed_loss = rec_loss/len(dataloader.dataset)
    return train_loss, kl_div_loss, reconstructed_loss


# Validation over test dataset
def validate(model, dataloader, beta, criterion, device):
    model.eval()  # network in evaluation mode
    running_loss = 0.0
    rec_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad():  # in validation we don't want to update weights
        for data in dataloader:
            data = data.to(device)
            reconstruction, mu, logvar, _ = model(data)
            reconstruction_loss = criterion(reconstruction, data)
            loss, kl = final_loss(mu, logvar, reconstruction_loss, beta)
            rec_loss += reconstruction_loss.item()
            kl_loss += kl.item()
            running_loss += loss.item()

    val_loss = running_loss / len(dataloader.dataset)
    kl_div_loss = kl_loss / len(dataloader.dataset)
    reconstructed_loss = rec_loss / len(dataloader.dataset)
    return val_loss, kl_div_loss, reconstructed_loss


# Defined cyclical training
def cyclical_training(model, loader_train, loader_test, epochs=110, cycles=3, initial_width=80, reduction=20,
                      beta=1.00, option="mse", learning_rate=0.0001):
    train_loss = []
    test_loss = []
    kl_loss_train = []
    kl_loss_test = []
    rec_loss_train = []
    rec_loss_test = []

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%H_%M")

    # Setup
    optimizer, criterion, device = setup(model, learning_rate, option)
    model = model.to(device)

    # Training cycles
    for cycle in range(cycles):
        print("Starting Cycle ", cycle)
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print("Training...")
            width = initial_width - cycle*reduction
            if epoch < width:
                beta_launcher = 0
            else:
                beta_slope = (1/400) * (epoch - width) * (epoch - width)
                beta_launcher = min(beta, beta_slope)

            train_epoch_loss, kl_train_loss, rec_train_loss = fit(model, loader_train, beta_launcher,
                                                                  optimizer, criterion, device)
            test_epoch_loss, kl_test_loss, rec_test_loss = validate(model, loader_test, beta_launcher,
                                                                    criterion, device)
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)
            kl_loss_train.append(kl_train_loss)
            kl_loss_test.append(kl_test_loss)
            rec_loss_train.append(rec_train_loss)
            rec_loss_test.append(rec_test_loss)
            f = open(f"TrainingReports/TrainingReport{dt_string}.txt", "a")
            f.write(f"\n Epoch {epoch + 1} of {epochs} \n")
            f.write("------------\n")
            f.write(f"Train Loss: {train_epoch_loss:.4f}\n")
            f.write(f"Train KL Loss: {kl_train_loss:.4f}\n")
            f.write(f"Train Rec Loss: {rec_train_loss:.4f}\n")
            f.write("------------\n")
            f.write(f"Test Loss: {test_epoch_loss:.4f}\n")
            f.write(f"Test KL Loss: {kl_test_loss:.4f}\n")
            f.write(f"Test Rec Loss: {rec_test_loss:.4f}\n")
            f.write("------------\n")
            f.write(f"Beta value: {beta_launcher: 4f}\n")
            f.write(f"Cycle: {cycle: 4f} \n")
            f.write("------------\n\n")
            f.close()

    # Save the model trained

    # path = f'trained_models/model{dt_string}.pth'
    # torch.save(model.state_dict(), path)

    return train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, dt_string, device


def data2tensor(train_path, test_path, batch_size):

    # Read train and test data
    read_train = pd.read_csv(train_path, sep=';', na_values=".")
    read_train = read_train.T
    read_test = pd.read_csv(test_path, sep=';', na_values=".")
    read_test = read_test.T

    # Read relevant genes
    genes = pd.read_csv('files/input_genes.csv', index_col=False, header=None)
    genes = list(genes[0])

    # Keep relevant genes
    read_train = read_train[genes]
    read_test = read_test[genes]

    genes_name = list(read_test)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(read_train)
    read_train = scaler.transform(read_train)

    scaler.fit(read_test)
    read_test = scaler.transform(read_test)

    # To Tensor and Dataloaders
    read_train = pd.DataFrame(read_train)
    read_test = pd.DataFrame(read_test)

    train_dataset = torch.tensor(read_train.values).float()
    test_dataset = torch.tensor(read_test.values).float()

    loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataset, loader_train, test_dataset, loader_test, genes_name


def loss_plots(train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, dt_string):
    f = plt.figure(1)
    plt.title('Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss)
    plt.plot(np.linspace(1, len(test_loss), len(test_loss)), test_loss)
    f.savefig(f"TrainingReports/Plots/Train_VS_Test_{dt_string}.png")

    g = plt.figure(2)
    plt.title('Rec Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(rec_loss_train), len(rec_loss_train)), rec_loss_train)
    plt.plot(np.linspace(1, len(rec_loss_test), len(rec_loss_test)), rec_loss_test)
    g.savefig(f"TrainingReports/Plots/Rec_Train_VS_Test_{dt_string}.png")

    h = plt.figure(3)
    plt.title('KL Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(kl_loss_train), len(kl_loss_train)), kl_loss_train)
    plt.plot(np.linspace(1, len(kl_loss_test), len(kl_loss_test)), kl_loss_test)
    h.savefig(f"TrainingReports/Plots/KL_Train_VS_Test_{dt_string}.png")
