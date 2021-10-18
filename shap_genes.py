import argparse
import torch.utils.data
from sklearn.model_selection import train_test_split
from general import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # VAE Definition
    parser.add_argument("--hidden_layer", help="hidden layer dimension")
    parser.add_argument("--features", help="bottleneck dimension")

    # Setup options
    parser.add_argument("--loss", help="loss function", choices=["bce", "mse"], default="mse")
    parser.add_argument("--lr", help="learning rate", default=0.0001)

    # Training options
    parser.add_argument("--epochs", help="epochs per cycle")
    parser.add_argument("--cycles", help="number of cycles")
    parser.add_argument("--initial_width", help="initial width of beta=0")
    parser.add_argument("--reduction", help="reduction of width per cycle")
    parser.add_argument("--beta", help="beta")

    # Data
    parser.add_argument("--train_path", help="path to training csv")
    parser.add_argument("--test_path", help="path to test csv")
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--colors_train", help="path to the train subgroups")
    parser.add_argument("--colors_test", help="path to the test subgroups")

    # Plots

    parser.add_argument("--plots", help="loss plots", choices=["0", "1"], default="0")

    # XGboost params

    parser.add_argument("--LV", help="number of LV selected", default=20)

    args = parser.parse_args()

    # input_dim = int(args.input_dim)
    mid_dim = int(args.hidden_layer)
    features = int(args.features)

    lr = float(args.lr)

    ch_epochs = int(args.epochs)
    ch_cycles = int(args.cycles)
    ch_width = int(args.initial_width)
    ch_reduction = int(args.reduction)
    ch_beta = float(args.beta)

    ch_batch_size = int(args.batch_size)

    ch_latent_variables = int(args.LV)

    # Load and preprocess the data
    train_data, train_loader, test_data, test_loader, genes_name = data2tensor(args.train_path, args.test_path,
                                                                               ch_batch_size)
    print("Data Preprocessing successfully")

    colors_train, colors_test = colors_preprocessing(args.colors_train, args.colors_test)

    # Chosen model with dimensions
    chosen_model = VAE(input_dim=len(train_data[0]), mid_dim=mid_dim, features=features)

    # Training
    tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string, dev = cyclical_training(chosen_model, train_loader,
                                                                             test_loader,
                                                                             epochs=ch_epochs,
                                                                             cycles=ch_cycles,
                                                                             initial_width=ch_width,
                                                                             reduction=ch_reduction,
                                                                             beta=ch_beta,
                                                                             option=args.loss,
                                                                             learning_rate=lr)

    if args.plots == "1":
        loss_plots(tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string)

    # XGBoost pipeline

    # The train-test split is random. As we will perform the pipeline several times, I prefer
    # split it everytime
    total_data, total_colors = xgboost_preprocessing(train_data, colors_train, test_data, colors_test)
    xgboost_train, xgboost_test, xgboost_train_colors, xgboost_test_colors = train_test_split(total_data,
                                                                                              total_colors,
                                                                                              test_size=0.3)

    # We create a dataset to work with SHAP that it's all the dataset without outliers
    SHAP_dataset = torch.cat((xgboost_train, xgboost_test))

    # Create Dataloaders
    xgboost_train_dataloader = torch.utils.data.DataLoader(
        xgboost_train,
        batch_size=ch_batch_size,
        shuffle=False,
    )

    xgboost_test_dataloader = torch.utils.data.DataLoader(
        xgboost_test,
        batch_size=ch_batch_size,
        shuffle=False,
    )

    shap_dataloader = torch.utils.data.DataLoader(  # Dataloader for SHAP computations. Batch size higher!
        xgboost_train,
        batch_size=256,
        shuffle=True,
    )

    # SHAP Pipeline

    # We get a entire batch of the shap_dataloader, i.e., 256 genomic profiles
    profiles = next(iter(shap_dataloader))

    index = xgboost_shap(chosen_model, xgboost_train_dataloader, xgboost_train_colors,
                         xgboost_test_dataloader, xgboost_test_colors, ch_latent_variables, dev)

    genes1 = shap_vae(chosen_model, profiles, index, dev)

    print("Genes founded :", len(genes1))
    print("Total number :", len(set(genes1)))