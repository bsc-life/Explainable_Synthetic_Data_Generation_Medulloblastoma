import argparse
import general

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

    # Plots

    parser.add_argument("--plots", help="loss plots", choices=["0", "1"], default="0")

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

    train_data, train_loader, test_data, test_loader, _ = general.data2tensor(args.train_path, args.test_path,
                                                                              ch_batch_size)
    print("Data Preprocessing successfully")

    chosen_model = general.VAE(input_dim=len(train_data[0]), mid_dim=mid_dim, features=features)
    # ch_optimizer, ch_criterion, ch_device = setup(chosen_model, lr, args.loss)

    tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string, dev = general.cyclical_training(chosen_model, train_loader,
                                                                                     test_loader,
                                                                                     epochs=ch_epochs,
                                                                                     cycles=ch_cycles,
                                                                                     initial_width=ch_width,
                                                                                     reduction=ch_reduction,
                                                                                     beta=ch_beta,
                                                                                     option=args.loss,
                                                                                     learning_rate=lr)

    if args.plots == "1":
        general.loss_plots(tr_l, tt_l, tr_kl, tt_kl, tr_r, tt_r, id_string)

