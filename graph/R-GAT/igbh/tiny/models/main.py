import argparse


def main():
    parser = argparse.ArgumentParser()

    # Input/output paths
    parser.add_argument('--path', type=str, default='/gnndataset/')
    parser.add_argument('--modelpath', type=str, default='gcn_19.pt')

    # Dataset selection
    parser.add_argument(
        '--dataset_size',
        type=str,
        default='experimental',
        choices=[
            'experimental',
            'small',
            'medium',
            'large',
            'full'])
    parser.add_argument(
        '--type_classes',
        type=int,
        default=19,
        choices=[
            19,
            292,
            2983])

    # Hyperparameters
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--fan_out', type=str, default='5,10')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--decay', type=int, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2048 * 16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument(
        '--model_type',
        type=str,
        default='gcn',
        choices=[
            'gat',
            'sage',
            'gcn'])
    parser.add_argument('--in_memory', type=int, default=0)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--device', type=str, default='1')
    args = parser.parse_args()

    print("Dataset_size: " + args.dataset_size)
    print("Model       : " + args.model)
    print("Num_classes : " + str(args.num_classes))
    print()

    device = f'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

    dataset = IGL260M_DGL(args)
    g = dataset[0]

    best_test_acc, train_acc, test_acc = track_acc(g, args)

    print(
        f"Train accuracy: {np.mean(train_acc):.2f} \u00B1 {np.std(train_acc):.2f} \t Best: {np.max(train_acc) * 100:.4f}%")
    print(
        f"Test accuracy: {np.mean(test_acc):.2f} \u00B1 {np.std(test_acc):.2f} \t Best: {np.max(test_acc) * 100:.4f}%")
    print()
    print(" -------- For debugging --------- ")
    print("Parameters: ", args)
    print(g)
    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)


if __name__ == '__main__':
    main()
