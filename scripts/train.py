from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from plr_exercise.models.cnn import Net

import wandb
import optuna
from plr_exercise import PLR_ROOT_DIR


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    Trains the NN on train_loader training data. Prints and logs training progress. 

    Inputs:
    - args : Configurations
    - model: Torch.nn.module : neural network model to train
    - device: cuda or cpu
    - train_loader: training dataset
    - optimizer: optimziatio for model parameters
    - epoch: current epoch number. 

    Returns:
    None
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            
            complete = 100.0 * batch_idx / len(train_loader) # complete percentage. 


            wandb.log({"Training Loss": loss.item()})
            
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    '''
    Evaluates the NN on provided test data

    Calculates loss and accuracy, and logs it. 

    Inputs:
    - model: torch.nn.Module: neural network model
    - device: cuda or cpu
    - test_loader (dataLoader): test dataset
    - epoch: Current epoch number

    Returns:
    float: average Loss over the dataset
    
    '''
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    accuracy = 100.0 *correct / len(test_loader.dataset)
    # Adds Wandb logging
    #wandb.log({"Test Loss": test_loss, "Test Accuracy": accuracy, "Epoch":epoch})
    wandb.log({"Test Loss": test_loss, "Test Accuracy": accuracy})
 
    return test_loss

def train_optuna(trial, args, model, device):
    '''
    Utiilizes Optuna for hyperparam optimization during training

    Suggests values for learning rate, batch size, gamma and total number of epochs during training
    intermedially outputs the best parameters of the study so far

    Inputs:
    trial: optuna.trial: trial object for suggesting hyperparameters
    args: configurations
    model: torch.nn.module: neural net
    device: torch.device: gpu or cpu

    Outputs:
    float: Final test set loss after training witht the suggested parameters
    
    '''
    optim_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optim_batch_size = trial.suggest_categorical("batch_size", [64,128,256])
    optim_gamma = trial.suggest_float("gamma", 0.5,0.99)
    optim_epoch = trial.suggest_int("epochs",0,7) # unused for now. How to use it? 

    # Code from the previous version of main()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    train_kwargs = {"batch_size": optim_batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=optim_lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=optim_gamma)
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        loss = test(model, device, test_loader, epoch)
        scheduler.step()

    return loss


def main():
    '''
    Main function

    - Parses training settings
    - Initializes wandb for tracking
    - Sets up dataloaders and transforms for MNIST dataset
    - Initializes neural net
    - Performs optuna hyperparameter optimzation
    - Prints best hyperparameters found during optimization
    - Saves the trained model and logs training code as a WandB artifact.

    Batch Size: Batch size for training: default 64
    Test Batch Size: default 1000
    epochs: Number of epochs to train. default 2.
    lr: learning rate
    gamma: learning rate step
    seed: random seed

    Inputs: None
    Returns: None
    
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #Setup wandb
    wandb.init(project="plr-exercise-jose",
               name="02_MNIST_RunOptuna",
               config={
                   "learning_rate": args.lr,
                   "epochs": args.epochs,
                   "batch_size": args.batch_size,
                   "test_batch_size":args.test_batch_size,
                   "gamma":args.gamma,
                   "network":"CNN",
               },
               )

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Net().to(device)

    opt_study = optuna.create_study(direction='minimize')
    opt_study.optimize(lambda trial: train_optuna(trial, args,model,device), n_trials=5)
    print(f'Output Parameters: {opt_study.best_params}')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    code_artifact = wandb.Artifact("training_code", type="code")
    code_artifact.add_file("scripts/train.py")
    wandb.log_artifact(code_artifact)


if __name__ == "__main__":
    main()
