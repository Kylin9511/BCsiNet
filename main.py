import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Tester
from utils import init_device, init_model
from dataset import Cost2100DataLoader


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the test data loader
    test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()

    # Define model
    model = init_model(args)
    model._fc_binarization()
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference
    Tester(model, device, criterion)(test_loader)


if __name__ == "__main__":
    main()
