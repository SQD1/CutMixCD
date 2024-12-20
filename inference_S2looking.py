import os
import argparse
from utils.func import Logger, format_logs
from utils.metrics import *
import torch
import torch.nn as nn
import os
import time
from torch.utils.data.dataloader import DataLoader
import models
from loaders.datasets import S2lookingDataset_all
from tqdm import tqdm
from utils.ValEpoch import ValEpoch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    parser.add_argument("--work_dirs", type=str, default='../semi_checkpoints/S2looking/CutMixCD')
    parser.add_argument("--log", type=str, default='cutmix_0.2')
    parser.add_argument("--weight_path", type=str, default='../semi_checkpoints/S2looking/CutMixCD/cutmix_0.2/best.pth')
    parser.add_argument("--data_root", type=str, default='/data0/qidi/S2looking')
    parser.add_argument("--test-batch-size", type=int, default=1,
                        help="test dataset batch size.")

    return parser.parse_args()


def test():
    num_classes = 2
    torch_device = torch.device('cuda')
    args = get_arguments()

    # logger
    save_dir = os.path.join(args.work_dirs, args.log)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, "test.log"))
    logger.write(str(args))

    # test dataset
    test_dataset = S2lookingDataset_all(args.data_root,"test")
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.test_batch_size,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True)

    # load model
    eval_net = models.ResNet50_CD(num_classes, pretrained=None).to(torch_device)
    checkpoint = torch.load(args.weight_path)
    eval_net.load_state_dict(checkpoint['net'])
    _ = eval_net.eval()

    CE_loss = nn.CrossEntropyLoss()

    # test
    metric = ChangeMetrics(False)
    test_runner = ValEpoch(num_classes, eval_net, CE_loss, metric)

    # Eval this epoch
    test_log = test_runner.run(test_loader)
    val_metric = test_log['f1']

    logger.write('Test:\t' + format_logs(test_log))
    print("Test:", test_log)

if __name__ == "__main__":
    test()