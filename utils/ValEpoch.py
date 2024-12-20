from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter
from utils.metrics import CMMeter
from utils.func import get_mem, list2device



class ValEpoch:
    def __init__(self, num_classes, net,
                criterion1,
                metric, device="cuda"):

        self.num_classes = num_classes
        self.net = net
        self.criterion = criterion1
        # self.criterion_consistency = criterion_consistency
        self.metric = metric
        self.device = device

        self._to_device()

    def _to_device(self):
        self.net.to(self.device)
        self.criterion.to(self.device)
        self.metric.to(self.device)

    @torch.no_grad()
    def run(self, dataloader):
        print(('\n' + '%10s' * 8) % ("val", 'gpu', 'loss', 'precision', 'recall', 'f1', 'iou', 'OA'))

        # 测试模式
        self.net.eval()

        # loss和指标
        loss_meter = AverageValueMeter()
        cm_meter = CMMeter()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, sample in pbar:
        # for step, (sample, _) in pbar:
            # x = x.to(self.device)
            x = list2device(sample['image'], self.device)
            label = sample['labels'].to(self.device)
            Chg = self.net(x)

            loss = self.criterion(Chg, label)
            loss_labeled_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_labeled_value)
            metrics = self.metric(Chg, label)
            cm_meter.add(metrics)
            precision, recall, f1, iou, oa = cm_meter.get_metrics()

            pbar.set_description(('%10s' * 2 + '%10.4g' * 6) % ("val", get_mem(), loss_labeled_value,
                                                                precision, recall, f1, iou, oa))
        precision, recall, f1, iou, oa = cm_meter.get_metrics()
        logs = {
            'loss': loss_meter.mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'oa': oa
        }
        return logs