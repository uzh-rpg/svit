class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update2(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


select_stat = dict()  # number of selected tokens at each layer
rescale_stat = dict()  # max(rescale ratio) / min(rescale ratio) at each layer
selector_record = []
attn_weights_record = []
weighting_loss_record = [None]
for i in range(12):
    select_stat[i] = AverageMeter()
    rescale_stat[i] = AverageMeter()