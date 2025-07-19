class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.keys: list[str] = []
        self.avg: dict[str, float] = {}
        self.sum: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def reset(self) -> None:
        for key in self.keys:
            self.avg[key] = 0
            self.sum[key] = 0
            self.count[key] = 0

    def update(self, val: float, key: str, n: int = 1) -> None:
        if key in self.keys:
            self.sum[key] += val * n
            self.count[key] += n
            self.avg[key] = self.sum[key] / self.count[key]
        else:
            self.keys.append(key)
            self.sum[key] = val * n
            self.count[key] = n
            self.avg[key] = self.sum[key] / self.count[key]
