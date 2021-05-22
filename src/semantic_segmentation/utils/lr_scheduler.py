class polylr(object):
    def __init__(self, optimizer, nb, lr):
        self.nb = nb
        self.lr = lr
        self.optimizer = optimizer
        self.iteration = 0

    def step(self):

        self.iteration += 1
        lr = self.calc_lr()
        self.update_lr(self.optimizer, lr)

    def calc_lr(self):
        lr = self.lr * ((1 - float(self.iteration) / self.nb) ** (0.9))
        return lr

    def update_lr(self, optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def state_dict(self):
        return {"iteration": self.iteration}

    def load_state_dict(self, state_dict):
        self.iteration = state_dict["iteration"]
