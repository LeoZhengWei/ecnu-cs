from __future__ import division
import torch


def resolve_device():
    """按当前机器选择训练设备：5090、3050（cuda:0）或 Mac 的 MPS。"""
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            if "5090" in torch.cuda.get_device_name(index):
                return torch.device(f"cuda:{index}")
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            if "3050" in name:
                return torch.device("cuda:0")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer(object):
    def __init__(self,train_loader,model, opt):
        self.args = opt
        self.train_loader = train_loader
        self.model = model
        self.criterion = torch.nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(lr=0.003, params=model.parameters())
        self.epochs = 200
        self.device = resolve_device()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            device_name = torch.cuda.get_device_name(self.device)
        else:
            device_name = self.device.type
        self.model.to(self.device)
        print(f"Using {self.device}: {device_name}")

    def train_model(self):
        for epoch in range(self.epochs):
            losses = []
            for i, (img, mask) in enumerate(self.train_loader):
                img = img.to(self.device)
                mask = mask.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if i % 10 == 0:
                    print(f'setp{i}---train_loss, ', loss.item())

            print(f'epoch{epoch}-----------loss:{sum(losses) / len(losses)}')
            if epoch % 10 == 0:
                torch.save(self.model, f'unet-{epoch}.pth.tar')

        torch.save(self.model, f'Unet-epochs{self.epochs}.pth')


