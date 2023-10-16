from torch.utils.tensorboard import SummaryWriter


class TFLogger:
    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalars(self, tag, scalar_values: dict, global_step=None):
        if self._writer is not None:
            self._writer.add_scalars(tag, scalar_values, global_step)

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def write_scalar_dict(self, data_dict: dict, global_step=None):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                self.write_scalars(k, v, global_step)
            else:
                self.write_scalar(k, v, global_step)
