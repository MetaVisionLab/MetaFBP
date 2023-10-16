import random
from typing import Any

import torch


class file_writer:
    def __init__(self, filename: str, mode: str = 'w', print2console: bool = True):
        self.filename = filename
        self.print2console = print2console
        self.file = open(filename, mode)

    def __print(self, content: Any):
        if self.print2console:
            print(content)

    def write(self, content: Any, flush: bool = False):
        self.__print(content)
        self.file.write(content)
        if flush:
            self.flush()

    def write_text_line(self, content, flush: bool = False):
        self.__print(content)
        self.file.write(str(content))
        self.file.write('\n')
        if flush:
            self.flush()

    def write_csv_line(self, content, flush: bool = False):
        self.__print(content)
        separator = ','
        content = [str(i) for i in content]
        self.file.write(separator.join(content))
        self.file.write('\n')
        if flush:
            self.flush()

    def flush(self):
        self.file.close()
        self.file = open(self.filename, 'a')

    def close(self):
        self.file.close()

    def get_handle(self):
        return self.file

class logger:
    def __init__(self, filename: str, mode: str = 'w', auto_save_seed: float = 0.5, ext: str = None, **kwargs):
        self.filename = filename + (ext if ext else '')
        self.auto_save_seed = auto_save_seed
        self.writer = file_writer(self.filename, mode, **kwargs)

    def _write(self, write_fn, content: Any):
        write_fn(content=content, flush=random.random() < self.auto_save_seed)

    def write(self, content):
        raise NotImplementedError()

    def close(self):
        self.writer.close()


class CSVLogger(logger):
    def __init__(self, filename: str, mode: str = 'w', **kwargs):
        super().__init__(filename, mode, print2console=False, **kwargs)

    def write(self, content):
        super()._write(self.writer.write_csv_line, content)


class TextLogger(logger):
    def __init__(self, filename: str, mode: str = 'w', **kwargs):
        super().__init__(filename, mode, **kwargs)

    def write(self, content):
        super()._write(self.writer.write_text_line, content)


class ValuesLog:
    def __init__(self):
        self.log = torch.Tensor()

    def write(self, line):
        line = torch.Tensor(line)
        line = line.unsqueeze(0)
        self.log = torch.cat([self.log, line])

    def state_dict(self):
        return self.log

    def load_state_dict(self, dict):
        self.log = dict

    def mean(self):
        return self.log.mean(dim=0)
