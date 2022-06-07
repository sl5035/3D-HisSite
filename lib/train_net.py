from torch.utils.data import DataLoader

from models import load_model
from lib.dataset import *
from lib.solver import Solver

def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            # kill_processes(train_queue, train_processes)
            # kill_processes(val_queue, val_processes)
            raise

    return func_wrapper


@cleanup_handle
def train_net():

    NetClass = load_model(cfg.CONST.NETWORK)
    net = NetClass()
    print('\nNetwork definition: ')
    print(net)

    if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
        raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using' \
                         'single-view reconstruction network')

    train_dataset = DatasetLoader()
    train_collate_fn = DatasetCollateFn()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.CONST.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKER,
        collate_fn=train_collate_fn,
        pin_memory=True
    )

    val_dataset = DatasetLoader(train=False)
    val_collate_fn = DatasetCollateFn(train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.CONST.BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    net.cuda()

    # Generate the solver
    solver = Solver(net)
    if cfg.WEIGHTS is not 'None':
        solver.load(cfg.WEIGHTS)

    # Train the network
    solver.train(train_loader, val_loader)

if __name__ == '__main__':
    train_net()
