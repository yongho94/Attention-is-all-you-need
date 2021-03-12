from torch.utils.tensorboard import SummaryWriter
import os

tb_dir = os.path.join('tmp')

tb_writer = SummaryWriter(tb_dir)

loss_list = [ 100 - i for i in range(100)]
steps_list = [ i for i in range(100)]
performance_list = [ i / 2 for i in range(100)]

for i in range(100):
    loss = loss_list[i]
    step = steps_list[i]
    perf = performance_list[i]
    print(step, loss)
    print(step, perf)
    if i % 10 == 0:
        tb_writer.add_scalar('train/loss', loss, step)
        tb_writer.add_scalar('train/performance', perf, step)
    tb_writer.flush()

tb_writer.close()