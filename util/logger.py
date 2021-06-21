__all__ = ['setup_logger']


class Logger(object):
    def __init__(self, log_file, command):
        self.log_file = log_file
        if command:
            #  self._print(command)
            self._write(command)

    def output(self, epoch, enc_losses, dec_losses, training_samples, testing_samples,
               enc_mAP, dec_mAP, running_time, debug=True, log=''):
        log += 'Epoch: {:2} | train enc_loss: {:.5f} dec_loss: {:.5f} | '.format(
            epoch,
            enc_losses['train'] / training_samples,
            dec_losses['train'] / training_samples,
        )
        log += 'test enc_loss: {:.5f} dec_loss: {:.5f} enc_mAP: {:.5f} dec_mAP: {:.5f} | '.format(
            enc_losses['test'] / testing_samples,
            dec_losses['test'] / testing_samples,
            enc_mAP,
            dec_mAP,
        ) if debug else ''
        log += 'running time: {:.2f} sec'.format(
            running_time,
        )

        self._print(log)
        self._write(log)

    def _print(self, log):
        print(log)

    def _write(self, log):
        with open(self.log_file, 'a+') as f:
            f.write(log + '\n')

    def output_print(self, log):
        self._print(log)
        self._write(log)


def setup_logger(log_file, command=''):
    return Logger(log_file, command)