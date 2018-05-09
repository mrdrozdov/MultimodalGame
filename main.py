import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim

from misc import read_data
from misc import embed
from misc import cbow

from agents import AgentConfig, Sender, Receiver
from baseline import Baseline
from exchange import Exchange, ExchangeModel
from trainer import Trainer

import gflags


FLAGS = gflags.FLAGS


def run():
    # Get Description Vectors

    ## Training
    descr_train, word_dict_train, dict_size_train, label_id_to_idx_train, idx_to_label_train = read_data(
        FLAGS.descr_train)

    def map_labels_train(x):
        return label_id_to_idx_train.get(x)

    word_dict_train = embed(word_dict_train, FLAGS.word_embedding_path)
    descr_train = cbow(descr_train, word_dict_train)
    desc_train = torch.cat([descr_train[i]["cbow"].view(1, -1)
                            for i in descr_train.keys()], 0)
    desc_train_set = torch.cat(
        [descr_train[i]["set"].view(-1, FLAGS.word_embedding_dim) for i in descr_train.keys()], 0)
    desc_train_set_lens = [len(descr_train[i]["desc"])
                           for i in descr_train.keys()]

    ## Development
    descr_dev, word_dict_dev, dict_size_dev, label_id_to_idx_dev, idx_to_label_dev = read_data(
        FLAGS.descr_dev)

    def map_labels_dev(x):
        return label_id_to_idx_dev.get(x)

    word_dict_dev = embed(word_dict_dev, FLAGS.word_embedding_path)
    descr_dev = cbow(descr_dev, word_dict_dev)
    desc_dev = torch.cat([descr_dev[i]["cbow"].view(1, -1)
                          for i in descr_dev.keys()], 0)
    desc_dev_set = torch.cat(
        [descr_dev[i]["set"].view(-1, FLAGS.word_embedding_dim) for i in descr_dev.keys()], 0)
    desc_dev_set_lens = [len(descr_dev[i]["desc"])
                         for i in descr_dev.keys()]

    desc_dev_dict = dict(
        desc=desc_dev,
        desc_set=desc_dev_set,
        desc_set_lens=desc_dev_set_lens)

    # Initialize Models
    exchange_model = ExchangeModel(config)
    sender = Sender(config)
    receiver = Receiver(config)
    baseline_sender = Baseline(config, 'sender')
    baseline_receiver = Baseline(config, 'receiver')
    exchange = Exchange(exchange_model, sender, receiver, baseline_sender, baseline_receiver, desc_train)
    trainer = Trainer(exchange)

    # Initialize Optimizer
    optimizer = optim.RMSprop(exchange.parameters(), lr=FLAGS.learning_rate)

    # Static Variables
    img_feat = "avgpool_512"

    # Run Epochs
    for epoch in range(FLAGS.max_epoch):
        source = "directory"
        path = FLAGS.train_file
        loader_config = DirectoryLoaderConfig.build_with("resnet18")
        loader_config.map_labels = map_labels_train
        loader_config.batch_size = FLAGS.batch_size
        loader_config.shuffle = True
        loader_config.cuda = FLAGS.cuda

        dataloader = DataLoader.build_with(path, source, loader_config).iterator()

        for i_batch, batch in enumerate(dataloader):

            data = batch[img_feat]
            target = batch["target"]
            trainer_loss = trainer.run_step(data, target)
            loss = trainer.calculate_loss(trainer_loss)

            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(exchange.parameters(), max_norm=1.)
            optimizer_rec.step()

            print(loss.item())


def flags():
    # Debug settings
    gflags.DEFINE_string("branch", None, "")
    gflags.DEFINE_string("sha", None, "")
    gflags.DEFINE_boolean("debug", False, "")

    # Performance settings
    gflags.DEFINE_boolean("cuda", False, "")

    # Display settings
    gflags.DEFINE_string("env", "main", "")
    gflags.DEFINE_string("experiment_name", None, "")

    # Data settings
    gflags.DEFINE_string("descr_train", "descriptions.csv", "")
    gflags.DEFINE_string("descr_dev", "descriptions.csv", "")
    gflags.DEFINE_string("train_data", "./utils/imgs/train", "")
    gflags.DEFINE_string("dev_data", "./utils/imgs/dev", "")
    gflags.DEFINE_integer("word_embedding_dim", 100, "")
    gflags.DEFINE_string("word_embedding_path", "~/data/glove/glove.6B.100d.txt", "")

    # Optimization settings
    gflags.DEFINE_enum("optim_type", "RMSprop", ["Adam", "SGD", "RMSprop"], "")
    gflags.DEFINE_integer("batch_size", 32, "Minibatch size for train set.")
    gflags.DEFINE_integer("batch_size_dev", 50, "Minibatch size for dev set.")
    gflags.DEFINE_float("learning_rate", 1e-4, "Used in optimizer.")
    gflags.DEFINE_integer("max_epoch", 500, "")


def default_flags():
    if not FLAGS.branch:
        FLAGS.branch = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not torch.cuda.is_available():
        FLAGS.cuda = False

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "experiment-{}-{}".format(FLAGS.sha[:6], timestamp)

    if FLAGS.debug:
        np.seterr(all='raise')

    # silly expanduser
    FLAGS.word_embedding_path = os.path.expanduser(FLAGS.word_embedding_path)


if __name__ == '__main__':
    flags()
    FLAGS(sys.argv)
    default_flags()
    run()
