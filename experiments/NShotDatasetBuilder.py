import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.MatchingNetwork import MatchingNetwork
from torch.autograd import Variable
from datasets import MatchingDataset


class NShotDatasetBuilder:

    def __init__(self, data_train: MatchingDataset, data_val: MatchingDataset, data_test: MatchingDataset,
                 batch_size: int = 8, n_ways: int = 5, k_shots: int = 1, channels: int = 3, fce: bool = True,
                 img_size: int = 256):
        """
        :param batch_size: The experiment batch size
        :param n_ways: An integer indicating the number of classes per support set
        :param k_shots: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :param data: A data provider class
        """
        self.data_train: MatchingDataset = data_train
        self.data_val: MatchingDataset = data_val
        self.data_test: MatchingDataset = data_test

        self.train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=batch_size,
                                                        shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(self.data_val, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.data_test, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.keep_prob = torch.FloatTensor(1)

        # Initialize model
        self.matchingNet = MatchingNetwork(batch_size=batch_size,
                                           keep_prob=self.keep_prob, num_channels=channels,
                                           fce=fce,
                                           num_classes_per_set=n_ways,
                                           num_samples_per_class=k_shots,
                                           nClasses=0, image_size=img_size)
        self.isCudaAvailable = torch.cuda.is_available()
        if self.isCudaAvailable:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(0)
            self.matchingNet.cuda()

        # Learning parameters
        self.optimizer = 'adam'
        self.lr = 1e-03
        self.current_lr = 1e-03
        self.lr_decay = 1e-6
        self.wd = 1e-4
        self.total_train_iter = 0

    def run_training_epoch(self):
        """
        Runs one training epoch
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        total_train_batches = len(self.train_loader)
        optimizer = self.__create_optimizer(self.matchingNet, self.lr)

        pbar = tqdm(enumerate(self.train_loader), total=total_train_batches, desc=f"Training", postfix={"loss": round(0, 4), "acc": f"{round(0, 4)}%"})
        for batch_idx, (x_support_set, y_support_set, x_target, target_y) in pbar:

            x_support_set = Variable(x_support_set).float()
            y_support_set = Variable(y_support_set, requires_grad=False).long()
            x_target = Variable(x_target.squeeze()).float()
            y_target = Variable(target_y.squeeze(), requires_grad=False).long()

            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                      self.n_ways).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)
            if self.isCudaAvailable:
                acc, c_loss_value = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                     x_target.cuda(), y_target.cuda())
            else:
                acc, c_loss_value = self.matchingNet(x_support_set, y_support_set_one_hot, x_target, y_target)
            optimizer.zero_grad()
            c_loss_value.backward()
            optimizer.step()
            self.__adjust_learning_rate(optimizer)

            # iter_out = "tr_loss: {}, tr_accuracy: {}%".format(round(c_loss_value.item(), 4), round(acc.item() * 100, 4))
            # pbar.set_description(iter_out)

            pbar.update(1)
            pbar.set_postfix({"loss": round(c_loss_value.item(), 4), "acc": f"{round(acc.item() * 100, 4)}%"})
            total_c_loss += c_loss_value.item()
            total_accuracy += acc.item()

            self.total_train_iter += 1
            if self.total_train_iter % 2000 == 0:
                self.lr /= 2
                print("change learning rate", self.lr)

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_validation_epoch(self):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.
        total_val_batches = len(self.val_loader)
        pbar = tqdm(enumerate(self.val_loader), total=total_val_batches, desc=f"Validation", postfix={"loss": round(0, 4), "acc": f"{round(0, 4)}%"})
        for batch_idx, (x_support_set, y_support_set, x_target, target_y) in pbar:

            x_support_set = Variable(x_support_set).float()
            y_support_set = Variable(y_support_set, requires_grad=False).long()
            x_target = Variable(x_target.squeeze()).float()
            y_target = Variable(target_y.squeeze(), requires_grad=False).long()

            # y_support_set: Add extra dimension for the one_hot
            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                      self.n_ways).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)

            if self.isCudaAvailable:
                acc, c_loss_value = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                     x_target.cuda(), y_target.cuda())
            else:
                acc, c_loss_value = self.matchingNet(x_support_set, y_support_set_one_hot, x_target, y_target)

            # iter_out = "val_loss: {}, val_accuracy: {}%".format(round(c_loss_value.item(), 4), round(acc.item() * 100, 4))
            # pbar.set_description(iter_out)
            pbar.update(1)
            pbar.set_postfix({"loss": round(c_loss_value.item(), 4), "acc": f"{round(acc.item() * 100, 4)}%"})

            total_val_c_loss += c_loss_value.item()
            total_val_accuracy += acc.item()

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        total_test_batches = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_test_batches, desc=f"Testing", postfix={"loss": round(0, 4), "acc": f"{round(0, 4)}%"})
        for batch_idx, (x_support_set, y_support_set, x_target, target_y) in pbar:

            x_support_set = Variable(x_support_set).float()
            y_support_set = Variable(y_support_set, requires_grad=False).long()
            x_target = Variable(x_target.squeeze()).float()
            y_target = Variable(target_y.squeeze(), requires_grad=False).long()

            # y_support_set: Add extra dimension for the one_hot
            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                      self.n_ways).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)

            if self.isCudaAvailable:
                acc, c_loss_value = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                     x_target.cuda(), y_target.cuda())
            else:
                acc, c_loss_value = self.matchingNet(x_support_set, y_support_set_one_hot, x_target, y_target)

            # iter_out = "test_loss: {}, test_accuracy: {}%".format(round(c_loss_value.item(), 4), round(acc.item() * 100, 4))
            # pbar.set_description(iter_out)
            pbar.update(1)
            pbar.set_postfix({"loss": round(c_loss_value.item(), 4), "acc": f"{round(acc.item() * 100, 4)}%"})

            total_test_c_loss += c_loss_value.item()
            total_test_accuracy += acc.item()

        total_test_c_loss = total_test_c_loss / total_test_batches
        total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy

    def __adjust_learning_rate(self, optimizer):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def __create_optimizer(self, model, new_lr):
        # setup optimizer
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=new_lr,
                                        momentum=0.9, dampening=0.9,
                                        weight_decay=self.wd)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr,
                                         weight_decay=self.wd)
        else:
            raise Exception('Not supported optimizer: {0}'.format(self.optimizer))
        return optimizer
