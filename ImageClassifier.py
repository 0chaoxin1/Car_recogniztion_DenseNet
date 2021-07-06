import json
import os.path
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tensorboardX import SummaryWriter
from IC_DenseNet161 import densenet161
import torch.nn as nn
# import EarlyStopping

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# TODO: Fix bug with the epoch print in the last batch of the epochs
# TODO: Change steps to number of images
# TODO: Extract dataset name from path

class ImageClassifier:
    def __init__(self):
        # Device to use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Device to bring back to cpu
        self.device_cpu = torch.device("cpu")

        # data_directory expects: /train /valid /test
        self.data_directory = './data'
        # dataset folder name
        self.dataset = 'data'

        # Transforms for the training, validation, and testing sets
        self.transform = {}
        # ImageFolder data training, validation, and testing sets
        self.data = {}
        # Data loaders for the training, validation, and testing sets
        self.batch_size = 4
        self.loader = {}

        # Normalization parameters
        # self.norm_mean = [0.485, 0.456, 0.406]
        # self.norm_std = [0.229, 0.224, 0.225]

        # DL model
        self.model = None
        # DL architecture to load (default value)
        self.arch = 'densenet161'
        # Hidden units of the classifier (default value)
        # self.hidden_units = [512, 256]
        # 暂时不添加全连接层
        self.hidden_units = []
        # Number of classes of the classifier (set from data['train'].class_to_idx)
        self.nclasses = 196

        # Criterion and probability function
        self.criterion = nn.CrossEntropyLoss()
        self.prob_func = nn.Softmax(dim=1)

        # Criterion and probability function
        # self.criterion = nn.NLLLoss()
        # self.prob_func = torch.exp()

        # Optimizer (SGD)
        self.optimizer = None
        # MultiStepLR
        self.scheduler = None

        # Optimizer learning_rate (default value)
        self.learning_rate = 0.001

        # Training settings
        self.trainer = {'epochs_to_train': 1000,  # 设置迭代次数
                        'print_every': 4,
                        'mute': False,
                        'train_losses': [],
                        'train_accuracy': [],
                        # 添加测试准确率,为了写logs
                        'test_loss':[],
                        'test_accuracy': [],

                        # 'validation_losses': [],
                        # 'accuracy_partials': [],
                        'valid_loss_min': np.inf,
                        'epoch_best': -1,
                        'epoch': [],
                        'step': [],
                        'epochs_acum': 0}

        # Training stats
        self.running_train_loss = 0
        self.step_cur = 0
        self.step_last = 0
        self.epochs_start = 0
        self.epochs_last = 0
        self.training_start_time = 0
        self.training_last_time = 0
        self.valid_time = 0

        # 测试
        self.test_accuracy_max = 0.0
        # 早停分数
        self.best_score = 0

        # Dictionaries
        self.class_to_idx = None
        self.idx_to_class = None
        self.class_to_name = None

        # Default checkpoint values
        self.save_directory = 'checkpoints'
        self.get_default_ckeckpoint_name = lambda: ('ckp_' + self.dataset
                                                    + '_' + self.arch
                                                    + '_' + "_".join(str(x) for x in self.hidden_units)
                                                    + '_' + str(self.learning_rate))
        # + '_' + str(self.trainer['epochs_acum']))

    def use_gpu(self, gpu):
        if gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def load_data(self, data_directory):
        try:
            self.data_directory = os.path.expanduser(data_directory)
            self.dataset = self.dataset  # TODO: get dataset name as the folder name of the dataset

            print("Loading dataset {} from {}".format(self.dataset, self.data_directory))

            train_dir = os.path.join(self.data_directory, 'train')
            # valid_dir = os.path.join(self.data_directory, 'valid')
            test_dir = os.path.join(self.data_directory, 'test')

            # Define your transforms for the training, validation, and testing sets
            self.transform['train'] = transforms.Compose([
                # 将输入的PIL图像调整到给定的大小
                # transforms.Resize(255),
                #  # 裁剪给定的PIL图像到随机大小和纵横比
                # transforms.RandomResizedCrop(224),
                # 按角度旋转图像
                transforms.RandomRotation(20),
                #  水平翻转给定的PIL图像，以给定的概率随机翻转
                transforms.RandomHorizontalFlip(),
                # 转换一个“PIL图像”或“numpy.ndarray”到一个“张量
                transforms.ToTensor()])
                # 用均值和标准差标准化一个张量图像
                # transforms.Normalize(self.norm_mean, self.norm_std)])

            # self.transform['valid'] = transforms.Compose([# transforms.Resize(255),
            #                                             # transforms.CenterCrop(224),
            #                                             transforms.ToTensor(),
            #                                             transforms.Normalize(self.norm_mean, self.norm_std)])

            self.transform['test'] = transforms.Compose([  # transforms.Resize(255),
                # transforms.CenterCrop(224),
                transforms.ToTensor()])
                # transforms.Normalize(self.norm_mean, self.norm_std)])

            # Load the datasets with ImageFolder
            # datasets.ImageFolder（）该接口默认你的训练数据是按照一个类别存放在一个文件夹下
            self.data['train'] = datasets.ImageFolder(train_dir, transform=self.transform['train'])
            # self.data['valid'] = datasets.ImageFolder(valid_dir, transform=self.transform['valid'])
            self.data['test'] = datasets.ImageFolder(test_dir, transform=self.transform['test'])

            # Using the image datasets and the trainforms, define the dataloaders
            # 使用图像数据集和trainforms定义dataloaders
            self.loader['train'] = torch.utils.data.DataLoader(self.data['train'], batch_size=self.batch_size,
                                                               shuffle=True)
            # self.loader['valid'] = torch.utils.data.DataLoader(self.data['valid'], batch_size=self.batch_size)
            self.loader['test'] = torch.utils.data.DataLoader(self.data['test'], batch_size=self.batch_size,
                                                              shuffle=True)

            # Save class_to_idx and idx_to_class
            self.class_to_idx = self.data['train'].class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            # set classifier number of classes
            # 设置类的分类器数量
            self.nclasses = len(self.data['train'].class_to_idx)

            return True
        except Exception as e:
            print("[ERR] Loading data:", str(e))
            return False

    def load_class_names(self, filepath):
        filepath = os.path.expanduser(filepath)
        try:
            with open(filepath, 'r') as f:
                self.class_to_name = json.load(f)
            return True
        except Exception as e:
            print("[ERR] Loading class names json:", str(e))
            return False

            # 处理图像：1/255.0 , 标准化 ，改变颜色通道位置

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio (thumbnail or resize)
        # 缩略图
        # image.thumbnail((256, 256))

        # Convert to numpy array 转换为numpy数组
        image_np = np.array(image)

        # Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1:
        # 图像的颜色通道通常编码为整数0-255，但是模型期望是浮点数0-1
        image_np = image_np / 255.0

        # Normalize
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_np = (image_np - mean) / std

        # Move color channel from 3rd to 1st
        image_np = image_np.transpose(2, 1, 0)

        return image_np

    # 在最后全连接层前添加额外的 fc
    def crete_output_classifier(self):
        # def crete_output_classifier(self, in_units):
        units = self.hidden_units.copy()
        # insert(index, object) 在索引前插入对象
        # (start_units)in_units 插入 units 前面
        # units.insert(0,in_units)
        layers_dict = OrderedDict([])
        # for i in range(len(units)-1):
        #     # Linear(Module) 对输入的数据应用线性变换 math:`y = xA^T + b`
        #     layers_dict['fc'+str(i+1)] = nn.Linear(units[i], units[i+1])
        #     # layers_dict['relu'+str(i+1)] = nn.ReLU(inplace=True)
        #     layers_dict['relu'+str(i+1)] = nn.ReLU()
        #     layers_dict['drop'+str(i+1)] = nn.Dropout(0.2)

        # layers_dict['AvgPool_7x7'] = nn.AdaptiveAvgPool2d((1, 2208))
        layers_dict['fc' + str(len(self.hidden_units) + 1)] = nn.Linear(units[-1], self.nclasses)
        # layers_dict['output'] =  nn.LogSoftmax(dim=1)
        return nn.Sequential(layers_dict)

    def create_model(self, arch=None, hidden_units=None):
        self.arch = 'IC_densenet161'
        self.hidden_units = hidden_units if hidden_units is not None else self.hidden_units

        print('Creating model:', self.arch,
              'and nclass:', self.nclasses)
        # print('Creating model:', self.arch, 'with hidden_units:', " ".join(str(x) for x in self.hidden_units), 'and nclass:', self.nclasses)

        self.model = densenet161()
        # self.model = models.densenet201(pretrained=True)
        # Freeze parameters so we don't backprop through them
        # 屏蔽預訓練模型的權重只訓練最後一層的全連接層的權重.
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # # Replace last part of the pre-trained network
        # # 替换训练前网络的最后一部分
        start_units = self.model.classifier.in_features  # 1920
        self.model.classifier = nn.Linear(start_units, self.nclasses)

        # self.model.classifier = self.crete_output_classifier(start_units)
        self.model.param_to_optimize = self.model.parameters()

        self.model = self.model.to(self.device)
        return self.model

    # def create_optimizer(self, lr=None):
    #     self.learning_rate = lr if lr is not None else self.learning_rate
    #
    #     # Only train the classifier parameters, feature parameters are frozen
    #     self.optimizer = optim.SGD(self.model.param_to_optimize, lr=self.learning_rate, momentum=0.9)
    #     # 设置epoch值，在该epoch下 lr * 0.1
    #     # lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10, verbose=True )
    #
    #     self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [40], 0.1)
    #     return self.scheduler

    def print_stats(self):
        e = self.trainer['epochs_acum']
        nstep = len(self.loader['train'])

        step_remaining = (self.epochs_last - e) * nstep - self.step_cur
        time_spend = time.time() - self.training_last_time
        speed = (self.step_cur - self.step_last) / time_spend

        print("Epoch: {}/{}.. ".format(e, self.epochs_last),
              # "Step: {}/{}.. ".format(self.step_cur, nstep),
              "Train Loss: {:.3f}.. ".format(self.trainer['train_losses'][-1]),
              "Train accuracy: {:.3f}..".format(self.trainer['train_accuracy'][-1]))

    def validation(self, save_ckp=False):
        valid_loss, accuracy = 0, 0
        self.valid_time = time.time()
        self.model.eval()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # torch.no_grad() 上下文管理器，禁用梯度计算
        with torch.no_grad():
            for images, labels in self.loader['valid']:
                # Move input and label tensors to the default self.device
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model.forward(images)
                # 计算交叉熵损失
                valid_loss += self.criterion(outputs, labels).item()

                # Calculate accuracy
                _, top_class = torch.max(outputs, 1)
                accuracy += torch.mean((top_class == labels.data).type(torch.FloatTensor)).item()
        self.model.train()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.valid_time = time.time() - self.valid_time

        valid_loss /= len(self.loader['valid'])
        accuracy /= len(self.loader['valid'])

        # Save results
        self.trainer['train_losses'].append(self.running_train_loss / (self.step_cur - self.step_last))
        self.trainer['validation_losses'].append(valid_loss)
        self.trainer['accuracy_partials'].append(accuracy)
        self.trainer['epoch'].append(self.trainer['epochs_acum'])
        # 当前step
        self.trainer['step'].append(self.step_cur)

        # Print results
        if not self.trainer['mute']:
            self.print_stats()

        # Save checkpoint
        if save_ckp:
            if valid_loss < self.trainer['valid_loss_min']:
                self.trainer['valid_loss_min'] = valid_loss
                self.trainer['epoch_best'] = self.trainer['epochs_acum']
                self.save_checkpoint(best=True)
            else:
                self.save_checkpoint(best=False)

    def EarlyStopping(self, test_accuracy, patience):
        score = test_accuracy

        if score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {patience}')
            if self.counter >= patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def test(self, topk = 2, save_ckp=False):
        # 加载模型权重
        # self.model.load_state_dict(torch.load(pthfile))

        print(f"Testing using: {str(self.device)}")

        corrects_acum, accuracy_count = 0, 0
        test_loss = 0
        # images, labels = next(iter(self.loader['test']))
        for images, labels in self.loader['test']:
            # Move input and label tensors to the default self.device
            images, labels = images.to(self.device), labels.to(self.device)

            # Disable dropouts and turn off gradients to speed up this part
            # 禁用dropouts和关闭梯度，以加快这部分
            # 设置模型为评估/测试模式
            self.model.eval()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # torch.no_grad():上下文管理器，禁用梯度计算
            with torch.no_grad():
                # 输出
                outputs = self.model.forward(images)
                # 测试loss
                test_loss += self.criterion(outputs, labels).item()
                # topk()函数取指定维度上的最大值(或最大几个)，第二个参数dim=1，为按行取，dim=0，为按列取
                # 获得预测值
                # ps = torch.exp(outputs)
                _, top_class = outputs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                # corrects_acum += torch.mean(equals.type(torch.FloatTensor)).item()
                corrects = equals.sum()
                # # 记录识别正确的图片数
                corrects_acum += corrects
                # 记录图片总数量
                accuracy_count += images.size(0)
                # accuracy = float(corrects) / images.size(0) * 100
                # print('Accuracy partial: {}/{}[{:.2f}%]'.format(corrects, images.size(0), accuracy))
        self.model.train()
        accuracy_total = float(corrects_acum) / accuracy_count * 100
        test_loss = test_loss/len(self.loader['test'])

        self.trainer['test_accuracy'].append(accuracy_total/100)
        self.trainer['test_loss'].append(test_loss)
        print('Test Loss: {}\n'.format(test_loss))
        print('Accuracy total: {}/{}[{:.2f}%]\n'.format(corrects_acum, accuracy_count, accuracy_total))
        # 设置早停参数
        # self.EarlyStopping(accuracy_total, 30)
        if self.test_accuracy_max < accuracy_total:
             self.test_accuracy_max = accuracy_total
             torch.save(self.model.state_dict(), "./models/" + 'epoch:{}'.format(self.trainer['epochs_acum'])+'-'+'{:.3f}'.format(accuracy_total) + ".pth")

    def train(self, epochs_to_train=None, save_directory=None, print_every=None):
        self.trainer['epochs_to_train'] = epochs_to_train if epochs_to_train is not None else self.trainer[
            'epochs_to_train']
        self.trainer['print_every'] = print_every if print_every is not None else self.trainer['print_every']
        self.save_directory = save_directory if save_directory is not None else self.save_directory
        self.verify_directory()

        optimizer = optim.SGD(self.model.param_to_optimize, lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        # 设置epoch值，在该epoch下 lr * 0.1
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True )
        # scheduler = lr_scheduler.MultiStepLR(optimizer, [30,70], gamma=0.1)

        print("Training {} epoch using {}".format(self.trainer['epochs_to_train'], self.device))

        # Set variables for the training  为训练设置变量
        self.running_train_loss, self.step_cur, self.step_last = 0, 0, 0
        self.train_accuracy = 0
        self.train_correct = 0
        self.training_start_time, self.training_last_time = time.time(), time.time()
        # 起始迭代次数
        self.epochs_start = self.trainer['epochs_acum']
        # 总迭代次数
        self.epochs_last = self.trainer['epochs_acum'] + self.trainer['epochs_to_train']
        writer = SummaryWriter('./logs')
        try:
            # model in training mode, dropout is on模型在训练模式下，dropout是使用的
            self.model.train()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # 迭代次数里循环
            for e in range(self.epochs_start, self.epochs_last):
                for images, labels in self.loader['train']:
                    self.step_cur += 1
                    # 将输入和标签张量移动到默认的self.device
                    # Move input and label tensors to the default self.device
                    images, labels = images.to(self.device), labels.to(self.device)
                    # 清除上一次自动求导的梯度信息
                    optimizer.zero_grad()

                    output = self.model.forward(images)
                    # 计算交叉熵损失
                    loss = self.criterion(output, labels)
                    # 反向传播
                    loss.backward()
                    # print('学习率：Lr = {}'.format(float(scheduler.)))
                    # optimizer.step()更新参数
                    optimizer.step()
                    # loss.item() 获得loss的值
                    self.running_train_loss += loss.item()

                    # Calculate Train accuracy
                    # _,train_class = output.topk(1, dim=1)
                    # self.train_correct +=  equal.sum()
                    _, train_class = torch.max(output, 1)
                    self.train_accuracy += torch.mean((train_class == labels.data).type(torch.FloatTensor)).item()
                    # 不用每个step都打印出来
                    # if self.step_cur % self.trainer['print_every'] == 0:
                    #     # 在每次打印结束时重置变量
                    #     # Reset train per end of print
                    #     # (self.step_cur - self.step_last) 计算每print_every step记录一次
                    #     self.trainer['train_losses'].append(self.running_train_loss / (self.step_cur - self.step_last))
                    #     self.trainer['epoch'].append(self.trainer['epochs_acum'])
                    #     self.trainer['step'].append(self.step_cur)
                    #
                    #     self.train_accuracy /= len(self.loader['train'])
                    #     self.trainer['train_accuracy'].append(self.train_accuracy)
                    #     # 打印print_stats()
                    #     if not self.trainer['mute']:
                    #         self.print_stats()
                    #
                    #     self.running_train_loss = 0
                    #     self.train_accuracy = 0
                    #     self.step_last = self.step_cur
                    #     self.training_last_time = time.time()
                # End of epoch
                # End of epoch
                else:
                    self.trainer['epochs_acum'] += 1
                    # Reset train per end of epoch
                    # (self.step_cur - self.step_last) 计算每print_every step记录一次
                    self.trainer['train_losses'].append(self.running_train_loss/len(self.loader['train']))
                    self.trainer['epoch'].append(self.trainer['epochs_acum'])
                    self.trainer['step'].append(self.step_cur)

                    self.train_accuracy = self.train_accuracy/len(self.loader['train'])
                    self.trainer['train_accuracy'].append(self.train_accuracy)

                    # 打印print_stats()
                    if not self.trainer['mute']:
                        self.print_stats()
                    # 在epoch结束时重置变量
                    self.running_train_loss = 0
                    self.train_accuracy = 0
                    self.step_cur = 0
                    self.step_last = 0
                    self.training_last_time = time.time()

                    # torch.save(self.model.state_dict(), "./models/model.pth")
                    # 一个epoch后运行测试传递、保存和结果
                    self.test(save_ckp=False)
                    scheduler.step(self.trainer['test_accuracy'][-1])

                    # 写日志
                    writer.add_scalar("eoch--trainloss", self.trainer['train_losses'][-1], self.trainer['epoch'][-1])
                    writer.add_scalar("eoch--trainaccuracy", self.trainer['train_accuracy'][-1], self.trainer['epoch'][-1])
                    writer.add_scalar("eoch--testloss", self.trainer['test_loss'][-1],self.trainer['epoch'][-1])
                    writer.add_scalar("eoch--testaccuracy", self.trainer['test_accuracy'][-1], self.trainer['epoch'][-1])

        finally:
            # Print the training time
            time_duration = time.time() - self.training_start_time
            print("Training duration: {}m{}s".format(int(time_duration // 60), int(time_duration % 60)))

    # 使用训练好的模型预测
    def predict(self, image_path, topk=1, show_image=True):
        ''' Predict the class (or classes) of an image using a trained deep learning self.model.
        '''
        image_path = os.path.expanduser(image_path)
        # Load image
        try:
            img_pil = Image.open(image_path)
        except Exception as e:
            print('[ERR] In predict opening image: ' + str(e))
            return None, None

        # Process image
        image_np = self.process_image(img_pil)
        image = torch.from_numpy(image_np).unsqueeze(0)

        print("Predict using: ", self.device)
        # input and label tensors to the default self.device
        image = image.to(self.device, dtype=torch.float)

        # Turn off gradients to speed up this part
        self.model.eval()  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        with torch.no_grad():
            # If model's output is log-softmax, take exponential to get the probabilities
            # If model's output is linear, take softmax to get the probabilities
            ps = self.prob_func(self.model.forward(image))
            top_p, top_class = ps.topk(topk, dim=1)

        # Bring back to CPU
        image, top_p, top_class = image.to(self.device_cpu), top_p.to(self.device_cpu), top_class.to(self.device_cpu)

        image, top_p, top_class = image.squeeze(0), top_p.squeeze(0), top_class.squeeze(0)

        if show_image:
            self.view_classify(image, top_p, top_class)

        return {'top_p': top_p, 'top_class': top_class}

    def print_predictions(self, predictions):
        top_class = predictions['top_class'].numpy()
        top_p = predictions['top_p'].numpy()

        top_class_print = [self.idx_to_class[i] for i in top_class]

        if self.class_to_name is not None:
            top_class_print = [self.class_to_name[i] for i in top_class_print]

        for p, c in zip(top_p, top_class_print):
            print('[{}]: {:.2f}%'.format(c, p * 100))

    # 用于查看图像及其预测类的函数
    def view_classify(self, img, top_p, top_class, correct=None):
        ''' Function for viewing an image and it's predicted classes.
        '''
        topk = len(top_p)

        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
        ax1 = self.imshow(img, ax1)
        ax1.axis('off')
        ax2.barh(np.arange(topk), top_p)
        ax2.set_yticks(np.arange(topk))
        ax2.set_aspect(0.1)
        if self.class_to_name is None:
            ax2.set_yticklabels(["{}[{}]".format(self.idx_to_class.get(i), i) for i in top_class.numpy()], size='small')
        else:
            ax2.set_yticklabels(
                ["{}[{}]".format(self.class_to_name.get(self.idx_to_class.get(i)), i) for i in top_class.numpy()],
                size='small')
        if correct is not None:
            if self.class_to_name is None:
                ax2.set_title('Class Prob. [correct:{}[{}]]'.format(self.idx_to_class.get(correct), correct))
            else:
                ax2.set_title(
                    'Class Prob. [correct:{}[{}]]'.format(self.class_to_name.get(self.idx_to_class.get(correct)),
                                                          correct))
        else:
            ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()

    def imshow(self, image, ax=None, title=None, normalize=True):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        image = image.numpy().transpose((1, 2, 0))

        if normalize:
            mean = np.array(self.norm_mean)
            std = np.array(self.norm_std)
            image = std * image + mean
            image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        return ax

    def verify_directory(self):
        if not os.path.isdir(self.save_directory):
            os.mkdir(self.save_directory)

    def save_checkpoint(self, save_directory=None, best=False):
        filepath_base = os.path.join(self.save_directory, self.get_default_ckeckpoint_name())
        filepath = filepath_base + "_last.pth"

        checkpoint = {
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'nclasses': self.nclasses,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,

            'trainer': self.trainer,
            'test_accuracy_max': self.test_accuracy_max
        }
        torch.save(checkpoint, filepath)
        print("Checkpoint saved:", filepath)

        if best:
            # 如果best为True：checkpoint保存为 ...._best.pth
            filepath = filepath_base + "_best.pth"
            torch.save(checkpoint, filepath)
            print("Checkpoint saved:", filepath)

    def load_checkpoint(self, filepath='checkpoint.pth'):
        filepath = os.path.expanduser(filepath)
        # 测试路径是否为常规文件
        if os.path.isfile(filepath):
            print("Loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)

            # Build a model with the checkpoint data
            # 用检查点数据构建一个模型
            self.arch = checkpoint['arch']
            # self.hidden_units = checkpoint['hidden_units']
            self.nclasses = checkpoint['nclasses']
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = checkpoint['idx_to_class']
            # create_model() needs to be here for model.to(device) be called before creating the optimizer...
            self.create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(self.model)

            self.learning_rate = checkpoint['learning_rate']
            print("Optimizer:\n", self.create_optimizer())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.trainer = checkpoint['trainer']

            # Print checkpoint info
            print('Trainer epoch_best/epochs_acum: {}/{}'.format(self.trainer['epoch_best'],
                                                                 self.trainer['epochs_acum']))
            print('Trainer min_loss:', self.trainer['valid_loss_min'])
            print('Trainer accuracy Last/Max: {:.2f}%/{:.2f}%'.format(self.trainer['accuracy_partials'][-1] * 100,
                                                                      np.max(self.trainer['accuracy_partials']) * 100))
            return True
        else:
            print("[ERR] Loading checkpoint path '{}'".format(filepath))
            return False
