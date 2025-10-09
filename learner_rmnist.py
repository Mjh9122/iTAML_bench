import os
import csv
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import copy
from resnet import *
import random
from radam import *

class ResNet_features(nn.Module):
    def __init__(self, original_model):
        super(ResNet_features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
class Learner():
    def __init__(self,model,args,trainloader,trainset, testloader, use_cuda, memory):
        self.model=model
        self.best_model=model
        self.args=args
        self.title='incremental-learning' + self.args.checkpoint.split("/")[-1]
        self.trainloader=trainloader 
        self.trainset = trainset
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.testloader=testloader
        self.test_loss=0.0
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0  
        self._data_memory, self._tasks_memory, self.age = memory
        
        meta_parameters = []
        normal_parameters = []
        for n,p in self.model.named_parameters():
            meta_parameters.append(p)
            p.requires_grad = True
            if("fc" in n):
                normal_parameters.append(p)
      
        if(self.args.optimizer=="radam"):
            self.optimizer = RAdam(meta_parameters, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        elif(self.args.optimizer=="adam"):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        elif(self.args.optimizer=="sgd"):
            self.optimizer = optim.SGD(meta_parameters, lr=self.args.lr, momentum=0.9, weight_decay=0.001)
            
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = [200], gamma = .1)
 

    def learn(self):
        logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_log.txt'), title=self.title)
        logger.set_names(['Support Loss', 'Context Loss', 'Support Acc.', 'Context Acc.'])
        
        # Set up bar
        batch_time = AverageMeter()
        train_loss = AverageMeter()
        test_loss = AverageMeter()
        train_top1 = AverageMeter()
        test_top1 = AverageMeter()
        end = time.time() 
        
        bar = Bar('Processing', max=len(self.trainloader))
        context_set = None
        next_context_set = None
         
        for batch_idx, batch in enumerate(self.trainloader):
            # Break on last batch if too small
            if len(batch[0]) != self.args.train_batch:
                break
            sX, sy, st, qX, qy, qt = self.make_batch(batch)
            next_context_set = qX, qy, qt
            batch_train_loss, batch_train_top1 = self.batch_train(self.model, sX, sy, st)
            if context_set is not None:
                qX, qy, qt = context_set
                batch_test_loss, batch_test_top1 = self.batch_test(self.model, qX, qy, qt)
                test_loss.update(batch_test_loss.item(), self.args.context_size)  
                test_top1.update(batch_test_top1.item(), self.args.context_size)
                
                logger.append([batch_train_loss.item(), batch_test_loss.item(), batch_train_top1.item(), batch_test_top1.item()])
       
            context_set = next_context_set
            # Update memory buffer
            self.batch_update(batch)
            
            train_loss.update(batch_train_loss.item(), self.args.train_batch)
            train_top1.update(batch_train_top1.item(), self.args.train_batch)
                     
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) | Total: {total:} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train top1: {train_top1:.4f} | Test top1: {test_top1:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        total=bar.elapsed_td,
                        train_loss=train_loss.avg,
                        test_loss=test_loss.avg,
                        train_top1=train_top1.avg,
                        test_top1=test_top1.avg
                        )
            bar.next()
        bar.finish()
        
        self.save_checkpoint(self.model.state_dict(), True, checkpoint=self.args.savepoint, filename='session_'+str(self.args.sess)+'_model.pth.tar')
        self.model = copy.deepcopy(self.best_model)
        return
        
        
    def make_batch(self, batch):
        sX, sy, st, s_idx = batch
       
        # Make query set for next batch 
        context_indices = np.random.choice(s_idx, self.args.context_size, replace = False)
        qX, qy, qt, q_idx = self.trainset[context_indices]
       
        # Make current support set with buffer 
        buffer_indices = self.get_batch_from_memory(self.args.train_batch//2)     
        if len(buffer_indices) > 0:
            buf_x, buf_y, buf_t, buf_idx = self.trainset[buffer_indices]
            replacement_indices = np.random.choice(range(self.args.train_batch), size = len(buffer_indices), replace = False)
                 
            for buf_i, batch_i in enumerate(replacement_indices):
                sX[batch_i] = buf_x[buf_i]
                sy[batch_i] = buf_y[buf_i]
                st[batch_i] = buf_t[buf_i]
                
        return sX, sy, st, qX, qy, qt 
    
    def batch_train(self, model, inputs, targets, tasks):
        model.train()
        sessions = []
        
        bi = self.args.num_class
        targets_one_hot = torch.FloatTensor(inputs.shape[0], bi)
        targets_one_hot.zero_()
        targets_one_hot.scatter_(1, targets[:,None], 1)

        if self.use_cuda:
            inputs, targets_one_hot, targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
        inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot),torch.autograd.Variable(targets)

        reptile_grads = {}            
        np_targets = targets.detach().cpu().numpy()
        np_tasks = tasks.detach().cpu().numpy()
        num_updates = 0
        
        outputs2, _ = model(inputs)
            
        model_base = copy.deepcopy(model)
        for task_idx in range(1+self.args.sess):
            # idx is all samples from the current task
            idx = np.where(np_tasks == task_idx)[0]

            ii = 0
            if(len(idx)>0):
                sessions.append([task_idx, ii])
                ii += 1
                for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
                    p=copy.deepcopy(q)
                    
                class_inputs = inputs[idx]
                class_targets_one_hot= targets_one_hot[idx]
                class_targets = targets[idx]
                
                self.args.r = 1
                    
                for kr in range(self.args.r):
                    _, class_outputs = model(class_inputs)

                    class_tar_ce=class_targets_one_hot.clone()
                    class_pre_ce=class_outputs.clone()
                    loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce) 
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                for i,p in enumerate(model.parameters()):
                    if(num_updates==0):
                        reptile_grads[i] = [p.data]
                    else:
                        reptile_grads[i].append(p.data)
                num_updates += 1
        
        for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
            alpha = np.exp(-self.args.beta*((1.0*self.args.sess)/self.args.num_task))
#                 alpha = np.exp(-0.05*self.args.sess)
            ll = torch.stack(reptile_grads[i])
#                 if(p.data.size()[0]==10 and p.data.size()[1]==256):
# #                     print(sessions)
#                     for ik in sessions:
# #                         print(ik)
#                         p.data[2*ik[0]:2*(ik[0]+1),:] = ll[ik[1]][2*ik[0]:2*(ik[0]+1),:]*(alpha) + (1-alpha)* q.data[2*ik[0]:2*(ik[0]+1),:]
#                 else:
            p.data = torch.mean(ll,0)*(alpha) + (1-alpha)* q.data  
            
        self.lr_scheduler.step()
        prec1, prec5 = accuracy(output=outputs2, target=targets.cuda().data, topk=(1, 1)) 
                    
        return loss, prec1

    def batch_test(self, model, inputs, targets, tasks):
        # switch to evaluate mode
        model.eval()

        targets_one_hot = torch.FloatTensor(inputs.shape[0], self.args.num_class)
        targets_one_hot.zero_()
        targets_one_hot.scatter_(1, targets[:,None], 1)
        target_set = np.unique(targets)
        
        if self.use_cuda:
            inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
        inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)

        outputs2, outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets_one_hot)    
        prec1, prec5 = accuracy(outputs2.data, targets.cuda().data, topk=(1, 1))

        return loss, prec1 
        
    def eval_set(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for inputs, targets, tasks, _ in loader:
            targets_one_hot = torch.FloatTensor(inputs.shape[0], self.args.num_class)
            targets_one_hot.zero_()
            targets_one_hot.scatter_(1, targets[:,None], 1)
            target_set = np.unique(targets)
            
            if self.use_cuda:
                    inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()

            _, outputs = model(inputs)
            class_pre_ce=outputs.clone()
            class_tar_ce=targets_one_hot.clone()

            loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce)
            
            pred = torch.argmax(outputs, 1, keepdim=False)
            pred = pred.view(1,-1)
            batch_correct = pred.eq(targets.view(1, -1).expand_as(pred)).view(-1) 
            correct += sum(batch_correct)
            total += len(inputs)
        
        

        return correct/total
            
        
        
    def meta_test(self, model, memory, inc_dataset):
        logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_meta_log.txt'), title=self.title)
        logger.set_names(['Context acc. pre', 'Context acc. post', 'Test acc. pre', 'Test acc. post'])
        # switch to evaluate mode
        model.eval()
        
        meta_models = []   
        base_model = copy.deepcopy(model)
        class_correct = np.zeros((self.args.num_task, self.args.num_class))
        class_total = np.zeros((self.args.num_task, self.args.num_class)) 
        meta_task_test_list = {}
        for task_idx in range(self.args.num_task):
            test_indices = inc_dataset.get_test_indices(inc_dataset.test_dataset.tids, [task_idx])
            
            context_set = np.random.choice(test_indices, size = self.args.context_size, replace = False)
            
            test_set = [idx for idx in test_indices if idx not in context_set]
             
            meta_model = copy.deepcopy(base_model)
            context_loader = inc_dataset.get_custom_loader_idx(context_set, mode="test", batch_size=256) 
            meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
            
            test_loader = inc_dataset.get_custom_loader_idx(test_set, mode="test", batch_size=256)
            meta_model.train()

            print("Training meta tasks:\t" , task_idx)
            
            pre_train_context_acc = self.eval_set(meta_model, context_loader)
            pre_train_test_acc = self.eval_set(meta_model, test_loader)
            
                
            #META training 
            bar = Bar('Processing', max=len(context_loader))
            for batch_idx, (inputs, targets, tasks, _) in enumerate(context_loader):
                assert all(torch.equal(tasks[0], t) for t in tasks)
                targets_one_hot = torch.FloatTensor(inputs.shape[0], self.args.num_class)
                targets_one_hot.zero_()
                targets_one_hot.scatter_(1, targets[:,None], 1)
                target_set = np.unique(targets)

                if self.use_cuda:
                    inputs, targets_one_hot,targets = inputs.cuda(), targets_one_hot.cuda(),targets.cuda()
                inputs, targets_one_hot, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets_one_hot) ,torch.autograd.Variable(targets)

                _, outputs = meta_model(inputs)
                class_pre_ce=outputs.clone()
                class_tar_ce=targets_one_hot.clone()

                loss = F.binary_cross_entropy_with_logits(class_pre_ce, class_tar_ce)

                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()
                bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f}'.format(
                                batch=batch_idx + 1,
                                size=len(context_loader),
                                total=bar.elapsed_td,
                                loss=loss)
                bar.next()
            bar.finish()

            post_train_context_acc = self.eval_set(meta_model, context_loader)
            
            #META testing with given knowledge on task
            meta_model.eval()   
            for batch_idx, (inputs, targets, tasks, _) in enumerate(test_loader):
                assert all(torch.equal(tasks[0], t) for t in tasks)
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                _, outputs = meta_model(inputs)

                pred = torch.argmax(outputs, 1, keepdim=False)
                pred = pred.view(1,-1)
                correct = pred.eq(targets.view(1, -1).expand_as(pred)).view(-1) 
                
                for i,p in enumerate(pred.view(-1)):
                    key = int(p.detach().cpu().numpy())
                    class_total[task_idx, key] += 1
                    if(correct[i]==1):
                        class_correct[task_idx, key] += 1 
                        
            post_train_test_acc = np.sum(class_correct[task_idx])/np.sum(class_total[task_idx]) 
            logger.append([pre_train_context_acc, post_train_context_acc, pre_train_test_acc, post_train_test_acc])

# #           META testing - no knowledge on task
#             meta_model.eval()   
#             for batch_idx, (inputs, targets, tasks) in enumerate(self.testloader):
#                 if self.use_cuda:
#                     inputs, targets = inputs.cuda(), targets.cuda()
#                 inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                             
#                 _, outputs = meta_model(inputs)
#                 outputs_base, _ = self.model(inputs)
#                 task_ids = outputs

#                 task_ids = task_ids.detach().cpu()
#                 outputs = outputs.detach().cpu()
#                 outputs = outputs.detach().cpu()
#                 outputs_base = outputs_base.detach().cpu()
                
#                 batch_size = inputs.size()[0]
#                 for i in range(batch_size):
#                     j = batch_idx*self.args.test_batch + i
#                     output_base_max = []
#                     for si in range(self.args.sess+1):
#                         sj = outputs_base[i][si* self.args.class_per_task:(si+1)* self.args.class_per_task]
#                         sq = torch.max(sj)
#                         output_base_max.append(sq)
                    
#                     task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
#                     task_max = outputs[i][ai:bi][task_argmax]
                    
#                     if ( j not in meta_task_test_list.keys()):
#                         meta_task_test_list[j] = [[task_argmax,task_max, output_base_max,targets[i]]]
#                     else:
#                         meta_task_test_list[j].append([task_argmax,task_max, output_base_max,targets[i]])
#             del meta_model

        class_accuracies = class_correct/class_total
        acc_task = 100 * class_correct.sum(axis = 1)/class_total.sum(axis = 1)
        
        if self.args.sess == 0:
            with open(self.args.checkpoint + "post_meta_training_acc.csv", "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['task'] + [f'task_{i}_acc' for i in range(20)])  
                      
        with open(self.args.checkpoint + "post_meta_training_acc.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([self.args.sess] + list(acc_task))
         

        print(f'Task accuracies: {acc_task}')
        print(f'Avg task accuracy: {acc_task.mean()}')
        
        with open(self.args.savepoint + "/meta_task_test_list_"+str(task_idx)+".pickle", 'wb') as handle:
            pickle.dump(meta_task_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return acc_task
        
    def batch_update(self, new_batch):
        _, _, _, new_batch_indices = new_batch
        if self.age < self.args.memory:
            remaining_space = self.args.memory - self.age
            samples_to_add = min(remaining_space, self.args.train_batch)
            self._data_memory = np.concatenate([self._data_memory, new_batch_indices[:samples_to_add]])
            self._tasks_memory = np.concatenate([self._tasks_memory, np.full(samples_to_add, self.args.sess, dtype=np.int32)])
            self.age += samples_to_add
            
            if samples_to_add < self.args.train_batch:
                remaining_indices = new_batch_indices[samples_to_add:]
                for idx in remaining_indices:
                    random_pos = np.random.randint(0, self.age + 1)
                    if random_pos < self.args.memory:
                        self._data_memory[random_pos] = idx
                        self._tasks_memory[random_pos] = self.args.sess
                    self.age += 1
        else:
            for idx in new_batch_indices:
                random_pos = np.random.randint(0, self.age + 1)
                if random_pos < self.args.memory:
                    self._data_memory[random_pos] = idx
                    self._tasks_memory[random_pos] = self.args.sess
                self.age += 1

        return list(self._data_memory.astype(np.int32)), list(self._tasks_memory.astype(np.int32))

    def get_batch_from_memory(self, batch_size):
        if len(self._data_memory) == 0:
            return np.array([], dtype = 'uint32')
        
        num_samples_to_retrieve = min(batch_size, len(self._data_memory))
        sampled_indices = np.random.choice(
            len(self._data_memory), 
            size = num_samples_to_retrieve, 
            replace = False
        )
        
        return self._data_memory[sampled_indices]
    
    def get_memory(self):
        return self._data_memory, self._tasks_memory, self.age

    def save_checkpoint(self, state, is_best, checkpoint, filename):
        if is_best:
            torch.save(state, os.path.join(checkpoint, filename))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']
