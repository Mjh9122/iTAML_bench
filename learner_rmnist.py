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
        self.best_acc = 0 
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
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])
            
        for epoch in range(0, self.args.epochs):
            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))

            self.train(self.model, epoch)
            self.test(self.model)
        
            # append logger file
            logger.append([self.state['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc, self.best_acc])

            # save model
            is_best = self.test_acc > self.best_acc
            if(is_best and epoch>self.args.epochs-10):
                self.best_model = copy.deepcopy(self.model)

            self.best_acc = max(self.test_acc, self.best_acc)
            if(epoch==self.args.epochs-1):
                self.save_checkpoint(self.best_model.state_dict(), True, checkpoint=self.args.savepoint, filename='session_'+str(self.args.sess)+'_model_best.pth.tar')
        self.model = copy.deepcopy(self.best_model)
        
        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.best_acc)
    
    def train(self, model, epoch):
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        
        #bi = self.args.class_per_task*(1+self.args.sess)
        # All classes always present
        bi = self.args.num_class
        bar = Bar('Processing', max=len(self.trainloader))
        
        for batch_idx, (inputs, targets, tasks, orig_idx) in enumerate(self.trainloader):
            if len(inputs) != self.args.train_batch:
                break
            # measure data loading time
            data_time.update(time.time() - end)
            sessions = []
            
            buffer_indices = self.get_batch_from_memory(self.args.train_batch//2)
            
            if len(buffer_indices) > 0:
                buf_in, buf_tar, buf_tas, buf_idx = self.trainset[buffer_indices]
                replacement_indices = np.random.choice(range(self.args.train_batch), size = len(buffer_indices), replace = False)
                 
                for buf_i, batch_i in enumerate(replacement_indices):
                    inputs[batch_i] = buf_in[buf_i]
                    targets[batch_i] = buf_tar[buf_i]
                    tasks[batch_i] = buf_tas[buf_i]
                    
                
            #print(inputs.shape, targets.shape, tasks.shape)
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
                
            self.batch_update(orig_idx)
            self.lr_scheduler.step()
            
                    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output=outputs2, target=targets.cuda().data, topk=(1, 1))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()

        self.train_loss,self.train_acc=losses.avg, top1.avg

    def test(self, model):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        task_class_counts = np.zeros((self.args.num_task, self.args.num_class))
        task_class_correct = np.zeros((self.args.num_task, self.args.num_class))
        
        # switch to evaluate mode
        model.eval()
        
        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets, tasks, idx) in enumerate(self.testloader):
            # measure data loading time
            data_time.update(time.time() - end)
#             print(targets)
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


            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            pred = torch.argmax(outputs2, 1, keepdim=False)
            pred = pred.view(1,-1)
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).view(-1) 
            correct_k = float(torch.sum(correct).detach().cpu().numpy())

            for i, p in enumerate(pred.view(-1)):
                task_val = int(tasks[i].detach().cpu().numpy())
                class_val = int(p.detach().cpu().numpy())

                task_class_counts[task_val, class_val] += 1
                if correct[i]:
                    task_class_correct[task_val, class_val] += 1
                        
                        
            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top1_task: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg
            
        acc_task = {}
        for i in range(self.args.num_task):
            if task_class_counts[i].sum() == 0:
                acc_task[i] = np.nan
            else:
                acc_task[i] = 100 * task_class_correct[i].sum()/task_class_counts[i].sum()
        
        if self.args.sess == 0:
            with open(self.args.checkpoint + "pre_meta_training_acc.csv", "w", newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['task'] + [f'task_{i}_acc' for i in range(20)])  
                      
        with open(self.args.checkpoint + "pre_meta_training_acc.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([self.args.sess] + [acc_task[i] for i in acc_task.keys()])
            
        print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]) )    
        #print(class_acc)s
        
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
            pre_train_test_acc = self.eval_set(meta_model, context_loader)
                
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

            print(f"Meta-Training Context Accuracy Delta {self.eval_set(meta_model, context_loader) - pre_train_context_acc :.4f}")
            
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
                        
            print(f'Meta-Training Testing Accuracy Delta {np.sum(class_correct[task_idx])/np.sum(class_total[task_idx]) - pre_train_test_acc: .4f}')

            
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
        
    def batch_update(self, new_batch_indices):
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
