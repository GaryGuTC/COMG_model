import logging
import os
from abc import abstractmethod
import multiprocessing

import torch
from numpy import inf
from datetime import datetime
import random
from torch import nn


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)        
        self.logger = logging.getLogger(__name__)
        
        # set hadler 
        dt = datetime.strftime(datetime.now(), "%Y-%m-%d_%H")
        experiment_type = "experiment"
        logname = "logfile_saved/train_{}_{}_{}.log".format(args.dataset_name, str(dt), experiment_type)
        file_handler = logging.FileHandler(logname, 'w')
        self.logger.addHandler(file_handler)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)
            
        self._train_epoch = self._train_epoch_IU_Xray if args.dataset_name == "iu_xray" else self._train_epoch_MIMIC_CXR

    @abstractmethod
    def _train_epoch_MIMIC_CXR(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch_IU_Xray(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            self.logger.info('#### The remaining training steps is {} ####'.format(self.early_stop - not_improved_count))
        self._print_best()

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        # save model info
        self.logger.info(args)
        self.cosin_simi_loss = nn.CosineEmbeddingLoss()

    def _train_epoch_MIMIC_CXR(self, epoch):

        self.logger.info('################# [{}/{}] MIMIC_CXR Start to train in the training set. #################'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (_, 
                        images, 
                        image_mask_bone, 
                        image_mask_lung, 
                        image_mask_heart, 
                        image_mask_mediastinum, 
                        reports_ids, 
                        reports_masks, 
                        disease_detected_bone, 
                        disease_detected_lung, 
                        disease_detected_heart, 
                        disease_detected_mediastinum) in enumerate(self.train_dataloader):
            
            cosin_sim_tgt = torch.ones(images.shape[0]).to(self.device) # 1 => 完全相关， -1 => 完全无关
            
            images = images.to(self.device)
            image_mask_bone = image_mask_bone.to(self.device)
            image_mask_lung = image_mask_lung.to(self.device)
            image_mask_heart = image_mask_heart.to(self.device)
            image_mask_mediastinum = image_mask_mediastinum.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            disease_detected_bone = disease_detected_bone.to(self.device)
            disease_detected_lung = disease_detected_lung.to(self.device)
            disease_detected_heart = disease_detected_heart.to(self.device)
            disease_detected_mediastinum = disease_detected_mediastinum.to(self.device)
            
            output,mask_feats, text_embeddings, disease_token_feats, disease_token_target = self.model(images = images, 
                                                                                                       image_mask_bone = image_mask_bone,
                                                                                                       image_mask_lung = image_mask_lung, 
                                                                                                       image_mask_heart = image_mask_heart, 
                                                                                                       image_mask_mediastinum = image_mask_mediastinum, 
                                                                                                       disease_detected_bone = disease_detected_bone,
                                                                                                       disease_detected_lung = disease_detected_lung,
                                                                                                       disease_detected_heart = disease_detected_heart,
                                                                                                       disease_detected_mediastinum = disease_detected_mediastinum, 
                                                                                                       targets = reports_ids, 
                                                                                                       mode='train')
            # Disease cosine similarity
            loss_disease_cs_bone = self.cosin_simi_loss(torch.mean(disease_token_feats["bone"], dim=1),torch.mean(disease_token_target["bone"], dim=1), cosin_sim_tgt) 
            loss_disease_cs_lung = self.cosin_simi_loss(torch.mean(disease_token_feats["lung"], dim=1),torch.mean(disease_token_target["lung"], dim=1), cosin_sim_tgt)
            loss_disease_cs_heart = self.cosin_simi_loss(torch.mean(disease_token_feats["heart"], dim=1),torch.mean(disease_token_target["heart"], dim=1), cosin_sim_tgt)
            loss_disease_cs_mediastinum = self.cosin_simi_loss(torch.mean(disease_token_feats["mediastinum"], dim=1),torch.mean(disease_token_target["mediastinum"], dim=1), cosin_sim_tgt)
            # Disease Cosine Similarity
            loss_disease_cs = (loss_disease_cs_bone + loss_disease_cs_lung + loss_disease_cs_heart + loss_disease_cs_mediastinum)/4
            # # Caption Cosine Similarity
            loss_cs = self.cosin_simi_loss(torch.mean(text_embeddings, dim=1),torch.mean(mask_feats, dim=1), cosin_sim_tgt)
            # # # Total loss
            loss = self.criterion(output, reports_ids, reports_masks) + 0.1 * loss_cs + 0.1 * loss_disease_cs
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}. Caption Cosine Similarity Loss: {:.5f}, Disease Cosine Similarity Loss: {:.5f}'
                                .format(epoch, self.epochs, batch_idx, len(self.train_dataloader), train_loss / (batch_idx + 1), loss_cs, loss_disease_cs))


        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('## [{}/{}] MIMIC_CXR Start to evaluate in the validation set. ##'.format(epoch, self.epochs))
        self.model.eval()  
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (_, 
                            images, 
                            image_mask_bone,
                            image_mask_lung, 
                            image_mask_heart, 
                            image_mask_mediastinum, 
                            reports_ids, 
                            reports_masks, 
                            _,
                            _,
                            _,
                            _) in enumerate(self.val_dataloader):
                
                images = images.to(self.device)
                image_mask_bone = image_mask_bone.to(self.device)
                image_mask_lung = image_mask_lung.to(self.device)
                image_mask_heart = image_mask_heart.to(self.device)
                image_mask_mediastinum = image_mask_mediastinum.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                output, _ = self.model(images, 
                                       image_mask_bone,
                                       image_mask_lung, 
                                       image_mask_heart, 
                                       image_mask_mediastinum, 
                                       mode='sample')
                
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            idx = random.randint(0, len(val_gts)-1)
            self.logger.info(">>>>> val Example predict: {}.".format(val_res[idx]))
            self.logger.info(">>>>> val Example target : {}.".format(val_gts[idx]))
            
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
                

        self.logger.info('## [{}/{}] MIMIC_CXR Start to evaluate in the test set. ##'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (_, 
                            images, 
                            image_mask_bone,
                            image_mask_lung, 
                            image_mask_heart, 
                            image_mask_mediastinum, 
                            reports_ids, 
                            reports_masks, 
                            _, 
                            _, 
                            _, 
                            _) in enumerate(self.test_dataloader):
                
                images = images.to(self.device)
                image_mask_bone = image_mask_bone.to(self.device)
                image_mask_lung = image_mask_lung.to(self.device)
                image_mask_heart = image_mask_heart.to(self.device)
                image_mask_mediastinum = image_mask_mediastinum.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                output, _ = self.model(images, 
                                       image_mask_bone,
                                       image_mask_lung, 
                                       image_mask_heart, 
                                       image_mask_mediastinum,
                                       mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            idx = random.randint(0, len(test_gts)-1)
            self.logger.info(">>>>> test Example predict: {}.".format(test_res[idx]))
            self.logger.info(">>>>> test Example target : {}.".format(test_gts[idx]))
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        return log

    def _train_epoch_IU_Xray(self, epoch):
        self.logger.info('################# [{}/{}] IU_Xray Start to train in the training set. #################'.format(epoch, self.epochs))
        train_loss = 0
        batch_num = len(self.train_dataloader)
        self.model.train()
        # with torch.autograd.set_detect_anomaly(True):
        for batch_idx, (_, 
                        images, 
                        image_mask_bone, 
                        image_mask_lung, 
                        image_mask_heart, 
                        image_mask_mediastinum, 
                        reports_ids, 
                        reports_masks, 
                        disease_detected) in enumerate(self.train_dataloader):
            
            cosin_sim_tgt = torch.ones(images.shape[0]).to(self.device)
            
            images = images.to(self.device)
            image_mask_bone = image_mask_bone.to(self.device)
            image_mask_lung = image_mask_lung.to(self.device)
            image_mask_heart = image_mask_heart.to(self.device)
            image_mask_mediastinum = image_mask_mediastinum.to(self.device)
            reports_ids = reports_ids.to(self.device)
            reports_masks = reports_masks.to(self.device)
            disease_detected = disease_detected.to(self.device)
            
            output, mask_feats, text_embeddings, disease_token_target, disease_token_feats = self.model(images = images,
                                                                                                        image_mask_bone = image_mask_bone,
                                                                                                        image_mask_lung = image_mask_lung, 
                                                                                                        image_mask_heart = image_mask_heart, 
                                                                                                        image_mask_mediastinum = image_mask_mediastinum, 
                                                                                                        disease_detected = disease_detected, 
                                                                                                        targets = reports_ids, 
                                                                                                        mode='train')
            # Disease cosine similarity
            loss_disease_cs = self.cosin_simi_loss(torch.mean(disease_token_feats, dim=1),torch.mean(disease_token_target, dim=1), cosin_sim_tgt) # bs, 512
            # # Caption cosine similarity
            loss_cs = self.cosin_simi_loss(torch.mean(text_embeddings, dim=1),torch.mean(mask_feats, dim=1), cosin_sim_tgt)
            # # # Total Loss
            loss = self.criterion(output, reports_ids, reports_masks) + 0.1 * loss_disease_cs  + 0.1 * loss_cs
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}, Caption Cosine Similarity: {:.5f}. Disease Consine Similarity: {:.5f}.'
                                .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                        train_loss / (batch_idx + 1), loss_cs, loss_disease_cs))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('## [{}/{}] IU_Xray Start to evaluate in the validation set. ##'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (_, 
                            images, 
                            image_mask_bone,
                            image_mask_lung, 
                            image_mask_heart, 
                            image_mask_mediastinum, 
                            reports_ids, 
                            reports_masks, 
                            _) in enumerate(self.val_dataloader):
                
                images = images.to(self.device)
                image_mask_bone = image_mask_bone.to(self.device)
                image_mask_lung = image_mask_lung.to(self.device) 
                image_mask_heart = image_mask_heart.to(self.device)
                image_mask_mediastinum = image_mask_mediastinum.to(self.device) 
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                output, _ = self.model(images, 
                                       image_mask_bone,
                                       image_mask_lung, 
                                       image_mask_heart, 
                                       image_mask_mediastinum, 
                                       mode='sample')
                
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            idx = random.randint(0, len(val_gts)-1)
            self.logger.info(">>>>> val Example predict: {}.".format(val_res[idx]))
            self.logger.info(">>>>> val Example target : {}.".format(val_gts[idx]))
            
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})   

        self.logger.info('## [{}/{}] IU_Xray Start to evaluate in the test set. ##'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (_, 
                            images, 
                            image_mask_bone,
                            image_mask_lung, 
                            image_mask_heart, 
                            image_mask_mediastinum, 
                            reports_ids, 
                            reports_masks, 
                            _) in enumerate(self.test_dataloader):
                
                images = images.to(self.device)
                image_mask_bone = image_mask_bone.to(self.device)
                image_mask_lung = image_mask_lung.to(self.device) 
                image_mask_heart = image_mask_heart.to(self.device)
                image_mask_mediastinum = image_mask_mediastinum.to(self.device) 
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)
                
                output, _ = self.model(images, 
                                       image_mask_bone,
                                       image_mask_lung, 
                                       image_mask_heart, 
                                       image_mask_mediastinum, 
                                       mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            idx = random.randint(0, len(test_gts)-1)
            self.logger.info(">>>>> test Example predict: {}.".format(test_res[idx]))
            self.logger.info(">>>>> test Example target : {}.".format(test_gts[idx]))
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        return log

