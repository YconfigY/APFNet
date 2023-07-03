import os
import time
import argparse
import numpy as np
import yaml, json
import cv2
import matplotlib.pyplot as plt

import torch

from APFNet.model_tracking import MDNet
from data.sample_generator import SampleGenerator

from utils.utils import overlap_ratio
from utils.log import get_logger
from utils.metric import BCELoss
from utils.optimizer import set_optimizer

from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config
from tracking.tracking_option import opts


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def read_test_data(dataset_path, list_file):
    dataset_list = open(list_file)
    seq_list = []
    while True:
        line = dataset_list.readline().strip()
        if line:
            seq_list.append(line)
        else:
            break
    dataset_list.close()
    for seq_name in seq_list:
        if seq_name not in os.listdir(dataset_path):
            seq_list.remove(seq_name)
    return seq_list


def forward_samples(model, image_v, image_i, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image_v, image_i, samples, opts)
    for i, (regions_v, regions_i) in enumerate(extractor):
        if opts['use_gpu']:
            regions_v = regions_v.cuda()
            regions_i = regions_i.cuda()
        with torch.no_grad():
            feat = model(regions_v, regions_i, out_layer=out_layer)
        if i == 0:
            features = feat.detach().clone()
        else:
            features = torch.cat((features, feat.detach().clone()), dim=0)

    return features


def train(model, criterion, optimizer, pos_features, neg_features, maxiter, in_layer='fc4'):
    """train model by using pos_features and neg_features

    Args:
        model (_type_): _description_
        criterion (_type_): _description_
        optimizer (_type_): _description_
        pos_features (_type_): _description_
        neg_features (_type_): _description_
        maxiter (_type_): _description_
        in_layer (str, optional): _description_. Defaults to 'fc4'.
    """
    
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_features.size(0))
    neg_idx = np.random.permutation(neg_features.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_features.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_features.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_features.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_features.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_features = pos_features[pos_cur_idx]
        batch_neg_features = neg_features[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_features[start:end], batch_neg_features[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_features = batch_neg_features[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_features, batch_pos_features, in_layer=in_layer)
        neg_score = model(batch_neg_features, batch_neg_features, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_model(img_list_visible, img_list_infrared, init_bbox, gt, args, savefig_dir='', display=False):
    """_summary_

    Args:
        img_list_visible (_type_): _description_
        img_list_infrared (_type_): _description_
        init_bbox (_type_): _description_
        gt (_type_): _description_
        args (_type_): _description_
        savefig_dir (str, optional): _description_. Defaults to ''.
        display (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list_visible), 4))
    result_bb = np.zeros((len(img_list_visible), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list_visible))
        overlap[0] = 1

    # Init model
    model = MDNet(args.model_path)
    if opts['use_gpu']:
        model = model.cuda()
    
    # Init criterion and optimizer 
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    model.get_learnable_params()
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_layer'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_layer'], opts['lr_mult'])

    tic = time.time()
    # Load first frame
    image_v = cv2.imread(img_list_visible[0])[:, :, ::-1]
    image_i = cv2.imread(img_list_infrared[0])[:, :, ::-1]

    # Generate pos/neg samples according to first frame
    pos_sample_generator = SampleGenerator('gaussian', image_v.size, opts['trans_pos'], opts['scale_pos'])
    ## Generate 500 positive samples
    pos_samples = pos_sample_generator(target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_sample_generator_1 = SampleGenerator('uniform', image_v.size, opts['trans_neg_init'], opts['scale_neg_init'])
    neg_sample_generator_2 = SampleGenerator('whole', image_v.size)
    
    ## Generate 5000 negative samples
    neg_samples = np.concatenate(
        [neg_sample_generator_1(target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
         neg_sample_generator_2(target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_samples = np.random.permutation(neg_samples)  # shuffle neg_samples
    
    # Extract pos/neg features
    pos_features = forward_samples(model, image_v, image_i, pos_samples)
    neg_features = forward_samples(model, image_v, image_i, neg_samples)
    torch.cuda.empty_cache()
    
    # Initial training
    train(model, criterion, init_optimizer, pos_features, neg_features, opts['maxiter_init'])
    # Train bbox regressor
    bbox_reg_sample_generator = SampleGenerator('uniform', image_v.size, opts['trans_bbreg'], 
                                                opts['scale_bbreg'], opts['aspect_bbreg'])
    bbox_reg_examples = bbox_reg_sample_generator(target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])

    bbox_reg_features = forward_samples(model, image_v, image_i, bbox_reg_examples)
    bbox_reg = BBRegressor(image_v.size)
    bbox_reg.train(bbox_reg_features, bbox_reg_examples, target_bbox)
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image_v.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image_v.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image_v.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_features = forward_samples(model, image_v, image_i, neg_examples)
    pos_features_all = [pos_features]
    neg_features_all = [neg_features]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image_v.size[0] / dpi, image_v.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image_v, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list_visible)):

        tic = time.time()
        # Load image
        image_v = cv2.imread(img_list_visible[i])[:, :, ::-1]
        image_i = cv2.imread(img_list_infrared[i])[:, :, ::-1]

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image_v, image_i, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbox_reg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbox_reg_samples = bbox_reg_samples[None, :]
            bbox_reg_features = forward_samples(model, image_v, image_i, bbox_reg_samples)
            bbox_reg_samples = bbox_reg.predict(bbox_reg_features, bbox_reg_samples)
            bbox_reg_bbox = bbox_reg_samples.mean(axis=0)
        else:
            bbox_reg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbox_reg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            
            pos_features = forward_samples(model, image_v, image_i, pos_examples)
            neg_features = forward_samples(model, image_v, image_i, neg_examples)
            
            pos_features_all.append(pos_features)
            neg_features_all.append(neg_features)
            
            if len(pos_features_all) > opts['n_frames_long']:
                del pos_features_all[0]
            if len(neg_features_all) > opts['n_frames_short']:
                del neg_features_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_features_all))
            pos_data = torch.cat(pos_features_all[-nframes:], 0)
            neg_data = torch.cat(neg_features_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            # we update the new templeate
            first_samples = np.expand_dims(target_bbox, axis=0)
            first_samples = first_samples.repeat(256, axis=0)
            # first image extractor
            first_extractor = RegionExtractor(image_v, image_i, first_samples, opts)
            pos_data = torch.cat(pos_features_all, 0)
            neg_data = torch.cat(neg_features_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            # im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                  .format(i, len(img_list_visible), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('\rFrame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                  .format(i, len(img_list_visible), overlap[i], target_score, spf), end="", flush=True)

    if gt is not None:
        print('')
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list_visible) / spf_total
    return overlap, result, result_bb, fps


def record_results(results_file, result_bb, is_create_txt=True):
    with open(args.result_path + '/' + results_file, 'a') as f:
        if is_create_txt:
            res = '{} {} {} {} {} {} {} {}\n'.format(
                result_bb[0], result_bb[1],
                result_bb[0] + result_bb[2], result_bb[1],
                result_bb[0] + result_bb[2], result_bb[1] + result_bb[3],
                result_bb[0], result_bb[1] + result_bb[3])
            f.write(res)
        else:
            for i in range(1, len(result_bb)):
                res = '{} {} {} {} {} {} {} {}\n'.format(
                    result_bb[i][0], result_bb[i][1],
                    result_bb[i][0] + result_bb[i][2], result_bb[i][1],
                    result_bb[i][0] + result_bb[i][2], result_bb[i][1] + result_bb[i][3], 
                    result_bb[i][0], result_bb[i][1] + result_bb[i][3])
                f.write(res)


def run_tracker(seq_list, start_index, tracker):
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    for seq_index, seq_name in enumerate(seq_list):
        torch.cuda.empty_cache()
        np.random.seed(123)
        torch.manual_seed(456)
        torch.cuda.manual_seed(789)
        torch.backends.cudnn.benchmark = True
        seq_path = dataset_path + '/' + seq_name
        results_file = seq_name + '.txt'
        if 'txt' in seq_name or results_file in os.listdir(args.result_path) or seq_index < -1:
            continue
        logger.info('tracker name:{} seq index:{} seq name:{} start track'
                    .format(tracker, seq_index+start_index, seq_name))
        # Generate sequence config
        img_list_v, img_list_i, gt = gen_config(seq_path, args.dataset)

        # Write first frame target bounding box to results file
        record_results(results_file, np.array(gt[0]), is_create_txt=True)
        # Run tracker
        iou_result, result, result_bb, fps = run_model(img_list_v, img_list_i, gt[0], gt, args)

        # Save result
        iou_list.append(iou_result.sum() / len(iou_result))
        fps_list[seq_name] = fps

        bb_result[seq_name] = result_bb
        bb_result_nobb[seq_name] = result

        logger.info('tracker name:{} seq index:{} seq name:{} aver IoU:{:.3f} total mIoU:{:.3f} fps:{:.3f}'
                    .format(tracker, seq_index+start_index, seq_name, iou_result.mean(), 
                            sum(iou_list) / len(iou_list), sum(fps_list.values()) / len(fps_list)))
        record_results(results_file, result_bb, is_create_txt=False)

    
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument("-dataset", default='RGBT234')  # testing dataset
    parser.add_argument("-model_path", default='./resume/stage_3/GTOT/ALL/ALL.pth')  # your model
    parser.add_argument("-result_path", default='./results/')  # your result path
    args = parser.parse_args()
    logger = get_logger()  # Record your log files
    model_name = args.model_path.split('/')[-1].split('.')[0]
    ##your result path
    args.result_path = os.path.join('./results/', args.dataset, model_name)
    # assert args.seq != '' or args.json != ''
    print(opts)

    dataset_path = os.path.join('/home/ubuntu/data/~/', args.dataset)  # dataset path
    dataset_list_file = '/home/ubuntu/data/~/RGBT234/RGBT234.txt'  # dataset list path

    seq_list = read_test_data(dataset_path, dataset_list_file)
    iou_list = []
    fps_list = dict()
    bb_result = dict()
    result = dict()
    bb_result_nobb = dict()
    num_test = len(seq_list)//5

    # with futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     for seq in zip(seq_list[num_test*4:], executor.map(run, seq_list[num_test*4:])):
    #         print(seq, 'done!')
    start_index, end_index = num_test*4, num_test*5
    
    run_tracker(seq_list[113:114], 113, 'tracker_1')