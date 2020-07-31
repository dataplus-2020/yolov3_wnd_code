import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         opt=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.png'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    lbl_path = data['valid_label']
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, lbl_path, img_size, batch_size, rect=True, single_cls=opt.single_cls)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once
    #fixme
    # coco91class = coco80_to_coco91_class()
    # img_id_map = json.load(open(glob.glob(os.path.join(opt.base_dir, 'xview_val_*cls_img_id_map.json'))[0]))

    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Plot images with bounding boxes
        f = 'test_batch%g.png' % batch_i  # filename
        if batch_i < 1 and not os.path.exists(f):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs

            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)  # nms
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                #fixme
                # image_id = int(Path(paths[si]).stem.split('_')[-1])
                image_name = os.path.basename(paths[si]).replace('.txt', '.jpg')
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_name,# img_id_map[image_name],
                                  'category_id': 0, # coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        # imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        # with open('results.json', 'w') as file:
        #     json.dump(jdict, file)
        result_json_file = 'results_{}.json'.format(opt.name)
        with open(os.path.join(opt.result_dir, result_json_file), 'w') as file:
            json.dump(jdict, file, ensure_ascii=False, indent=2, cls=MyEncoder)

        # imgIds = [id for id in img_id_map.values()]
        # try:
        #    from pycocotools.coco import COCO
        #    from pycocotools.cocoeval import COCOeval
        # except:
        #     print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')
        #
        # # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        # cocoGt = COCO(glob.glob(os.path.join(opt.base_dir, '*_xtlytlwh.json'))[0])  # initialize COCO ground truth api
        # cocoDt = cocoGt.loadRes(os.path.join(opt.result_dir, result_json_file)) # initialize COCO pred api
        #
        # cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        # cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        # cocoEval.evaluate()
        # cocoEval.accumulate()
        # cocoEval.summarize()
        # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    # print(map)
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # print(maps)
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls_syn.cfg', help='*.cfg path cfg/yolov3-spp.cfg')
    parser.add_argument('--data', type=str, default='/home/jovyan/work/data_xview/{}_cls/{}/xview_{}_{}.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='/home/jovyan/work/code/yxu-yolov3-xview/weights/1_cls/syn_mixed_seed17/hgiou1_1gpu_seed17_2020-06-17_05.54/last_seed17.pt',
                        help='path to weights file')
    # parser.add_argument('--weights', type=str, default='weights/{}_cls/{}_seed{}/', help='path to weights file')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='16 size of each image batch')
    parser.add_argument('--img-size', type=int, default=608, help='512 inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help=' 0.001 object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='0.6 IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', default=True, help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', default=True, help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--class_num', type=int, default=1, help='class number')
    parser.add_argument('--base_dir', type=str, default='data_xview/{}_cls/{}/', help='without syn data path')
    parser.add_argument('--weights_dir', type=str, default='weights/{}_cls/{}_seed{}/', help='to save weights path')
    parser.add_argument('--result_dir', type=str, default='result_output/{}_cls/{}_seed{}/{}/', help='to save result files path')
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    '''
    xview_syn_xview_bkg* test on model
    '''
    sd=17
    # comments = ['xview_syn_xview_bkg_px23whr3_6groups_models_color', 'xview_syn_xview_bkg_px23whr3_6groups_models_mixed']
    # # comments = ['px23whr3']
    # base_cmt = 'px23whr3_seed{}'.format(sd)

    # comments = ['px23whr4']
    # base_cmt = 'px23whr4_seed{}'.format(sd)
    comments = ['syn_mixed']
    base_cmt = 'seed{}'.format(sd)

    # hyp_cmt = 'hgiou1_fitness'
    hyp_cmt = 'hgiou1'

    # opt.name = 'seed{}_on_original'.format(sd)
    opt.name = 'seed{}_on_original_model'.format(sd)
    for cmt in comments:

        # opt.weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}'.format(hyp_cmt, sd), 'best_{}.pt'.format(base_cmt)))[-1]
        opt.weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*', 'last_{}.pt'.format(base_cmt)))[-1]
        # opt.weights = glob.glob(os.path.join(opt.weights_dir.format(opt.class_num, cmt, sd),  '*_{}_seed{}'.format(hyp_cmt, sd), 'backup80.pt'))[-1]
        print(opt.weights)
        # opt.data = 'data_xview/{}_cls/{}_seed{}/xview_{}_seed{}.data'.format(opt.class_num, cmt, sd, cmt, sd)
        opt.data = '/home/jovyan/work/data/syn_background_gt_bbox/syn_mixed_seed17.data'
        opt.result_dir = opt.result_dir.format(opt.class_num, cmt, sd, 'test_on_original_model_{}_seed{}'.format(hyp_cmt, sd))

        if not os.path.exists(opt.result_dir):
            os.makedirs(opt.result_dir)
        opt.base_dir = opt.base_dir.format(opt.class_num, base_cmt)
        print(opt)
        # task = 'test', 'study', 'benchmark'
        if opt.task == 'test':  # (default) test normally
            test(opt.cfg,
                 opt.data,
                 opt.weights,
                 opt.batch_size,
                 opt.img_size,
                 opt.conf_thres,
                 opt.iou_thres,
                 opt.save_json,
                 opt.single_cls,
                 opt.augment,
                 opt = opt)

        elif opt.task == 'benchmark':  # mAPs at 320-608 at conf 0.5 and 0.7
            y = []
            for i in [320, 416, 512, 608]:  # img-size
                for j in [0.5, 0.7]:  # iou-thres
                    t = time.time()
                    r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json, opt = opt)[0]
                    y.append(r + (time.time() - t,))
            np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

        elif opt.task == 'study':  # Parameter study
            y = []
            x = np.arange(0.4, 0.9, 0.05)  # iou-thres
            for i in x:
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, i, opt.save_json, opt = opt)[0]
                y.append(r + (time.time() - t,))
            np.savetxt('study.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

            # Plot
            fig, ax = plt.subplots(3, 1, figsize=(6, 6))
            y = np.stack(y, 0)
            ax[0].plot(x, y[:, 2], marker='.', label='mAP@0.5')
            ax[0].set_ylabel('mAP')
            ax[1].plot(x, y[:, 3], marker='.', label='mAP@0.5:0.95')
            ax[1].set_ylabel('mAP')
            ax[2].plot(x, y[:, -1], marker='.', label='time')
            ax[2].set_ylabel('time (s)')
            for i in range(3):
                ax[i].legend()
                ax[i].set_xlabel('iou_thr')
            fig.tight_layout()
            plt.savefig('study.jpg', dpi=200)
