import numpy as np
import math
from keras_retinanet.utils.image import preprocess_image, resize_image

TEXT_NORMAL = 'OK'
TEXT_DEFECT = 'NOK'


def predict_retina_net_chip(model, img, th):
    min_side, max_side = 512, 512
    list_normal = [
        # 0,  # crack
        # 1,  # foreign
        # 2,  # missing
        3,  # normal capacitor / resistor
        # 4,  # position
        # 5,  # solder
        6,  # tombstone
        # 7,  # wrong
    ]

    results = predict_retina_net(model, img, th, min_side, max_side)
    results = pick_up_best(results)

    if results:
        img_width, img_height = img.size
        image_center = img_width / 2, img_height / 2
        list_center = list(((left + right) / 2, (top + bottom) / 2) for (left, top, right, bottom) in
                           (boxes for (boxes, label, score) in results))
        list_dist = list(cal_distance(image_center, center) for center in list_center)
        min_dist = min(list_dist)
        min_dist_idx = list_dist.index(min_dist)
        box, label, score = results[min_dist_idx]

        b_defect = label not in list_normal
    else:
        b_defect = True

    return TEXT_DEFECT if b_defect else TEXT_NORMAL


def predict_retina_net_tab(model, img, th):
    min_side, max_side = 800, 1000
    list_normal = [
        0,  # normal
    ]
    min_normal = 2
    ratio_wh, ratio_w, ratio_gap = 1.3, 1.1, 2 / 3
    iou = 0.1

    results = predict_retina_net_slide(model, img, th, ratio_wh, ratio_w, ratio_gap, min_side, max_side)
    results = pick_up_best(results, iou)

    result_normal = list(result for result in results if result[1] in list_normal and ((1 - th) * 2) < result[2])
    result_defect = list(result for result in results if result[1] not in list_normal)

    if bool(result_defect):
        b_defect = True
    else:
        if len(result_normal) < min_normal:
            b_defect = True
        else:
            b_defect = False

    return TEXT_DEFECT if b_defect else TEXT_NORMAL


def predict_retina_net_ground(model, img, th):
    min_side, max_side = 800, 1333
    ratio_wh, ratio_w, ratio_gap = 1.3, 1.1, 2 / 3

    results = predict_retina_net_slide(model, img, th, ratio_wh, ratio_w, ratio_gap, min_side, max_side)

    b_defect = bool(results)

    return TEXT_DEFECT if b_defect else TEXT_NORMAL


def predict_retina_net_edge(model, img, th):
    min_side, max_side = 200, 2000

    img_width, img_height = img.size
    smallest_side = min(img_width, img_height)
    largest_side = max(img_width, img_height)
    iou = 0.1

    results = []
    if largest_side <= smallest_side * 1.5:
        crop_size = int(smallest_side * 0.1)

        list_crop_range = [
            [0, 0, crop_size, img_height],  # Left
            [0, 0, img_width, crop_size],  # Top
            [img_width - crop_size, 0, img_width, img_height],  # Right
            [0, img_height - crop_size, img_width, img_height],  # Bottom
        ]

        guide_iou_th = 0.5
        guide_ratio = 0.3
        list_guide_range = [
            [0, 0, int(crop_size * guide_ratio), img_height],  # Left_L
            # [int(crop_size * (1 - guide_ratio)), 0, crop_size, img_height],  # Left_R
            # [img_width - crop_size, 0, img_width - int(crop_size * (1 - guide_ratio)), img_height],  # Right_L
            [img_width - int(crop_size * guide_ratio), 0, img_width, img_height],  # Right_R
        ]

        for (idx, crop_range) in enumerate(list_crop_range):
            [left, top, right, bottom] = crop_range

            img_crop = img.crop(crop_range)
            results_crop = predict_retina_net(model, img_crop, th, min_side, max_side)
            results += list([list(x + y for (x, y) in zip(result[0], [left, top, left, top])), *result[1:]] for result in results_crop)

        list_idx = []
        for guide_range in list_guide_range:
            [left_guide, top_guide, right_guide, bottom_guide] = guide_range
            for (idx_result, result) in enumerate(results):
                ((left_result, top_result, right_result, bottom_result), label_result, score_result) = result

                (left_IoU, top_IoU, right_IoU, bottom_IoU) = (max(left_guide, left_result),
                                                              max(top_guide, top_result),
                                                              min(right_guide, right_result),
                                                              min(bottom_guide, bottom_result))

                size_overlap = max(0, (right_IoU - left_IoU)) * max(0, (bottom_IoU - top_IoU))
                size_result = (right_result - left_result) * (bottom_result - top_result)
                guide_iou = size_overlap / size_result
                if guide_iou_th < guide_iou:
                    list_idx.append(idx_result)

        if bool(list_idx):
            results = list(result for (idx_result, result) in enumerate(results) if idx_result not in list_idx)
    else:
        results = predict_retina_net(model, img, th, min_side, max_side)

    results = pick_up_best(results, iou)

    b_defect = bool(results)

    return TEXT_DEFECT if b_defect else TEXT_NORMAL


def predict_retina_net_slide(model, img, th, ratio_wh, ratio_w, ratio_gap, min_side=800, max_side=1333):
    img_width, img_height = img.size

    results = []
    if ratio_wh < (img_width / img_height):
        new_width = img_height * ratio_w
        img_gap = int(new_width * ratio_gap)
        ratio = (img_width - img_height) / img_gap
        decimal = ratio - int(ratio)

        for idx in range(math.ceil(ratio)):
            left = img_gap * idx
            img_crop = img.crop((left, 0, left + img_height, img_height))
            results += predict_retina_net(model, img_crop, th, min_side, max_side)

        if decimal:
            img_crop = img.crop((img_width - new_width, 0, img_width, img_height))
            results += predict_retina_net(model, img_crop, th, min_side, max_side)
    else:
        results = predict_retina_net(model, img, th, min_side, max_side)

    return results


def predict_retina_net(model, img, th, min_side=800, max_side=1333):
    image = preprocess_image(np.asarray(img)[..., ::-1].copy())
    image, scale = resize_image(image, min_side, max_side)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale

    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    indices = np.where(scores[:] >= (1 - th))[0]
    boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

    results = [[list(map(int, box)), label, score] for (box, score, label) in zip(boxes, scores, labels)]

    return results


def cal_distance(pt_from, pt_to):
    return math.sqrt(pow(pt_from[0] - pt_to[0], 2) + pow(pt_from[1] - pt_to[1], 2))


def pick_up_best(results, iou=0.0):
    idx_1 = 0
    while idx_1 in range(len(results)):
        ((left_1, top_1, right_1, bottom_1), label_1, score_1) = results[idx_1]
        set_idx = set()
        for idx_2, ((left_2, top_2, right_2, bottom_2), label_2, score_2) in enumerate(results):
            if idx_1 != idx_2:
                left, top, right, bottom = max(left_1, left_2), max(top_1, top_2), min(right_1, right_2), min(bottom_1,
                                                                                                              bottom_2)
                size_overlap = max(0, (right - left)) * max(0, (bottom - top))
                size_union = (right_1 - left_1) * (bottom_1 - top_1) + (right_2 - left_2) * (
                        bottom_2 - top_2) - size_overlap
                IoU = size_overlap / size_union

                if iou < IoU:
                    idx = idx_2 if score_2 <= score_1 else idx_1
                    set_idx.add(idx)
        list_idx = sorted(list(set_idx))
        if list_idx:
            for idx in reversed(list_idx):
                results.pop(idx)
        else:
            idx_1 += 1

    return results
