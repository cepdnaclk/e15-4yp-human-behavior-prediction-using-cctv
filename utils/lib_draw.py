def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(0,0,0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    image_h, image_w, _ = image.shape

    num_ids = 10
    hsv_tuples = [(1.0 * x / num_ids, 1., 1.) for x in range(num_ids)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    cur_point = [0, 0]
    cv2.polylines(image, [pts], True, (0, 0, 255), 2)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])

        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[int(bbox[4])], bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), colors[int(bbox[4])], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

def draw_path(floor_plan, bboxes, human_paths):
    num_ids = 10
    hsv_tuples = [(1.0 * x / num_ids, 1., 1.) for x in range(num_ids)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i, bbox in enumerate(bboxes):
        cur_point_conv = cv2.perspectiveTransform(bbox[6], h_mat)
        if cur_point_conv is not None:
            cur_point = cur_point_conv[0][0]
            track_id = int(bbox[4])
            pre_point = human_paths[track_id]
            if pre_point is not None:
                pre_point_conv = cv2.perspectiveTransform(pre_point, h_mat)
                pre_point = pre_point_conv[0][0]
                #print(f'{track_id}----> ({pre_point[0]} :: {pre_point[1]}) ({cur_point[0]} :: {cur_point[1]})')
                cv2.line(floor_plan, (pre_point[0], pre_point[1]), (cur_point[0], cur_point[1]), colors[track_id], 5)

    return floor_plan
