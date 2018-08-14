import glob
import os
import cv2
import numpy as np
import time
import argparse
import xml.etree.ElementTree as ET

template = """<annotation>
	<folder>data</folder>
	<filename></filename>
	<path></path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width></width>
		<height></height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
</annotation>
"""

objectTemplate = """<object>
		<name></name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin></xmin>
			<ymin></ymin>
			<xmax></xmax>
			<ymax></ymax>
		</bndbox>
	</object>
"""

offsetMap = {0:(-1, -1), 1:(-1, 0), 2:(-1, 1), 3:(0, -1), 4:(0, 0), 5:(0, 1), 6:(1, -1), 7:(1, 0), 8:(1, 1)}

bndbox_line_color = (255, 0, 0)
bndbox_line_width = 2
square_block_color = (0, 0, 255)
square_block_width = 2


def get_arguments():
    parser = argparse.ArgumentParser(description="ImageSimilarity Network")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--image-format", type=str)
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--margin", type=int)
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


if __name__ == '__main__':
    # from PIL import Image
    # img = Image.open('./coordinate_183.jpg')
    # img_resize = img.resize((300, 300))
    # img_resize.save('data/coordinate_183_300.jpg')

    start = time.time()
    args = get_arguments()
    target = os.path.join(args.data_dir, '*.' + args.image_format)
    results_dir = args.results_dir
    window_size = args.window_size
    margin = args.margin
    threshold = args.threshold

    print("Settings:")
    print(' - [target] : ' + target)
    print(' - [results dir] : ' + results_dir)
    print(' - [window size] : ' + str(window_size))
    print(' - [margin] : ' + str(margin))
    print(' - [threshold] : ' + str(threshold))
    print('\n')


    for path in glob.iglob(target):
        xmlMap = {}
        dir, filename = os.path.split(path)
        print('start processing : ' + filename)

        # revise org image
        basefilename = os.path.splitext(filename)[0]
        img = cv2.imread(path)
        height, width, channels = img.shape
        remain_height = height % (window_size - margin)
        remain_width = width % (window_size - margin)
        add_height = np.ones((window_size - remain_height, width, 3), np.uint8)
        add_width = np.ones((height + add_height.shape[0], window_size - remain_width, 3), np.uint8)
        processed_img = cv2.vconcat([img, add_height])
        processed_img = cv2.hconcat([processed_img, add_width])
        sample_img = processed_img.copy()
        new_height, new_width = processed_img.shape[0],processed_img.shape[1]
        num_vert = ((new_height - window_size) // (window_size - margin)) + 1
        num_horizon = ((new_width - window_size) // (window_size - margin)) + 1

        # transform xml
        tree = ET.parse(os.path.join(args.data_dir, basefilename + '.xml'))
        objects = tree.findall('object')
        print("  - Number of initial BBox: " + str(len(objects)))
        for object in objects:
            name = object.findtext('name')
            xmin = int(object.findtext('bndbox/xmin'))
            xmax = int(object.findtext('bndbox/xmax'))
            ymin = int(object.findtext('bndbox/ymin'))
            ymax = int(object.findtext('bndbox/ymax'))
            cv2.rectangle(sample_img, (xmin, ymin), (xmax, ymax), bndbox_line_color, bndbox_line_width)
            bndbox_width = xmax - xmin
            bndbox_height = ymax - ymin

            org_area = bndbox_width * bndbox_height

            x_base_pos = xmin // (window_size - margin)
            y_base_pos = ymin // (window_size - margin)

            target_list = {}
            for key in range(len(offsetMap)):
                x_val_pos = x_base_pos + offsetMap[key][1]
                y_val_pos = y_base_pos + offsetMap[key][0]

                if x_val_pos < 0 or y_val_pos < 0 or x_val_pos == num_horizon or y_val_pos == num_vert:
                    continue

                transformed_x_min = xmin - (window_size - margin) * x_val_pos
                transformed_x_max = xmax - (window_size - margin) * x_val_pos
                transformed_y_min = ymin - (window_size - margin) * y_val_pos
                transformed_y_max = ymax - (window_size - margin) * y_val_pos

                if key == 0 or key == 1 or key == 3 or key == 4:
                    if transformed_x_min >= window_size or transformed_y_min >= window_size:
                        continue

                    if transformed_x_max > window_size:
                        transformed_x_max = window_size

                    if transformed_y_max > window_size:
                        transformed_y_max = window_size

                elif key == 2 or key == 5:
                    if 0 >= transformed_x_max or transformed_y_min >= window_size:
                        continue

                    if transformed_x_min < 0:
                        transformed_x_min = 0

                    if transformed_x_max > window_size:
                        transformed_x_max = window_size

                    if transformed_y_max > window_size:
                        transformed_y_max = window_size

                elif key == 6 or key == 7:
                    if transformed_x_min >= window_size or transformed_y_max <= 0:
                        continue

                    if transformed_x_max > window_size:
                        transformed_x_max = window_size
                    if transformed_y_min < 0:
                        transformed_y_min = 0

                elif key == 8:
                    if transformed_x_max <= 0 or transformed_y_max <= 0:
                        continue

                    if transformed_x_min < 0:
                        transformed_x_min = 0
                    if transformed_y_min < 0:
                        transformed_y_min = 0

                transformed_area = (transformed_y_max - transformed_y_min) * (transformed_x_max - transformed_x_min)

                if transformed_area / float(org_area) < threshold:
                    continue

                xmlFileName = basefilename + '-' + str(y_val_pos) + '-' + str(x_val_pos) + '.xml'
                if not xmlFileName in xmlMap:
                    baseXML = ET.fromstring(template)
                    baseXML.find('filename').text = xmlFileName
                    baseXML.find('size/width').text = str(window_size)
                    baseXML.find('size/height').text = str(window_size)
                    xml = ET.ElementTree(baseXML)
                    xmlMap[xmlFileName] = xml
                else:
                    xml = xmlMap[xmlFileName]

                newObj = ET.fromstring(objectTemplate)
                newObj.find('name').text = name
                newObj.find('bndbox/xmin').text = str(transformed_x_min)
                newObj.find('bndbox/xmax').text = str(transformed_x_max)
                newObj.find('bndbox/ymin').text = str(transformed_y_min)
                newObj.find('bndbox/ymax').text = str(transformed_y_max)
                xml.getroot().append(newObj)
                # xmlFileTree.write(sys.stdout)


        # create xml files
        xml_list = []
        for f, x in xmlMap.items():
            xml_list.append(f)
            x.write(os.path.join(results_dir, f), encoding='utf-8')

        print("  - Number of XML files: " + str(len(xmlMap)) + " " + str(xml_list) )


        # divide org image
        print("  - Number of images: " + str(num_vert * num_horizon))
        for i in range(num_vert):
            for j in range(num_horizon):
                height_start = i * (window_size - margin)
                height_end = height_start + window_size
                width_start = j * (window_size - margin)
                width_end = width_start + window_size

                cv2.rectangle(sample_img, (width_start, height_start), (width_end, height_end), square_block_color, square_block_width)
                cv2.putText(sample_img, str(i) + '-' + str(j), ((width_end - width_start)/3 + width_start, (height_end - height_start)/2 + height_start), cv2.FONT_HERSHEY_PLAIN, 1, square_block_color, square_block_width, cv2.LINE_AA)

                clopped = processed_img[height_start:height_end, width_start:width_end]
                newfile = basefilename + '-' + str(i) + '-' + str(j) + '.jpg'
                cv2.imwrite(os.path.join('results', newfile), clopped)

        cv2.imwrite(os.path.join('results', 'sample-' + filename), sample_img)

    elapsed_time = time.time() - start

    print('\n')
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")