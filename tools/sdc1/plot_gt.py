import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import xml.etree.ElementTree as ET
import pdb
from PIL import Image


for img_f in os.listdir("/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h_png"):
    if not img_f.endswith('.png'):
       continue
    file_name_d = img_f
    print(file_name_d)
    tree = ET.parse("/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h_png/"  + os.path.splitext(file_name_d)[0] + ".xml")
    root = tree.getroot()
    image = Image.open(os.path.join("/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h_png", img_f))
    im = cv2.imread(os.path.join("/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h_png", img_f))
    thickness = (image.size[0] + image.size[1]) // 300

    show_img_size = image.size[0]
    my_dpi = 96
    fig = plt.figure()
    fig.set_size_inches(show_img_size/my_dpi, show_img_size/my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim([0, image.size[1]])
    ax.set_ylim([image.size[0], 0])
    ax.imshow(im[:, :, (2, 1, 0)], aspect='auto')
    #colors = {'1s_1c':'r','2s_1c':'k','2s_2c':'m','2s_3c':'g', '3s_3c':'b'}

    for box in root.iter('bndbox'):
        x1 = float(box.find('xmin').text)
        y1 = float(box.find('ymin').text)
        x2 = float(box.find('xmax').text)
        y2 = float(box.find('ymax').text)

        x1 = x1
        y1 = y1
        x2 = x2
        y2 = y2

        #label = '{} {:.2f}'.format(predicted_class_re[i], score_re[i])

        top, left, bottom, right = y1, x1, y2, x2
        #top = max(0, np.floor(top + 0.5).astype('int32'))
        #left = max(0, np.floor(left + 0.5).astype('int32'))
        #bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        #right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        ax.add_patch(
            plt.Rectangle((left, top),
                          abs(left - right),
                          abs(top - bottom), fill=False,
                          edgecolor='g', linewidth=0.5)
                          )
        #ax.text(left, top-1, label,
        #             bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
        #              fontsize=4, color='y')
        plt.rcParams["font.family"] = "Times New Roman"
        outdir = "/o9000/MWA/GLEAM/data_challenge1/split_B1_1000h_png_gt_new"
    pngfile = os.path.splitext(file_name_d)[0] + "_pred.pdf"
    plt.savefig(os.path.join(outdir, pngfile))

