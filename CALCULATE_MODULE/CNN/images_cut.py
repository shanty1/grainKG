# ! /usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
from PIL import Image

def cut_image(image, row_num, col_num):
    width, height = image.size
    item_width = width / row_num
    item_height = height / col_num

    box_list = []
    for row in range(0, row_num):
        for col in range(0, col_num):
            box = (col * item_width, row * item_height, (col + 1) * item_width, (row + 1) * item_height)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list

def save_images(image_list, save_dir, father_file):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index, image in enumerate(image_list):
        save_file = os.path.join(save_dir, "{}({}).png".format(father_file[:father_file.rindex(".")], index+1))
        image.save(save_file, 'PNG')


def main(cut_row=3, cut_col=3, image_root="./image_data/full_graph"):
    """
        执行切割图片
        【注意文件目录必须满足】
        : 样本编号/样本块号/样本块的EBSD图文件 （每个子目录下应只有一个文件，因为一个样本块只有一张EBSD图）
    """
    image_root = os.path.join(os.path.dirname(__file__), image_root)
    save_root = image_root[:image_root.rindex("/full_graph")] + os.sep + "subgraph" + str(cut_row*cut_col) # ./subgraph{sub_pic_num}
    save_dir = save_root + os.sep + "{}" + os.sep + "{}"  # 格式：./subgraph{sub_pic_num}/sample_id/block_id

    for sample_dir in os.listdir(image_root):
        for block_dir in os.listdir(os.path.join(image_root, sample_dir)):
            for file in os.listdir(os.path.join(image_root, sample_dir, block_dir)):
                image = Image.open(os.path.join(image_root, sample_dir, block_dir, file))
                image_list = cut_image(image, cut_row, cut_col)
                save_images(image_list, save_dir.format(sample_dir, block_dir), file)
    print("切图完成！分割后的图片存储在：{}，路径格式为“样本编号/块编号/原文件名(子图编号).png”".format(os.path.relpath(save_root)))

if __name__ == '__main__':
    main(cut_row=3, cut_col=3)
