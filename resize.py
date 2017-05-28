import glob
import scipy as sp
from PIL import Image
import os, sys
from resizeimage import resizeimage


size = 28, 28
ite = 0
'''
for file in glob.iglob('./Realism/*.*'):
    a = open(file, 'rw')
    img = Image.open(a)#.convert('L')
    #print file
    if img.width <= img.height:
        small = img.width
    else:
        small = img.height
    img = resizeimage.resize_crop(img, [small, small])
    img = img.resize((64, 64), Image.ANTIALIAS)
    out = "./realism_output/" + str(ite) + ".jpeg"
    print out
    img.save(out, img.format)
    a.close()
    ite += 1
'''

for file in glob.iglob('./Unabstract/*.*'):
    a = open(file, 'rw')
    img = Image.open(a)#.convert('L')
    #print file
    if img.width <= img.height:
        small = img.width
    else:
        small = img.height
    img = resizeimage.resize_crop(img, [small, small])
    img = img.resize((28, 28), Image.ANTIALIAS)
    out = "./unabstract_ouput/" + str(ite) + "1.jpeg"
    print out
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./unabstract_ouput/" + str(ite) + "2.jpeg"
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./unabstract_ouput/" + str(ite) + "3.jpeg"
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./unabstract_ouput/" + str(ite) + "4.jpeg"
    img.save(out, img.format)
    a.close()
    ite += 1


'''

for file in glob.iglob('./Expressionism/*.*'):
    a = open(file, 'rw')
    img = Image.open(a)
    #print file
    if img.width <= img.height:
        small = img.width
    else:
        small = img.height
    img = resizeimage.resize_crop(img, [small, small])
    img = img.resize((28, 28), Image.ANTIALIAS)
    out = "./expressionism_output/" + str(ite) + ".jpeg"
    print out
    img.save(out, img.format)
    a.close()
    ite += 1
'''
for file in glob.iglob('./Abstract/*.*'):
    a = open(file, 'rw')
    img = Image.open(a)
    #print file
    if img.width <= img.height:
        small = img.width
    else:
        small = img.height
    img = resizeimage.resize_crop(img, [small, small])
    img = img.resize((28, 28), Image.ANTIALIAS)
    out = "./abstract_output/" + str(ite) + "1.jpeg"
    print out
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./abstract_output/" + str(ite) + "2.jpeg"
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./abstract_output/" + str(ite) + "3.jpeg"
    img.save(out, img.format)

    img = img.rotate(90)
    out = "./abstract_output/" + str(ite) + "4.jpeg"
    img.save(out, img.format)
    a.close()
    ite += 1
