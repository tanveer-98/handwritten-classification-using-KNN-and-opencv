import cv2 as cv
import numpy as np

digits =cv.imread('D:/images/digits.png',cv.IMREAD_GRAYSCALE)
r,c=digits.shape[:2]
#test digits
test_digit=cv.imread('D:/images/test2.png',cv.IMREAD_GRAYSCALE)

#vsplit means vertical split

rows =np.vsplit(digits,50)

#seperating all the numbers and storing into cells
#so now we have all the 2500 numbers in 2500 cells 
#0-2499 

cells=[]
for row in rows:
    row_cells=np.hsplit(row,50)
    for cell in row_cells:
        cell=cell.flatten()
        cells.append(cell)
 #algorithm wont work for multiple dimension array
#so we need to convert it into single array with the help of flattening 
#cells is just a normal array 
#so we need to convert it into numpy as it is faster else the code wont work for opencv 
cells=np.array(cells,dtype=np.float32)     
#k for labeling or indexing each horizontal row

k=np.arange(10)
cells_labels=np.repeat(k,250)

'''
1 2 3 4 5 6 7 8 9 10
1 2 3 4 5 6 7 8 9 10
.
.
.
1 2 3 4 5 6 7 8 9 10

this is done 250 to generate the labels\
    
'''  
r,c=test_digit.shape[:2]
test=np.vsplit(test_digit,50)
test_cells=[]
for d in test:
    d=d.flatten()
    test_cells.append(d)
test_cells=np.array(test_cells,dtype=np.float32)


#KNN 
knn=cv.ml.KNearest_create()
knn.train(cells,cv.ml.ROW_SAMPLE,cells_labels)

#TESTING THE ALGORITHNM BY PASSING THE DIGITS
ret,result,neighbours,dist=knn.findNearest(test_cells,k=3)

print(result)
cv.imshow('numbers for test',test_digit)
cv.waitKey()
cv.destroyAllWindows()

