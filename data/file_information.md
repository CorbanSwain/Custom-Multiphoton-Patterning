## test_z_mask_01
* shape: (128, 128, 24)
* unknown, some random trianlge and dots

## test_z_mask_02
* shape: (128, 128, 5)
* z_0 = 250 
* z_4 = 254
* z_incriment = 1
* pages
  * 0 = blank
  * 1 = single mark at (0, 0)
  * 2 = single mark at (1, 0)
  * 3 = single mark at (118, 121)
  * 4 = blank

## test_z_mask_03
* shape: (256, 256, 5)
* z_0 = 250 
* z_4 = 254
* z_incriment = 1
* pages
  * 0 = blank
  * 1 = single mark at (0, 0)
  * 2 = single mark at (1, 0)
  * 3 = single mark at (118, 121)
  * 4 = blank


## test_z_mask_04
* shape: (128, 128, 5)
* z_0 = 250 
* z_4 = 259
* z_incriment = 1
* pages
  * 0 = blank
  * 1 = blank
  * 2 = single mark at (0, 0)
  * 3 = single mark at (1, 0)
  * 4 = single mark at (0, 1)
  * 5 = single mark at (2, 0)
  * 6 = single mark at (3, 0)
  * 7 = 5 marks from (0, 0) to (4, 0)
  * 8 = 3 marks at (0, 0), (1, 0), (0, 1)
  * 9 = single mark at (1, 1)

## test_z_mask_05
* shape: (128, 128, 5)
* z_0 = 250 
* z_4 = 259
* z_incriment = 1
* pages
  * 0 = blank
  * 1 = blank
  * 2 = rect from (0, 0) to (5, 0)
  * 3 = rect from (0, 0) to (5, 1)
  * 4 = rect from (0, 0) to (5, 2)
  * 5 = rect from (0, 0) to (5, 3)
  * 6 = rect from (0, 0) to (7, 3)
  * 7 = rect from (2, 0) to (9, 3)
  * 8 = rect from (2, 2) to (9, 5)
  * 9 = rect from (2, 2) to (9, 5), (12, 20) to (17, 24), point at (23, 12), point at (33, 4)


## test_z_mask_06
* shape: (128, 128, 5)
* z_0 = 250 
* z_4 = 259
* z_incriment = 1
* pages
  * 0 = blank
  * 1 = blank
  * 2 = all selected
  * 3 = all selected
  * 4 = blank
  * 5 = blank
  * 6 = blank
  * 7 = blank
  * 8 = blank
  * 9 = blank

## test_z_mask_07
* shape: (64, 64, 10)
* pages
  * 0 = blank
  * 1 = blank 
  * 2 = rect form (0, 0) to (0, 0)
  * 3 = rect from (0, 0) to (1, 0)
  * 4 = rect from (0, 0) to (2, 0)
  * 5 = rect from (1, 0) to (5, 1)
  * 6 = rect from (1, 0) to (5, 1); rect from (10, 1) to (15, 1)
  * 7 = rect from (0, 0) to (0, 8)
  * 8 = rect from (3, 4) to (16, 14); rect from (38, 11) to (41, 14); rect from (3, 19) to (11, 26)
  * 9 = blank

## test_z_mask_08
* shape: (128, 128, 5)
* pages
  * 0 = blank
  * 1 = blank 
  * 2 = rect form (0, 0) to (0, 0)
  * 3 = rect from (0, 0) to (1, 0)
  * 4 = rect from (1, 0) to (5, 1); rect from (10, 1) to (15, 1)

## test_z_mask_09
* shape : (128, 128, 10)
* pages (point(s) at ...)
  * 0 = blank
  * 1 = blank
  * 2 = (0, 0)
  * 3 = (1, 0)
  * 4 = (1, 1)
  * 5 = (0, 0); (1, 1)
  * 6 = (0, 0); (1, 0)
  * 7 = (0, 0); (1, 1); (2, 2)
  * 8 = (3, 4); (5, 4); (7, 8)
  * 9 = (2, 7); (3, 7); (4, 7); (5, 7)

## test_z_mask_10
* shape: (1024, 1024)
* pages (point(s) at ...)
  * 0 = blank
  * 1 = blank
  * 2 = (0, 0)
  * 3 = (1, 0)
  * 4 = (0, 0); (1, 1)


