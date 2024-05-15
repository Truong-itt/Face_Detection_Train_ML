import numpy as np
# dau la mot matrix 3d nhu sau 
matrix_3d = np.random.rand(3, 3, 3)
print(matrix_3d)


bien = np.array(
    [[[5, 5 ,5],
  [5  ,5 ,5],
  [5 ,5 ,5]],

#  [[5  ,5  ,5],
#   [5 ,5 ,5],
#   [5 ,5 ,5 ]],

 [[5 ,5  ,5],
  [5 ,5 ,5 ],
  [5  ,5 ,5 ]]])


print(bien)
print(bien.shape)
# 2 3 3  : trong do 2 la so luong cua matrix 3d so lop layer, 3 3 la kich thuoc cua matrix 2d
# nguoc voi  anh la co so lop player cuoi cung la chieu sau cua anh