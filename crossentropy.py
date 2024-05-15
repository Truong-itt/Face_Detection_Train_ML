import numpy as np



P = [0.2,0.4,0.4]
Q = [0.1,0.1,0.8]

#  entropy , crossentropy : do khong dong deu cua hai tham so , kl drivergence : he so phan tan

entropy_P = -sum([P[i] * np.log(P[i]) for i in range(len(P))])
entropy_Q = -sum([Q[i] * np.log(Q[i]) for i in range(len(Q))])
print(entropy_P, entropy_Q)

# crossentropy_PQ
crossentropy_PQ =  -sum([P[i] * np.log(Q[i]) for i in range(len(P))])
crossentropy_QP = -sum([Q[i] * np.log(P[i]) for i in range(len(Q))])
print(crossentropy_PQ, crossentropy_QP)

# klDrivergence_pq = crossentropy_PQ - crossentropy_P
klDrivergence_pq = sum([P[i] * np.log(P[i]/Q[i]) for i in range(len(P))])
klDrivergence_qp = sum([Q[i] * np.log(Q[i]/P[i]) for i in range(len(Q))])

print(klDrivergence_pq, klDrivergence_qp)
print(klDrivergence_pq," = ", crossentropy_PQ - entropy_P)
#  ta co the chung minh nguo lai 
print(klDrivergence_qp," = ", crossentropy_QP - entropy_Q)



# MSE : mean square error
# y du doan - y thuc ta binh phuong khong anh chia cho tong so y
#  -1 khi so phan tu it khong can -1 khi so phan tu du nhieu
sum((y_hat - y)**2) / (sophantu -1 ) 


# Likelyhood : xac suat
goood = [0, 1, 0, 0]
predict = [0.1,0.6,0.2, 0.1]
#  0.9 * 0.6 * 0.8 * 0.9 =0.38

#  Crossentropy : do khong dong deu cua hai tham so

y_good = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
predict = [0.1, 0.1, 0.2, 0.3, 0.1, 0.2]
#  cong thuc chinh 
cross_entropy_y_yhat = -sum([y_good[i] * np.log(predict[i]) for i in range(len(y_good))])
#  do thuc hiẹn encoding ơ bo train 
cross_entropy_y_yhat = 1*np.log(yhat[i])   # can co so i de xac dinh vi tri cua yhatdu doan