from scipy.spatial.distance import pdist, squareform
a=[1,2,3,5,4]
b=[1.1,4.9,3.2,5.5,4.3]
data=[a,b]
dist=pdist(data,'cosine')
print(dist)