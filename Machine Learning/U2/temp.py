import math

global clusters
clusters = []

def rangeQuery(i, eps, dataset):
    # get all distances from other points to i-th point
    i_distances = euclidean_distance(dataset, dataset[i])
    
    # get all neighbors with the certain condition
    neighbors = np.extract(i_distances <= eps, i_distances)
    
    indices = np.where(i_distances <= eps)
    
    return neighbors, indices

def DBSCAN_algo(eps, minPts, dataset):
    global clusters
    
    # get number of points
    length = dataset.shape[0]
    labels = np.zeros([length, 1])
    cores = []
    cores_counter = 0
    
    #print("length: " + str(length))
    
    for i in range(length):
        '''
        0 stands for the default value(undefined)
        -1 stands for noise
        for all numbers n greater than 0 stands for the id of the core
        '''
        
        if(labels[i] != 0):
            continue
        
        # get its neigbors(distance not pairs) and indices
        i_neighbors, indices = rangeQuery(i, eps, dataset)   
            
        if(i_neighbors.shape[0] < minPts):
            labels[i] = -1
            continue
            
        cores_counter += 1
        labels[i] = cores_counter
        #print("labels[i]: " + str(labels[i]) + "and i: " + str(i))
        
        #print("cores_counter" + str(cores_counter))
        
        # remove the core itself
        index = np.where(i_neighbors == 0)
        
        seedSet = np.delete(i_neighbors, index[0])
        
        for j in range(seedSet.shape[0]):
            if(labels[j] == -1):
                labels[j] = labels[i]
                
            if(labels[j] != 0):
                continue
                    
            labels[j] = labels[i]
            j_neighbors, temp = rangeQuery(j, eps, dataset)
            if(j_neighbors.shape[0] >= minPts):
                np.concatenate((seedSet, j_neighbors), axis=0) 
                
        clusters.extend([indices, dataset[indices]]) 

        #print("#-----------------------------------------------------------#")
    
    #-----------------------------------------------------------#
    # np.concatenate((np.asarray(clusters), dataset[indices]), axis = 0)
    # get all labels of its neigbors
    neighbor_labels = np.concatenate([labels[clusters[0]], labels[clusters[2]]], axis = 0)
    print(neighbor_labels.shape)
#     neighbor_pairs = dataset[indices]
    
    #labels = np.hstack((np.zeros(clusters),np.ones(clusters)))
    #plt.scatter(clusters[:,0], clusters[:,1])
    
    #plt.axis('equal')
    #plt.show()
    #print(dataset[indices])
    #print("length of clusters: " + str(len(clusters)))
    #print(clusters)
    return labels


#(6,353) (4,194)
a = 6
b = 353

my_label = DBSCAN_algo(a, b, dataset)
print("max: " + str(np.max(DBSCAN_algo(a, b, dataset))))
#print(my_label)
counter = 0
for i in range(my_label.shape[0]):
    if(my_label[i][0] == 1):
        my_label[i][0] = 0
        counter += 1
    elif((my_label[i][0] == -1)):
        my_label[i][0] = 1
        
#    else:
        #print(my_label[i][0])
#     if(my_label[i][0] != 1 and my_label[i][0] != -1):
#         print("strange value: " + str(my_label[i][0]))
#         counter += 1
counter2 = 0
for i in range(labels.shape[0]):        
    if(labels[i] == 0):
        counter2 += 1
   
# print(counter)
# print(counter2)
# print(np.mean(my_label==labels))
# print(my_label.shape)
# print(labels.shape)
print(1-abs(counter - counter2)/1000)
