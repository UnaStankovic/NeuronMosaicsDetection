import numpy as np
import pandas as pd

from sklearn import cluster

ARR_LEN = 230
DESC_LEN = 128

#function takes vector whose each component is number of cluster of the descriptor
#the length of the vector is a number of descriptors of the image
#return value is a vector of length 230 containing frequencies of each cluster between
#the descriptor of that image
def make_array_230(descriptor_predicted):
    result = np.zeros((ARR_LEN))
    unique, counts = np.unique(descriptor_predicted, return_counts=True)
    dict_occurs = dict(zip(unique, counts))
    for key, val in dict_occurs.iteritems():
        result[key] = val
    return result


def construct_all_230(data, output_filename='sift_230.csv'):
    indexes = [str(i) for i in range(0,DESC_LEN)]
    desc_data = data[indexes]

    kmeans = cluster.KMeans(n_clusters=ARR_LEN, random_state=7)
    kmeans.fit(desc_data)

    image_indexes_dict = data.groupby(['image_name']).groups
    
    all_230 = np.empty((0, ARR_LEN+2))
    column_list = ['image_name', 'class'] + range(0,ARR_LEN)

    data_230 = pd.DataFrame(columns=column_list)
    for img_name, indexes in image_indexes_dict.iteritems():
        #after grouping we make a 2D array of all rows containing descriptors for current image
        desc_mapped = np.array([desc_data.iloc[i] for i in indexes])
        
        #apply kmeans to the current image descriptors
        desc_predicted = kmeans.predict(desc_mapped)
        
        #count frequences of all clusters in the descriptors of current image
        array_230 = make_array_230(desc_predicted)
        
        #take the class of the image (True/False)
        class_name = np.array(data[data['image_name'] == img_name]['class'])
        class_name = class_name[0]
        
        #make new row consisting of image_name, class and array of length 230
        new_row = np.hstack(([img_name, class_name], array_230))
        
        #append that row to current array
        all_230 = np.vstack((all_230, new_row))
    data_230 = pd.DataFrame(all_230, columns=column_list)
    data_230.to_csv(output_filename)
	
	
data = pd.read_csv('sift_data.csv')
construct_all_230(data)


