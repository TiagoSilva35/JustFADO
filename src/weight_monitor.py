import numpy as np

class ClassWeightMonitor:
    def __init__(self, total_num_samples, num_classes=2, alpha=1.0, _lambda=1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.total_num_samples = total_num_samples
        self._lambda = _lambda
        self.class_counts = np.zeros(num_classes, dtype=np.float32)
        self.class_weights = np.ones(num_classes, dtype=np.float32)
    # update class weights based on the batch (size 1 in online setting)    
    def update_weights(self, sample_label):
        N_t = self.total_num_samples # total number of samples seen so far
        num_classes = self.num_classes
        alpha = self.alpha
        N_t_classes = self.class_counts  
        self.class_counts[sample_label] += 1
        for i in range(num_classes):
            if N_t_classes[i] > 0:
                N_c_lambda_t = N_t_classes[i]*self._lambda + 1*(i == sample_label)
                self.class_weights[i] = N_t / (self.num_classes*((N_c_lambda_t) + alpha))
            else:
                self.class_weights[i] = 1.0

    def get_stats(self):
        print("Class Counts:", self.class_counts)
        print("Class Weights:", self.class_weights)
        print("Total Samples:", self.total_num_samples)
