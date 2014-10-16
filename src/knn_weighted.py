import Orange
import numpy
import random
import math
import logging

class KnnWeightedLearner(Orange.classification.Learner):
    def __new__(cls, examples=None, **kwargs):
        learner = Orange.classification.Learner.__new__(cls, **kwargs)
        if examples:
            # force init and return classifier
            learner.__init__(**kwargs)
            return learner.__call__(examples)
        else:
            # invoke init
            return learner

    def __init__(self,
                 k=0,
                 alpha=1, 
                 distance_constructor=Orange.distance.Euclidean(),
                 exp_weight=False, 
                 name='knn weighted'):
        self.k = k
        self.alpha = alpha
        self.distance_constructor = distance_constructor
        self.name = name
        self.exp_weight = exp_weight
        
    def __call__(self, data,  weight=0):
        assert isinstance(data.domain.class_var, Orange.feature.Continuous)
        
        if not data.domain.class_var:
            raise ValueError('classless domain')
        
        fnc = Orange.classification.knn.FindNearestConstructor()
        fnc.distance_constructor = self.distance_constructor
        did = Orange.feature.Descriptor.new_meta_id()
        
        fn = fnc(data, 0, did)
        
        k = self.k
        if k == 0:
            k = int(math.sqrt( len(data)))
        
        return KnnWeightedClassifier(data.domain, k, fn, self.alpha,  self.exp_weight)

## FIXME Orange.classification.Classifier (which should be there)
## is commented because if it is not, pickling does not work...
class KnnWeightedClassifier: #(Orange.classification.Classifier):
    def __init__(self, domain, k, find_nearest, alpha,  exp_weight):
        self.domain = domain
        self.domain_f = Orange.data.Domain(domain.features)
        self.k = k
        self.find_nearest = find_nearest
        self.alpha = alpha
        self.exp_weight = exp_weight
        
        
    def __call__(self, instance, resultType=Orange.core.GetValue):
        if not instance.domain != self.domain_f:
            raise ValueError("instance has wrong domain")
        
        def get_dist(nb):
            return 
        
        nbs = self.find_nearest(instance, self.k)
        
        # distances
        dsts = numpy.array( [ nb[self.find_nearest.distance_ID]
                                for nb in nbs ])
        # target variables
        clss = numpy.array( [ nb.get_class()
                                for nb in nbs ])
        if 0 in dsts:
            #logging.warn('0 in distances, add epsilon')
            dsts +=  1e-5
            
        # compute the weights
        if not self.exp_weight:
            # inversely proportional
            w = dsts ** ( - self.alpha ) 
        else:
            assert 0.0 < self.alpha < 1.0
            # weird exp.
            w = self.alpha ** dsts
        
        # normalize to 1
        w = w / w.sum()
        # lin combination
        res = (w * clss).sum()
        
        value = self.domain.class_var(res)
        
        dist = Orange.statistics.distribution.Continuous(self.domain.class_var)
        dist[value] = 1.
        
        if resultType == Orange.core.GetValue:
            return value
        if resultType == Orange.core.GetProbabilities:
            return dist
        return (value,  dist)

##
## tests and examples
##
        
def test_housing():
    data = Orange.data.Table("housing")

    from fann_neural import FannNeuralLearner
    
    learners = [
                 KnnWeightedLearner( k=4, alpha=2 ),
                 KnnWeightedLearner( k=4, alpha=1 ),
                 KnnWeightedLearner( k=4, alpha=0 ),
                 Orange.classification.knn.kNNLearner(k=4, name='knn 4'), 
                 Orange.classification.knn.kNNLearner(k=4, name='knn 4, False', rank_weight=False), 
                 ]
    
    cv = Orange.evaluation.testing.cross_validation(learners, data, folds=5)
    
    for l, score in zip(learners, Orange.evaluation.scoring.RMSE(cv)):
        print "%s: %.8f" % (l.name , score) 

def plot_im() :
    """
    this is somewhat inspired by
    http://quasiphysics.wordpress.com/2011/12/13/visualizing-k-nearest-neighbor-regression/
    
    """
    import Image, ImageDraw
        
    attrs = [ Orange.feature.Continuous(name) for name in ['X', 'Y', 'color'] ]
    insts = []    
    random.seed(50)
    for num in xrange(10):
        color = 255 * int(2 * random.random() )
        
        def get_point():
            return 0.25 + random.random() / 2
        
        x, y = get_point(), get_point()
        
        insts.append([x, y, color])
    
    data = Orange.data.Table(Orange.data.Domain(attrs), insts)
    
    def get_inst(a, b):
        return Orange.data.Instance(Orange.data.Domain(data.domain.features),[a, b])
    
    for k in xrange(1, 11):
        for alpha in xrange(4):
            for dist in [Orange.distance.Euclidean() ]: #, Orange.distance.Manhattan()   ]:
                
                    l = KnnWeightedLearner( k=k, alpha=alpha, distance_constructor=dist)
                    #l = Orange.classification.knn.kNNLearner( k=k )
                    knn = l(data)
                    
                    size = 200
                    
                    a = []
                    for X in xrange(size):
                        for Y in xrange(size):
                            val = int(knn(get_inst(float(X)/size, float(Y)/size)))
                            a.append(val)
                            
                    arr = numpy.array(a,  dtype=numpy.uint8 )
                    arr = arr.reshape((size, size))
                    
                    im = Image.fromarray(arr).convert("RGB")
                    for inst in data:
                        y, x = int(size * inst[0] ),  int(size * inst[1])
                        color = int(inst[2])
                        
                        draw = ImageDraw.Draw(im)
                        r = size / 50
                        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 0, 0), fill=(color, color, color))
                    
                    fn = "knn_w/k=%d_alpha=%d_dist=%s.ppm" % (k, alpha, dist.name)
                    print fn
                    im.save(fn)
                    #im.show()
    
if __name__ == "__main__":
    #plot_im()
    test_housing()
    
    pass
    
