import Orange
import re
import itertools

"""
This tests the idea to build the level 1 learners from the
features independently so that the size of the l2 domain space is

#num_features x #num_base_learners

This idea sucks. It takes forever and does not have good a performance.

"""


def construct_featurewise_subdomains(domain):
    dd = {}
    for f in domain.features:
        ma = re.match('^(f[0-9]*)\(', f.name)
        if not ma:
            raise RuntimeError('invalid featurename: ' + f.name)
        fgr = ma.groups(1)
        l = dd.setdefault(fgr, [])
        l.append(f.name)
    
    subdomains = []
    for key, attrs in sorted(dd.iteritems()):
        attrs.append(domain.class_var.name)
        subdomains.append( (key, Orange.data.Domain(attrs, domain)) )
        
    return subdomains

class FeaturewiseStackedClassificationLearner(Orange.classification.Learner):
    """
    """
    def __new__(cls, learners, data=None, weight=0, **kwds):
        if data is None:
            self = Orange.classification.Learner.__new__(cls)
            return self
        else:
            self = cls(learners, **kwds)
            return self(data, weight)

    def __init__(self, learners, meta_learner, folds=10, name='featurewise_stacking'):
        self.learners = learners
        self.meta_learner = meta_learner
        self.name = name
        self.folds = folds

    def __call__(self, data, weight=0):
        assert isinstance(data.domain.class_var, Orange.feature.Continuous)
        subdomains = construct_featurewise_subdomains(data.domain)
        
        # [ f1_l1, f1_l2, f2_l1, f2_l2]
        features = [ Orange.feature.Continuous("%s,%d" % (sub_name, il))
                        for sub_name, il in itertools.product( 
                                    [name for name, sub in subdomains], 
                                    range(len(self.learners))
                                    )]
        classifiers = []
        random_seed = 0
        domainwise = []
        
        def cbf():
            print "one fold out of",  self.folds
            
        
        for name, subdomain in subdomains:
            print '-----------------'
            print "subdomain:", name
            # table with feature 1
            subtable = Orange.data.Table(subdomain, data)
            classifiers.append( [ l(subtable, weight) for l in self.learners ] )
                
            res = Orange.evaluation.testing.cross_validation(self.learners, subtable,
                                                             self.folds,
                                                             random_generator=random_seed,
                                                             callback=cbf)
            print "subdomain:", name
            scores = Orange.evaluation.scoring.RMSE(res)
            for sc, le in zip(scores, self.learners):
                print le.name, sc 
                
            l = []
            for r in res.results:
                l.append( (r.classes, r.actual_class) )
            domainwise.append( l )
        
        domain = Orange.data.Domain(features + [data.domain.class_var])
        p_data = Orange.data.Table(domain)
        
        assert len(data) ==  min(map(len, domainwise)) == max(map(len, domainwise))
        assert len(domainwise)
        
        for resi in xrange(len(domainwise[0])):
            one_data = []
            clss = []
            for fi in xrange(len(domainwise)):
                feats, cls = domainwise[fi][resi]
                one_data.extend(feats)
                clss.append(cls)
            assert len(clss)
            assert all( cl == clss[0] for cl in clss )
            one_data.append(clss[0])
            
            assert len(one_data) == len(domain)
            
            p_data.append(one_data) 
        
        assert sum(map(len, classifiers)) == len(features)
        
        self.p_data = p_data
        meta_classifier = self.meta_learner(p_data)
        
        return FeaturewiseStackedClassifier(
            classifiers, meta_classifier, subdomains, name=self.name)

class FeaturewiseStackedClassifier:
    """
    """
    def __init__(self, classifiers, meta_classifier, subdomains, **kwds):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.subdomains = subdomains
        self.domain = Orange.data.Domain(self.meta_classifier.domain.features, False)
        self.__dict__.update(kwds)

    def __call__(self, instance, resultType=Orange.core.GetValue):
        values = []
        
        for classifiers, (name, subdomain) in zip(self.classifiers, self.subdomains):
            subinstance = Orange.data.Instance(subdomain, instance)
            values.extend( [
                float(cl(instance, Orange.core.GetValue))
                    for cl in classifiers  ] )
        
        assert len(values) == len(self.domain)
        ps = Orange.data.Instance(self.domain, values)
        return self.meta_classifier(ps, resultType)

def test_stack_reggression():
    base_learners = [
        Orange.regression.linear.LinearRegressionLearner(name='linear'), 
        Orange.regression.pls.PLSRegressionLearner(name='PLS'),
        Orange.classification.knn.kNNLearner(k=20,  name='knn 20'), 
        Orange.classification.knn.kNNLearner(k=30,  name='knn 30')
        #Orange.ensemble.forest.RandomForestLearner(name='random forrest')
    ]
    
    stack = FeaturewiseStackedClassificationLearner(base_learners, 
                                             #meta_learner=Orange.ensemble.forest.RandomForestLearner(name='meta random forrest'), 
                                             meta_learner=Orange.classification.knn.kNNLearner(k=20,  name='meta knn 20'),
                                             folds=4, 
                                             name='stacking')
    
    learners =    [ stack ] + base_learners
    
    data = Orange.data.Table('feature_wise.tab')
    res = Orange.evaluation.testing.cross_validation(learners, data, folds=4)
    
    print "\n".join(["%8s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.RMSE(res), learners)])
    
if __name__ == "__main__":
    test_stack_reggression()
    