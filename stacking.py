import Orange

class StackedClassificationLearner(Orange.classification.Learner):
    """Stacking by inference of meta classifier from class probability estimates
    on cross-validation held-out data for level-0 classifiers developed on held-in data sets.

    :param learners: level-0 learners.
    :type learners: list

    :param meta_learner: meta learner (default: :class:`~Orange.classification.bayes.NaiveLearner`).
    :type meta_learner: :class:`~Orange.classification.Learner`

    :param folds: number of iterations (folds) of cross-validation to assemble class probability data for meta learner.

    :param name: learner name (default: stacking).
    :type name: string

    :rtype: :class:`~Orange.ensemble.stacking.StackedClassificationLearner` or
        :class:`~Orange.ensemble.stacking.StackedClassifier`
    """
    def __new__(cls, learners, data=None, weight=0, **kwds):
        if data is None:
            self = Orange.classification.Learner.__new__(cls)
            return self
        else:
            self = cls(learners, **kwds)
            return self(data, weight)

    def __init__(self, learners, meta_learner=Orange.classification.bayes.NaiveLearner(), folds=10, name='stacking'):
        self.learners = learners
        self.meta_learner = meta_learner
        self.name = name
        self.folds = folds

    def __call__(self, data, weight=0):
        res = Orange.evaluation.testing.cross_validation(self.learners, data, self.folds)
        
        if isinstance(data.domain.class_var, Orange.feature.Discrete):
            features = [Orange.feature.Continuous("%d" % i) for i in range(len(self.learners) * (len(data.domain.class_var.values) - 1))]
            
        elif isinstance(data.domain.class_var, Orange.feature.Continuous):
            features = [Orange.feature.Continuous("%d" % i) for i in range(len(self.learners))]
            
        else:
            raise RuntimeError("unknown class_var type")
            
        domain = Orange.data.Domain(features + [data.domain.class_var])
        p_data = Orange.data.Table(domain)
        
        if isinstance(data.domain.class_var, Orange.feature.Discrete):
            for r in res.results:
                p_data.append([p for ps in r.probabilities for p in list(ps)[:-1]] + [r.actual_class])
        else:
            assert isinstance(data.domain.class_var, Orange.feature.Continuous)
            
            for r in res.results:
                p_data.append( r.classes + [r.actual_class])
            
            assert len(p_data[0]) == len(domain)
            
        meta_classifier = self.meta_learner(p_data)
        classifiers = [l(data, weight) for l in self.learners]
        
        #feature_domain = Orange.data.Domain(features)
        return StackedClassifier(classifiers, meta_classifier, name=self.name, meta_domain=p_data.domain)

class StackedClassifier:
    """
    A classifier for stacking. Uses a set of level-0 classifiers to induce class probabilities, which
    are an input to a meta-classifier to predict class probability for a given data instance.

    :param classifiers: a list of level-0 classifiers.
    :type classifiers: list

    :param meta_classifier: meta-classifier.
    :type meta_classifier: :class:`~Orange.classification.Classifier`
    """
    def __init__(self, classifiers, meta_classifier, meta_domain, **kwds):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.meta_domain = meta_domain
        self.domain = Orange.data.Domain(self.meta_domain.features, False)
        self.__dict__.update(kwds)

    def __call__(self, instance, resultType=Orange.core.GetValue):
        if isinstance(self.meta_domain.class_var, Orange.feature.Discrete):
        #if isinstance(self.meta_classifier.domain.class_var, Orange.feature.Discrete):
            ps = Orange.data.Instance(self.domain, [p for cl in self.classifiers for p in list(cl(instance, Orange.core.GetProbabilities))[:-1]])
        else:
            assert isinstance(self.meta_domain.class_var, Orange.feature.Continuous)
            #assert isinstance(self.meta_classifier.domain.class_var, Orange.feature.Continuous)
            ps = Orange.data.Instance(self.domain, [ float(cl(instance, Orange.core.GetValue)) for cl in self.classifiers ])
        
        return self.meta_classifier(ps, resultType)


##
## tests and examples
##

def test_stack_reggression():
    base_learners = [
        Orange.regression.linear.LinearRegressionLearner(name='linear'), 
        Orange.regression.pls.PLSRegressionLearner(name='PLS'),
        Orange.classification.knn.kNNLearner(k=20,  name='knn 20'), 
        Orange.classification.knn.kNNLearner(k=30,  name='knn 30')
        #Orange.ensemble.forest.RandomForestLearner(name='random forrest')
    ]
    
    stack = StackedClassificationLearner(base_learners, 
                                             #meta_learner=Orange.ensemble.forest.RandomForestLearner(name='meta random forrest'), 
                                             meta_learner=Orange.classification.knn.kNNLearner(k=20,  name='meta knn 20'),
                                             folds=10, 
                                             name='stacking')
    
    learners =    [ stack ] + base_learners
    
    data = Orange.data.Table("housing")
    res = Orange.evaluation.testing.cross_validation(learners, data, folds=10)
    
    print "\n".join(["%8s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.RMSE(res), learners)])
    
def test_stack_classification():
    data = Orange.data.Table("promoters")

    bayes = Orange.classification.bayes.NaiveLearner(name="bayes")
    tree = Orange.classification.tree.SimpleTreeLearner(name="tree")
    lin = Orange.classification.svm.LinearLearner(name="lr")
    knn = Orange.classification.knn.kNNLearner(name="knn")
    
    base_learners = [bayes, tree, lin, knn]
    stack = StackedClassificationLearner(base_learners)
    
    learners = [stack, bayes, tree, lin, knn]
    res = Orange.evaluation.testing.cross_validation(learners, data, 3)
    print "\n".join(["%8s: %5.3f" % (l.name, r) for r, l in zip(Orange.evaluation.scoring.CA(res), learners)])
    
if __name__ == "__main__":
    test_stack_reggression()
    #test_stack_classification()
    
