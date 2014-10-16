"""
Wrapper for the Fast Artificial Neural Network library:
    http://leenissen.dk/fann/wp/

This module mainly contains FannNeuralLearner and FannNeuralClassifier,
the classifier supports both classification (both normal and multilabel)
and reggresion.

The size of domains for Continuous classes is
limited by the range of activation functions of the neurons.

"""
import Orange
import Orange.core

import numpy
import tempfile
import itertools
import logging

from pyfann import libfann

__author__ = "Josef Moudrik"
__credits__ = [ 'Authors of the Fann library, http://leenissen.dk/fann/wp/' ]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Josef Moudrik"
__email__ = "J.Moudrik@gmail.com"


class FannNeuralNetPickable:
    def __init__(self, filename=None):
        self.ann = libfann.neural_net()
        if filename != None:
            self.ann = libfann.neural_net.create_from_file(filename)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ann']
        odict['fann_save'] = fake_file_call_f2s( self.ann.save )

        return odict

    def __setstate__(self, odict):
        ann = libfann.neural_net()
        fake_file_call_s2f( ann.create_from_file,
                            odict.pop('fann_save') )

        self.__dict__.update(odict)
        self.ann = ann

    def __getattr__(self, key):
        return self.ann.__getattribute__(key)

class FannNeuralLearner(Orange.classification.Learner):
    """
    """

    def __new__(cls, examples=None, name='Fann neural', **kwargs):
        self = Orange.classification.Learner.__new__(cls, **kwargs)
        if examples:
            self.__init__(**kwargs)
            return self.__call__(examples, **kwargs)
        else:
            return self

    def __init__(self, name='Fann neural',  **kwargs):
        """

        See:
        for parameters and methods of the ANN
            http://leenissen.dk/fann/html/files/fann-h.html

        for parameters and methods of the train data
            http://leenissen.dk/fann/html/files/fann_train-h.html
        """
        self.name = name
        # default parameters for the learner
        self.def_params = {
                 "nn_type":'standard',
                 # disable the check for data to be in <-1,1>
                 "allow_out_of_range":False,
                 "autorescale_output": False,
                 # dicts for setting properties of ann, and train data
                 "ann_prop":{},
                 "train_prop":{},
                 # custom postprocessing functions for more complicated modifications
                 # see the __call__ below
                 "ann_postprocess":None,  # will be called: ann_postprocess(ann)
                 "train_postprocess":None,   # will be called: train_postprocess(train_data)
                 # parameters:
                 # CREATION
                 "hidden_layers":[], # number of neurons in each of the hidden layers
                 # sparse
                 "connection_rate":0.5,
                 # TRAINING
                 "desired_error":0.0001,
                 # normal training
                 "max_epochs":2000,
                 "iterations_between_reports":0, # 0 turns it off
                 # cascade training
                 "max_neurons":20,
                 "neurons_between_reports":0, # 0 turns it off
        }
        self.def_params.update(kwargs)

    def __call__(self, data,
                 weight=None,
                 **kwargs ):
        """
        Learn from the given table of data instances.

        The learning proceeds as follows:
        1. data are transformed into pairs of input and output vectors, the size
           of these vector corresponding to number of neurons in input/output
           layers. The number of input neurons is determined by number of cols
           in Table.to_numpy, number of output neurons is as follows:
             1 output neuron for each Continuous class attribute (reggression)
             N output neurons for each Discrete class, where N is the number of
               possible class values.


            Because the omain of the neurons' output function is usually <-1,1>
            the reggression task only works if the data is scaled to this interval.
            If you wanna use the NN and you have different range of the output
            variables, you should do some scaling. This wrapper has one cannonical
            scaling available, if the autorescale_output option is set to True,
            the output is linearly scaled onto <-1,1> (min of the values to -1,
            max to 1, guys in the middle linearly in between). The Min and Max
            is learned from the training set, so if larger values are present
            later when testing, this will not work optimally. Though, it usually
            works well. (if this option is used, the output values from running
            the actual reggression are rescaled back, so this is transparent to the user)

        2. FANN training data struct (call it train_data) is made from these
           input/output pairs. The train_data is then posprocessed by:
                (a) for each iter pair (key, value) from the params.train_prop
                the train_data.key(value) is called. This is used to set up
                FANN properties of the train data, as specified in

                    http://leenissen.dk/fann/html/files/fann_train-h.html

                (b) if params.train_postprocess function is given, than the
                params.train_postprocess(train_data) is called. This param
                may be used to set up a hook for some complicated FANN
                train_data transformations.

        3. The Neural Network (ANN) is then created. Fann offers 3 types of
           network types, 'standard', 'sparse' and 'shortcut', as described in

                    http://leenissen.dk/fann/html/files/fann-h.html

           along with 'cascade' type (I've added; discussed in the point 4.,
           below) these can be specified in the params.nn_type.

           The network is postprocessed (similar to train_data postprocessing):
                (a) for each iter pair (key, value) from the params.ann_prop
                the ANN.key(value) is called. This is used to set up
                FANN properties of the network, as specified in the FANN reference.

                    for example, setting the kw parameter
                    ann_prop = {
                     'set_activation_function_hidden' : libfann.SIGMOID_STEPWISE ,
                     'set_activation_function_output' : libfann.SIGMOID_STEPWISE,
                     'set_training_algorithm' : FANN_TRAIN_QUICKPROP
                    }
                    will override the default activation function
                    libfann.SIGMOID_SYMMETRIC with its linear stepwise approximation,
                    and will change the default learning gradient learning
                    algorithm RPROP to QUICKPROP.

                (b) if params.ann_postprocess function is given, than the
                params.ann_postprocess(ANN) is called.

        4. The network is then trained. There are two different approaches to
           training in FANN:
                (a) fixed topology training: this is the "usual" way of training,
                the number of neurons and connections in the network is fixed, we
                only choose the learning algorithm which iteratively changes
                the weights.

                (b) cascade training (training with evolving topology): this
                approach starts with an empty network and adds promising neurons
                into the network. See

                    http://leenissen.dk/fann/html/files/fann_cascade-h.html

                for details. When using cascade training, the network type can
                only be the shortcut type with no hidden layers on the start.
                Here, in the FannNeuralLearner, you can specify that you want
                to do the cascade learning by setting the params.nn_type to
                'cascade'. This triggers the shortcut topology and trains using
                the FANN Cascade algorithm. Use params.nn_type = 'shortcut' if
                you want the standard fixed topology training.

        5. The classifier is returned. Surprisingly, huh? See the __doc__
        there for more stuff.

        """
        # params for this run of __call__ are the default Learner's params
        # overriden by the __call__ kwargs
        class Params(object):
            pass
        params = Params()
        params.__dict__.update(self.def_params)
        params.__dict__.update(kwargs)

        if not params.nn_type in ['standard',  'sparse',  'shortcut', 'cascade']:
            raise ValueError('Unknown network type "%s"'%params.nn_type)

        ## Create the training input/output pairs
        # Step 1 in the __call__.__doc__
        X, Y =  table_to_XY(data)

        def wrong_range(array):
            return not ((array >= -1.0) & (array <=  1.0)).all()

        # no scaling by default
        autoscaler = None
        if wrong_range(Y):
            if params.autorescale_output:
                lower = params.__dict__.get('autorescale_lower_bound', -1.0)
                upper = params.__dict__.get('autorescale_upper_bound', 1.0)

                autoscaler = AutoScaler(Y, lower, upper)
                Y = autoscaler(Y)
            elif not params.allow_out_of_range:
                raise RuntimeError("The training data for the neural net are not scaled"
                                   "to <-1,1>. This will probably result to poor performance"
                                   "of the reggression."
                                   "Set allow_out_of_range to True to disable the check, or"
                                   "set autorescale_output to True to perform the automatic scaling"
                                   "(and descaling of output), or do some scaling yourself.")

        ## Create and postprocess the training data
        # Step 2 in the __doc__
        train_data = XY_to_fann_train_data(X, Y)

        # set properties
        fann_setter(train_data, params.train_prop)

        # posprocess if relevant
        if params.train_postprocess:
            train_postprocess(train_data)

        ## Create the ANN
        # Step 3 in the __doc__

        ann = FannNeuralNetPickable()
        # this could be used instead, but we use the wrapper, so that the classifier
        # is pickable
        #ann = libfann.neural_net()

        # topology = [ number of input neurons,
        #              number of neurons in 1.st layer ,
        #              number of neurons in 2.nd layer ,
        #               etc,
        #              number of output neurons ]
        topology = (len(X[0]), ) + tuple(params.hidden_layers) + (len(Y[0]), )
        if params.nn_type == 'standard':
            ann.create_standard_array( topology )
        elif params.nn_type == 'sparse':
            ann.create_sparse_array(params.connection_rate, topology )
        elif params.nn_type == 'shortcut':
            ann.create_shortcut_array( topology )
        elif params.nn_type == 'cascade':
            if params.hidden_layers:
                raise ValueError("The cascade-trained network must not have any hidden layers on startup.")
            ann.create_shortcut_array( topology )
        else:
            assert False

        # set the properties
        # some defaults
        ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)

        # override by
        fann_setter(ann, params.ann_prop)

        # posprocess if relevant
        if params.ann_postprocess:
            ann_postprocess(ann)

        ## Train the ANN
        # Step 4 in the __doc__

        if params.nn_type == 'cascade':
            ann.cascadetrain_on_data(train_data,
                                     params.max_neurons,
                                     params.neurons_between_reports,
                                     params.desired_error
                                     )
        else:
            ann.train_on_data(train_data,
                              params.max_epochs,
                              params.iterations_between_reports,
                              params.desired_error
                              )

        return FannNeuralClassifier(ann, data.domain, autoscaler)


def table_to_XY(data):
    """Converts the Orange.Table data to pairs of input and output vectors
    (represented row-wise in two numpy.arrays X, Y)
    suitable to be used as a training/testing set for a Artificial neural network.

    The attributes are created by the Table.to_numpy method. The class attribute(s)
    are transformed as follows:
        - each Continuous class attribute (regression), is assinged one output neuron
            (no scaling is performed on this step)
        - each Discrete class attribute (classification), is assinged one output neuron
            for each discrete value of this class. E.g. in the iris dataset
            (one discrete class attribute noting the name of the flower), we have
            3 neurons.
    """
    if not len(data):
        return numpy.array(), numpy.aray()

    ## prepare the training data
    # classes

    cls_descriptors = filter( lambda desc: desc, [data.domain.class_var] + list(data.domain.class_vars))

    def get_unfolder(descriptor):
        """Unfolds class variable into a number of output neurons' output """
        if isinstance(descriptor, Orange.feature.Continuous):
            def unfold(value):
                return [float(value)]

        elif isinstance(descriptor, Orange.feature.Discrete):
            def unfold(value):
                l = [-1.0] * len(descriptor.values)
                l[int(value)] = 1.0
                return l

        else:
            raise ValueError("Unsupported class variable type '%s'. Must be either Discrete or Continuous."%descriptor.var_type)

        return unfold

    unfolders = map(get_unfolder, cls_descriptors)

    def get_class_values(instance):
        l = []
        if data.domain.class_var:
            l = [instance.get_class()]
        return l + instance.get_classes()

    y = []

    # flatten([[0,0,0,1], [0.44], [1,0]]) =
    # [ 0, 0, 0, 1, 0.44, 1, 0 ]
    flatten = lambda it: list(itertools.chain.from_iterable(it))

    # multi_map([lambda x: x + 1, lambda x: x * 2], [0, 10]) =
    # [1, 20]
    multi_map = lambda Fs, Args : [ f(arg) for f, arg in zip(Fs,  Args) ]

    for instance in data:
        values = get_class_values(instance)
        y.append( flatten(multi_map( unfolders, values )) )

    # attributes
    X = data.to_numpy()[0]
    # classes
    Y = numpy.array(y)

    """
    print "X"
    for instance in data:
        print len(instance)
        print instance
    print "Y"
    print Y
    """
    return X, Y

def XY_to_fann_train_data(X, Y):
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of lines.")

    train_data = libfann.training_data()

    if len(X):
        dim_X, dim_Y = len(X[0]), len(Y[0])

        tmp = tempfile.NamedTemporaryFile(delete=False)
        with tmp:
            tmp.write("%d %d %d\n"%(len(X), dim_X,  dim_Y))
            for i in xrange(len(X)):
                for line in [ X[i], Y[i] ]:
                    tmp.write("%s\n"% ' '.join( str(float(val)) for val in line ))

        train_data.read_train_from_file(tmp.name)
        tmp.unlink(tmp.name)

    return train_data

class RawScaler:
    def __init__(self, MIN, MAX, a, b):
        self.MIN = MIN
        self.MAX = MAX
        self.a = a
        self.b = b

    def __call__(self, number):
        assert self.a <= self.b

        if number < self.MIN or number > self.MAX:
            logging.warn("The MIN and MAX estimated from the train set"
                         " do not reflect real MIN and MAX from the test set."
                         " (%.2f < %.2f) or (%.2f > %.2f)"%(number, self.MIN,
                                                            number, self.MAX) )

        if self.MIN == self.MAX:
            # return average value of the set
            return float(self.a + self.b) / 2

        return self.a + (number - self.MIN) * ( float(self.b - self.a) / (self.MAX -  self.MIN) )


class AutoScaler:
    def __init__(self, train_array,  a=-1,  b=1):
        assert a <= b
        self.a = a
        self.b = b
        self.train(train_array)

    def train(self, array):
        rows, cols = array.shape

        #self.MIN,  self.MAX = [], []
        self.trans = []
        self.trans_back = []

        for col in xrange(cols):
            column = array[:, col]
            mi, ma =  column.min(),  column.max()
            #print mi,  ma
            #self.MIN.append(mi)
            #self.MAX.append(ma)
            self.trans.append(RawScaler(mi, ma, self.a, self.b))
            self.trans_back.append(RawScaler(self.a, self.b, mi, ma))

    def scale(self, vector):
        return self._scale(vector, self.trans)

    def scale_back(self, vector):
        return self._scale(vector, self.trans_back)

    def _scale(self, vector, fcs):
        vector = numpy.array(vector)
        cols, = vector.shape
        assert cols == len(fcs)
        return numpy.array([fcs[i](vector[i]) for i in xrange(cols) ])

    def scale_array(self, array):
        return self._scale_array(array, self.trans)

    def scale_array_back(self, array):
        return self._scale_array(array, self.trans_back)

    def _scale_array(self, array, fcs):
        by_rows = [ self._scale(vector, fcs) for vector in array ]
        return numpy.hstack(by_rows).reshape(array.shape)

    def __call__(self, array):
        return self.scale_array(array)

## FIXME Orange.classification.Classifier (which should be there)
## is commented because if it is not, pickling does not work...
class FannNeuralClassifier: #Orange.classification.Classifier):
    """
    """
    def __init__(self, ann, domain, autoscaler=None):
        assert isinstance(ann, FannNeuralNetPickable)

        self.ann = ann
        self.domain = domain
        self.autoscaler = autoscaler

    def raw_response(self, instance ):
        instance = list(instance)
        if self.domain.class_var:
            instance = instance[:len(self.domain)-1]

        if len(instance) !=  self.ann.get_num_input():
            raise ValueError("Instance '%s' has wrong length (%d instead of %d)."%(str(instance),
                                                                                   len(instance),
                                                                                   self.ann.get_num_input()))

        input_vector = map(float, instance)

        ## run the input throught the ANN
        output_vector = self.ann.run(input_vector)

        if self.autoscaler:
            output_vector = self.autoscaler.scale_back(output_vector)

        return output_vector

    def _get_responses(self, instance ):
        # basically the opposite of unfolding in table_to_XY

        output_vector = self.raw_response(instance)

        cls_descriptors = filter( lambda desc: desc, [self.domain.class_var] + list(self.domain.class_vars))

        def get_folder(descriptor):
            """Folds neurons' output into target value.
            returns a tuple (F, num), where F is function that takes list
            of len num (num is the consumed number
            """
            if isinstance(descriptor, Orange.feature.Continuous):
                def fold(outputs):
                    value = descriptor(outputs[0])
                    dist = Orange.statistics.distribution.Continuous(descriptor)
                    dist[value] = 1.
                    return value, dist
                return fold, 1

            elif isinstance(descriptor, Orange.feature.Discrete):
                def fold(outputs):
                    # the output neurons' range is <-1, 1>, where
                    # - 1 says this class is not likely
                    # 1 says this class is likely
                    # so we transform i to <0,2>, so that we do not have
                    # "negative" probabiliies after the normalization
                    outputs = [ o + 1 for o in outputs]
                    cprob = Orange.statistics.distribution.Discrete(outputs)
                    cprob.normalize()

                    mt_prob = cprob
                    mt_value = Orange.data.Value(descriptor, cprob.values().index(max(cprob)))
                    return mt_value, mt_prob
                return fold, len(descriptor.values)

            else:
                raise ValueError("Unsupported class variable type '%s'. Must be either Discrete or Continuous."%descriptor.var_type)

        responses = []
        for folder, input_size in map(get_folder, cls_descriptors):
            responses.append( folder(output_vector[:input_size]) )
            output_vector = output_vector[input_size:]

        return responses

    def __call__(self, instance,
                 result_type=Orange.classification.Classifier.GetValue):
        """Classify a new instance.
        """
        ## Handles the ugly result_type discussion

        ## see the self._get_responses

        responses = self._get_responses(instance)

        values, probs =  [], []
        for value, prob in responses:
            values.append(value)
            probs.append(prob)

        # multilabel
        if self.domain.class_vars :
            if result_type == Orange.classification.Classifier.GetValue:
                return values
            #if any( prob == None for prob in probs):
                #raise ValueError("Wrong result_type for reggresion task")
            if result_type == Orange.classification.Classifier.GetProbabilities:
                return probs
            if result_type == Orange.classification.Classifier.GetBoth:
                return (tuple(values), tuple(probs))
            assert False

        assert len(values) == 1
        value, prob = values[0], probs[0]

        if result_type == Orange.classification.Classifier.GetValue:
            return value
        #if prob == None:
            #raise ValueError("Wrong result_type for reggresion task")
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return prob
        if result_type == Orange.classification.Classifier.GetBoth:
            return (value, prob)

        assert False

## Utility functions

def fann_setter(obj, set_dict):
    """Small utility function for calling setters of FANN objects."""
    for key, val in set_dict.iteritems():
        setter = obj.__getattribute__(key)
        if not isinstance(val, tuple):
            val = (val, )

        setter(*val)

def fake_file_call_s2f(func, string):
    """saves string into a file and calls
    func(filename)

    delete the file afterwards
    """

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(string)
    f.close()

    func(f.name)
    f.unlink(f.name)

def fake_file_call_f2s(func):
    """lets function save something in a file and then returns the filecontent

    f(filename)
    return filecontent

    delete the file afterwards
    """
    f = tempfile.NamedTemporaryFile()
    func(f.name)
    ret = f.read()
    f.close() # and delete
    return ret

##
## tests and examples
##

def test_xor():
    """
    Test simple reggression by learning the XOR function, famous problem,
    imppossible for 1 layered network (without hidden layers).
    """
    ## the data
    attrs = [ Orange.feature.Continuous(name) for name in ['X', 'Y', 'X^Y'] ]
    insts =  [ [x, y, x ^ y] for x, y in itertools.product([0, 1], [0, 1])]
    data = Orange.data.Table(Orange.data.Domain(attrs), insts)

    ## the NeuralNetwork
    print "\n   Test: Xor Function\n" + test_xor.__doc__

    classifier = FannNeuralLearner( data,
                                    # one hidden layer with 2 neurons...
                                    # XOR cannot be done without hidden layer
                                    hidden_layers=[3],
                                    desired_error=0.0001,
                                    iterations_between_reports=500,
                                    max_epochs=5000 )

    for inst in data:
        print "%d xor %d = %d, nn(%d, %d) = %.2f " % (
                        inst[0],  inst[1],  inst[2],
                        inst[0],  inst[1],  classifier(inst) )

def test_iris():
    """
    Test simple classification by learning to classify the iris dataset.
    """
    data = Orange.data.Table("iris.tab")

    print "\n   Test: Iris Dataset\n" + test_iris.__doc__
    classifier = FannNeuralLearner( data,
                                    hidden_layers=[5],
                                    max_epochs=2000,
                                    desired_error=0.005,
                                    iterations_between_reports=200
                                    )

    show_predictions(classifier, data,  probs=True)

def show_predictions(classifier, data,  top=5,  probs=False):
    print
    if probs:
        print "Probability key:\n", data.domain.class_var.values
        print
    print "Random five classifications%s:" % (' and probabilities' if probs else '')
    print
    cnt = 0
    data.shuffle()

    for num, inst in enumerate(data):
        pred, prob = classifier(inst, Orange.classification.Classifier.GetBoth)
        cls = inst.get_class()

        if num < top:
            if probs:
                print prob
            print "%d: Instance %s predicted as %s" % (num + 1, cls, pred)
            print

        if cls !=  pred:
            cnt += 1

    print "\nMissed: %d out of %d examples = %.1f%%" % (cnt, len(data), 100.0 * cnt / len(data))

def test_cascade():
    """
    Test classification on the voting dataset and GetProbabilities output. Also,
    the learning method used to train the neural net is the cascade learning:
        See: http://leenissen.dk/fann/html/files/fann_cascade-h.html
    """
    data = Orange.data.Table("voting.tab")
    # Impute
    data = Orange.data.imputation.ImputeTable(data, method=Orange.feature.imputation.AverageConstructor())
    # take half as train data
    selection = Orange.data.sample.SubsetIndices2(data, 0.5)
    train_data = data.select(selection, 0)
    test_data = data.select(selection, 1)

    print "\n   Test: Cascade Train, Voting Dataset and GetProbabilities\n" + test_cascade.__doc__
    classifier = FannNeuralLearner( train_data,
                                    nn_type='cascade',
                                    max_neurons=5,
                                    neurons_between_reports=2, # 0 turns it off
                                    desired_error=0.005,
                                    )

    print
    print "Possible classes:", data.domain.classVar.values
    print "Probabilities for democrats:"
    print """
    (Note that this are not really 'probabilities';
    more like a measure of sureness of the network.
    This basically are normed neurons' outputs.)"""
    print

    test_data.shuffle()
    show_predictions( classifier,  test_data,  probs=True)

def test_compare():
    iris = Orange.data.Table("iris")
    learners = [
            Orange.classification.knn.kNNLearner(),
            Orange.classification.bayes.NaiveLearner(),
            Orange.classification.majority.MajorityLearner(),
            FannNeuralLearner()
    ]

    cv = Orange.evaluation.testing.cross_validation(learners, iris, folds=5)
    print ["%.4f" % score for score in Orange.evaluation.scoring.CA(cv)]

def test_housing():
    """
    Test reggression together with automatic scaling -- when the output
    domain is out of range <-1,1>.
    """
    data = Orange.data.Table("housing")

    # rescale the domain to -1.2, 1.2
    # default, X=1
    X =  1.2

    #print "\n   Test: Iris Dataset\n" + test_iris.__doc__
    learner = FannNeuralLearner(
                                 hidden_layers=[50],
                                 max_epochs=2000,
                                 desired_error=0.005,
                                 iterations_between_reports=0,
                                 allow_out_of_range=False,
                                 autorescale_output=True,
                                 autorescale_lower_bound=-X,
                                 autorescale_upper_bound=X,
                                )

    #show_predictions(classifier, data,  probs=True)
    cv = Orange.evaluation.testing.cross_validation([
        learner,
        Orange.regression.linear.LinearRegressionLearner()
        ], data, folds=5)

    print '\n'.join("%s : %.4f" % (text, score)
            for score, text in zip(Orange.evaluation.scoring.RMSE(cv),
                                    ["ann", "linear"])
                    )

def equal_within_epsilon(a, b, epsilon=1e-10):
    if a.shape !=  b.shape:
        return False
    return ( numpy.abs(a - b) <= epsilon ).all()

def test_autoscale():
    data = 10 * numpy.random.random((40,4))
    test, train =  data[:5], data[5:]

    # we could also specify smaller domain
    #at = AutoScaler(train, -0.8, 0.8 )
    at = AutoScaler(train) # (-1,1) by default

    print "train scaled"
    print at(train)
    print "test scaled"
    print at(test)

    to_list =  lambda arr : map(list,list(arr))

    #print "test - to and fro"
    #print at.scale_array_back(at(test))
    #print "test normal"
    #print test

    assert equal_within_epsilon(test,  at.scale_array_back(at(test)) )

def test_pickle():
    """
    Test pickling on the xor network
    """
    ## the data
    attrs = [ Orange.feature.Continuous(name) for name in ['X', 'Y', 'X^Y'] ]
    insts =  [ [x, y, x ^ y] for x, y in itertools.product([0, 1], [0, 1])]
    data = Orange.data.Table(Orange.data.Domain(attrs), insts)

    ## the NeuralNetwork
    print "\n   Test: Xor Function\n" + test_xor.__doc__

    classifier = FannNeuralLearner( data,
                                    # one hidden layer with 2 neurons...
                                    # XOR cannot be done without hidden layer
                                    hidden_layers=[3],
                                    desired_error=0.0001,
                                    iterations_between_reports=500,
                                    max_epochs=5000 )
    import pickle
    with open("OUT.pkl", 'wb') as fout:
        pickle.dump(classifier, fout)

    print 'saved'
    with open("OUT.pkl", 'rb') as fin:
        print pickle.load(fin)


if __name__ == "__main__":
    test_xor()
    #test_iris()
    #test_cascade()
    #test_compare()
    #test_housing()
    #test_autoscale()
    #test_pickle()



