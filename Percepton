# Ethayananth, Monica Rani
# 1001-417-942
#2016-09-29

import numpy as np
import Tkinter as Tk
import matplotlib
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pdb
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys
import scipy.misc
import numpy as np
import pdb
FOLDER = "mnist_images" #please enter the mnist_images path
class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples= None,targets=None):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        # Farhad Kamangar 2016_09_05

        for s,p,f in os.walk(FOLDER):
            Images = []
            Labels = []
            np.random.shuffle(f)
            for imgs in f:
                path = os.path.join(s,imgs)
                #pdb.set_trace()
                image = scipy.misc.imread(path,np.float32)
                image /= 255.0
                image = np.append(image.reshape(-1),1.0)
                Images.append(image)
                label = np.zeros(10,dtype=np.float32)
                label[int(imgs.split("_")[0])] = 1.0
                #Labels.append(imgs.split("_")[0])
                Labels.append(label)

        self.samples = np.array(Images).T
        self.desired_target_vectors = np.array(Labels).T
        #print self.samples.shape,self.desired_target_vectors.shape


nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs": 785,  # number of inputs to the network
    "learning_rate": 0.00001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_classes': 10,
    'number_of_samples_in_each_class': 3
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."



    def adjust_weights(self,learning_rule):
        self.neural_network.adjust_weights(self.data_set.samples,
                                           self.data_set.desired_target_vectors,learning_rule)


class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.learning_rate = self.nn_experiment.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 1
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Hebbian Learning")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Hebbian Learning")
        self.axes = self.figure.add_subplot(111)
        plt.title("Hebbian Learning")

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # self.number_of_classes_slider_label = Tk.Label(self.sliders_frame, text="Number of Classes")
        # self.number_of_classes_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_classes_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=5, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_classes_slider.set(self.number_of_classes)
        # self.number_of_classes_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_classes_slider_callback())
        # self.number_of_classes_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider = Tk.Scale(self.sliders_frame, variable=ivar, orient=Tk.HORIZONTAL,
        #                                          from_=2, to_=20, bg="#DDDDDD",
        #                                          activebackground="#FF0000",
        #                                          highlightcolor="#00FFFF", width=10)
        # self.number_of_samples_slider_label = Tk.Label(self.sliders_frame, text="Number of Samples")
        # self.number_of_samples_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.number_of_samples_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_samples_slider_callback())
        # self.number_of_samples_slider.set(self.number_of_samples_in_each_class)
        # self.number_of_samples_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.create_new_samples_bottun = Tk.Button(self.buttons_frame,
        #                                            text="Create New Samples",
        #                                            bg="yellow", fg="red",
        #                                            command=lambda: self.create_new_samples_bottun_callback())
        # self.create_new_samples_bottun.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # self.randomize_weights_button = Tk.Button(self.buttons_frame,
        #                                           text="Randomize Weights",
        #                                           bg="yellow", fg="red",
        #                                           command=lambda: self.randomize_weights_button_callback())
        # self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_method_variable = Tk.StringVar()
        self.learning_method_dropdown = Tk.OptionMenu(self.buttons_frame, self.learning_method_variable,
                                                      "Filtered Learning",
                                                      "Delta Rule", "Unsupervised Learning",
                                                      command=lambda event: self.learning_method_dropdown_callback())
        self.learning_method_variable.set("Filtered Learning")
        self.learning_rule = "Filtered Learning"
        self.learning_method_dropdown.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.x = []
        self.y = []

        self.reset_button = Tk.Button(self.buttons_frame, text="RESET",
                                      bg="red", fg="yellow",
                                      command=lambda: self.reset_button_callback())
        self.reset_button.grid(row=5,column=0,sticky=Tk.N+Tk.E+Tk.S+Tk.W)
        self.current_epoch=0
        self.initialize()

        #self.refresh_display()

    def initialize(self):
        #self.nn_experiment.create_samples()
        self.nn_experiment.neural_network.randomize_weights()
        self.neighborhood_colors = plt.cm.get_cmap('Accent')
        self.sample_points_colors = plt.cm.get_cmap('Dark2')
        self.xx, self.yy = np.meshgrid(np.arange(self.xmin, self.xmax + 0.5 * self.step_size, self.step_size),
                                       np.arange(self.ymin, self.ymax + 0.5 * self.step_size, self.step_size))
        self.convert_binary_to_integer = []
        for k in range(0, self.nn_experiment.neural_network.layers[-1].number_of_neurons):
            self.convert_binary_to_integer.append(2 ** k)

    def learning_method_dropdown_callback(self):

        self.learning_rule  = self.learning_method_variable.get()

    def refresh_display(self):
        #self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)
        self.display_neighborhoods()

    def display_neighborhoods(self):
        input_samples= self.nn_experiment.data_set.samples
        targets = self.nn_experiment.data_set.desired_target_vectors
        outputs = self.nn_experiment.neural_network.calculate_output(input_samples)
        #print outputs.shape,targets.shape
        #pdb.set_trace()
        def to_integer(vector):
            return np.where(vector==max(vector))
        count = 0
        size = input_samples.shape[1]
        for i in range(size):
            #pdb.set_trace()
            if to_integer(targets[:,i]) == to_integer(outputs[:,i]):
                count+=1
                #print to_integer(outputs[:,i])
        error = float(size-count)*100.0/size
        self.x.append(self.current_epoch)
        self.y.append(error)
        plt.plot(self.x,self.y,"b")
        self.canvas.draw()

    def initialize_plot_variables(self):
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)



    def reset_button_callback(self):
        self.x = []
        self.y = []
        self.current_epoch = 0
        plt.cla()
        self.initialize_plot_variables()
        self.canvas.draw()

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        #self.refresh_display()


    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        for k in range(100):
            self.current_epoch +=1
            self.nn_experiment.adjust_weights(learning_rule=self.learning_rule)
            self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        # self.nn_experiment.neural_network.display_network_parameters()
        # self.nn_experiment.run_forward_pass()
        #self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        #self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()


neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs": 785,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10,
                              "activation_function": "linear"}]  # list of dictionaries
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    def adjust_weights(self, input_samples, targets,learning_rule):
        if learning_rule == "Filtered Learning":
            for index,layer in enumerate(self.layers):
                samples_size,dim = input_samples.shape
                for i in range(samples_size):
                    current_inp = input_samples[:,i].reshape(-1,1)
                    current_target = targets[:,i].reshape(-1,1)
                    layer.weights = (1-self.learning_rate)*layer.weights + np.dot(current_target,current_inp.T)
        elif learning_rule == "Delta Rule":
            for index,layer in enumerate(self.layers):
                samples_size,dim = input_samples.shape
                for i in range(samples_size):
                    current_inp = input_samples[:,i].reshape(-1,1)
                    current_target = targets[:,i].reshape(-1,1)
                    current_output = layer.calculate_output(current_inp).reshape(-1,1)
                    layer.weights += self.learning_rate * np.dot((current_target-current_output),current_inp.T)
                   # print layer.weights

        elif learning_rule == "Unsupervised Learning":
            for index,layer in enumerate(self.layers):
                samples_size, dim = input_samples.shape
                for i in range(samples_size):
                    current_inp = input_samples[:, i].reshape(-1, 1)
                    current_output = layer.calculate_output(current_inp).reshape(-1, 1)
                    layer.weights += self.learning_rate * np.dot(current_output, current_inp.T)



single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -1.0,  # minimum initial weight
    "max_initial_weights": 1.0,  # maximum initial weight
    "number_of_inputs_to_layer": 785,  # number of input signals
    "number_of_neurons": 10,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (self.number_of_neurons, self.number_of_inputs_to_layer))

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        # Farhad Kamangar Sept. 4, 2016
        net = np.dot(self.weights,input_values)
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -1.0,  # minimum initial weight
        "max_initial_weights": 1.0,  # maximum initial weight
        "number_of_inputs": 785,  # number of inputs to the network
        "learning_rate": 0.001,  # learning rate
        "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClDataSet(),
        'number_of_classes': 10,

        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Hebb Rule")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
