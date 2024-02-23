import math
import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)


class Kohonen:
    """
    Kohonen SOM network implementation

    ...

    Attributes
    ----------
    output_layer : OutputLayer
        specify and OutputLayer object, or pass None to create a new output layer
    input_size : int
        the number of colours in the input layer
    width : int
        the width of the output layer
    height : int
        the height of the output layer
    learning_rate : float
        the learning rate of the current iteration
    iterations : int
        the number of iterations to train the network

    """

    MAX_ITERATIONS = 300
    NUM_COLOURS = 20
    NUM_DIMENSIONS = 3
    MAP_WIDTH = 100
    MAP_HEIGHT = 100
    INITIAL_NEIGHBOURHOOD_RADIUS = max(MAP_WIDTH, MAP_HEIGHT) / 2
    TIME_CONSTANT = MAX_ITERATIONS / np.log(INITIAL_NEIGHBOURHOOD_RADIUS)
    INITIAL_LEARNING_RATE = 0.1

    def __init__(
        self,
        input_size=NUM_COLOURS,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        learning_rate=INITIAL_LEARNING_RATE,
        iterations=MAX_ITERATIONS,
    ):
        """
        Initializes the Kohonen network
        """
        self.input_size = input_size
        self.width = width
        self.height = height
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.neighbourhood_radius = max(self.width, self.height) / 2
        self.init_neighbourhood_radius = max(self.width, self.height) / 2
        self.time_constant = self.calculate_time_constant()
        self.input_layer = self.InputLayer(num_colours=input_size)
        self.output_layer = OutputLayer(width=width, height=height)

    def print_progress(self, iteration, progress_nodes, start_time):
        """
        Prints the progress of the Kohonen network training process
        ...

        Attributes
        ----------
        iteration : int
            the current iteration
        progress_nodes : list
            node information for interval parts of the iteration, this is to be have the ability display the state of
            the network at different stages of the training process
        start_time : float
            the start time of the training process

        Returns
        -------
        progress_nodes
            return progress_nodes for client that want to display training progress
        """
        if (
            iteration == 0
            or iteration == self.iterations - 1
            or iteration % (self.iterations / 4) == 0
        ):
            nodes = np.copy(self.output_layer.nodes)
            progress_nodes.append(
                {
                    "iteration": iteration,
                    "neighbourhood_radius": self.neighbourhood_radius,
                    "learning_rate": self.learning_rate,
                    "nodes": nodes,
                }
            )
            plt.figure()
            plt.axis("off")
            plt.imshow(nodes, aspect="1")
            plt.title(
                f"Kohonen SOM ({self.output_layer.width}x{self.output_layer.height}), iteration: {iteration}"
            )
            plt.show(block=False)

            print(
                f"Iteration: {iteration:03}/{self.iterations-1} | Learning rate: {round(self.learning_rate, 4)} | Radius: {round(self.neighbourhood_radius, 4)}"
            )
            progress_end_time = time.time()
            self.print_elapsed_time(start_time, progress_end_time)

        return progress_nodes

    def print_elapsed_time(self, start_time, end_time):
        """
        Prints the elapsed time
        """
        elapsed_time_seconds = end_time - start_time
        elapsed_hours = int(elapsed_time_seconds // 3600)
        elapsed_minutes = int(elapsed_time_seconds // 60)
        elapsed_seconds = int(elapsed_time_seconds % 60)
        elapsed_milliseconds = int(
            (elapsed_time_seconds - int(elapsed_time_seconds)) * 1000
        )

        print(
            f"Execution time (hh:mm:ss.ms): {elapsed_hours:02}:{elapsed_minutes:02}:{elapsed_seconds:02}.{elapsed_milliseconds}"
        )

    async def fit(self):
        """
        Trains the Kohonen network using a vectorised approach
        ...

        Description
        -----------
        1. Iterate through the input layer vectors
        2. The euclidian distance is calclulated between the input vector and all the output nodes
        3. The Best Matching Unit (BMU) determined from this euclidian distance matrix
        4. The influence is calculated for each node in the output layer, based on the BMU and the neighbourhood radius
        5. The weights of the nodes are updated
        6. The neighbourhood radius and learning rate are updated
        7. The process is repeated for the number of iterations specified
        Note:
        - Iteration has to be done through a loop since the iteration is critical to the Kohonen algorithm
        - We are also lopping through the input layer vectors, and this has to be done in sequential order of the input layer vectors
        """
        progress_nodes = []
        start_time = time.time()
        for iteration in range(self.iterations):
            for current_input_vector in self.input_layer.vectors[0]:
                self.output_layer.update_weights(
                    current_input_vector, self.neighbourhood_radius, self.learning_rate
                )

            self.print_progress(iteration, progress_nodes, start_time)

            self.learning_rate = self.calculate_learning_rate(iteration + 1)
            self.neighbourhood_radius = self.calculate_neighbourhood_radius(
                iteration + 1
            )

        print(
            f"Kohonen training completed, input size: {self.input_size}, output size: {self.width}x{self.height}, iterations: {self.iterations}"
        )

        return progress_nodes

    def calculate_neighbourhood_radius(self, iteration):
        """
        Calculates the decaying neighbourhood radius
        """
        return self.init_neighbourhood_radius * np.exp(-iteration / self.time_constant)

    def calculate_learning_rate(self, iteration):
        """
        Calculates the decaying learning rate
        """
        return Kohonen.INITIAL_LEARNING_RATE * np.exp(-iteration / self.time_constant)

    def calculate_time_constant(self):
        """
        Calculates the time constant
        """
        return self.iterations / np.log(self.init_neighbourhood_radius)

    class InputLayer:
        """
        Class that represents the input layer of the Kohonen network
        """

        def __init__(self, num_colours):
            """
            Initializes the input layer
            """
            self.num_colours = num_colours
            self.name = "InputLayer"
            self.vectors = self.generate_input_vectors()

        def generate_input_vectors(self):
            """
            Returns a array of floats in the range [0.0, 1.0], where each vector represents a sample from our dataset.
            This 3 dimensional input vector actually represent a colour in the RGB colour space.
            """
            return np.random.random_sample(
                (1, self.num_colours, Kohonen.NUM_DIMENSIONS)
            )


class OutputLayer:
    """
    Class that represents the output layer of the Kohonen network
    """

    def __init__(self, width, height):
        """
        Initializes the output layer
        ...
        Attributes
        ----------
        width : int
            the width of the output layer
        height : int
            the height of the output layer

        """
        self.width = width
        self.height = height
        self.name = "OutputLayer"
        self.nodes = self.create_node_vectors()
        self.euclidean_distances = np.zeros((self.width, self.height))
        self.node_coordinates = np.indices((self.width, self.height)).transpose(
            (1, 2, 0)
        )

    def create_node_vectors(self):
        """
        Returns a array of floats in the range [0.0, 1.0]. This is 3-dimension array that represents the RGB colour space,
        and the size specified in the width and height attributes.
        """
        return np.random.random_sample(
            (self.width, self.height, Kohonen.NUM_DIMENSIONS)
        )

    def calculate_euclidean_matrix(self, input_vector):
        """
        Returns the Euclidean distance between two 3-dimensional vectors
        """
        self.euclidean_distances = np.sqrt(
            np.sum(((self.nodes - input_vector) ** 2), axis=-1)
        )

    def get_coordinates_of_bmu(self):
        """
        Returns the smallest euclidian distance (bmu) in this euclidian matrix
        """
        min_indices = np.unravel_index(
            np.argmin(self.euclidean_distances), self.euclidean_distances.shape
        )
        return min_indices[0], min_indices[1]

    def get_radial_distance(self, node_x, node_y, bmu_x, bmu_y):
        """
        Returns the distance between the node and the BMU using pythagoras theorem
        """
        return math.sqrt((node_x - bmu_x) ** 2 + (node_y - bmu_y) ** 2)

    def get_radial_distance_matrix(self, nodes, bmu):
        """
        Returns the distance matrix between the node and the BMU using pythagoras theorem
        """
        return np.sqrt(np.sum((nodes - bmu) ** 2, axis=-1))

    def get_nodes(self):
        """
        Returns the nodes in the output layer
        """
        return np.copy(self.nodes)

    def update_weights(self, current_input_vector, neighbourhood_radius, learning_rate):
        """
        Updates the weights of the nodes in the output layer
        """
        self.calculate_euclidean_matrix(current_input_vector)
        influence_matrix = self.calculate_influence(neighbourhood_radius)

        influence = np.expand_dims(influence_matrix, axis=-1)
        self.nodes = self.nodes + (
            learning_rate * influence * (current_input_vector - self.nodes)
        )

    def calculate_influence(self, neighbourhood_radius):
        """
        Calculates the influence of learning on the nodes
        """
        bmu_x, bmu_y = self.get_coordinates_of_bmu()

        dist_from_nodes_to_bmu = self.get_radial_distance_matrix(
            self.node_coordinates,
            np.array((bmu_x, bmu_y)),
        )

        return np.exp(-(dist_from_nodes_to_bmu**2) / (2 * neighbourhood_radius**2))


class NonVectorisedKohonen(Kohonen):
    """
    Deprecated Kohonen network implementation - where the algorithm works as expected, however
    it uses a naive approach with nested for-loops to update the weights of the nodes in the output layer. This
    implementation is kept for comparison purposes.
    """

    def __init__(
        self,
        input_size=Kohonen.NUM_COLOURS,
        width=Kohonen.MAP_WIDTH,
        height=Kohonen.MAP_HEIGHT,
        learning_rate=Kohonen.INITIAL_LEARNING_RATE,
        iterations=Kohonen.MAX_ITERATIONS,
    ):
        super().__init__(
            input_size=input_size,
            width=width,
            height=height,
            learning_rate=learning_rate,
            iterations=iterations,
        )
        self.output_layer = NonVectorisedOutputLayer(width=width, height=height)

    async def fit(self):
        """
        Trains the Kohonen network using Python nested loops

        ...

        Description
        -----------
        1. Iterate through the input layer vectors
        2. The euclidian distance is calclulated between the input vector and all the output nodes
        3. The Best Matching Unit (BMU) determined from this euclidian distance matrix
        4. The influence is calculated for each node in the output layer, based on the BMU and the neighbourhood radius
        5. The weights of the nodes are updated, by iterating through all the nodes in the output layer
        6. The neighbourhood radius and learning rate are updated
        7. The process is repeated for the number of iterations specified
        """
        progress_nodes = []
        start_time = time.time()
        for iteration in range(self.iterations):
            for current_input_vector in self.input_layer.vectors[0]:
                self.output_layer.update_weights(
                    current_input_vector,
                    self.neighbourhood_radius,
                    self.learning_rate,
                )

            progress_nodes = self.print_progress(iteration, progress_nodes, start_time)

            self.learning_rate = self.calculate_learning_rate(iteration + 1)
            self.neighbourhood_radius = self.calculate_neighbourhood_radius(
                iteration + 1
            )

        print(
            f"Non-Vectorised Kohonen training completed, input size: {self.input_size}, output size: {self.width}x{self.height}, iterations: {self.iterations}"
        )

        return progress_nodes


class NonVectorisedOutputLayer(OutputLayer):
    """
    Deprecated OutputLayer implementation - where the algorithm works as expected, however
    it uses a naive approach with nested for-loops to update the weights of the nodes in the output layer. This
    implementation is kept for comparison purposes.
    """

    def update_weights(
        self,
        current_input_vector,
        neighbourhood_radius,
        learning_rate,
    ):
        """
        Updates the weights of the nodes in the output layer
        """
        self.calculate_euclidean_matrix(current_input_vector)

        bmu_x, bmu_y = self.get_coordinates_of_bmu()
        for i in range(self.width):
            for j in range(self.height):
                distance_to_bmu = self.get_radial_distance(
                    i,
                    j,
                    bmu_x,
                    bmu_y,
                )
                current_weight = self.nodes[i, j]
                self.nodes[i, j] = current_weight + (
                    learning_rate
                    * self.calculate_influence(distance_to_bmu, neighbourhood_radius)
                    * (current_input_vector - current_weight)
                )

    def calculate_influence(self, distance_to_bmu, neighbourhood_radius):
        """
        Calculates the influence of learning on the nodes
        """

        return math.exp(-(distance_to_bmu**2) / (2 * neighbourhood_radius**2))
