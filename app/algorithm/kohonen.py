# from random import randint
import math
import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)


class Kohonen:
    """
    Kohonen network implementation
    """

    MAX_ITERATIONS = 300
    NUM_COLOURS = 20
    NUM_DIMENSIONS = 3
    MAP_WIDTH = 100
    MAP_HEIGHT = 100
    INITIAL_NEIGHBOURHOOD_RADIUS = max(MAP_WIDTH, MAP_HEIGHT) / 2
    TIME_CONSTANT = MAX_ITERATIONS / np.log(INITIAL_NEIGHBOURHOOD_RADIUS)
    INITIAL_LEARNING_RATE = 0.1
    RANDOM = True

    def __init__(
        self,
        output_layer,
        random=RANDOM,
        input_size=NUM_COLOURS,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        learning_rate=INITIAL_LEARNING_RATE,
        iterations=MAX_ITERATIONS,
    ):
        """
        Initializes the Kohonen network
        """
        self.random = random
        self.input_size = input_size
        self.width = width
        self.height = height
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.neighbourhood_radius = max(self.width, self.height) / 2
        self.init_neighbourhood_radius = max(self.width, self.height) / 2
        self.time_constant = self.calculate_time_constant()
        self.input_layer = self.InputLayer(random=random, num_colours=input_size)
        self.output_layer = (
            output_layer
            if output_layer is not None
            else OutputLayer(width=width, height=height)
        )

    def get_input_layer(self):
        """
        Returns the input layer of the Kohonen network
        """
        return self.input_layer

    def get_num_colours(self):
        """
        Returns the number of colours in the input layer
        """
        return self.input_layer.num_colours

    def get_output_layer(self):
        """
        Returns the output layer of the Kohonen network
        """
        return self.output_layer

    def print_progress(self, iteration, progress_nodes, start_time):
        """
        Prints the progress of the Kohonen network
        """
        if (
            iteration == 0
            or iteration == self.iterations - 1
            or iteration % (self.iterations / 4) == 0
        ):
            nodes = np.copy(self.get_output_layer().nodes)
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
            plt.imshow(nodes.astype("uint8"), aspect="1")
            plt.title(
                f"Kohonen SOM ({self.get_output_layer().width}x{self.get_output_layer().height}), iteration: {iteration}"
            )
            plt.show(block=False)

            print(
                f"Iteration: {iteration:03}/{self.iterations-1} | Learning rate: {round(self.learning_rate, 4)} | Radius: {round(self.neighbourhood_radius, 4)}"
            )
            progress_end_time = time.time()
            self.print_elapsed_time(start_time, progress_end_time)

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
        Trains the Kohonen network
        """
        progress_nodes = []
        start_time = time.time()
        for iteration in range(self.iterations):

            for current_input_vector in self.get_input_layer().vectors[0]:
                self.get_output_layer().update_weights_matrix(
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

        def __init__(self, random, num_colours):
            """
            Initializes the input layer
            """
            self.random = random
            self.num_colours = num_colours
            self.name = "InputLayer"
            self.vectors = self.generate_input_vectors()

        def generate_input_vectors(self):
            """
            Returns a array of RGB values
            """
            if self.random:

                random_vectors = self.generate_random_vectors()
                return random_vectors

            fixed_vectors = self.generate_fixed_vectors()
            self.num_colours = fixed_vectors.shape[1]
            return fixed_vectors

        def generate_fixed_vectors(self):
            """
            Returns a array of 30 fixed RGB values, to create a deterministic output
            """
            rgb_values = [
                [
                    [247, 3, 0],
                    [252, 5, 0],
                    [239, 8, 3],
                    [232, 0, 3],
                    [255, 3, 0],
                    [255, 3, 3],
                    [249, 5, 3],
                    [247, 3, 5],
                    [234, 8, 3],
                    [247, 3, 5],
                    [3, 255, 0],
                    [0, 247, 3],
                    [0, 252, 5],
                    [3, 239, 8],
                    [3, 234, 4],
                    [0, 232, 3],
                    [3, 255, 0],
                    [0, 250, 5],
                    [0, 250, 5],
                    [0, 250, 5],
                    [3, 0, 255],
                    [3, 0, 247],
                    [5, 0, 252],
                    [8, 3, 239],
                    [8, 5, 250],
                    [0, 3, 232],
                    [3, 3, 255],
                    [5, 3, 249],
                    [3, 5, 247],
                    [3, 7, 240],
                ]
            ]

            rgb_array = np.array(rgb_values)
            return rgb_array

        def generate_random_vectors(self):
            """
            Returns a array of num_colours random RGB values
            """
            return np.random.randint(
                0, 255, (1, self.num_colours, Kohonen.NUM_DIMENSIONS)
            )


class OutputLayer:
    """
    Class that represents the output layer of the Kohonen network
    """

    def __init__(self, width, height):
        """
        Initializes the output layer
        """
        self.width = width
        self.height = height
        self.name = "OutputLayer"
        self.nodes = self.create_node_vectors()
        self.euclidean_distances = np.zeros((self.width, self.height))
        self.vector_bmu_coordinates = []
        self.influence = 1
        self.node_coordinates = np.indices((self.width, self.height)).transpose(
            (1, 2, 0)
        )

    def create_node_vectors(self):
        """
        Returns a array of nodes where the values are in the RGB colour space
        """
        return np.random.randint(
            0, 255, (self.width, self.height, Kohonen.NUM_DIMENSIONS)
        )

    def euclidean_distance(self, input_vector, node_vector):
        """
        Returns the Euclidean distance between two RGB vectors
        """
        return np.sqrt(np.sum(((node_vector - input_vector) ** 2), axis=-1))

    def get_euclidean_matrix(self, input_vector):
        """
        Returns the all the Euclidean distances between the input vector and the node vectors
        and store them in a grid
        """
        self.euclidean_distances = self.euclidean_distance(input_vector, self.nodes)

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

    def update_weights(
        self,
        current_input_vector,
        neighbourhood_radius,
        learning_rate,
    ):
        """
        Updates the weights of the nodes in the output layer
        """
        self.get_euclidean_matrix(current_input_vector)

        bmu_x, bmu_y = self.get_coordinates_of_bmu()
        for i in range(self.width):
            for j in range(self.height):
                distance_to_bmu = self.get_radial_distance(
                    i,
                    j,
                    bmu_x,
                    bmu_y,
                )

                if distance_to_bmu < neighbourhood_radius:

                    current_weight = self.nodes[i, j]
                    self.nodes[i, j] = current_weight + (
                        learning_rate
                        * self.calculate_influence(
                            distance_to_bmu, neighbourhood_radius
                        )
                        * (current_input_vector - current_weight)
                    )

    def update_weights_matrix(self, input_vector, neighbourhood_radius, learning_rate):
        """
        Updates the weights of the nodes in the output layer
        """
        self.get_euclidean_matrix(input_vector)
        influence_matrix = self.calculate_influence_matrix(neighbourhood_radius)

        influence = np.expand_dims(influence_matrix, axis=-1)
        self.nodes = self.nodes + (
            learning_rate * influence * (input_vector - self.nodes)
        )

    def calculate_influence(self, distance_to_bmu, neighbourhood_radius):
        """
        Calculates the influence of learning on the nodes
        """
        # bmu_x, bmu_y = self.get_coordinates_of_bmu()
        # distance_to_bmu = self.get_radial_distance(
        #     node_coordinate[0], node_coordinate[1], bmu_x, bmu_y
        # )

        return math.exp(-(distance_to_bmu**2) / (2 * neighbourhood_radius**2))

    def calculate_influence_matrix(self, neighbourhood_radius):
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

    async def fit(self):
        """
        Trains the Kohonen network
        """
        progress_nodes = []
        start_time = time.time()
        for iteration in range(self.iterations):
            for current_input_vector in self.get_input_layer().vectors[0]:

                self.get_output_layer().update_weights(
                    current_input_vector,
                    self.neighbourhood_radius,
                    self.learning_rate,
                )

            self.print_progress(iteration, progress_nodes, start_time)

            self.learning_rate = self.calculate_learning_rate(iteration + 1)
            self.neighbourhood_radius = self.calculate_neighbourhood_radius(
                iteration + 1
            )

        print(
            f"Non-Vectorised Kohonen training completed, input size: {self.input_size}, output size: {self.width}x{self.height}, iterations: {self.iterations}"
        )

        return progress_nodes
