from agents.agent import Agent
from agents.artificial_neural_network import ArtificialNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize the inference model and load in the saved trained ANN model 
        weights and the HashingVectorizer from Hugging Face.
        """
        self.log("Neural Network Agent is initializing")
        self.neural_network = ArtificialNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load("sadjei65320/ProductPriceAI-ArtificialNeuralNetwork")
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str) -> float:
        """
        Use the Artificial Neural Network to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
