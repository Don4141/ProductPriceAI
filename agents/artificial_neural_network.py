import torch
import torch.nn as nn
import joblib
import logging
from huggingface_hub import hf_hub_download


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.layer1(x))
        output2 = self.relu(self.layer2(output1))
        output3 = self.relu(self.layer3(output2))
        output4 = self.layer4(output3)
        return output4


class ArtificialNeuralNetworkInference:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.device = None

    def setup(self):
        self.model = NeuralNetwork(4000)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        self.model.to(self.device)

    def load(self, repo_id):
        # Download the trained ANN weights and associated vectorizer
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="ann_model.pt"
        )

        vectorizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="ann_vectorizer.joblib"
        )

        # Load the trained weights and vectorizer
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.vectorizer = joblib.load(vectorizer_path)
        self.model.eval()

    def inference(self, text):
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            result = self.model(vector)[0].item()

        return max(0, result)