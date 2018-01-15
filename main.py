"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import numpy as np

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks):
    """Train each network.

    Args:
        networks (list): Current population of networks
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        pbar.update(1)
    pbar.close()

def get_np_losses(networks):
    """Get the average loss for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average loss of a population of networks.

    """
    losses = []
    for network in networks:
        losses.append(network.loss)

    np_losses = np.array(losses)

    return np_losses

def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        print("\r***Doing generation %d of %d***" %
                     (i + 1, generations))


        counter = 0
        for network in networks:
            network.network['number'] = counter
            counter += 1

        # Train and get loss for networks.
        train_networks(networks)

        # Get the average loss for this generation.
        np_loss = get_np_losses(networks)

        # Print out the average loss each generation.
        logging.info("Generation average: %.3f" % (np.mean(np_loss)))
        logging.info("Generation maximum: %.3f" % (np.max(np_loss)))
        logging.info("Generation minimum: %.3f" % (np.min(np_loss)))
        logging.info('-'*80)

        print("\rGeneration average: %.3f" % (np.mean(np_loss)))
        print("Generation maximum: %.3f" % (np.max(np_loss)))
        print("Generation minimum: %.3f" % (np.min(np_loss)))
        print('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.loss, reverse=False)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 30  # Number of networks in each generation.

    nn_param_choices = {
         # 'nb_neurons': [64, 128, 256, 512, 768, 1024],
         # 'nb_layers': [1, 2, 3, 4, 5, 6, 7, 8],
        'activation': ['relu', 'PReLU', 'ELU', 'ThresholdedReLU', 'selu'],
        # 'optimizer': ['RMSprop', 'Adam', 'Adagrad', 'SGD', 'Adadelta', 'Adamax', 'Nadam'],
        'optimizer': ['Adam', 'Adagrad','Adadelta', 'Adamax', 'Nadam' ],
        'batch_size': [512, 1024],
        # 'batch_size': [32, 64, 128, 256, 512],
        'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        # 'model_type': ['mae'],
        'model_type': ['mae', 'mape', 'mae_mape'],
        'result': ['mae'],
        # 'epochs':[1, 2, 5, 10000],
        'int_layer': [30],
        'log_y': [True, False],
        # 'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'he_normal', 'he_uniform'],
        'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
        # 'activation': ['tanh',  'softmax', 'softplus', 'softsign', 'relu', 'selu', 'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU'],
        'hidden_layers': [
            # [1],
            # [1, 0.1],
            # [1, 1, 1],
            # [1, 0.5, 0.1],
            [2],
            [5],
            [1, 1, 1, 1, 1],
            [1, 0.5, 0.25, 0.1, 0.05],
            [5, 5, 5, 5, 5],
            [5, 4, 3, 2, 1],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
            [6, 6, 6, 6],
            [7, 7, 7, 7],
            [4, 3, 2, 1],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [3, 2, 1],
            [1, 1, 1, 1],
            [1, 1]
        ],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()
