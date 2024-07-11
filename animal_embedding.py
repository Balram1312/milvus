import numpy as np

# Define the animal properties without size
animal_properties = {
    'lion': [1, 4, 1, 1, 1, 1],  # carnivorous, 4 legs, predator, social, savannah, land
    'elephant': [0, 4, 0, 2, 1, 1],  # herbivorous, 4 legs, prey, social, forest, land
    'tiger': [1, 4, 1, 1, 1, 1],  # carnivorous, 4 legs, predator, social, forest, land
    'giraffe': [0, 4, 0, 2, 1, 1],  # herbivorous, 4 legs, prey, social, savannah, land
    'polar_bear': [1, 4, 1, 1, 2, 1],  # carnivorous, 4 legs, predator, social, tundra, land
    'kangaroo': [0, 2, 0, 2, 1, 1],  # herbivorous, 2 legs, prey, social, savannah, land
    'crocodile': [1, 4, 1, 1, 3, 2],  # carnivorous, 4 legs, predator, social, freshwater, water
    'whale': [0, 0, 0, 0, 4, 3],  # herbivorous, 0 legs, prey, solitary, ocean, water
    'eagle': [1, 2, 1, 0, 5, 1],  # carnivorous, 2 legs, predator, solitary, sky, land
    'zebra': [0, 4, 0, 2, 1, 1],  # herbivorous, 4 legs, prey, social, savannah, land
    'hippopotamus': [0, 4, 0, 2, 3, 1],  # herbivorous, 4 legs, prey, social, freshwater, water
    'penguin': [1, 2, 0, 0, 6, 3],  # carnivorous, 2 legs, prey, social, polar, water
    'gorilla': [0, 2, 0, 2, 1, 1],  # herbivorous, 2 legs, prey, social, forest, land
    'komodo_dragon': [1, 4, 1, 1, 5, 1],  # carnivorous, 4 legs, predator, solitary, savannah, land
    'dolphin': [0, 0, 0, 0, 6, 3]  # herbivorous, 0 legs, prey, solitary, ocean, water
}


# Function to normalize a vector
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# Function to convert animal properties into normalized vectors
def animal_vectors():
    return {animal: normalize_vector(properties) for animal, properties in animal_properties.items()}
