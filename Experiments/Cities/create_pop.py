from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np

# Настройки популяции
location = "seattle_metro"  # Локация
state_location = "Washington"
country_location = "usa"

sizes = np.load('data/sizes.npy', allow_pickle=True)
labels = np.load('data/labels.npy', allow_pickle=True)

for i, pop_size in enumerate(sizes):
    # Генерация популяции
    popdict = sp.make_population(n=pop_size, location=location, state_location=state_location, country_location=country_location)

    # Сохранение в файл .ppl
    with open(f"pops/{labels[i]}.ppl", "wb") as f:
        pickle.dump(popdict, f)