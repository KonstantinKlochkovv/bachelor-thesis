from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np

# Настройки популяции
location = "seattle_metro"  # Локация
state_location = "Washington"
country_location = "usa"

population = 200000

sizes = np.arange(10000, population, 10000)

for size in sizes:
    popdict = sp.make_population(n=size, location=location, state_location=state_location, country_location=country_location)

    with open(f"pops/{size}.ppl", "wb") as f:
        pickle.dump(popdict, f)