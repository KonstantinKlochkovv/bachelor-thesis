from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np

# Настройки популяции
location = "seattle_metro"  # Локация
state_location = "Washington"
country_location = "usa"

for big_size in [5e4, 6e4, 7e4, 8e4, 9e4]:
    popdict = sp.make_population(n=big_size, location=location, state_location=state_location, country_location=country_location)
    
    with open(f"pops/big_{int(big_size)}.ppl", "wb") as f:
        pickle.dump(popdict, f)

    popdict = sp.make_population(n=1e5-big_size, location=location, state_location=state_location, country_location=country_location)

    with open(f"pops/small_{int(1e5-big_size)}.ppl", "wb") as f:
        pickle.dump(popdict, f)