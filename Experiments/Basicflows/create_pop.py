from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle
import numpy as np

# Настройки популяции
location = "seattle_metro"  # Локация
state_location = "Washington"
country_location = "usa"

popdict = sp.make_population(n=1e5, location=location, state_location=state_location, country_location=country_location)
    
with open(f"pops/100k.ppl", "wb") as f:
     pickle.dump(popdict, f)