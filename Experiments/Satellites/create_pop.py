# import sys
# sys.path.append('../../../poem/covasim_webapp')
# sys.path.append('../../../poem/covasim_webapp/synthpops')

from covasim.tourist_layer import TourismParameters
import covasim as cv
import synthpops as sp
import pickle

# Настройки популяции
pop_size = 660000  # Размер популяции
location = "seattle_metro"  # Локация
state_location = "Washington"
country_location = "usa"

# Генерация популяции
popdict = sp.make_population(n=pop_size, location=location, state_location=state_location, country_location=country_location)

# Сохранение в файл .ppl
with open("pops/msk.ppl", "wb") as f:
    pickle.dump(popdict, f)